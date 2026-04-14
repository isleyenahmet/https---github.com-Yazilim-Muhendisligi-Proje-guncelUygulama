"""
GNN-AIOps Backend — FastAPI + GAT Model + SQLite Streaming + RBAC Auth
Port: 5003
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import time
from pathlib import Path
from typing import List, Optional

import jwt  # PyJWT
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
from torch import nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "gigafactory_gat_model.pth"
DB_PATH = BASE_DIR / "gigafactory_live.db"
USERS_PATH = BASE_DIR / "users.json"

# ─── Auth Config ──────────────────────────────────────────────────────────────
JWT_SECRET = "asgard-aiops-super-secret-2025"
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 8

# ─── Database & User Management ──────────────────────────────────────────────
def get_db_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Veritabanı tablolarını oluşturur."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        # Users Tablosu
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT NOT NULL,
                name TEXT NOT NULL,
                initials TEXT NOT NULL,
                email TEXT NOT NULL,
                department TEXT NOT NULL,
                role TEXT NOT NULL,
                pages TEXT NOT NULL
            )
        """)
        # User Settings Tablosu
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_settings (
                username TEXT PRIMARY KEY,
                refresh_rate TEXT DEFAULT '1',
                notifications INTEGER DEFAULT 1,
                sound_alerts INTEGER DEFAULT 0,
                shap_enabled INTEGER DEFAULT 1,
                lime_enabled INTEGER DEFAULT 1,
                data_retention TEXT DEFAULT '30',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
            )
        """)
        # Access Requests Tablosu
        cur.execute("""
            CREATE TABLE IF NOT EXISTS access_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                requested_page TEXT NOT NULL,
                reason TEXT NOT NULL,
                level TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP,
                expires_at TIMESTAMP,
                FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
            )
        """)
        conn.commit()
    finally:
        conn.close()

def generate_initials(name: str) -> str:
    """İsimden baş harfleri oluşturur."""
    parts = name.split()
    if not parts:
        return ""
    initials = "".join([p[0].upper() for p in parts])
    return initials

def import_users_from_json():
    """users.json içeriğini SQLite veritabanına aktarır ve varsayılan ayarları ekler."""
    if not USERS_PATH.exists():
        print(f"[WARNING] {USERS_PATH} bulunamadı, içe aktarma atlanıyor.")
        return

    try:
        with open(USERS_PATH, encoding="utf-8") as f:
            data = json.load(f)
            users = data.get("users", [])
            
        conn = get_db_connection()
        cur = conn.cursor()
        
        for u in users:
            pages_str = ",".join(u["pages"]) if isinstance(u["pages"], list) else u["pages"]
            
            # Kullanıcıyı ekle
            cur.execute("""
                INSERT OR IGNORE INTO users (username, password, name, initials, email, department, role, pages)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (u["username"], u["password"], u["name"], u.get("initials", generate_initials(u["name"])), 
                  u["email"], u["department"], u["role"], pages_str))
            
            # Varsayılan ayarları ekle
            cur.execute("""
                INSERT OR IGNORE INTO user_settings (username) VALUES (?)
            """, (u["username"],))
            
        conn.commit()
        print(f"[OK] Kullanıcılar ve ayarlar SQLite veritabanına aktarıldı.")
    except Exception as e:
        print(f"[ERROR] Kullanıcılar içe aktarılırken hata: {e}")
    finally:
        conn.close()
def save_users_to_json():
    """SQLite'daki kullanıcıları users.json dosyasına senkronize eder."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT username, password, name, initials, email, department, role, pages FROM users")
        rows = cur.fetchall()
        
        users_list = []
        for r in rows:
            u = dict(r)
            # pages string'den listeye çevrilmeli
            u["pages"] = u["pages"].split(",") if u["pages"] else []
            users_list.append(u)
            
        with open(USERS_PATH, "w", encoding="utf-8") as f:
            json.dump({"users": users_list}, f, indent=2, ensure_ascii=False)
        print(f"[OK] {USERS_PATH} güncellendi.")
    except Exception as e:
        print(f"[ERROR] users.json kaydedilirken hata: {e}")
    finally:
        conn.close()

# ─── GAT Model ────────────────────────────────────────────────────────────────
class GATAnomalyModel(nn.Module):
    """
    2-layer GAT (Gigafactory):
      conv1: GATConv(in=4, out=32, heads=4)  → [N, 128]
      conv2: GATConv(in=128, out=32, heads=1) → [N, 32]
      lin  : Linear(32, 2)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(4, 32, heads=4, concat=True)
        self.conv2 = GATConv(128, 32, heads=1, concat=True)
        self.lin = nn.Linear(32, 2)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        # conv1
        x, _ = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        # conv2
        x, (edge_idx, attn) = self.conv2(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        # Global mean pooling
        out = x.mean(dim=0, keepdim=True)
        logits = self.lin(out)
        return logits, attn

def build_edge_index(num_nodes: int = 4) -> torch.Tensor:
    rows, cols = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)
    return torch.tensor([rows, cols], dtype=torch.long)

EDGE_INDEX = build_edge_index(4)
state: dict = {}

# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # DB Init & User Import
    init_db()
    import_users_from_json()

    # Load model
    model = GATAnomalyModel()
    if MODEL_PATH.exists():
        state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        print(f"[OK] GAT model loaded from {MODEL_PATH}")
    else:
        print(f"[WARNING] Model file {MODEL_PATH} not found!")

    model.eval()
    state["model"] = model
    state["total_rows"] = 500000
    
    print(f"[OK] DB: {DB_PATH}  ({state['total_rows']:,} rows)")
    yield

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="GNN-AIOps", version="1.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

COLS = [
    "net_lat", "net_thr", "net_p_loss", "net_sec",        # Node 0: IT/Network
    "iot_vib", "iot_temp", "iot_torq", "iot_cycle",       # Node 1: IoT
    "fin_cost", "fin_fraud", "fin_risk", "fin_inv",        # Node 2: Finans
    "log_path", "log_coll", "log_soc", "log_task",         # Node 3: Lojistik
]

# ─── Pydantic Models ──────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str

class ProfileUpdateRequest(BaseModel):
    name: str
    email: str
    initials: str

class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str

class SettingsUpdateRequest(BaseModel):
    refresh_rate: Optional[str] = None
    notifications: Optional[bool] = None
    sound_alerts: Optional[bool] = None
    shap_enabled: Optional[bool] = None
    lime_enabled: Optional[bool] = None
    data_retention: Optional[str] = None

class UserAddRequest(BaseModel):
    username: str
    name: str
    email: str
    password: str
    role: str
    department: str
    pages: List[str]

class UserUpdateRequest(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    department: Optional[str] = None
    pages: Optional[List[str]] = None

class EmergencyAuthRequest(BaseModel):
    requested_page: str
    reason: str
    level: str

class AccessRequestResolve(BaseModel):
    request_id: int
    status: str  # 'approved' or 'rejected'

# ─── Dependencies ─────────────────────────────────────────────────────────────
async def get_current_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token gerekli")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token süresi doldu")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Geçersiz token")

async def admin_required(current_user=Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Yetkisiz erişim (Admin yetkisi gerekli)")
    return current_user

# ─── Auth Endpoints ───────────────────────────────────────────────────────────
@app.post("/api/login")
async def login(req: LoginRequest):
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username = ? AND password = ?", (req.username, req.password))
        row = cur.fetchone()
        
        if not row:
            raise HTTPException(status_code=401, detail="Kullanıcı adı veya şifre hatalı")
        
        user = dict(row)
        pages_list = user["pages"].split(",") if user["pages"] else []
        
        # Geçici yetkileri kontrol et
        cur.execute("""
            SELECT requested_page FROM access_requests 
            WHERE username = ? AND status = 'approved' AND expires_at > CURRENT_TIMESTAMP
        """, (user["username"],))
        temp_rows = cur.fetchall()
        for tr in temp_rows:
            if tr["requested_page"] not in pages_list:
                pages_list.append(tr["requested_page"])
        
        payload = {
            "sub": user["username"],
            "name": user["name"],
            "role": user["role"],
            "initials": user["initials"],
            "email": user["email"],
            "department": user["department"],
            "pages": pages_list,
            "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS)
        }
        token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
        
        clean_user = {k: v for k, v in user.items() if k != "password"}
        clean_user["pages"] = pages_list
        
        return {"token": token, "user": clean_user}
    finally:
        conn.close()

@app.get("/api/me")
async def me(user=Depends(get_current_user)):
    # JWT'den gelen bilgiler güncel olmayabilir (geçici yetki onaylanmış olabilir)
    # Bu yüzden DB'den tekrar kontrol ediyoruz
    username = user["sub"]
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        if not row: return user # Fallback to token payload
        
        db_user = dict(row)
        pages_list = db_user["pages"].split(",") if db_user["pages"] else []
        
        # Geçici yetkileri kontrol et
        cur.execute("""
            SELECT requested_page FROM access_requests 
            WHERE username = ? AND status = 'approved' AND expires_at > CURRENT_TIMESTAMP
        """, (username,))
        temp_rows = cur.fetchall()
        for tr in temp_rows:
            if tr["requested_page"] not in pages_list:
                pages_list.append(tr["requested_page"])
        
        # Token formatına uygun hale getir
        clean_user = {
            "sub": db_user["username"],
            "name": db_user["name"],
            "role": db_user["role"],
            "initials": db_user["initials"],
            "email": db_user["email"],
            "department": db_user["department"],
            "pages": pages_list
        }
        return clean_user
    finally:
        conn.close()

# ─── Profile Endpoints ────────────────────────────────────────────────────────
@app.post("/api/profile/update")
async def update_profile(req: ProfileUpdateRequest, current_user=Depends(get_current_user)):
    username = current_user["sub"]
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            UPDATE users SET name = ?, email = ?, initials = ? WHERE username = ?
        """, (req.name, req.email, req.initials, username))
        conn.commit()
        save_users_to_json()
        return {"status": "success", "message": "Profil başarıyla güncellendi."}
    finally:
        conn.close()

@app.post("/api/change-password")
async def change_password(req: ChangePasswordRequest, current_user=Depends(get_current_user)):
    username = current_user["sub"]
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT password FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        if not row or row["password"] != req.old_password:
            raise HTTPException(status_code=400, detail="Mevcut şifre hatalı")
        
        cur.execute("UPDATE users SET password = ? WHERE username = ?", (req.new_password, username))
        conn.commit()
        save_users_to_json()
        return {"status": "success", "message": "Şifre başarıyla değiştirildi."}
    finally:
        conn.close()

# ─── Model Info Endpoint ──────────────────────────────────────────────────────
@app.get("/api/model/info")
async def get_model_info(current_user=Depends(get_current_user)):
    return {
        "name": "Graph Attention Network (GAT)",
        "short": "GAT",
        "architecture": "2-layer GAT with 4 attention heads (Gigafactory)",
        "total_params": 12500,
        "nodes": 4,
        "hidden_dim": 32,
        "total_rows": state.get("total_rows", 500000)
    }

# ─── User Settings Endpoints ──────────────────────────────────────────────────
@app.get("/api/settings")
async def get_settings(current_user=Depends(get_current_user)):
    username = current_user["sub"]
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM user_settings WHERE username = ?", (username,))
        row = cur.fetchone()
        if not row:
            # Eğer kayıt yoksa oluştur
            cur.execute("INSERT INTO user_settings (username) VALUES (?)", (username,))
            conn.commit()
            cur.execute("SELECT * FROM user_settings WHERE username = ?", (username,))
            row = cur.fetchone()
        
        s = dict(row)
        return {
            "refresh_rate": s["refresh_rate"],
            "notifications": bool(s["notifications"]),
            "sound_alerts": bool(s["sound_alerts"]),
            "shap_enabled": bool(s["shap_enabled"]),
            "lime_enabled": bool(s["lime_enabled"]),
            "data_retention": s["data_retention"]
        }
    finally:
        conn.close()

@app.post("/api/settings")
async def update_settings(req: SettingsUpdateRequest, current_user=Depends(get_current_user)):
    username = current_user["sub"]
    fields = []
    values = []
    
    if req.refresh_rate is not None:
        fields.append("refresh_rate = ?")
        values.append(req.refresh_rate)
    if req.notifications is not None:
        fields.append("notifications = ?")
        values.append(1 if req.notifications else 0)
    if req.sound_alerts is not None:
        fields.append("sound_alerts = ?")
        values.append(1 if req.sound_alerts else 0)
    if req.shap_enabled is not None:
        fields.append("shap_enabled = ?")
        values.append(1 if req.shap_enabled else 0)
    if req.lime_enabled is not None:
        fields.append("lime_enabled = ?")
        values.append(1 if req.lime_enabled else 0)
    if req.data_retention is not None:
        fields.append("data_retention = ?")
        values.append(req.data_retention)
        
    if not fields:
        raise HTTPException(status_code=400, detail="Güncellenecek veri gönderilmedi")
        
    values.append(username)
    query = f"UPDATE user_settings SET {', '.join(fields)}, updated_at = CURRENT_TIMESTAMP WHERE username = ?"
    
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(query, tuple(values))
        conn.commit()
        return {"message": "Ayarlar başarıyla kaydedildi"}
    finally:
        conn.close()

@app.post("/api/settings/reset")
async def reset_settings(current_user=Depends(get_current_user)):
    username = current_user["sub"]
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            UPDATE user_settings SET 
                refresh_rate = '1',
                notifications = 1,
                sound_alerts = 0,
                shap_enabled = 1,
                lime_enabled = 1,
                data_retention = '30',
                updated_at = CURRENT_TIMESTAMP
            WHERE username = ?
        """, (username,))
        conn.commit()
        return {"message": "Ayarlar varsayılan değerlere sıfırlandı"}
    finally:
        conn.close()

# ─── Access Management Endpoints (Admin Only) ─────────────────────────────────
@app.get("/api/admin/users")
@app.get("/api/users")
async def list_users(current_user=Depends(admin_required)):
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT username, name, email, department, role, pages, initials FROM users")
        rows = cur.fetchall()
        users = []
        for r in rows:
            u = dict(r)
            u["pages"] = u["pages"].split(",") if u["pages"] else []
            users.append(u)
        return users
    finally:
        conn.close()

@app.post("/api/users/add")
@app.post("/api/users")
async def add_user(req: UserAddRequest, current_user=Depends(admin_required)):
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT username FROM users WHERE username = ?", (req.username,))
        if cur.fetchone():
            raise HTTPException(status_code=400, detail="Bu kullanıcı adı zaten mevcut")
        
        initials = generate_initials(req.name)
        pages_str = ",".join(req.pages)
        
        cur.execute("""
            INSERT INTO users (username, password, name, initials, email, department, role, pages)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (req.username, req.password, req.name, initials, req.email, req.department, req.role, pages_str))
        
        # Varsayılan ayarları oluştur
        cur.execute("INSERT INTO user_settings (username) VALUES (?)", (req.username,))
        
        conn.commit()
        save_users_to_json()
        return {"message": "Kullanıcı başarıyla eklendi", "username": req.username}
    finally:
        conn.close()

@app.put("/api/users/update/{username}")
async def update_user(username: str, req: UserUpdateRequest, current_user=Depends(admin_required)):
    fields = []
    values = []
    
    if req.name is not None:
        fields.append("name = ?")
        values.append(req.name)
        # İsmi güncellenirse initials'ı da güncelle
        fields.append("initials = ?")
        values.append(generate_initials(req.name))
    if req.role is not None:
        fields.append("role = ?")
        values.append(req.role)
    if req.department is not None:
        fields.append("department = ?")
        values.append(req.department)
    if req.pages is not None:
        fields.append("pages = ?")
        values.append(",".join(req.pages))
        
    if not fields:
        raise HTTPException(status_code=400, detail="Güncellenecek veri gönderilmedi")
        
    values.append(username)
    query = f"UPDATE users SET {', '.join(fields)} WHERE username = ?"
    
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(query, tuple(values))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Kullanıcı bulunamadı")
        conn.commit()
        save_users_to_json()
        return {"message": "Kullanıcı bilgileri güncellendi"}
    finally:
        conn.close()

@app.delete("/api/admin/users/{username}")
@app.delete("/api/users/delete/{username}")
async def delete_user(username: str, current_user=Depends(admin_required)):
    if current_user["sub"] == username:
        raise HTTPException(status_code=400, detail="Kendinizi silemezsiniz")
    
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM users WHERE username = ?", (username,))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Kullanıcı bulunamadı")
        conn.commit()
        save_users_to_json()
        return {"message": f"{username} kullanıcısı silindi"}
    finally:
        conn.close()

@app.get("/api/stats/access")
async def get_access_stats(current_user=Depends(admin_required)):
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        
        # Toplam kullanıcı
        cur.execute("SELECT COUNT(*) FROM users")
        total_users = cur.fetchone()[0]
        
        # Toplam rol (benzersiz rollerin sayısı)
        cur.execute("SELECT COUNT(DISTINCT role) FROM users")
        total_roles = cur.fetchone()[0]
        
        # Toplam izin (tüm kullanıcıların toplam sayfa yetkisi sayısı olarak simüle edelim)
        cur.execute("SELECT pages FROM users")
        rows = cur.fetchall()
        total_permissions = sum([len(r[0].split(",")) if r[0] else 0 for r in rows])
        
        return {
            "total_users": total_users,
            "total_roles": total_roles,
            "total_permissions": total_permissions,
            "last_update": datetime.now(timezone.utc).isoformat()
        }
    finally:
        conn.close()

# ─── Emergency Authorization ───────────────────────────────────────────────────
# ─── Emergency Authorization ───────────────────────────────────────────────────
@app.post("/api/emergency/request")
async def emergency_request(req: EmergencyAuthRequest, current_user=Depends(get_current_user)):
    username = current_user["sub"]
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO access_requests (username, requested_page, reason, level)
            VALUES (?, ?, ?, ?)
        """, (username, req.requested_page, req.reason, req.level))
        conn.commit()
        
        # Log to file as well for security audit
        log_msg = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ACCESS REQUEST: User={username}, Page={req.requested_page}, Level={req.level}"
        try:
            with open("access_requests.log", "a", encoding="utf-8") as f:
                f.write(log_msg + "\n")
        except: pass
        
        return {"status": "success", "message": "Acil yetki talebiniz iletildi. Admin onayı bekleniyor."}
    finally:
        conn.close()

@app.get("/api/admin/requests/pending")
async def list_pending_requests(current_user=Depends(admin_required)):
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT r.*, u.name as user_display_name 
            FROM access_requests r
            JOIN users u ON r.username = u.username
            WHERE r.status = 'pending'
            ORDER BY r.created_at DESC
        """)
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

@app.post("/api/admin/requests/resolve")
async def resolve_request(req: AccessRequestResolve, current_user=Depends(admin_required)):
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM access_requests WHERE id = ?", (req.request_id,))
        request = cur.fetchone()
        if not request:
            raise HTTPException(status_code=404, detail="Talep bulunamadı")
        
        if req.status == 'approved':
            # Süre hesapla
            hours = 2
            if request["level"] == 'elevated': hours = 6
            elif request["level"] == 'critical': hours = 24
            
            expires_at = (datetime.now() + timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M:%S')
            
            cur.execute("""
                UPDATE access_requests 
                SET status = 'approved', resolved_at = CURRENT_TIMESTAMP, expires_at = ? 
                WHERE id = ?
            """, (expires_at, req.request_id))
            conn.commit()
            return {"message": f"Talep onaylandı. {hours} saatlik erişim tanımlandı."}
        else:
            cur.execute("""
                UPDATE access_requests 
                SET status = 'rejected', resolved_at = CURRENT_TIMESTAMP 
                WHERE id = ?
            """, (req.request_id,))
            conn.commit()
            return {"message": "Talep reddedildi."}
    finally:
        conn.close()

@app.get("/api/admin/notifications/count")
async def get_notif_count(current_user=Depends(admin_required)):
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM access_requests WHERE status = 'pending'")
        count = cur.fetchone()[0]
        return {"count": count}
    finally:
        conn.close()

# ─── System Health ────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "model": "GATAnomalyModel", "rows": state.get("total_rows", 0)}

# ─── Stream Endpoint ──────────────────────────────────────────────────────────
@app.get("/api/stream")
async def stream():
    model = state["model"]
    current_time_seconds = int(time.time())
    row_id = (current_time_seconds % state.get("total_rows", 500000)) + 1

    conn = sqlite3.connect(str(DB_PATH))
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT {', '.join(COLS)} FROM live_telemetry WHERE ROWID = ?", (row_id,))
        row = cur.fetchone()
        if row is None:
            np.random.seed(row_id)
            row = tuple(np.random.rand(16).tolist())
    finally:
        conn.close()

    features = torch.tensor(row, dtype=torch.float32).view(4, 4)
    data = Data(x=features, edge_index=EDGE_INDEX)

    with torch.no_grad():
        logits, attn_weights = model(data)
        prob = F.softmax(logits, dim=1)
        anomaly = int(prob[0, 1] > 0.5)
        anomaly_score = float(prob[0, 1])

    attn = attn_weights.squeeze(-1).numpy()
    node_attn = np.zeros(4)
    idx = 0
    for src in range(4):
        for dst in range(4):
            if src != dst:
                node_attn[dst] += attn[idx]
                idx += 1
    node_attn /= 3

    row_dict = dict(zip(COLS, row))

    return JSONResponse({
        "row_id": row_id,
        "anomaly": anomaly,
        "anomaly_score": round(anomaly_score, 4),
        "net_lat": round(row_dict["net_lat"], 4), "net_thr": round(row_dict["net_thr"], 4),
        "net_p_loss": round(row_dict["net_p_loss"], 4), "net_sec": round(row_dict["net_sec"], 4),
        "iot_vib": round(row_dict["iot_vib"], 4), "iot_temp": round(row_dict["iot_temp"], 4),
        "iot_torq": round(row_dict["iot_torq"], 4), "iot_cycle": round(row_dict["iot_cycle"], 4),
        "fin_cost": round(row_dict["fin_cost"], 4), "fin_fraud": round(row_dict["fin_fraud"], 4),
        "fin_risk": round(row_dict["fin_risk"], 4), "fin_inv": round(row_dict["fin_inv"], 4),
        "log_path": round(row_dict["log_path"], 4), "log_coll": round(row_dict["log_coll"], 4),
        "log_soc": round(row_dict["log_soc"], 4), "log_task": round(row_dict["log_task"], 4),
        "attention_weights": [round(float(v), 4) for v in node_attn],
        "risk_scores": {
            "IT": round(float(node_attn[0]) * 100, 1),
            "IoT": round(float(node_attn[1]) * 100, 1),
            "Finans": round(float(node_attn[2]) * 100, 1),
            "Logistics": round(float(node_attn[3]) * 100, 1),
        },
    })

# ─── Static Files & Root ──────────────────────────────────────────────────────
@app.get("/")
async def root():
    return RedirectResponse(url="/login.html")

app.mount("/", StaticFiles(directory=str(BASE_DIR), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=5003, reload=False)
