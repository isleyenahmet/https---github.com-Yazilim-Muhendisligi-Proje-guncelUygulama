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
from pydantic import BaseModel
from torch import nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "gigafactory_gat_model.pth"
DB_PATH = BASE_DIR / "gigafactory_live.db"
USERS_PATH = BASE_DIR / "users.json"

# ─── Auth Config ──────────────────────────────────────────────────────────────
JWT_SECRET = "nexus-aiops-super-secret-2025"
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 8

# ─── Database & User Management ──────────────────────────────────────────────
def get_db_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Kullanıcı tablosunu oluşturur (eğer yoksa)."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT,
                name TEXT,
                initials TEXT,
                email TEXT,
                department TEXT,
                role TEXT,
                pages TEXT
            )
        """)
        conn.commit()
    finally:
        conn.close()

def import_users_from_json():
    """users.json içeriğini SQLite veritabanına aktarır (sadece eksikleri)."""
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
            # pages listesi virgülle ayrılmış TEXT'e çevrilir
            pages_str = ",".join(u["pages"]) if isinstance(u["pages"], list) else u["pages"]
            
            cur.execute("""
                INSERT OR IGNORE INTO users (username, password, name, initials, email, department, role, pages)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (u["username"], u["password"], u["name"], u["initials"], u["email"], u["department"], u["role"], pages_str))
            
        conn.commit()
        print(f"[OK] {len(users)} kullanıcı SQLite veritabanına aktarıldı (varsa atlandı).")
    except Exception as e:
        print(f"[ERROR] Kullanıcılar içe aktarılırken hata: {e}")
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
        x, (edge_idx1, attn1) = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        # conv2 — captures per-edge attention scalars
        x, (edge_idx2, attn2) = self.conv2(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        # Global mean pooling
        out = x.mean(dim=0, keepdim=True)
        logits = self.lin(out)
        return logits, attn2  # attn2: [num_edges, 1]

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
app = FastAPI(title="GNN-AIOps", version="1.0.0", lifespan=lifespan)

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
NODE_NAMES = ["IT", "IoT", "Finans", "Lojistik"]

# ─── Auth Models ──────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str

class ProfileUpdateRequest(BaseModel):
    name: str
    email: str
    initials: str

class ChangePasswordRequest(BaseModel):
    oldPassword: str
    newPassword: str

class CreateUserRequest(BaseModel):
    username: str
    password: str
    name: str
    email: str
    department: str
    role: str
    initials: str
    pages: str

class EmergencyAuthRequest(BaseModel):
    reason: str

# ─── Dependency ───────────────────────────────────────────────────────────────
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
        # pages string'ini listeye çevir
        pages_list = user["pages"].split(",") if user["pages"] else []
        
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
        
        # password'ü çıkararak kullanıcı bilgisini dön
        clean_user = {k: v for k, v in user.items() if k != "password"}
        clean_user["pages"] = pages_list
        
        return {"token": token, "user": clean_user}
    finally:
        conn.close()

@app.get("/api/me")
async def me(user=Depends(get_current_user)):
    return user

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
        return {"status": "success", "message": "Profil başarıyla güncellendi."}
    finally:
        conn.close()

@app.post("/api/change-password")
async def change_password(req: ChangePasswordRequest, current_user=Depends(get_current_user)):
    username = current_user["sub"]
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        # Eski şifreyi kontrol et
        cur.execute("SELECT password FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        if not row or row["password"] != req.oldPassword:
            raise HTTPException(status_code=400, detail="Mevcut şifre hatalı")
        
        # Yeni şifreyi güncelle
        cur.execute("UPDATE users SET password = ? WHERE username = ?", (req.newPassword, username))
        conn.commit()
        return {"status": "success", "message": "Şifre başarıyla değiştirildi."}
    finally:
        conn.close()

# ─── Admin User Management Endpoints ──────────────────────────────────────────
@app.get("/api/admin/users")
async def list_admin_users(current_user=Depends(get_current_user)):
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Yetkisiz erişim")
    
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users ORDER BY name ASC")
        rows = cur.fetchall()
        
        users = []
        for row in rows:
            u = dict(row)
            u["pages"] = u["pages"].split(",") if u["pages"] else []
            # Şifreyi güvenlik için listede göstermiyoruz
            u.pop("password")
            users.append(u)
        return users
    finally:
        conn.close()

@app.delete("/api/admin/users/{username}")
async def delete_user(username: str, current_user=Depends(get_current_user)):
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Yetkisiz erişim")
    
    if current_user["sub"] == username:
        raise HTTPException(status_code=400, detail="Kendinizi silemezsiniz")
    
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.commit()
        return {"status": "success", "message": f"{username} kullanıcısı silindi."}
    finally:
        conn.close()

@app.post("/api/users")
async def create_user(req: CreateUserRequest, current_user=Depends(get_current_user)):
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Yetkisiz erişim")
    
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        # Kullanıcı zaten var mı?
        cur.execute("SELECT username FROM users WHERE username = ?", (req.username,))
        if cur.fetchone():
            raise HTTPException(status_code=400, detail="Bu kullanıcı adı zaten alınmış")
        
        cur.execute("""
            INSERT INTO users (username, password, name, initials, email, department, role, pages)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (req.username, req.password, req.name, req.initials, req.email, req.department, req.role, req.pages))
        conn.commit()
        return {"status": "success", "message": "Yeni kullanıcı oluşturuldu."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.get("/api/users")
async def get_users_list(current_user=Depends(get_current_user)):
    # Bu endpoint hem admin hem de diğer kullanıcılar için (raporlama vb. için)
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT username, name, email, department, initials FROM users")
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

# ─── Emergency Authorization ───────────────────────────────────────────────────
@app.post("/api/emergency-auth")
async def emergency_auth(req: EmergencyAuthRequest, current_user=Depends(get_current_user)):
    # Simülasyon: Acil yetki talebini günlüğe kaydet
    log_msg = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] EMERGENCY AUTH REQUEST: User={current_user['sub']}, Reason={req.reason}"
    print(log_msg)
    
    # İsteğe bağlı olarak bir log dosyasına da yazabiliriz
    try:
        with open("emergency_auth.log", "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")
    except:
        pass

    return {
        "status": "success", 
        "message": "Acil yetki talebiniz sistem yöneticisine iletildi ve loglandı."
    }

# ─── Other Endpoints ───────────────────────────────────────────────────────────
class SendReportRequest(BaseModel):
    emails: list[str]
    interval: str
    departments: list[str]

@app.post("/api/send-report")
async def send_report(req: SendReportRequest, current_user=Depends(get_current_user)):
    print(f"\n[EMAIL SIMULATION] Sending report to: {', '.join(req.emails)}")
    print(f"[EMAIL SIMULATION] From: {current_user['sub']}")
    print(f"[EMAIL SIMULATION] Status: Success\n")
    return {"status": "success", "message": f"Rapor {len(req.emails)} kişiye başarıyla gönderildi."}

@app.get("/api/health")
async def health():
    return {"status": "ok", "model": "GATAnomalyModel", "rows": state.get("total_rows", 0)}

@app.get("/api/stream")
async def stream():
    model: GATAnomalyModel = state["model"]
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
            "HR": round(float(node_attn[3]) * 100, 1),
        },
    })

# ─── Static Files ─────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return RedirectResponse(url="/login.html")

app.mount("/", StaticFiles(directory=str(BASE_DIR), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=5003, reload=False)
