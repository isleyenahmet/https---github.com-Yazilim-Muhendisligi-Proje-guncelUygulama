"""
GNN-AIOps Backend — FastAPI + GAT Model + SQLite Streaming + RBAC Auth
Port: 5003
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import sqlite3
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jwt  # PyJWT
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from torch import nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "gat_aiops_model_final.pth"
DB_PATH = BASE_DIR / "aiops_stream.db"
USERS_PATH = BASE_DIR / "users.json"

# ─── Auth Config ──────────────────────────────────────────────────────────────
JWT_SECRET = "nexus-aiops-super-secret-2025"
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 8

# ─── Load Users ───────────────────────────────────────────────────────────────
def load_users() -> list:
    with open(USERS_PATH, encoding="utf-8") as f:
        return json.load(f)["users"]

# ─── GAT Model ────────────────────────────────────────────────────────────────
class GATAnomalyModel(nn.Module):
    """
    2-layer GAT:
      conv1: GATConv(in=4, out=64, heads=4)  → [N, 256]
      conv2: GATConv(in=256, out=64, heads=1) → [N, 64]
      lin  : Linear(64, 2)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(4, 64, heads=4, concat=True)
        self.conv2 = GATConv(256, 64, heads=1, concat=True)
        self.lin = nn.Linear(64, 2)

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


# ─── 4-node fully-connected edge index ────────────────────────────────────────
def build_edge_index(num_nodes: int = 4) -> torch.Tensor:
    """All directed edges among 4 nodes (excluding self-loops)."""
    rows, cols = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)
    return torch.tensor([rows, cols], dtype=torch.long)


EDGE_INDEX = build_edge_index(4)

# ─── Shared State ─────────────────────────────────────────────────────────────
state: dict = {}
stream_lock = threading.Lock()


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model
    model = GATAnomalyModel()
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    state["model"] = model
    state["cursor_id"] = 0          # sequential row pointer
    state["total_rows"] = 175341
    print(f"✅  GAT model loaded from {MODEL_PATH}")
    print(f"✅  DB: {DB_PATH}  ({state['total_rows']:,} rows)")
    yield
    # Cleanup (nothing heavy needed)


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="GNN-AIOps", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Column mapping ───────────────────────────────────────────────────────────
COLS = [
    "net_dur", "net_rate", "net_sbytes", "net_dbytes",   # Node 0: IT
    "iot_temp", "iot_torque", "iot_speed", "iot_wear",   # Node 1: IoT
    "hr_age", "hr_daily_rate", "hr_income", "hr_satisfaction",  # Node 2: HR
    "fin_amount", "fin_oldbal", "fin_newbal", "fin_destbal",     # Node 3: Fin
]

NODE_NAMES = ["IT", "IoT", "HR", "Finans"]


# ─── Auth Models ──────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str


# ─── /api/login ───────────────────────────────────────────────────────────────
@app.post("/api/login")
async def login(req: LoginRequest):
    users = load_users()
    user = next((u for u in users if u["username"] == req.username and u["password"] == req.password), None)
    if not user:
        raise HTTPException(status_code=401, detail="Kullanıcı adı veya şifre hatalı")

    payload = {
        "sub": user["username"],
        "name": user["name"],
        "role": user["role"],
        "initials": user["initials"],
        "email": user["email"],
        "department": user["department"],
        "pages": user["pages"],
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS)
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return {"token": token, "user": {k: v for k, v in user.items() if k != "password"}}


# ─── /api/me ──────────────────────────────────────────────────────────────────
@app.get("/api/me")
async def me(authorization: str = Header(None)):
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


# ─── /api/health ──────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "model": "GATAnomalyModel", "rows": state.get("total_rows", 0)}


# ─── /api/stream ──────────────────────────────────────────────────────────────
@app.get("/api/stream")
async def stream():
    model: GATAnomalyModel = state["model"]

    with stream_lock:
        row_id = state["cursor_id"]
        state["cursor_id"] = (row_id + 1) % state["total_rows"]

    # Fetch one row from SQLite
    conn = sqlite3.connect(str(DB_PATH))
    try:
        cur = conn.cursor()
        cur.execute(
            f"SELECT {', '.join(COLS)} FROM live_telemetry WHERE id = ?",
            (row_id,),
        )
        row = cur.fetchone()
        if row is None:
            # Fallback: random normalized row
            row = tuple(np.random.rand(16).tolist())
    finally:
        conn.close()

    # Build 4×4 feature tensor
    features = torch.tensor(row, dtype=torch.float32).view(4, 4)
    data = Data(x=features, edge_index=EDGE_INDEX)

    with torch.no_grad():
        logits, attn_weights = model(data)
        prob = F.softmax(logits, dim=1)
        anomaly = int(prob[0, 1] > 0.5)
        anomaly_score = float(prob[0, 1])

    # Attention weights: one scalar per edge → aggregate to per-node
    attn = attn_weights.squeeze(-1).numpy()  # shape: [12]
    # Map edges back to destination nodes → mean attention received
    node_attn = np.zeros(4)
    num_nodes = 4
    idx = 0
    for src in range(num_nodes):
        for dst in range(num_nodes):
            if src != dst:
                node_attn[dst] += attn[idx]
                idx += 1
    node_attn /= (num_nodes - 1)  # normalize

    row_dict = dict(zip(COLS, row))

    return JSONResponse({
        "row_id": row_id,
        "anomaly": anomaly,
        "anomaly_score": round(anomaly_score, 4),
        # Raw features per department
        "net_dur":    round(row_dict["net_dur"], 4),
        "net_rate":   round(row_dict["net_rate"], 4),
        "net_sbytes": round(row_dict["net_sbytes"], 4),
        "net_dbytes": round(row_dict["net_dbytes"], 4),
        "iot_temp":   round(row_dict["iot_temp"], 4),
        "iot_torque": round(row_dict["iot_torque"], 4),
        "iot_speed":  round(row_dict["iot_speed"], 4),
        "iot_wear":   round(row_dict["iot_wear"], 4),
        "hr_age":          round(row_dict["hr_age"], 4),
        "hr_daily_rate":   round(row_dict["hr_daily_rate"], 4),
        "hr_income":       round(row_dict["hr_income"], 4),
        "hr_satisfaction": round(row_dict["hr_satisfaction"], 4),
        "fin_amount":  round(row_dict["fin_amount"], 4),
        "fin_oldbal":  round(row_dict["fin_oldbal"], 4),
        "fin_newbal":  round(row_dict["fin_newbal"], 4),
        "fin_destbal": round(row_dict["fin_destbal"], 4),
        # Attention weights per node
        "attention_weights": [round(float(v), 4) for v in node_attn],
        # Department risk scores (0-100)
        "risk_scores": {
            "IT":     round(float(node_attn[0]) * 100, 1),
            "IoT":    round(float(node_attn[1]) * 100, 1),
            "HR":     round(float(node_attn[2]) * 100, 1),
            "Finans": round(float(node_attn[3]) * 100, 1),
        },
    })


# ─── Static Files (HTML Pages) ────────────────────────────────────────────────
@app.get("/")
async def root():
    return RedirectResponse(url="/login.html")

app.mount("/", StaticFiles(directory=str(BASE_DIR), html=True), name="static")


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=5003, reload=False)
