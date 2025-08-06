from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Dict, List
from datetime import datetime
from fastapi import Body

import os
import torch
import numpy as np

from database import SessionLocal, init_db
from models import User, PendingMessage
from NCA_model import NCAGenerator, get_config

# --------------------------- INIT ---------------------------

app = FastAPI()
init_db()

# -------------------------- DB ------------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

generator_model = None  # Global model instance


@app.get("/delete-db")
def delete_database():
    db_path = "/opt/render/project/src/users_v2.db"  # <- absolute path
    if os.path.exists(db_path):
        os.remove(db_path)
        return {"status": "success", "message": "Database deleted."}
    else:
        return {"status": "error", "message": f"Database file not found at {db_path}."}



# -------------------------- CORS ----------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- Model Loading on Startup ----------------

@app.on_event("startup")
def load_model_on_startup():
    global generator_model
    cfg = get_config()
    generator_model = NCAGenerator(
        channels=cfg["channels"],
        hidden=cfg["hidden"],
        steps=cfg["steps"],
        dropout=cfg["dropout"],
        length=cfg["length"]
    )
    generator_model.load_state_dict(torch.load("best_generator_g2.pt", map_location="cpu"))
    generator_model.eval()
    print("Model loaded ✅")

# ---------------------- RNG Endpoint ------------------------
@app.post("/generate-session-keys")
def generate_session_keys(from_phone: str, to_phone: str, db: Session = Depends(get_db)):
    global generator_model
    if generator_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    def get_256bit_hex():
        z = torch.randn(1, 128)
        out = generator_model(z)
        bits = (torch.sigmoid(out) > 0.5).int().squeeze().cpu().numpy()
        bits = bits[:256] if len(bits) > 256 else np.pad(bits, (0, 256 - len(bits)), mode='constant')
        return np.packbits(bits).tobytes().hex()

    # Generate 2 session keys
    K1 = get_256bit_hex()
    K2 = get_256bit_hex()

    # Compose messages
    sender_key = K1 + K2
    receiver_key = K2 + K1

    # Dispatch helper
    def dispatch(sender, receiver, key_data):
        msg = f"SESSION_KEY:{sender}:{key_data}"
        if receiver in manager.active_connections:
            asyncio.create_task(manager.send_personal_message(msg, receiver))
        else:
            pending = PendingMessage(
                sender_phone=sender,
                receiver_phone=receiver,
                message=msg,
                timestamp=datetime.utcnow()
            )
            db.add(pending)
            db.commit()

    # Send to both parties
    dispatch(from_phone, to_phone, receiver_key)  # receiver gets K2 + K1
    dispatch(to_phone, from_phone, sender_key)    # sender gets K1 + K2

    return {
        "sender_hex": sender_key,
        "receiver_hex": receiver_key,
        "status": "session keys sent or stored"
    }



@app.get("/random-256")
def generate_random():
    global generator_model
    if generator_model is None:
        return {"error": "Model not loaded"}

    with torch.no_grad():
        cfg = get_config()
        z = torch.randn(1, cfg["channels"], cfg["length"])
        output = generator_model(z)
        probs = torch.sigmoid(output)
        bits = (probs > 0.5).int().cpu().numpy().flatten()
        bits = bits[:256]  # Now this is truly 256 bits

        # bits = (probs > 0.5).int().squeeze().cpu().numpy()
        # bits = bits[:256] if len(bits) > 256 else np.pad(bits, (0, 256 - len(bits)), mode='constant')
        byte_array = np.packbits(bits)
        return {
            "random_bits": bits.tolist(),
            "random_hex": byte_array.tobytes().hex()
        }


# ----------------------- Request Schemas --------------------

class RegisterUserRequest(BaseModel):
    phone: str
    name: str
    publickey: str  # 🔧 Added this

class DeleteUserRequest(BaseModel):
    phone: str

# ------------------- WebSocket Manager ----------------------

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, phone: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[phone] = websocket

    def disconnect(self, phone: str):
        if phone in self.active_connections:
            del self.active_connections[phone]

    async def send_personal_message(self, message: str, receiver_phone: str):
        if receiver_phone in self.active_connections:
            await self.active_connections[receiver_phone].send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)

manager = ConnectionManager()

# ---------------------- REST APIs ---------------------------

@app.post("/register-user")
def register_user(request_data: RegisterUserRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.phone == request_data.phone).first()
    if user:
        return {"status": "exists"}
    new_user = User(
        phone=request_data.phone,
        name=request_data.name,
        publickey=request_data.publickey
    )
    db.add(new_user)
    db.commit()
    return {"status": "registered"}

@app.post("/delete-user")
def delete_user(request_data: DeleteUserRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.phone == request_data.phone).first()
    if user:
        db.delete(user)
        db.commit()
        return {"status": "deleted"}
    else:
        raise HTTPException(status_code=404, detail="User not found")

@app.get("/list-users")
def list_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return {"users": [{"phone": user.phone, "name": user.name} for user in users]}

@app.get("/get-user")
def get_user(phone: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.phone == phone).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"name": user.name, "phone": user.phone}

@app.get("/search-users")
def search_users(query: str = Query(...), exclude_phone: str = Query(...), db: Session = Depends(get_db)):
    results = db.query(User).filter(
        User.name.ilike(f"%{query}%"),
        User.phone != exclude_phone
    ).all()
    return {"users": [{"phone": user.phone, "name": user.name} for user in results]}

@app.get("/resolve-users")
def resolve_users(phones: List[str] = Query(...), db: Session = Depends(get_db)):
    users = db.query(User).filter(User.phone.in_(phones)).all()
    return {"users": [{"name": user.name, "phone": user.phone} for user in users]}

@app.get("/debug-pending/{receiver_phone}")
def get_pending_messages(receiver_phone: str, db: Session = Depends(get_db)):
    messages = db.query(PendingMessage).filter(PendingMessage.receiver_phone == receiver_phone).all()
    return [
        {"from": msg.sender_phone, "message": msg.message, "timestamp": msg.timestamp.isoformat()}
        for msg in messages
    ]

# ------------------ WebSocket Chat --------------------------

@app.websocket("/ws/{phone}")
async def websocket_endpoint(websocket: WebSocket, phone: str):
    await manager.connect(phone, websocket)
    db = SessionLocal()
    try:
        pending = db.query(PendingMessage).filter(PendingMessage.receiver_phone == phone).all()
        for msg in pending:
            await websocket.send_text(f"{msg.sender_phone}:{msg.message}")
            db.delete(msg)
        db.commit()

        while True:
            data = await websocket.receive_text()
            print(f"Message from {phone}: {data}")

            if ":" in data:
                receiver_phone, message = data.split(":", 1)
                if receiver_phone in manager.active_connections:
                    await manager.send_personal_message(f"{phone}:{message}", receiver_phone)
                else:
                    pending_msg = PendingMessage(
                        sender_phone=phone,
                        receiver_phone=receiver_phone,
                        message=message,
                        timestamp=datetime.utcnow()
                    )
                    db.add(pending_msg)
                    db.commit()

    except WebSocketDisconnect:
        manager.disconnect(phone)
        print(f"{phone} disconnected")
    finally:
        db.close()

# ---------------------- Healthcheck -------------------------

@app.get("/")
def root():
    return {"message": "Server running with SQLite, WebSocket, and Random Number Generator"}

