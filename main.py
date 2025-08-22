import os
import base64
import sqlite3
import torch
import numpy as np
import traceback
from datetime import datetime
from typing import Dict, List
import json

from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query, Form
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import SessionLocal, init_db, DATABASE_URL, engine
from models import Base, User, PendingMessage, SessionKey, PendingSession
from NCA_model import NCAGenerator, get_config

from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes, serialization

# --------------------------- INIT ---------------------------
app = FastAPI()

@app.on_event("startup")
def startup_event():
    global engine
    if DATABASE_URL.startswith("sqlite:////"):
        db_path = DATABASE_URL.replace("sqlite:////", "/", 1)
    elif DATABASE_URL.startswith("sqlite:///"):
        db_path = DATABASE_URL.replace("sqlite:///", "", 1)
    else:
        raise ValueError(f"Unsupported DATABASE_URL format: {DATABASE_URL}")

    parent_dir = os.path.dirname(db_path)
    os.makedirs(parent_dir, exist_ok=True)

    if not os.path.exists(db_path):
        open(db_path, "a").close()
        print(f"[INFO] Database file created at {db_path}")

    try:
        os.chmod(db_path, 0o666)
        print(f"[INFO] Database permissions set to 666")
    except PermissionError:
        print(f"[WARNING] Could not change permissions for {db_path}")

    Base.metadata.create_all(bind=engine)
    print("[INFO] Database tables created/verified.")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------------- CORS ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- Model Loading on Startup ----------------
generator_model = None

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

# ---------------------- Helper -------------------------------
def get_public_key_for(phone_number: str) -> str | None:
    db_path = DATABASE_URL.replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT publickey FROM users WHERE phone = ?", (phone_number,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

# ------------------- Request Schemas -------------------------
class RegisterUserRequest(BaseModel):
    phone: str
    name: str
    publickey: str

class DeleteUserRequest(BaseModel):
    phone: str

# ------------------- WebSocket Manager -----------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, phone: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[phone] = websocket

    def disconnect(self, phone: str):
        self.active_connections.pop(phone, None)
        
    async def send_session_keys(self, phone1: str, phone2: str, receiver: str, key1: bytes, key2: bytes, timestamp: datetime):
        """Send session keys to a connected client"""
        if receiver in self.active_connections:
            await self.active_connections[receiver].send_text(json.dumps({
                "type": "session_key",
                "phone1": phone1,
                "phone2": phone2,
                "receiver": receiver,
                "key1": base64.b64encode(key1).decode(),
                "key2": base64.b64encode(key2).decode(),
                "timestamp": timestamp.isoformat()
            }))
            print("sent to receiver", base64.b64encode(key1).decode(), base64.b64encode(key2).decode())
    async def send_personal_message(self, message_type: str, sender: str, receiver: str,
                                    message: str = "", keys: dict | None = None):
        payload = {
            "from": sender,
            "to": receiver,
            "type": message_type,
            "message": message,
            "keys": keys or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        if receiver in self.active_connections:
            await self.active_connections[receiver].send_text(json.dumps(payload))

    async def broadcast(self, message_type: str, sender: str, message: str, keys: dict | None = None):
        for receiver, conn in self.active_connections.items():
            payload = {
                "from": sender,
                "to": receiver,
                "type": message_type,
                "message": message,
                "keys": keys or {},
                "timestamp": datetime.utcnow().isoformat()
            }
            await conn.send_text(json.dumps(payload))

manager = ConnectionManager()

# ---------------------- Endpoints ----------------------------
@app.get("/where-is-db")
def find_database():
    candidates = []
    for root, dirs, files in os.walk("/"):
        for file in files:
            if file.endswith(".db"):
                candidates.append(os.path.join(root, file))
    return {"type": "info", "payload": {
        "found_db_files": candidates,
        "cwd": os.getcwd(),
        "expected_path": os.path.abspath(DATABASE_URL.replace("sqlite:///", ""))
    }}

@app.get("/delete-db")
def delete_database():
    db_path = "/data/users_v3.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        return {"type": "success", "payload": {"message": "Database deleted."}}
    return {"type": "error", "payload": {"message": f"Database file not found at {db_path}."}}

# ---------------------- Session Key / Random -----------------
@app.post("/generate-session-keys-test")
async def generate_session_keys_test(
    sender: str = Form(...),
    receiver: str = Form(...),
    db: Session = Depends(get_db)
):
    print(f"🔑 Session key request: sender={sender}, receiver={receiver}")
    try:
        phone1, phone2 = sorted([sender, receiver])

        existing = db.query(SessionKey).filter(
            SessionKey.phone1 == phone1,
            SessionKey.phone2 == phone2
        ).first()

        if existing:
            print("✅ Existing session found in DB")
            key1 = existing.key1
            key2 = existing.key2
            timestamp = existing.created_at
        else:
            print("🆕 No session found, generating new one...")

            key_bytes = (123).to_bytes(1, "big").rjust(32, b'\x00')

            raw_sender = db.query(User.publickey).filter(User.phone == sender).scalar()
            raw_receiver = db.query(User.publickey).filter(User.phone == receiver).scalar()
            if not raw_sender or not raw_receiver:
                raise HTTPException(status_code=404, detail="Sender or receiver public key not found")

            def load_pubkey(raw_base64: str):
                raw = "".join(raw_base64.strip().split())
                if raw_base64.strip().startswith("-----BEGIN"):
                    return serialization.load_pem_public_key(raw_base64.encode("utf-8"))
                decoded = base64.b64decode(raw)
                try:
                    return serialization.load_der_public_key(decoded)
                except Exception:
                    pem = (
                        b"-----BEGIN PUBLIC KEY-----\n"
                        + base64.encodebytes(decoded)
                        + b"-----END PUBLIC KEY-----\n"
                    )
                    return serialization.load_pem_public_key(pem)

            pub_sender = load_pubkey(raw_sender)
            pub_receiver = load_pubkey(raw_receiver)

            enc_for_sender = pub_sender.encrypt(
                key_bytes,
                padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
            )
            enc_for_receiver = pub_receiver.encrypt(
                key_bytes,
                padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
            )

            timestamp = datetime.utcnow()
            db.add(SessionKey(
                phone1=phone1,
                phone2=phone2,
                key1=enc_for_sender,
                key2=enc_for_receiver,
                created_at=timestamp
            ))
            db.commit()
            print("💾 Session stored in DB")

            key1, key2 = enc_for_sender, enc_for_receiver

        # Send to sender
        if sender in manager.active_connections:
            await manager.send_session_keys(
                phone1=phone1,
                phone2=phone2,
                receiver=sender,
                key1=key1,
                key2=key2,
                timestamp=timestamp
            )
            print(f"📤 Sent session key to {sender}")
        else:
            db.add(PendingSession(
                receiver_phone=sender,
                key1=key1,
                key2=key2,
                phone1=phone1,
                phone2=phone2,
                created_at=timestamp
            ))

        # Send to receiver
        if receiver in manager.active_connections:
            await manager.send_session_keys(
                phone1=phone1,
                phone2=phone2,
                receiver=receiver,
                key1=key1,
                key2=key2,
                timestamp=timestamp
            )
            print(f"📤 Sent session key to {receiver}")
        else:
            db.add(PendingSession(
                receiver_phone=receiver,
                key1=key1,
                key2=key2,
                phone1=phone1,
                phone2=phone2,
                created_at=timestamp
            ))

        db.commit()
        return {"type": "success"}

    except Exception as e:
        db.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/random-256")
def generate_random():
    global generator_model
    if generator_model is None:
        return {"type": "error", "payload": {"message": "Model not loaded"}}
    with torch.no_grad():
        cfg = get_config()
        z = torch.randn(1, cfg["channels"], cfg["length"])
        output = generator_model(z)
        probs = torch.sigmoid(output)
        bits = (probs > 0.5).int().cpu().numpy().flatten()[:256]


        byte_array = np.packbits(bits)
    return {"type": "random_bits", "payload": {"bits": bits.tolist(), "hex": byte_array.tobytes().hex()}}

# ------------------- User Management Endpoints ----------------
@app.post("/register-user")
def register_user(request: RegisterUserRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.phone == request.phone).first()
    if user:
        return {"type": "exists", "payload": {}}
    db.add(User(phone=request.phone, name=request.name, publickey=request.publickey))
    db.commit()
    return {"type": "registered", "payload": {}}

@app.post("/delete-user")
def delete_user(request: DeleteUserRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.phone == request.phone).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(user)
    db.commit()
    return {"type": "deleted", "payload": {}}

@app.get("/list-users")
def list_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return {"type": "list_users", "payload": [{"phone": u.phone, "name": u.name, "publickey": u.publickey} for u in users]}

@app.get("/get-user")
def get_user(phone: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.phone == phone).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"type": "get_user", "payload": {"name": user.name, "phone": user.phone}}

@app.get("/search-users")
def search_users(query: str, exclude_phone: str, db: Session = Depends(get_db)):
    results = db.query(User).filter(User.name.ilike(f"%{query}%"), User.phone != exclude_phone).all()
    return {"type": "search_users", "payload": [{"phone": u.phone, "name": u.name} for u in results]}

@app.get("/resolve-users")
def resolve_users(phones: List[str] = Query(...), db: Session = Depends(get_db)):
    users = db.query(User).filter(User.phone.in_(phones)).all()
    return {"type": "resolve_users", "payload": [{"name": u.name, "phone": u.phone} for u in users]}

@app.get("/clear-pending-messages")
def clear_pending_messages(db: Session = Depends(get_db)):
    try:
        num_deleted = db.query(PendingMessage).delete()
        db.commit()
        return {"type": "success", "payload": {"deleted_rows": num_deleted}}
    except Exception as e:
        db.rollback()
        return {"type": "error", "payload": {"message": str(e)}}

@app.get("/debug-pending/{receiver_phone}")
def get_pending_messages(receiver_phone: str, db: Session = Depends(get_db)):
    messages = db.query(PendingMessage).filter(PendingMessage.receiver_phone == receiver_phone).all()
    return {"type": "pending_messages", "payload": [{"from": m.sender_phone, "message": m.message, "timestamp": m.timestamp.isoformat()} for m in messages]}

# ------------------- WebSocket Endpoint ----------------------
@app.websocket("/ws/{phone}")
async def websocket_endpoint(websocket: WebSocket, phone: str):
    await manager.connect(phone, websocket)
    db = SessionLocal()
    try:
        # 🔹 Handle pending session keys for this user
        await handleSession(db, websocket, phone)

        # 🔹 Handle pending chat messages + new messages
        #await handleChatMessage(db, websocket, phone)

    except WebSocketDisconnect:
        manager.disconnect(phone)
        print(f"{phone} disconnected")
    finally:
        db.close()

async def handleSession(self, websocket: WebSocket, phone: str, db: Session):
    """
    Deliver all pending session keys for this phone and remove them from PendingSession.
    """
    try:
        pending = (
            db.query(PendingSession)
            .filter((PendingSession.phone1 == phone) | (PendingSession.phone2 == phone))
            .all()
        )

        for ps in pending:
            await self.send_session_keys(
                phone1=ps.phone1,
                phone2=ps.phone2,
                receiver=phone,   # current user receiving
                key1=ps.key1,
                key2=ps.key2,
                timestamp=ps.created_at,
            )
            
            db.delete(ps)
            db.commit()

    except Exception as e:
        print(f"❌ Error delivering session keys to {phone}: {e}")
        db.rollback()


@app.get("/")
def root():
    return {"type": "info", "payload": {"message": "Server running with SQLite, WebSocket, and Random Number Generator"}}
