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
from models import Base, User, PendingMessage, SessionKey
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
# @app.post("/generate-session-keys-test")
# async def generate_session_keys_test(
#     sender: str = Form(...),
#     receiver: str = Form(...),
#     db: Session = Depends(get_db)
# ):
#     try:
#         global generator_model
#         if generator_model is None:
#             raise HTTPException(status_code=500, detail="Generator model not loaded")

#         user1, user2 = sorted([sender, receiver])

#         existing = db.query(SessionKey).filter(
#             SessionKey.user1 == user1,
#             SessionKey.user2 == user2
#         ).first()
#         if existing:
#             enc_for_sender = existing.key_sender_to_receiver
#             enc_for_receiver = existing.key_receiver_to_sender
#         else:
#             key_bytes = (123).to_bytes(1, "big").rjust(32, b'\x00')

#             raw_sender = db.query(User.publickey).filter(User.phone == sender).scalar()
#             raw_receiver = db.query(User.publickey).filter(User.phone == receiver).scalar()
#             if not raw_sender or not raw_receiver:
#                 raise HTTPException(status_code=404, detail="Sender or receiver public key not found")

#             def load_pubkey(raw_base64: str):
#                 try:
#                     raw = "".join(raw_base64.strip().split())
#                     if raw_base64.strip().startswith("-----BEGIN"):
#                         return serialization.load_pem_public_key(raw_base64.encode("utf-8"))
#                     decoded = base64.b64decode(raw)
#                     try:
#                         return serialization.load_der_public_key(decoded)
#                     except Exception:
#                         pem = b"-----BEGIN PUBLIC KEY-----\n" + base64.encodebytes(decoded) + b"-----END PUBLIC KEY-----\n"
#                         return serialization.load_pem_public_key(pem)
#                 except Exception as e:
#                     raise ValueError(f"Failed to load public key: {e}")

#             pub_sender = load_pubkey(raw_sender)
#             pub_receiver = load_pubkey(raw_receiver)

#             enc_for_sender = pub_sender.encrypt(
#                 key_bytes,
#                 padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
#             )
#             enc_for_receiver = pub_receiver.encrypt(
#                 key_bytes,
#                 padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
#             )

#             db.add(SessionKey(
#                 user1=user1,
#                 user2=user2,
#                 key_sender_to_receiver=enc_for_sender,
#                 key_receiver_to_sender=enc_for_receiver,
#                 created_at=datetime.utcnow()
#             ))
#             db.commit()

#         DELIM = "||"
#         payload_for_sender = f"SESSION_KEY:{base64.b64encode(enc_for_sender).decode()}{DELIM}{base64.b64encode(enc_for_receiver).decode()}"
#         payload_for_receiver = f"SESSION_KEY:{base64.b64encode(enc_for_receiver).decode()}{DELIM}{base64.b64encode(enc_for_sender).decode()}"

#         if sender in manager.active_connections:
#             await manager.send_personal_message("SESSION_KEY", sender, sender, payload_for_sender, keys={"sender_encrypted": base64.b64encode(enc_for_sender).decode(), "receiver_encrypted": base64.b64encode(enc_for_receiver).decode()})
#         else:
#             db.add(PendingMessage(sender_phone=sender, receiver_phone=sender, message=payload_for_sender, timestamp=datetime.utcnow()))

#         if receiver in manager.active_connections:
#             await manager.send_personal_message("SESSION_KEY", sender, receiver, payload_for_receiver, keys={"sender_encrypted": base64.b64encode(enc_for_sender).decode(), "receiver_encrypted": base64.b64encode(enc_for_receiver).decode()})
#         else:
#             db.add(PendingMessage(sender_phone=sender, receiver_phone=receiver, message=payload_for_receiver, timestamp=datetime.utcnow()))

#         db.commit()
#         return {"type": "success", "payload": {"sender_encrypted": base64.b64encode(enc_for_sender).decode(), "receiver_encrypted": base64.b64encode(enc_for_receiver).decode()}}

#     except Exception as e:
#         db.rollback()
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-session-keys-test")
async def generate_session_keys_test(
    sender: str = Form(...),
    receiver: str = Form(...),
    db: Session = Depends(get_db)
):
    try:
        global generator_model
        if generator_model is None:
            raise HTTPException(status_code=500, detail="Generator model not loaded")

        user1, user2 = sorted([sender, receiver])

        existing = db.query(SessionKey).filter(
            SessionKey.user1 == user1,
            SessionKey.user2 == user2
        ).first()

        if existing:
            enc_for_sender = existing.key_sender_to_receiver
            enc_for_receiver = existing.key_receiver_to_sender
        else:
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
                    pem = b"-----BEGIN PUBLIC KEY-----\n" + base64.encodebytes(decoded) + b"-----END PUBLIC KEY-----\n"
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

            db.add(SessionKey(
                user1=user1,
                user2=user2,
                key_sender_to_receiver=enc_for_sender,
                key_receiver_to_sender=enc_for_receiver,
                created_at=datetime.utcnow()
            ))
            db.commit()

        # Send individually: sender gets keys encrypted for them
        if sender in manager.active_connections:
            await manager.send_personal_message(
                message_type="SESSION_KEY",
                sender=receiver,  # "from" is the other user
                receiver=sender,
                message="",
                keys={
                    "myKey": base64.b64encode(enc_for_sender).decode(),
                    "otherKey": base64.b64encode(enc_for_receiver).decode()
                }
            )
        else:
            db.add(PendingMessage(
                sender_phone=receiver,  # "from" is other user
                receiver_phone=sender,
                message="SESSION_KEY",
                timestamp=datetime.utcnow()
            ))

        if receiver in manager.active_connections:
            await manager.send_personal_message(
                message_type="SESSION_KEY",
                sender=sender,  # "from" is the other user
                receiver=receiver,
                message="",

                
                keys={
                    "myKey": base64.b64encode(enc_for_receiver).decode(),
                    "otherKey": base64.b64encode(enc_for_sender).decode()
                }
            )
        else:
            db.add(PendingMessage(
                sender_phone=sender,
                receiver_phone=receiver,
                message="SESSION_KEY",
                timestamp=datetime.utcnow()
            ))

        db.commit()
        return {
            "type": "success",
            "payload": {
                "sender_encrypted": base64.b64encode(enc_for_sender).decode(),
                "receiver_encrypted": base64.b64encode(enc_for_receiver).decode()
            }
        }

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
        pending = db.query(PendingMessage).filter(PendingMessage.receiver_phone == phone).all()
        for msg in pending:
            await websocket.send_text(json.dumps({
                "from": msg.sender_phone,
                "to": phone,
                "type": "pending_message",
                "message": msg.message,
                "keys": {},
                "timestamp": msg.timestamp.isoformat()
            }))
            db.delete(msg)
        db.commit()

        while True:
            data = await websocket.receive_text()
            print(f"Message from {phone}: {data}")
            if ":" in data:
                receiver_phone, message = data.split(":", 1)
                payload_keys = None
                message_type = "chat_message"

                if receiver_phone in manager.active_connections:
                    await manager.send_personal_message(message_type, phone, receiver_phone, message, keys=payload_keys)
                else:
                    db.add(PendingMessage(sender_phone=phone, receiver_phone=receiver_phone, message=message, timestamp=datetime.utcnow()))
                    db.commit()

    except WebSocketDisconnect:
        manager.disconnect(phone)
        print(f"{phone} disconnected")
    finally:
        db.close()

@app.get("/")
def root():
    return {"type": "info", "payload": {"message": "Server running with SQLite, WebSocket, and Random Number Generator"}}
