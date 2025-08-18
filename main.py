import os
import base64
import sqlite3
import torch
import numpy as np
import traceback
from datetime import datetime
from typing import Dict, List

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
#init_db()

# Remove the top-level init_db() call
# init_db()  <-- remove this

@app.on_event("startup")
def startup_event():
    global engine
    # Determine DB path from DATABASE_URL
    if DATABASE_URL.startswith("sqlite:////"):
        db_path = DATABASE_URL.replace("sqlite:////", "/", 1)
    elif DATABASE_URL.startswith("sqlite:///"):
        db_path = DATABASE_URL.replace("sqlite:///", "", 1)
    else:
        raise ValueError(f"Unsupported DATABASE_URL format: {DATABASE_URL}")

    # Ensure parent directory exists
    parent_dir = os.path.dirname(db_path)
    os.makedirs(parent_dir, exist_ok=True)

    # Create DB file if missing
    if not os.path.exists(db_path):
        open(db_path, "a").close()
        print(f"[INFO] Database file created at {db_path}")

    # Ensure writable permissions
    try:
        os.chmod(db_path, 0o666)
        print(f"[INFO] Database permissions set to 666")
    except PermissionError:
        print(f"[WARNING] Could not change permissions for {db_path}")

    # Create tables if missing
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

    async def send_personal_message(self, message: str, receiver_phone: str):
        if receiver_phone in self.active_connections:
            await self.active_connections[receiver_phone].send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)

manager = ConnectionManager()

# ---------------------- Endpoints ----------------------------

@app.get("/where-is-db")
def find_database():
    candidates = []
    for root, dirs, files in os.walk("/"):
        for file in files:
            if file.endswith(".db"):
                candidates.append(os.path.join(root, file))
    return {
        "found_db_files": candidates,
        "cwd": os.getcwd(),
        "expected_path": os.path.abspath(DATABASE_URL.replace("sqlite:///", ""))
    }

@app.get("/delete-db")
def delete_database():
    db_path = "/data/users_v3.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        return {"status": "success", "message": "Database deleted."}
    return {"status": "error", "message": f"Database file not found at {db_path}."}

# @app.post("/generate-session-keys")
# async def generate_session_keys(
#     sender: str = Form(...),
#     receiver: str = Form(...),
#     db: Session = Depends(get_db)
# ):
#     global generator_model
#     try:
#         if generator_model is None:
#             return {"error": "Model not loaded"}

#         with torch.no_grad():
#             cfg = get_config()
#             z = torch.randn(1, cfg["channels"], cfg["length"])
#             output = generator_model(z)
#             probs = torch.sigmoid(output)
#             bits = (probs > 0.5).int().cpu().numpy().flatten()[:256]
#             shared_key_bytes = np.packbits(bits)

#         raw_sender = get_public_key_for(sender)
#         raw_receiver = get_public_key_for(receiver)
#         if not raw_sender or not raw_receiver:
#             raise HTTPException(status_code=404, detail="Sender or receiver public key not found")

#         def load_pubkey(raw_base64: str):
#             pem = "-----BEGIN PUBLIC KEY-----\n"
#             pem += "\n".join(raw_base64[i:i+64] for i in range(0, len(raw_base64), 64))
#             pem += "\n-----END PUBLIC KEY-----\n"
#             return serialization.load_pem_public_key(pem.encode("utf-8"))

#         pub_key_sender = load_pubkey(raw_sender)
#         pub_key_receiver = load_pubkey(raw_receiver)

#         enc_sender = pub_key_sender.encrypt(
#             shared_key_bytes.tobytes(),
#             padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
#         )
#         enc_receiver = pub_key_receiver.encrypt(
#             shared_key_bytes.tobytes(),
#             padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
#         )

#         DELIMITER = "||"
#         payload_for_sender = f"SESSION_KEY:{base64.b64encode(enc_sender).decode()}{DELIMITER}{base64.b64encode(enc_receiver).decode()}"
#         payload_for_receiver = f"SESSION_KEY:{base64.b64encode(enc_receiver).decode()}{DELIMITER}{base64.b64encode(enc_sender).decode()}"

#         if sender in manager.active_connections:
#             await manager.send_personal_message(payload_for_sender, sender)
#         else:
#             db.add(PendingMessage(sender_phone=receiver, receiver_phone=sender, message=payload_for_sender, timestamp=datetime.utcnow()))
        
#         if receiver in manager.active_connections:
#             await manager.send_personal_message(payload_for_receiver, receiver)
#         else:
#             db.add(PendingMessage(sender_phone=sender, receiver_phone=receiver, message=payload_for_receiver, timestamp=datetime.utcnow()))
        
#         db.commit()
#         return {"status": "success", "shared_key_hex": shared_key_bytes.tobytes().hex()}

#     except Exception as e:
#         print("Exception in generate-session-keys:", repr(e))
#         raise HTTPException(status_code=500, detail=f"Server error: {e}")
@app.post("/generate-session-keys-test")
async def generate_session_keys_test(
    sender: str = Form(...),
    receiver: str = Form(...),
    db: Session = Depends(get_db)
):
    try:
        # Order users consistently
        user1, user2 = sorted([sender, receiver])

        # Check if session key already exists
        existing = db.query(SessionKey).filter(
            SessionKey.user1 == user1,
            SessionKey.user2 == user2
        ).first()
        if existing:
            return {"status": "exists", "message": "Session key already exists"}

        # Use fixed key for testing
        shared_key_bytes = b"123"

        # Get public keys
        raw_sender = db.query(User.publickey).filter(User.phone == sender).scalar()
        raw_receiver = db.query(User.publickey).filter(User.phone == receiver).scalar()
        if not raw_sender or not raw_receiver:
            raise HTTPException(status_code=404, detail="Sender or receiver public key not found")
        def load_pubkey(raw_base64: str):
            try:
                # clean base64 (remove spaces/newlines just in case)
                raw_base64 = "".join(raw_base64.strip().split())
                pem = "-----BEGIN PUBLIC KEY-----\n"
                pem += "\n".join(raw_base64[i:i+64] for i in range(0, len(raw_base64), 64))
                pem += "\n-----END PUBLIC KEY-----\n"
                return serialization.load_pem_public_key(pem.encode("utf-8"))
            except Exception as e:
                print("❌ Failed to load public key:", e)
                raise

        pub_sender = load_pubkey(raw_sender)
        pub_receiver = load_pubkey(raw_receiver)

        # Encrypt the shared key
        enc_sender = pub_sender.encrypt(
            shared_key_bytes,
            padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
        )
        enc_receiver = pub_receiver.encrypt(
            shared_key_bytes,
            padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
        )

        # Save to SessionKey table
        db.add(SessionKey(
            user1=user1,
            user2=user2,
            key_sender_to_receiver=enc_sender,
            key_receiver_to_sender=enc_receiver,
            created_at=datetime.utcnow()
        ))
        db.commit()

        # Return encrypted keys as base64
        return {
            "status": "success",
            "sender_encrypted": base64.b64encode(enc_sender).decode(),
            "receiver_encrypted": base64.b64encode(enc_receiver).decode()
        }

    except Exception as e:
        db.rollback()
        print("❌ ERROR in /generate-session-keys-test:")
        traceback.print_exc()  # prints full stacktrace to console

        raise HTTPException(status_code=500, detail=str(e))
    
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
        bits = (probs > 0.5).int().cpu().numpy().flatten()[:256]
        byte_array = np.packbits(bits)
    return {"random_bits": bits.tolist(), "random_hex": byte_array.tobytes().hex()}

@app.post("/register-user")
def register_user(request: RegisterUserRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.phone == request.phone).first()
    if user:
        return {"status": "exists"}
    db.add(User(phone=request.phone, name=request.name, publickey=request.publickey))
    db.commit()
    return {"status": "registered"}

@app.post("/delete-user")
def delete_user(request: DeleteUserRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.phone == request.phone).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(user)
    db.commit()
    return {"status": "deleted"}

@app.get("/list-users")
def list_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return {"users": [{"phone": u.phone, "name": u.name, "publickey": u.publickey} for u in users]}

@app.get("/get-user")
def get_user(phone: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.phone == phone).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"name": user.name, "phone": user.phone}

@app.get("/search-users")
def search_users(query: str, exclude_phone: str, db: Session = Depends(get_db)):
    results = db.query(User).filter(User.name.ilike(f"%{query}%"), User.phone != exclude_phone).all()
    return {"users": [{"phone": u.phone, "name": u.name} for u in results]}

@app.get("/resolve-users")
def resolve_users(phones: List[str] = Query(...), db: Session = Depends(get_db)):
    users = db.query(User).filter(User.phone.in_(phones)).all()
    return {"users": [{"name": u.name, "phone": u.phone} for u in users]}

@app.get("/clear-pending-messages")
def clear_pending_messages(db: Session = Depends(get_db)):
    try:
        num_deleted = db.query(PendingMessage).delete()
        db.commit()
        return {"status": "success", "deleted_rows": num_deleted}
    except Exception as e:
        db.rollback()
        return {"status": "error", "message": str(e)}

@app.get("/debug-pending/{receiver_phone}")
def get_pending_messages(receiver_phone: str, db: Session = Depends(get_db)):
    messages = db.query(PendingMessage).filter(PendingMessage.receiver_phone == receiver_phone).all()
    return [{"from": m.sender_phone, "message": m.message, "timestamp": m.timestamp.isoformat()} for m in messages]

@app.websocket("/ws/{phone}")
async def websocket_endpoint(websocket: WebSocket, phone: str):
    await manager.connect(phone, websocket)
    db = SessionLocal()
    try:
        pending = db.query(PendingMessage).filter(PendingMessage.receiver_phone == phone).all()
        for msg in pending:
            if msg.message.startswith("SESSION_KEY"):
                await websocket.send_text(msg.message)  # send raw message
            else:
                await websocket.send_text(f"{msg.sender_phone}:{msg.message}")
            #await websocket.send_text(f"{msg.sender_phone}:{msg.message}")
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
                    db.add(PendingMessage(sender_phone=phone, receiver_phone=receiver_phone, message=message, timestamp=datetime.utcnow()))
                    db.commit()
    except WebSocketDisconnect:
        manager.disconnect(phone)
        print(f"{phone} disconnected")
    finally:
        db.close()

@app.get("/")
def root():
    return {"message": "Server running with SQLite, WebSocket, and Random Number Generator"}
