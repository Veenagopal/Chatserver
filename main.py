import os
import base64
import sqlite3
import torch
import numpy as np
import traceback
from datetime import datetime
from typing import Dict, List
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query, Form
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

# ----------------------- DATABASE SETUP ---------------------
@app.on_event("startup")
def startup_event():
    global engine
    # Ensure database path exists
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

# ------------------- MODEL LOADING --------------------------
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
    print("✅ Model loaded")


# ---------------------- HELPERS ----------------------------
def get_public_key_for(phone_number: str) -> str | None:
    db_path = DATABASE_URL.replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT publickey FROM users WHERE phone = ?", (phone_number,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


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


# ------------------- REQUEST SCHEMAS -----------------------
class RegisterUserRequest(BaseModel):
    phone: str
    name: str
    publickey: str


class DeleteUserRequest(BaseModel):
    phone: str


# ------------------- CONNECTION MANAGER --------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, phone: str):
        await websocket.accept()
        self.active_connections[phone] = websocket
        print(f"✅ Connected: {phone}")

    def disconnect(self, phone: str):
        self.active_connections.pop(phone, None)
        print(f"❌ Disconnected: {phone}")

    async def send_session_keys(
        self,
        phone1: str,
        phone2: str,
        receiver: str,
        key1: bytes,
        key2: bytes,
        timestamp: datetime,
    ):
        websocket = self.active_connections.get(receiver)
        if websocket:
            await websocket.send_text(json.dumps({
                "type": "session_key",
                "phone1": phone1,
                "phone2": phone2,
                "receiver": receiver,
                "key1": key1,
                "key2": key2,
                "timestamp": int(timestamp.timestamp() * 1000)  
            }))
            print(f"📤 Sent session keys to {receiver}")

    # async def send_personal_message(self, message_type: str, sender: str, receiver: str,
    #                                 message: str = "", keys: dict | None = None):
    #     payload = {
    #         "from": sender,
    #         "to": receiver,
    #         "type": message_type,
    #         "message": message,
    #         "keys": keys or {},
    #         "timestamp": datetime.utcnow().isoformat()
    #     }
    #     websocket = self.active_connections.get(receiver)
    #     if websocket:
    #         await websocket.send_text(json.dumps(payload))
async def send_personal_message(self, message_type: str, receiver: str, payload: dict):
    """
    Forward an encrypted chat message or other payload to a specific receiver.
    `payload` already contains 'from', 'to', 'ct', 'iv', 'kpack', etc.
    """
    payload['timestamp'] = datetime.utcnow().isoformat()

    outer = {
        "type": message_type,
        "payload": payload
    }

    websocket = self.active_connections.get(receiver)
    if websocket:
        await websocket.send_text(json.dumps(outer))



manager = ConnectionManager()


# ---------------------- SESSION / CHAT HANDLING ------------
async def handle_pending_sessions(db: Session, phone: str):
    pending = db.query(PendingSession).filter(
        (PendingSession.phone1 == phone) | (PendingSession.phone2 == phone)
    ).all()

    for ps in pending:
        try:
            await manager.send_session_keys(
                phone1=ps.phone1,
                phone2=ps.phone2,
                receiver=phone,
                key1=ps.key1,
                key2=ps.key2,
                timestamp=ps.created_at
            )
        except Exception as e:
            print(f" Failed sending keys to {phone}: {e}")
        finally:
            db.delete(ps)

    db.commit()


async def handle_chat_messages(db: Session, websocket: WebSocket, phone: str):
    try:
        # Send pending messages first
        pending = db.query(PendingMessage).filter(PendingMessage.receiver_phone == phone).all()
        for msg in pending:
            await websocket.send_text(json.dumps({
                "from": msg.sender_phone,
                "to": phone,
                "type": "pending_message",
                "message": msg.message,
                "keys": {},  # can be replaced with actual keys if needed
                "timestamp": msg.timestamp.isoformat()
            }))
            db.delete(msg)
        db.commit()

        # Handle new incoming messages
        while True:
            try:
                data = await websocket.receive_text()
                obj = json.loads(data)

                if obj.get("type") == "chat_message":
                    payload = obj.get("payload", {})
                    receiver_phone = payload.get("to")
                    if receiver_phone in manager.active_connections:
                        # Forward entire payload with sender info
                        await manager.send_personal_message(
                            "chat_message",
                            receiver_phone=receiver_phone,
                            payload=payload  # full encrypted payload
                        )
                    else:
                        # Store full payload for offline delivery
                        db.add(PendingMessage(
                            sender_phone=phone,
                            receiver_phone=receiver_phone,
                            payload=json.dumps(payload),
                            timestamp=datetime.utcnow()
                        ))
                        db.commit()

            except WebSocketDisconnect as ii:
                print(f"WS disconnected for {phone}: {ii}")
                manager.disconnect(phone)
                break
            except Exception as e:
                print(f"Error handling message from {phone}: {e}")
                traceback.print_exc()

    except Exception as outer_e:
        print(f"Error in handle_chat_messages for {phone}: {outer_e}")
        traceback.print_exc()


# ------------------- ENDPOINTS -----------------------------
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
    db_path = DATABASE_URL.replace("sqlite:///", "")
    if os.path.exists(db_path):
        os.remove(db_path)
        return {"type": "success", "payload": {"message": "Database deleted."}}
    return {"type": "error", "payload": {"message": f"Database not found at {db_path}"}}


@app.post("/generate-session-keys-test")
async def generate_session_keys_test(
    sender: str = Form(...),
    receiver: str = Form(...),
    db: Session = Depends(get_db)
):
    print(f"🔑 Session key request: sender={sender}, receiver={receiver}")
    try:
        phone1, phone2 = sorted([sender, receiver])

        # Check for existing session
        existing = db.query(SessionKey).filter(
            SessionKey.phone1 == phone1,
            SessionKey.phone2 == phone2
        ).first()

        if existing:
            key1, key2 = existing.key1, existing.key2
            timestamp = existing.created_at
            print("✅ Existing session found")
        else:
            # Generate 32-byte session key
            key_bytes = (123).to_bytes(1, "big").rjust(32, b'\x00')

            # Load public keys from DB
            raw_sender = db.query(User.publickey).filter(User.phone == sender).scalar()
            raw_receiver = db.query(User.publickey).filter(User.phone == receiver).scalar()
            if not raw_sender or not raw_receiver:
                raise HTTPException(status_code=404, detail="Sender or receiver public key not found")

            # Deserialize public keys
            pub_sender = load_pubkey(raw_sender)
            pub_receiver = load_pubkey(raw_receiver)

            # Encrypt session key using OAEP SHA-256 / MGF1 SHA-256
            enc_for_sender = pub_sender.encrypt(
                key_bytes,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            enc_for_receiver = pub_receiver.encrypt(
                key_bytes,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )

            # Base64-encode for transport
            enc_for_sender_b64 = base64.b64encode(enc_for_sender).decode("ascii")
            enc_for_receiver_b64 = base64.b64encode(enc_for_receiver).decode("ascii")

            # Store session in DB
            timestamp = datetime.utcnow()
            db.add(SessionKey(
                phone1=sender,
                phone2=receiver,
                key1=enc_for_sender_b64,
                key2=enc_for_receiver_b64,
                created_at=timestamp
            ))
            db.commit()
            print("💾 Session stored in DB")
            key1, key2 = enc_for_sender_b64, enc_for_receiver_b64

        # Deliver keys
        for user in [sender, receiver]:
            if user in manager.active_connections:
                await manager.send_session_keys(sender, receiver, user, key1, key2, timestamp)
            else:
                db.add(PendingSession(
                    receiver_phone=user,
                    phone1=sender,
                    phone2=receiver,
                    key1=key1,
                    key2=key2,
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


@app.post("/register-user")
def register_user(request: RegisterUserRequest, db: Session = Depends(get_db)):
    # Check if user already exists
    user = db.query(User).filter(User.phone == request.phone).first()
    if user:
        return {"type": "exists", "payload": {}}

    # Load the uploaded public key
    try:
        pub_key = load_pubkey(request.publickey)  # your existing load_pubkey function
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid public key: {e}")

    # Serialize to DER/X.509 and Base64-encode for storage
    pub_bytes = pub_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    pub_b64 = base64.b64encode(pub_bytes).decode("ascii")

    # Store user with DER Base64 public key
    db.add(User(phone=request.phone, name=request.name, publickey=pub_b64))
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
    num_deleted = db.query(PendingMessage).delete()
    db.commit()
    return {"type": "success", "payload": {"deleted_rows": num_deleted}}


@app.get("/debug-pending/{receiver_phone}")
def get_pending_messages(receiver_phone: str, db: Session = Depends(get_db)):
    messages = db.query(PendingMessage).filter(PendingMessage.receiver_phone == receiver_phone).all()
    return {"type": "pending_messages", "payload": [{"from": m.sender_phone, "message": m.message, "timestamp": m.timestamp.isoformat()} for m in messages]}


# ------------------- WEBSOCKET ENDPOINT --------------------
@app.websocket("/ws/{phone}")
async def websocket_endpoint(websocket: WebSocket, phone: str):
    await manager.connect(websocket, phone)
    db = SessionLocal()
    try:
        # Deliver pending sessions
        await handle_pending_sessions(db, phone)
        # Handle chat messages
        await handle_chat_messages(db, websocket, phone)
    except WebSocketDisconnect as ee:
        print(f"WS error for {phone}: {ee}")
        manager.disconnect(phone)
    finally:
        db.close()


# ------------------- ROOT -----------------------------
@app.get("/")
def root():
    return {"type": "info", "payload": {"message": "Server running with SQLite, WebSocket, and Random Number Generator"}}
