from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Dict, List
from datetime import datetime
from fastapi import Body

import os
import torch
import numpy as np
import asyncio

from database import SessionLocal, init_db
from models import User, PendingMessage
from NCA_model import NCAGenerator, get_config
from database import DATABASE_URL
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes, serialization
import numpy as np
import torch
import traceback



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

@app.get("/where-is-db")
def find_database():
    import os

    candidates = []
    for root, dirs, files in os.walk("/"):  # search entire container
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
    db_path = "/data/users_v3.db"  # <- absolute path
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
import sqlite3



def get_public_key_for(phone_number: str) -> str | None:
    db_path = DATABASE_URL.replace("sqlite:///", "")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT publickey FROM users WHERE phone = ?", (phone_number,))
    row = cursor.fetchone()
    conn.close()
    if row:
        
        return row[0]  # public_key
    return None



@app.post("/generate-session-keys")
async def generate_session_keys(
    sender: str = Form(...),
    receiver: str = Form(...),
    db: Session = Depends(get_db)
):
    global generator_model

    try:
        if generator_model is None:
            return {"error": "Model not loaded"}

        # 1. Generate shared 256-bit key
        with torch.no_grad():
            cfg = get_config()
            z = torch.randn(1, cfg["channels"], cfg["length"])
            output = generator_model(z)
            probs = torch.sigmoid(output)
            bits = (probs > 0.5).int().cpu().numpy().flatten()[:256]
            shared_key_bytes = np.packbits(bits)  # 32 bytes

        # 2. Load public keys from DB
        public_key_sender_str = get_public_key_for(sender)
        print("PUB KEY RAW SENDER:", repr(public_key_sender_str))

        public_key_receiver_str = get_public_key_for(receiver)
        print("PUB KEY RAW RECEIVER:", repr(public_key_receiver_str))
        if not public_key_sender_str or not public_key_receiver_str:
            raise HTTPException(status_code=404, detail="Sender or receiver public key not found")

        # convert to bytes
        pub_key_sender = serialization.load_pem_public_key(public_key_sender_str.encode("utf-8"))
        pub_key_receiver = serialization.load_pem_public_key(public_key_receiver_str.encode("utf-8"))

        # 3. Encrypt shared key for both
        enc_sender = pub_key_sender.encrypt(
            shared_key_bytes.tobytes(),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        enc_receiver = pub_key_receiver.encrypt(
            shared_key_bytes.tobytes(),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        # 4. Prepare payloads with SESSION_KEY prefix
        payload_for_sender = f"SESSION_KEY:{base64.b64encode(enc_sender).decode()}{base64.b64encode(enc_receiver).decode()}"
        payload_for_receiver = f"SESSION_KEY:{base64.b64encode(enc_receiver).decode()}{base64.b64encode(enc_sender).decode()}"

        # 5. Send via WebSocket if online, else store in PendingMessage table
        if sender in manager.active_connections:
            await manager.send_personal_message(payload_for_sender, sender)
        else:
            db.add(PendingMessage(
                sender_phone=receiver,
                receiver_phone=sender,
                message=payload_for_sender,
                timestamp=datetime.utcnow()
            ))

        if receiver in manager.active_connections:
            await manager.send_personal_message(payload_for_receiver, receiver)
        else:
            db.add(PendingMessage(
                sender_phone=sender,
                receiver_phone=receiver,
                message=payload_for_receiver,
                timestamp=datetime.utcnow()
            ))

        db.commit()

        return {
            "status": "success",
            "shared_key_hex": shared_key_bytes.tobytes().hex()
        }
    except Exception as e:
        print("🔥 Exception in generate-session-keys:", e)
        traceback.print_exc()
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
    return {"users": [{"phone": user.phone, "name": user.name,"publickey":user.publickey} for user in users]}

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

