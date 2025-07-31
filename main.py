from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Dict, List
from datetime import datetime

from database import SessionLocal, init_db
from models import User, PendingMessage
import os
print("FILES IN /data at startup:", os.listdir("/data"))

# --------------------------- INIT ---------------------------

app = FastAPI()
init_db()

# -------------------------- CORS ----------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------- DB ------------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ----------------------- Request Schemas --------------------

class RegisterUserRequest(BaseModel):
    phone: str
    name: str

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
    new_user = User(phone=request_data.phone, name=request_data.name)
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

# 🧪 Optional Debug: View pending messages for a user
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
        # 🔁 Step 1: Deliver pending messages
        pending = db.query(PendingMessage).filter(PendingMessage.receiver_phone == phone).all()
        for msg in pending:
            await websocket.send_text(f"{msg.sender_phone}:{msg.message}")
            db.delete(msg)
        db.commit()

        # 🔁 Step 2: Handle incoming messages
        while True:
            data = await websocket.receive_text()
            print(f"Message from {phone}: {data}")

            if ":" in data:
                receiver_phone, message = data.split(":", 1)

                if receiver_phone in manager.active_connections:
                    # Receiver online
                    await manager.send_personal_message(f"{phone}:{message}", receiver_phone)
                else:
                    # Receiver offline → store
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
    return {"message": "Server running with SQLite and WebSocket support"}
