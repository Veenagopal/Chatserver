from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi import Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from fastapi import Query
from typing import List, Dict

# Database
DATABASE_URL = "sqlite:////data/users_v2.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# User table
from models import User, Base
Base.metadata.create_all(bind=engine)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# FastAPI app
app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class RegisterUserRequest(BaseModel):
    phone: str
    name: str

class DeleteUserRequest(BaseModel):
    phone: str

# WebSocket connection manager
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

# Register user
@app.post("/register-user")
def register_user(request_data: RegisterUserRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.phone == request_data.phone).first()
    if user:
        return {"status": "exists"}
    new_user = User(phone=request_data.phone, name=request_data.name)
    db.add(new_user)
    db.commit()
    return {"status": "registered"}

# Delete user
@app.post("/delete-user")
def delete_user(request_data: DeleteUserRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.phone == request_data.phone).first()
    if user:
        db.delete(user)
        db.commit()
        print(f"Deleted user: {request_data.phone}")
        return {"status": "deleted"}
    else:
        raise HTTPException(status_code=404, detail="User not found")

# List users
@app.get("/list-users")
def list_users(db: Session = Depends(get_db)):
    try:
        users = db.query(User).all()
        return {
            "users": [
                {"phone": user.phone, "name": user.name} for user in users
            ]
        }
    except Exception as e:
        print(f"ERROR: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Get user
@app.get("/get-user")
def get_user(phone: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.phone == phone).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"name": user.name, "phone": user.phone}

# Search users
@app.get("/search-users")
def search_users(
    query: str = Query(...),
    exclude_phone: str = Query(...),
    db: Session = Depends(get_db)
):
    try:
        results = db.query(User).filter(
            User.name.ilike(f"%{query}%"),
            User.phone != exclude_phone
        ).all()
        print("Search results:")
        for user in results:
            print(f"- Name: {user.name}, Phone: {user.phone}")
        return {
            "users": [
                {"phone": user.phone, "name": user.name} for user in results
            ]
        }
    except Exception as e:
        print(f"Search error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# WebSocket endpoint for chat
@app.websocket("/ws/{phone}")
async def websocket_endpoint(websocket: WebSocket, phone: str):
    await manager.connect(phone, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Message from {phone}: {data}")
            if ":" in data:
                receiver_phone, message = data.split(":", 1)
                await manager.send_personal_message(f"{phone}:{message}", receiver_phone)
    except WebSocketDisconnect:
        manager.disconnect(phone)
        print(f"{phone} disconnected")

# Root
@app.get("/")
def root():
    return {"message": "Server running with SQLite and WebSocket support"}
