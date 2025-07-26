from fastapi import FastAPI, HTTPException
from fastapi import Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Database
#DATABASE_URL = "sqlite:///./users.db"
DATABASE_URL = "sqlite:///./users_v2.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
#Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# User table
from models import User , Base 
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
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class RegisterUserRequest(BaseModel):
    phone: str
    name:str

class DeleteUserRequest(BaseModel):
    phone: str

# Register user
@app.post("/register-user")
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

@app.get("/get-user")
def get_user(phone: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.phone == phone).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"name": user.name, "phone": user.phone}

# Root
@app.get("/")
def root():
    return {"message": "Server running with SQLite"}
