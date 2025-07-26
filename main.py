from fastapi import FastAPI, HTTPException
from fastapi import Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Database
DATABASE_URL = "sqlite:///./users.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# User table
class User(Base):
    __tablename__ = "users"
    phone = Column(String, primary_key=True, unique=True)

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

# Register user
@app.post("/register-user")
def register_user(request_data: RegisterUserRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.phone == request_data.phone).first()
    if user:
        return {"status": "exists"}
    db.add(User(phone=request_data.phone))
    db.commit()
    return {"status": "registered"}

# Delete user
@app.post("/delete-user")
def delete_user(request_data: RegisterUserRequest, db: Session = Depends(get_db)):
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
    users = db.query(User).all()
    return {"users": [user.phone for user in users]}

# Root
@app.get("/")
def root():
    return {"message": "Server running with SQLite"}
