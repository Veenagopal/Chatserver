from sqlalchemy import (
    Column, Integer, String, Text, DateTime, LargeBinary, UniqueConstraint
)
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()



class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    phone = Column(String(20), unique=True, index=True)
    publickey = Column(String)  # 🔐 Store user’s public key
    created_at = Column(DateTime, default=datetime.utcnow)


class PendingMessage(Base):
    __tablename__ = "pending_messages"

    id = Column(Integer, primary_key=True, index=True)
    sender_phone = Column(String(20), nullable=False)
    receiver_phone = Column(String(20), nullable=False)
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


class SessionKey(Base):
    __tablename__ = "session_keys"

    id = Column(Integer, primary_key=True, index=True)
    key1 = Column(LargeBinary, nullable=False)  # sender → receiver
    key2 = Column(LargeBinary, nullable=False)  # receiver → sender
    phone1 = Column(String(20), nullable=False)  # smaller phone number
    phone2 = Column(String(20), nullable=False)  # larger phone number
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("phone1", "phone2", name="_phone_pair_uc"),
    )


class PendingSession(Base):
    __tablename__ = "pending_sessions"

    id = Column(Integer, primary_key=True, index=True)
    receiver_phone = Column(String(20), nullable=False)
    key1 = Column(LargeBinary, nullable=False)
    key2 = Column(LargeBinary, nullable=False)
    phone1 = Column(String(20), nullable=False)
    phone2 = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
