from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)  
    phone = Column(String(20), unique=True, index=True)
    publickey = Column(String)  # 🔐 Added field to store public key

    created_at = Column(DateTime, default=datetime.utcnow)

    messages = relationship("Message", back_populates="sender")


class Message(Base):
    __tablename__ = 'messages'

    id = Column(Integer, primary_key=True, index=True)
    sender_id = Column(Integer, ForeignKey('users.id'))
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

    sender = relationship("User", back_populates="messages")

class PendingMessage(Base):
    __tablename__ = "pending_messages"

    id = Column(Integer, primary_key=True, index=True)
    sender_phone = Column(String(20))  # We don't need FK here
    receiver_phone = Column(String(20))
    message = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
