from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, LargeBinary, UniqueConstraint
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

class SessionKey(Base):
    __tablename__ = "session_keys"

    id = Column(Integer, primary_key=True, index=True)
    user1 = Column(String(20), nullable=False)  # smaller phone number
    user2 = Column(String(20), nullable=False)  # larger phone number
    key_sender_to_receiver = Column(LargeBinary, nullable=False)
    key_receiver_to_sender = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (UniqueConstraint("user1", "user2", name="_user_pair_uc"),)
# Always store user1 < user2 alphabetically or numerically.
# Unique constraint ensures no duplicate session keys for the same user pair.