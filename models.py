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

    #messages = relationship("Message", back_populates="sender")


# class Message(Base):
#     __tablename__ = 'messages'

#     id = Column(Integer, primary_key=True, index=True)
#     sender_id = Column(Integer, ForeignKey('users.id'))
#     content = Column(Text)
#     timestamp = Column(DateTime, default=datetime.utcnow)

#     sender = relationship("User", back_populates="messages")

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

    # Combined string: "+919999999999||BASE64KEY"
    key1 = Column(LargeBinary, nullable=False)
    key2 = Column(LargeBinary, nullable=False)
    # Internal fields (not exposed in API)
    phone1 = Column(String(20), nullable=False)
    phone2 = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Ensure uniqueness at phone number level
    __table_args__ = (
        UniqueConstraint("phone1", "phone2", name="_phone_pair_uc"),
    )
class PendingSession(Base):
    id = Column(Integer, primary_key=True, index=True)
    receiver_phone = Column(String(20), nullable=False)
    key1 = Column(LargeBinary, nullable=False)
    key2 = Column(LargeBinary, nullable=False)
    # Internal fields (not exposed in API)
    phone1 = Column(String(20), nullable=False)
    phone2 = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)