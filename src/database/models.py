from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Integer, DateTime, ForeignKey, JSON, Text
from sqlalchemy.sql import func
import uuid
from typing import Optional, List

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(100))
    role: Mapped[str] = mapped_column(String(20), default='staff')
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    started_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())
    ended_at: Mapped[Optional[DateTime]] = mapped_column(DateTime)

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("chat_sessions.id"))
    role: Mapped[str] = mapped_column(String(10), nullable=False) # 'user' or 'assistant'
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[Optional[dict]] = mapped_column(JSON)
    route_type: Mapped[Optional[str]] = mapped_column(String(20))
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())

class Escalation(Base):
    __tablename__ = "escalations"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    message_id: Mapped[int] = mapped_column(Integer, ForeignKey("chat_messages.id"))
    status: Mapped[str] = mapped_column(String(20), default='pending')
    staff_reply: Mapped[Optional[str]] = mapped_column(Text)
    resolved_by: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"))
    resolved_at: Mapped[Optional[DateTime]] = mapped_column(DateTime)
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())

class Document(Base):
    __tablename__ = "documents"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    file_size: Mapped[Optional[int]] = mapped_column(Integer)
    num_chunks: Mapped[Optional[int]] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20), default='processing')
    uploaded_by: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"))
    uploaded_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())
