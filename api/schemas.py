from pydantic import BaseModel
from typing import List, Optional, Literal, Any
from datetime import datetime

class Citation(BaseModel):
    source: str
    page: Optional[int]
    snippet: str

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation] = []
    route_type: Optional[str] = None
    session_id: str

class FeedbackRequest(BaseModel):
    message_id: int
    feedback: Literal["positive", "negative"]

class LoginRequest(BaseModel):
    username: str
    password: str

class UserInfo(BaseModel):
    id: int
    username: str
    full_name: Optional[str]
    role: str
    
    class Config:
        from_attributes = True

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserInfo

class EscalationReplyRequest(BaseModel):
    reply: str

class PaginatedResponse(BaseModel):
    items: List[Any]
    total: int
    page: int
    limit: int

class DocumentResponse(BaseModel):
    id: int
    filename: str
    file_size: Optional[int]
    num_chunks: Optional[int]
    status: str
    uploaded_by: Optional[int]
    uploaded_at: datetime
    
    class Config:
        from_attributes = True
