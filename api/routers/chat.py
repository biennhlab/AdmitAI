from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
import uuid
from api.schemas import ChatRequest, ChatResponse, FeedbackRequest
from src.database.connection import get_db

router = APIRouter(prefix="/api/chat", tags=["chat"])

@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    # Phase 0 Stub
    session_id = request.session_id or str(uuid.uuid4())
    return ChatResponse(
        answer="This is a stub answer from Phase 0.",
        citations=[],
        route_type="rag",
        session_id=session_id
    )

@router.post("/feedback")
async def chat_feedback(request: FeedbackRequest, db: AsyncSession = Depends(get_db)):
    # Phase 0 Stub
    return {"status": "ok"}
