from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import logging
import uuid

from api.schemas import ChatRequest, ChatResponse, FeedbackRequest, Citation
from src.database.connection import get_db
from src.database.models import ChatSession, ChatMessage
from src.generation.session_memory import SessionMemory
from src.generation.rag_chain import RAGChain
from src.generation.llm_client import LLMClient
from src.retrieval.dense_search import NaiveDenseSearch
from src.retrieval.embedder import Embedder
from src.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

# Global instances for in-memory session and RAG
session_memory = SessionMemory()
llm_client = LLMClient(api_key=settings.NVIDIA_API_KEY, model=settings.LLM_MODEL)
embedder = Embedder(model_name=settings.EMBEDDING_MODEL)
retriever = NaiveDenseSearch(embedder=embedder, chunks=[])
rag_chain = RAGChain(retriever=retriever, llm_client=llm_client)

@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message cannot be empty")
        
    session_id = request.session_id
    
    # 1. Get or create session
    is_new_session = False
    if not session_id or session_id not in session_memory.sessions:
        session_id = session_memory.create_session()
        is_new_session = True
        
        # Save new session to DB
        new_session_db = ChatSession(id=session_id)
        db.add(new_session_db)
        await db.commit()
    
    try:
        # Save user message to memory & DB
        session_memory.add_message(session_id, "user", request.message)
        
        user_msg_db = ChatMessage(
            session_id=session_id,
            role="user",
            content=request.message
        )
        db.add(user_msg_db)
        await db.commit()
        
        # 2. Get session history (excluding current user msg to avoid duplication if RAG handles it differently?
        # Actually RAGChain expects history before current message, or handles history internally.
        # Wait, RAGChain.answer(question, session_history=None). So history should NOT include the current question.
        # Let's get history before adding? Or just pass history minus the last message.
        raw_history = session_memory.get_history(session_id, max_turns=5)
        # Exclude the very last message since it's the current user prompt
        history_for_rag = raw_history[:-1] if raw_history else []
        
        # 3. Call RAG Chain
        try:
            rag_response = rag_chain.answer(question=request.message, session_history=history_for_rag)
        except Exception as e:
            logger.error(f"RAGChain error: {str(e)}")
            # Graceful degradation
            rag_response = None
            
        if rag_response is None:
            answer = "Sorry, I am currently experiencing technical difficulties. Please try again later."
            citations = []
            route_type = "error"
        else:
            answer = rag_response.answer
            citations = rag_response.citations
            route_type = rag_response.route_type
            
        # 4. Save assistant message to memory & DB
        session_memory.add_message(session_id, "assistant", answer)
        
        # Build citations dict for DB
        citations_db = [{"source": c} for c in citations]
        
        bot_msg_db = ChatMessage(
            session_id=session_id,
            role="assistant",
            content=answer,
            citations=citations_db,
            route_type=route_type
        )
        db.add(bot_msg_db)
        await db.commit()
        
        # Convert to API response model
        api_citations = [Citation(source=c, page=None, snippet="") for c in citations]
        
        return ChatResponse(
            answer=answer,
            citations=api_citations,
            route_type=route_type,
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"Chat API error: {str(e)}")
        # If DB error or other unexpected error
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.post("/feedback")
async def chat_feedback(request: FeedbackRequest, db: AsyncSession = Depends(get_db)):
    # Phase 0 Stub
    return {"status": "ok"}
