from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any
import json
import queue
import threading

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas import ChatRequest, ChatResponse, Citation, FeedbackRequest
from src.config import settings
from src.database.connection import get_db
from src.database.models import ChatMessage, ChatSession
from src.generation.llm_client import (
    LLMAuthenticationError,
    LLMClient,
    LLMConnectionError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from src.generation.rag_chain import RAGChain, RAGRetrievalError
from src.generation.session_memory import SessionMemory
from src.query_transform import AbbreviationNormalizer, QueryRewriter
from src.retrieval import Embedder, NaiveDenseSearch, load_dense_index

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])

session_memory = SessionMemory()
rag_chain: RAGChain | None = None
rag_manifest: dict[str, Any] | None = None
rag_initialization_error: str | None = None
rag_public_error: str | None = None

RAG_UNAVAILABLE_MESSAGE = "Dịch vụ tư vấn đang tạm thời chưa sẵn sàng."
LLM_TIMEOUT_MESSAGE = "Yêu cầu xử lý mất nhiều thời gian hơn dự kiến."
LLM_UNAVAILABLE_MESSAGE = "Dịch vụ xử lý câu hỏi đang tạm thời chưa sẵn sàng."
INTERNAL_ERROR_MESSAGE = "Không thể xử lý yêu cầu lúc này."


def _service_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    )


def initialize_rag(
    index_dir: str | Path | None = None,
    *,
    retriever: Any | None = None,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the process-wide RAG chain from a loaded or injected retriever."""
    global rag_chain, rag_manifest, rag_initialization_error, rag_public_error
    try:
        is_injected = retriever is not None
        if is_injected:
            if manifest is None:
                raise ValueError("manifest is required when injecting a retriever")
            loaded_manifest = manifest
        else:
            chunks, embeddings, loaded_manifest = load_dense_index(
                index_dir or settings.INDEX_DIR
            )
            embedder = Embedder(model_name=settings.EMBEDDING_MODEL)
            retriever = NaiveDenseSearch(
                embedder,
                chunks,
                chunk_embeddings=embeddings,
            )

        indexed_model = loaded_manifest.get("embedding_model")
        if indexed_model != settings.EMBEDDING_MODEL:
            raise ValueError(
                f"Index model '{indexed_model}' differs from configured model '{settings.EMBEDDING_MODEL}'. "
                "Run scripts/ingest.py again."
            )
        llm_client = LLMClient(
            api_key=settings.LLM_API_KEY,
            model=settings.LLM_MODEL,
            base_url=settings.LLM_BASE_URL,
            timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
            max_retries=settings.LLM_MAX_RETRIES,
        )
        rag_chain = RAGChain(
            retriever,
            llm_client,
            top_k=settings.RETRIEVAL_TOP_K,
            # Dense cosine thresholds are not meaningful for RRF scores.
            min_score=None if is_injected else settings.RETRIEVAL_MIN_SCORE,
            min_lexical_coverage=settings.RETRIEVAL_MIN_LEXICAL_COVERAGE,
            query_rewriter=QueryRewriter(llm_client),
            abbreviation_normalizer=AbbreviationNormalizer(),
        )
        rag_manifest = loaded_manifest
        rag_initialization_error = None
        rag_public_error = None
        logger.info(
            "RAG ready: %s documents, %s chunks, backend=%s",
            loaded_manifest["document_count"],
            loaded_manifest["chunk_count"],
            loaded_manifest.get("retrieval_backend", loaded_manifest.get("backend")),
        )
        return loaded_manifest
    except Exception as exc:
        rag_chain = None
        rag_manifest = None
        rag_initialization_error = str(exc)
        rag_public_error = RAG_UNAVAILABLE_MESSAGE
        logger.exception("RAG initialization failed")
        raise


def mark_rag_unavailable(
    exc: Exception,
    public_error: str = RAG_UNAVAILABLE_MESSAGE,
) -> None:
    """Expose startup failures through health and chat responses."""
    global rag_chain, rag_manifest, rag_initialization_error, rag_public_error
    rag_chain = None
    rag_manifest = None
    rag_initialization_error = str(exc)
    rag_public_error = public_error


def rag_status() -> dict[str, Any]:
    return {
        "ready": rag_chain is not None,
        "error": rag_public_error,
        "backend": (
            rag_manifest.get("retrieval_backend", rag_manifest.get("backend"))
            if rag_manifest
            else None
        ),
        "documents": rag_manifest.get("document_count") if rag_manifest else 0,
        "chunks": rag_manifest.get("chunk_count") if rag_manifest else 0,
        "embedding_model": rag_manifest.get("embedding_model") if rag_manifest else None,
    }


@router.post("")
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message cannot be empty")
    if rag_chain is None:
        raise _service_error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "RAG_UNAVAILABLE",
            rag_public_error or RAG_UNAVAILABLE_MESSAGE,
        )

    session_id = request.session_id
    is_new_session = not session_id or session_id not in session_memory.sessions
    if is_new_session:
        session_id = session_memory.create_session()

    history_for_rag = session_memory.get_history(session_id, max_turns=5)
    
    q = queue.Queue()
    
    def worker():
        try:
            for item in rag_chain.answer_stream(request.message, history_for_rag):
                q.put(item)
            q.put(None)
        except Exception as e:
            q.put(e)
            
    threading.Thread(target=worker, daemon=True).start()
    
    async def async_generator():
        full_answer = ""
        citations_payload = []
        route_type_payload = "general"
        
        while True:
            item = await asyncio.to_thread(q.get)
            if item is None:
                break
            if isinstance(item, Exception):
                exc = item
                if isinstance(exc, LLMTimeoutError):
                    logger.exception("Chat generation timed out")
                    yield f"event: error\ndata: {json.dumps({'code': exc.code, 'message': LLM_TIMEOUT_MESSAGE}, ensure_ascii=False)}\n\n"
                elif isinstance(exc, LLMAuthenticationError):
                    logger.exception("Chat generation failed because provider authentication was rejected")
                    yield f"event: error\ndata: {json.dumps({'code': exc.code, 'message': LLM_UNAVAILABLE_MESSAGE}, ensure_ascii=False)}\n\n"
                elif isinstance(exc, LLMRateLimitError):
                    logger.exception("Chat generation was rate limited by the provider")
                    yield f"event: error\ndata: {json.dumps({'code': exc.code, 'message': LLM_UNAVAILABLE_MESSAGE}, ensure_ascii=False)}\n\n"
                elif isinstance(exc, LLMConnectionError):
                    logger.exception("Chat generation could not reach the provider")
                    yield f"event: error\ndata: {json.dumps({'code': exc.code, 'message': LLM_UNAVAILABLE_MESSAGE}, ensure_ascii=False)}\n\n"
                elif isinstance(exc, LLMProviderError):
                    logger.exception("Chat generation failed at the provider")
                    yield f"event: error\ndata: {json.dumps({'code': exc.code, 'message': LLM_UNAVAILABLE_MESSAGE}, ensure_ascii=False)}\n\n"
                elif isinstance(exc, RAGRetrievalError):
                    logger.exception("Chat generation dependency failed")
                    yield f"event: error\ndata: {json.dumps({'code': 'RAG_UNAVAILABLE', 'message': RAG_UNAVAILABLE_MESSAGE}, ensure_ascii=False)}\n\n"
                else:
                    logger.exception("Unexpected chat generation failure")
                    yield f"event: error\ndata: {json.dumps({'code': 'INTERNAL_ERROR', 'message': INTERNAL_ERROR_MESSAGE}, ensure_ascii=False)}\n\n"
                return
                
            if item["type"] == "metadata":
                citations_payload = item["citations"]
                route_type_payload = item["route_type"]
                payload = {
                    "type": "metadata",
                    "session_id": session_id,
                    "citations": citations_payload,
                    "route_type": route_type_payload
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            elif item["type"] == "chunk":
                chunk_text = item["text"]
                full_answer += chunk_text
                yield f"data: {json.dumps({'type': 'chunk', 'text': chunk_text}, ensure_ascii=False)}\n\n"
                
        # Save to database
        session_memory.add_message(session_id, "user", request.message)
        session_memory.add_message(session_id, "assistant", full_answer)
        
        records = [
            ChatMessage(session_id=session_id, role="user", content=request.message),
            ChatMessage(
                session_id=session_id,
                role="assistant",
                content=full_answer,
                citations=citations_payload,
                route_type=route_type_payload,
            ),
        ]
        if is_new_session:
            records.insert(0, ChatSession(id=session_id))
        db.add_all(records)
        try:
            await db.commit()
        except Exception:
            await db.rollback()
            logger.exception("Could not persist chat messages")
            
    return StreamingResponse(async_generator(), media_type="text/event-stream")


@router.post("/feedback")
async def chat_feedback(request: FeedbackRequest, db: AsyncSession = Depends(get_db)) -> dict[str, str]:
    return {"status": "ok"}
