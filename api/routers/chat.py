from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas import ChatRequest, ChatResponse, Citation, FeedbackRequest
from src.config import settings
from src.database.connection import get_db
from src.database.models import ChatMessage, ChatSession
from src.generation.llm_client import LLMClient
from src.generation.rag_chain import RAGChain
from src.generation.session_memory import SessionMemory
from src.retrieval import Embedder, NaiveDenseSearch, load_dense_index

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])

session_memory = SessionMemory()
rag_chain: RAGChain | None = None
rag_manifest: dict[str, Any] | None = None
rag_initialization_error: str | None = None


def initialize_rag(index_dir: str | Path | None = None) -> dict[str, Any]:
    """Load the persisted corpus and vectors once during application startup."""
    global rag_chain, rag_manifest, rag_initialization_error
    try:
        chunks, embeddings, manifest = load_dense_index(index_dir or settings.INDEX_DIR)
        indexed_model = manifest.get("embedding_model")
        if indexed_model != settings.EMBEDDING_MODEL:
            raise ValueError(
                f"Index model '{indexed_model}' differs from configured model '{settings.EMBEDDING_MODEL}'. "
                "Run scripts/ingest.py again."
            )
        embedder = Embedder(model_name=settings.EMBEDDING_MODEL)
        retriever = NaiveDenseSearch(embedder, chunks, chunk_embeddings=embeddings)
        llm_client = LLMClient(api_key=settings.NVIDIA_API_KEY, model=settings.LLM_MODEL)
        rag_chain = RAGChain(
            retriever,
            llm_client,
            top_k=min(5, settings.RETRIEVAL_TOP_K),
            min_score=settings.RETRIEVAL_MIN_SCORE,
            min_lexical_coverage=settings.RETRIEVAL_MIN_LEXICAL_COVERAGE,
        )
        rag_manifest = manifest
        rag_initialization_error = None
        logger.info("RAG ready: %s documents, %s chunks", manifest["document_count"], manifest["chunk_count"])
        return manifest
    except Exception as exc:
        rag_chain = None
        rag_manifest = None
        rag_initialization_error = str(exc)
        logger.exception("RAG initialization failed")
        raise


def rag_status() -> dict[str, Any]:
    return {
        "ready": rag_chain is not None,
        "error": rag_initialization_error,
        "backend": rag_manifest.get("backend") if rag_manifest else None,
        "documents": rag_manifest.get("document_count") if rag_manifest else 0,
        "chunks": rag_manifest.get("chunk_count") if rag_manifest else 0,
        "embedding_model": rag_manifest.get("embedding_model") if rag_manifest else None,
    }


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_db)) -> ChatResponse:
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message cannot be empty")
    if rag_chain is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"RAG index is not ready: {rag_initialization_error or 'run scripts/ingest.py'}",
        )

    session_id = request.session_id
    if not session_id or session_id not in session_memory.sessions:
        session_id = session_memory.create_session()
        db.add(ChatSession(id=session_id))
        try:
            await db.commit()
        except Exception:
            await db.rollback()
            logger.exception("Could not create chat session")
            raise HTTPException(status_code=500, detail="Could not create chat session")

    history_for_rag = session_memory.get_history(session_id, max_turns=5)
    try:
        response = await asyncio.to_thread(
            rag_chain.answer,
            request.message,
            history_for_rag,
        )
    except Exception as exc:
        logger.exception("RAG generation failed")
        raise HTTPException(status_code=503, detail=f"RAG generation failed: {exc}")

    session_memory.add_message(session_id, "user", request.message)
    session_memory.add_message(session_id, "assistant", response.answer)
    citations_payload = response.citations
    db.add_all(
        [
            ChatMessage(session_id=session_id, role="user", content=request.message),
            ChatMessage(
                session_id=session_id,
                role="assistant",
                content=response.answer,
                citations=citations_payload,
                route_type=response.route_type,
            ),
        ]
    )
    try:
        await db.commit()
    except Exception:
        await db.rollback()
        logger.exception("Could not persist chat messages")
        raise HTTPException(status_code=500, detail="Could not persist chat messages")

    return ChatResponse(
        answer=response.answer,
        citations=[Citation(**citation) for citation in citations_payload],
        route_type=response.route_type,
        session_id=session_id,
    )


@router.post("/feedback")
async def chat_feedback(request: FeedbackRequest, db: AsyncSession = Depends(get_db)) -> dict[str, str]:
    return {"status": "ok"}
