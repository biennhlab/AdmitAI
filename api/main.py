import asyncio
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient
from slowapi.errors import RateLimitExceeded
from api.routers import chat, auth, admin
from src.config import settings
from src.database.connection import init_db
from src.retrieval import (
    BM25Search,
    Embedder,
    HybridRetriever,
    QdrantDenseSearch,
    load_dense_index,
)
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
logger = logging.getLogger(__name__)

app = FastAPI(title="AdmitAI API")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_hybrid_retriever() -> tuple[QdrantClient, HybridRetriever, dict[str, Any]]:
    """Load and validate both retrieval branches once for this API process."""
    chunks, _embeddings, manifest = load_dense_index(settings.INDEX_DIR)
    if manifest.get("module") != 3:
        raise RuntimeError(
            f"Index at '{settings.INDEX_DIR}' was built by module {manifest.get('module')}; "
            "run scripts/ingest.py --module 3 before starting the API"
        )
    if manifest.get("embedding_model") != settings.EMBEDDING_MODEL:
        raise RuntimeError(
            f"Index model '{manifest.get('embedding_model')}' differs from configured "
            f"model '{settings.EMBEDDING_MODEL}'; run module 3 ingestion again"
        )

    client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    try:
        if not client.collection_exists(settings.QDRANT_COLLECTION):
            raise RuntimeError(
                f"Qdrant collection '{settings.QDRANT_COLLECTION}' does not exist; "
                "run scripts/ingest.py --module 3"
            )
        qdrant_count = int(
            client.count(collection_name=settings.QDRANT_COLLECTION, exact=True).count
        )
        if qdrant_count != len(chunks):
            raise RuntimeError(
                f"Qdrant collection '{settings.QDRANT_COLLECTION}' has {qdrant_count} chunks, "
                f"but the BM25 snapshot has {len(chunks)}; run module 3 ingestion again"
            )

        embedder = Embedder(model_name=settings.EMBEDDING_MODEL)
        dense_search = QdrantDenseSearch(
            client,
            settings.QDRANT_COLLECTION,
            embedder,
        )
        dense_search.validate_ready(int(manifest["embedding_dimension"]))
        sparse_search = BM25Search()
        sparse_search.index(chunks)
        retriever = HybridRetriever(dense_search, sparse_search)
        return client, retriever, manifest
    except Exception:
        client.close()
        raise

@app.on_event("startup")
async def startup_event():
    await init_db()
    try:
        client, retriever, manifest = await asyncio.to_thread(load_hybrid_retriever)
        try:
            await asyncio.to_thread(
                chat.initialize_rag,
                retriever=retriever,
                manifest=manifest,
            )
        except Exception:
            client.close()
            raise
        app.state.qdrant_client = client
        app.state.hybrid_retriever = retriever
        logger.info(
            "HybridRetriever loaded once at startup: collection=%s, chunks=%s",
            settings.QDRANT_COLLECTION,
            manifest["chunk_count"],
        )
    except Exception as exc:
        chat.mark_rag_unavailable(exc)
        # Health reports the failure and /api/chat returns 503; the API itself
        # stays available so operators can ingest and restart it.
        logger.exception("HybridRetriever startup failed; API is running in degraded mode")


@app.on_event("shutdown")
async def shutdown_event():
    client = getattr(app.state, "qdrant_client", None)
    if client is not None:
        await asyncio.to_thread(client.close)
        app.state.qdrant_client = None
        app.state.hybrid_retriever = None
        logger.info("Qdrant client closed")

app.include_router(chat.router)
app.include_router(auth.router)
app.include_router(admin.router)

@app.get("/api/health")
async def health_check():
    rag = chat.rag_status()
    return {"status": "healthy" if rag["ready"] else "degraded", "rag": rag}


frontend_dir = Path(__file__).resolve().parents[1] / "frontend" / "chatbot"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="chatbot")
