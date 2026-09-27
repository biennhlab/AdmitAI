import asyncio
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from sqlalchemy import text
from slowapi.errors import RateLimitExceeded
from api.routers import chat, auth, admin
from src.config import settings
from src.database.connection import engine, init_db
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

_NOT_STARTED = {"ready": False, "error": "Startup check has not completed."}
app.state.components = {
    "database": dict(_NOT_STARTED),
    "local_index": dict(_NOT_STARTED),
    "qdrant": dict(_NOT_STARTED),
    "llm": {"configured": False},
}

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in settings.CORS_ORIGINS.split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _llm_is_configured() -> bool:
    key = settings.LLM_API_KEY.strip()
    return bool(key) and key.lower() not in {"placeholder", "your_gemini_api_key_here"}


def load_local_index() -> tuple[list[Any], dict[str, Any]]:
    """Load and validate the local snapshot without contacting other services."""
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
    return chunks, manifest


def load_hybrid_retriever(
    chunks: list[Any] | None = None,
    manifest: dict[str, Any] | None = None,
) -> tuple[QdrantClient, HybridRetriever, dict[str, Any]]:
    """Load and validate both retrieval branches once for this API process."""
    if chunks is None or manifest is None:
        chunks, manifest = load_local_index()

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
        retriever = HybridRetriever(
            dense_search, 
            sparse_search,
            parent_chunks=manifest.get("parent_chunks")
        )
        return client, retriever, manifest
    except Exception:
        client.close()
        raise

@app.on_event("startup")
async def startup_event():
    app.state.components = {
        "database": {"ready": False, "error": "Database initialization failed."},
        "local_index": {"ready": False, "error": "Local index is unavailable or invalid."},
        "qdrant": {"ready": False, "error": "Qdrant is unavailable or out of sync."},
        "llm": {"configured": _llm_is_configured()},
    }

    try:
        await init_db()
        app.state.components["database"] = {"ready": True}
    except Exception:
        logger.exception("Database startup check failed; API is running in degraded mode")

    try:
        chunks, manifest = await asyncio.to_thread(load_local_index)
        app.state.components["local_index"] = {
            "ready": True,
            "documents": int(manifest.get("document_count", 0)),
            "chunks": len(chunks),
            "embedding_model": manifest.get("embedding_model"),
        }
    except Exception as exc:
        chat.mark_rag_unavailable(exc, "Chỉ mục tuyển sinh đang tạm thời chưa sẵn sàng.")
        logger.exception("Local index startup check failed; API is running in degraded mode")
        return

    try:
        client, retriever, manifest = await asyncio.to_thread(
            load_hybrid_retriever,
            chunks,
            manifest,
        )
        app.state.components["qdrant"] = {
            "ready": True,
            "collection": settings.QDRANT_COLLECTION,
            "chunks": len(chunks),
        }
    except Exception as exc:
        chat.mark_rag_unavailable(exc, "Kho dữ liệu tuyển sinh đang tạm thời chưa sẵn sàng.")
        logger.exception("Qdrant startup check failed; API is running in degraded mode")
        return

    if not app.state.components["llm"]["configured"]:
        client.close()
        chat.mark_rag_unavailable(
            RuntimeError("LLM_API_KEY is not configured"),
            "Dịch vụ xử lý câu hỏi đang tạm thời chưa sẵn sàng.",
        )
        logger.error("LLM_API_KEY is not configured; API is running in degraded mode")
        return

    try:
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
    components = {name: dict(value) for name, value in app.state.components.items()}

    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        components["database"] = {"ready": True}
    except Exception:
        components["database"] = {
            "ready": False,
            "error": "Database is unavailable.",
        }

    if components["local_index"]["ready"]:
        index_dir = Path(settings.INDEX_DIR)
        if not (index_dir / "manifest.json").is_file() or not (index_dir / "embeddings.npy").is_file():
            components["local_index"] = {
                "ready": False,
                "error": "Local index files are unavailable.",
            }

    qdrant_client = getattr(app.state, "qdrant_client", None)
    
    # Auto-reconnect Qdrant and RAG if they failed during startup
    if (not components["qdrant"]["ready"] or qdrant_client is None) and components["local_index"]["ready"]:
        try:
            chunks, manifest = await asyncio.to_thread(load_local_index)
            client, retriever, _ = await asyncio.to_thread(
                load_hybrid_retriever, chunks, manifest
            )
            app.state.qdrant_client = client
            app.state.hybrid_retriever = retriever
            qdrant_client = client
            await asyncio.to_thread(
                chat.initialize_rag, retriever=retriever, manifest=manifest
            )
            app.state.components["qdrant"] = {
                "ready": True,
                "collection": settings.QDRANT_COLLECTION,
                "chunks": len(chunks),
            }
            components["qdrant"] = dict(app.state.components["qdrant"])
            components["llm"]["configured"] = _llm_is_configured()
            app.state.components["llm"]["configured"] = components["llm"]["configured"]
        except Exception:
            pass
    if components["qdrant"]["ready"] and qdrant_client is not None:
        try:
            collection_exists = await asyncio.to_thread(
                qdrant_client.collection_exists,
                settings.QDRANT_COLLECTION,
            )
            count = await asyncio.to_thread(
                qdrant_client.count,
                collection_name=settings.QDRANT_COLLECTION,
                exact=True,
            )
            expected_chunks = int(components["local_index"].get("chunks", 0))
            actual_chunks = int(count.count)
            if not collection_exists or (expected_chunks and actual_chunks != expected_chunks):
                raise RuntimeError("Qdrant collection is missing or out of sync")
            components["qdrant"] = {
                "ready": True,
                "collection": settings.QDRANT_COLLECTION,
                "chunks": actual_chunks,
            }
        except Exception:
            components["qdrant"] = {
                "ready": False,
                "error": "Qdrant is unavailable or out of sync.",
            }

    rag = chat.rag_status()
    if not components["local_index"]["ready"] or not components["qdrant"]["ready"]:
        rag = {
            **rag,
            "ready": False,
            "error": "Dịch vụ dữ liệu tuyển sinh đang tạm thời chưa sẵn sàng.",
        }
    components["rag"] = rag
    all_ready = (
        components["database"]["ready"]
        and components["local_index"]["ready"]
        and components["qdrant"]["ready"]
        and components["llm"]["configured"]
        and components["rag"]["ready"]
    )
    return {"status": "healthy" if all_ready else "degraded", "components": components}
