import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient
from sqlalchemy import text
from slowapi.errors import RateLimitExceeded
from api.routers import chat, auth, admin
from src.config import settings
from src.runtime_resources import StartupCheckError, failure_code
from src.qdrant_integrity import inspect_qdrant_points
from src.database.connection import engine, init_db
from src.retrieval import (
    BM25Search,
    Embedder,
    HybridRetriever,
    QdrantDenseSearch,
    load_dense_index,
)
from src.retrieval.index_version import (
    INDEX_BUILD_REVISION,
    index_build_fingerprint,
    manifest_build_stale_reasons,
)
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
logger = logging.getLogger(__name__)
_rag_recovery_lock = asyncio.Lock()

app = FastAPI(title="AdmitAI API")
app.state.rag_recovery = {"blocked": False, "error_code": None, "retry_at": 0.0}


def _record_startup_failure(stage: str, exc: BaseException) -> None:
    code = failure_code(exc, stage)
    blocked = code in {
        "INSUFFICIENT_MEMORY", "MODEL_UNAVAILABLE",
        "INDEX_MISSING", "INDEX_INVALID", "INDEX_FINGERPRINT_MISSING",
        "INDEX_INCOMPATIBLE", "COLLECTION_MISMATCH", "QDRANT_INDEX_MISMATCH",
    }
    app.state.rag_recovery = {
        "blocked": blocked, "error_code": code,
        "retry_at": time.monotonic() + settings.RAG_RECOVERY_COOLDOWN_SECONDS,
    }
    if stage in app.state.components:
        app.state.components[stage]["error_code"] = code
    logger.error("startup_check stage=%s code=%s automatic_retry=%s", stage, code, not blocked)

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _llm_is_configured() -> bool:
    key = settings.LLM_API_KEY.strip()
    return bool(key) and key.lower() not in {"placeholder", "your_gemini_api_key_here"}


def _validate_manifest_collection(manifest: dict[str, Any]) -> None:
    manifest_collection = manifest.get("qdrant_collection")
    if manifest_collection != settings.QDRANT_COLLECTION:
        raise StartupCheckError("COLLECTION_MISMATCH",
            f"Index collection '{manifest_collection}' differs from configured Qdrant "
            f"collection '{settings.QDRANT_COLLECTION}'; update INDEX_DIR and "
            "QDRANT_COLLECTION together"
        )


def load_local_index() -> tuple[list[Any], dict[str, Any]]:
    """Load and validate the local snapshot without contacting other services."""
    try:
        chunks, _embeddings, manifest = load_dense_index(settings.INDEX_DIR)
    except FileNotFoundError as exc:
        raise StartupCheckError("INDEX_MISSING", "Configured index files are missing; check INDEX_DIR before considering any rebuild.") from exc
    if manifest.get("module") != 3:
        raise StartupCheckError("INDEX_INCOMPATIBLE",
            f"Index at '{settings.INDEX_DIR}' was built by module {manifest.get('module')}; "
            "run scripts/ingest.py --module 3 before starting the API"
        )
    if manifest.get("embedding_model") != settings.EMBEDDING_MODEL:
        raise StartupCheckError("INDEX_INCOMPATIBLE",
            f"Index model '{manifest.get('embedding_model')}' differs from configured "
            f"model '{settings.EMBEDDING_MODEL}'; run module 3 ingestion again"
        )
    _validate_manifest_collection(manifest)
    expected_build_fingerprint = index_build_fingerprint(
        3,
        embedding_model=settings.EMBEDDING_MODEL,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        table_chunk_size=settings.TABLE_CHUNK_SIZE,
        parent_chunk_size=settings.PARENT_CHUNK_SIZE,
        child_chunk_size=settings.CHILD_CHUNK_SIZE,
    )
    stale_reasons = manifest_build_stale_reasons(
        manifest,
        expected_corpus_fingerprint=str(manifest.get("corpus_fingerprint") or ""),
        expected_build_fingerprint=expected_build_fingerprint,
        expected_revision=INDEX_BUILD_REVISION,
    )
    if stale_reasons:
        code = "INDEX_FINGERPRINT_MISSING" if any(reason.endswith("_missing") for reason in stale_reasons) else "INDEX_INCOMPATIBLE"
        raise StartupCheckError(code,
            f"Index at '{settings.INDEX_DIR}' is stale ({', '.join(stale_reasons)}); "
            "select the validated index/collection pair or run the read-only migration preflight. "
            "Do not fabricate fingerprints or rebuild a valid existing index."
        )
    return chunks, manifest


def load_hybrid_retriever(
    chunks: list[Any] | None = None,
    manifest: dict[str, Any] | None = None,
) -> tuple[QdrantClient, HybridRetriever, dict[str, Any]]:
    """Load and validate both retrieval branches once for this API process."""
    if chunks is None or manifest is None:
        chunks, manifest = load_local_index()
    _validate_manifest_collection(manifest)

    client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    try:
        if not client.collection_exists(settings.QDRANT_COLLECTION):
            raise StartupCheckError("QDRANT_UNAVAILABLE",
                f"Qdrant collection '{settings.QDRANT_COLLECTION}' does not exist; "
                "verify the configured index/collection pair; no data was changed"
            )
        qdrant_count = int(
            client.count(collection_name=settings.QDRANT_COLLECTION, exact=True).count
        )
        if qdrant_count != len(chunks):
            raise StartupCheckError("QDRANT_INDEX_MISMATCH",
                f"Qdrant collection '{settings.QDRANT_COLLECTION}' has {qdrant_count} chunks, "
                f"but the BM25 snapshot has {len(chunks)}; no data was changed"
            )

        dense_search = QdrantDenseSearch(
            client,
            settings.QDRANT_COLLECTION,
            None,
        )
        try:
            dense_search.validate_ready(int(manifest["embedding_dimension"]))
        except ValueError as exc:
            raise StartupCheckError("QDRANT_INDEX_MISMATCH", "Qdrant vector configuration does not match the local index") from exc
        records = [{"chunk_id": chunk.chunk_id, "content": chunk.content, "metadata": chunk.metadata} for chunk in chunks]
        integrity = inspect_qdrant_points(client, settings.QDRANT_COLLECTION, records)
        if not integrity["compatible"]:
            raise StartupCheckError("QDRANT_INDEX_MISMATCH", "Qdrant point IDs or payloads differ from the local searchable index; no data was changed")
        dense_search.embedder = Embedder(model_name=settings.EMBEDDING_MODEL)
        sparse_search = BM25Search()
        sparse_search.index(chunks)
        retriever = HybridRetriever(
            dense_search, 
            sparse_search,
            parent_chunks=manifest.get("parent_chunks")
        )
        return client, retriever, manifest
    except Exception:
        try:
            client.close()
        except Exception:
            logger.warning("Qdrant client cleanup failed during startup")
        raise

@app.on_event("startup")
async def startup_event():
    app.state.rag_recovery = {"blocked": False, "error_code": None, "retry_at": 0.0}
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
        _record_startup_failure("local_index", exc)
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
        _record_startup_failure("qdrant", exc)
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
        _record_startup_failure("rag", exc)
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
    
    # Auto-reconnect Qdrant and RAG if they failed during startup.
    # Apply the same LLM configuration check used by startup before loading
    # expensive retrieval resources or making the chat service available.
    components["llm"]["configured"] = _llm_is_configured()
    app.state.components["llm"]["configured"] = components["llm"]["configured"]
    if (
        (not components["qdrant"]["ready"] or qdrant_client is None)
        and components["local_index"]["ready"]
        and components["llm"]["configured"]
        and not app.state.rag_recovery["blocked"]
        and time.monotonic() >= app.state.rag_recovery["retry_at"]
    ):
        async with _rag_recovery_lock:
            # Another health request may have completed recovery while waiting.
            qdrant_client = getattr(app.state, "qdrant_client", None)
            if (not app.state.rag_recovery["blocked"]
                and time.monotonic() >= app.state.rag_recovery["retry_at"]
                and (not app.state.components["qdrant"]["ready"] or qdrant_client is None)):
                candidate_client = None
                try:
                    chunks, manifest = await asyncio.to_thread(load_local_index)
                    candidate_client, retriever, _ = await asyncio.to_thread(
                        load_hybrid_retriever, chunks, manifest
                    )
                    await asyncio.to_thread(
                        chat.initialize_rag, retriever=retriever, manifest=manifest
                    )
                    old_client = qdrant_client
                    app.state.qdrant_client = candidate_client
                    app.state.hybrid_retriever = retriever
                    qdrant_client = candidate_client
                    app.state.components["local_index"] = {
                        "ready": True,
                        "documents": int(manifest.get("document_count", 0)),
                        "chunks": len(chunks),
                        "embedding_model": manifest.get("embedding_model"),
                    }
                    app.state.components["qdrant"] = {
                        "ready": True,
                        "collection": settings.QDRANT_COLLECTION,
                        "chunks": len(chunks),
                    }
                    candidate_client = None
                    app.state.rag_recovery = {"blocked": False, "error_code": None, "retry_at": 0.0}
                    if old_client is not None and old_client is not qdrant_client:
                        await asyncio.to_thread(old_client.close)
                except Exception as exc:
                    _record_startup_failure("qdrant", exc)
                    if candidate_client is not None:
                        await asyncio.to_thread(candidate_client.close)
                    logger.exception("RAG recovery failed; API remains in degraded mode")
            components["local_index"] = dict(app.state.components["local_index"])
            components["qdrant"] = dict(app.state.components["qdrant"])
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
    revision = os.getenv("ADMITAI_APPLICATION_REVISION", "")
    return {
        "status": "healthy" if all_ready else "degraded", "components": components,
        "runtime": {
            "pid": os.getpid(), "application_revision": revision if len(revision) == 64 and all(char in "0123456789abcdef" for char in revision) else None,
            "local_dev_profile": settings.LOCAL_DEV_PROFILE,
            "rerank_backend": settings.RERANK_BACKEND,
            "neural_reranker_module_loaded": "src.retrieval.reranker" in sys.modules,
            "recovery_blocked": app.state.rag_recovery["blocked"],
            "startup_error_code": app.state.rag_recovery["error_code"],
        },
    }


frontend_dir = Path(__file__).resolve().parents[1] / "frontend" / "chatbot"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="chatbot")
