import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded
from api.routers import chat, auth, admin
from src.database.connection import init_db
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

@app.on_event("startup")
async def startup_event():
    await init_db()
    try:
        await asyncio.to_thread(chat.initialize_rag)
    except Exception:
        # Health reports the failure and /api/chat returns 503; the API itself
        # stays available so operators can ingest and restart it.
        logger.exception("API started without a ready RAG index")

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
