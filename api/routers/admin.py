from fastapi import APIRouter, Depends, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from api.schemas import EscalationReplyRequest
from src.database.connection import get_db
from src.auth.dependencies import get_current_user
from src.database.models import User

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(get_current_user)])

@router.get("/chat-history")
async def get_chat_history(page: int = 1, limit: int = 20, status: str = "all", db: AsyncSession = Depends(get_db)):
    # Phase 0 Stub
    return {"items": [], "total": 0, "page": page, "limit": limit}

@router.get("/escalations")
async def get_escalations(page: int = 1, limit: int = 20, status: str = "pending", db: AsyncSession = Depends(get_db)):
    # Phase 0 Stub
    return {"items": [], "total": 0, "page": page, "limit": limit}

@router.post("/escalations/{id}/reply")
async def reply_escalation(id: int, request: EscalationReplyRequest, db: AsyncSession = Depends(get_db)):
    # Phase 0 Stub
    return {"status": "stub"}

@router.get("/documents")
async def list_documents(db: AsyncSession = Depends(get_db)):
    # Phase 0 Stub
    return {"items": [], "total": 0}

@router.post("/documents/upload", status_code=202)
async def upload_document(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    # Phase 0 Stub
    return {"id": 1, "filename": file.filename, "status": "processing", "message": "Document uploaded. Indexing in progress..."}

@router.get("/documents/{id}")
async def get_document(id: int, db: AsyncSession = Depends(get_db)):
    # Phase 0 Stub
    return {"id": id, "status": "indexed"}

@router.delete("/documents/{id}")
async def delete_document(id: int, db: AsyncSession = Depends(get_db)):
    # Phase 0 Stub
    return {"message": f"Document {id} deleted successfully"}
