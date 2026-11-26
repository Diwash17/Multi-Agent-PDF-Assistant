import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.agents.upload_agent import UploadAgent
from backend.agents.file_edit_agent import FileEditAgent
from backend.core.config import settings
from backend.core.exceptions import InvalidPDFError, DocumentNotFoundError, EditParseError
from backend.db.models import Conversation, Document
from backend.db.session import get_db

router = APIRouter(prefix="/conversations", tags=["Documents"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class DocumentOut(BaseModel):
    id: uuid.UUID
    conversation_id: uuid.UUID
    filename: str
    original_path: str
    edited_path: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "/{conversation_id}/documents",
    response_model=DocumentOut,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    conversation_id: uuid.UUID,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> Document:
    """Upload a PDF to a conversation and index it for RAG retrieval."""
    # Verify conversation exists
    convo = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Validate MIME type
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        # Also allow octet-stream since some clients don't set content-type correctly;
        # UploadAgent will reject it via fitz if it's not actually a PDF.
        pass
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=422,
            detail="Only PDF files are accepted. Please upload a .pdf file.",
        )

    # Validate file size
    file_bytes = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=422,
            detail=f"File too large. Maximum allowed size is {settings.max_upload_size_mb} MB.",
        )

    # Ingest via UploadAgent
    try:
        agent = UploadAgent()
        document = agent.ingest(
            file_bytes=file_bytes,
            filename=file.filename,
            conversation_id=str(conversation_id),
            db=db,
        )
    except InvalidPDFError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return document


@router.get("/{conversation_id}/documents", response_model=list[DocumentOut])
def list_documents(
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> list[Document]:
    """List all documents uploaded to a conversation."""
    convo = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return (
        db.query(Document)
        .filter(Document.conversation_id == conversation_id)
        .order_by(Document.created_at.desc())
        .all()
    )


@router.get("/{conversation_id}/documents/{document_id}/download-edited")
def download_edited_document(
    conversation_id: uuid.UUID,
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> FileResponse:
    """Download the edited version of a document (created by the File Edit Agent)."""
    doc = (
        db.query(Document)
        .filter(
            Document.id == document_id,
            Document.conversation_id == conversation_id,
        )
        .first()
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if not doc.edited_path:
        raise HTTPException(
            status_code=404,
            detail="No edited version exists for this document yet.",
        )

    return FileResponse(
        path=doc.edited_path,
        media_type="application/pdf",
        filename=f"edited_{doc.filename}",
    )


# ── Edit endpoint ─────────────────────────────────────────────────────────────

class EditRequest(BaseModel):
    instruction: str


@router.post("/{conversation_id}/documents/{document_id}/edit", response_model=DocumentOut)
def edit_document(
    conversation_id: uuid.UUID,
    document_id: uuid.UUID,
    body: EditRequest,
    db: Session = Depends(get_db),
) -> Document:
    """Apply a natural-language edit instruction to a PDF document.

    Supported operations (auto-detected from instruction):
    - Replace text: "Replace 'Confidential' with 'Public'"
    - Redact text:  "Redact all phone numbers" / "Black out 'John Doe'"
    - Remove pages: "Remove page 3" / "Delete pages 1 and 5"

    Original file is never modified. Edited copy saved to storage/edited/.
    """
    # Verify conversation + document belong together
    doc = (
        db.query(Document)
        .filter(
            Document.id == document_id,
            Document.conversation_id == conversation_id,
        )
        .first()
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        agent = FileEditAgent()
        updated_doc = agent.edit(
            instruction=body.instruction,
            document_id=str(document_id),
            db=db,
        )
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except EditParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Edit failed: {exc}") from exc

    return updated_doc
