import os
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.agents.file_edit_agent import FileEditAgent
from backend.agents.orchestrator import build_orchestrator
from backend.core.config import settings
from backend.core.exceptions import InvalidPDFError, DocumentNotFoundError, EditParseError
from backend.core.state import GraphState
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
    """Upload a PDF to a conversation and index it via the orchestrator.

    Flow:
      1. Validate file (mime type, size) — API layer
      2. Save raw bytes to storage/originals/ as a temp file
      3. Invoke orchestrator with intent="upload" (skips LLM classifier)
      4. Orchestrator routes to UploadAgent node → extract → chunk → embed → pgvector
      5. Return created Document metadata
    """
    # 1. Verify conversation exists
    convo = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # 2. Validate file type and size
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=422,
            detail="Only PDF files are accepted. Please upload a .pdf file.",
        )
    file_bytes = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=422,
            detail=f"File too large. Maximum allowed size is {settings.max_upload_size_mb} MB.",
        )

    # 3. Save bytes to a temp path so the upload node can read it
    #    (LangGraph state is a plain dict — no binary bytes)
    os.makedirs(settings.storage_originals_dir, exist_ok=True)
    temp_id = uuid.uuid4()
    temp_path = os.path.join(settings.storage_originals_dir, f"tmp_{temp_id}.pdf")
    with open(temp_path, "wb") as f_out:
        f_out.write(file_bytes)

    # 4. Build state with intent="upload" — orchestrator skips classifier
    state = GraphState(
        conversation_id=str(conversation_id),
        message="",
        history=[],
        documents=[],
        intent="upload",          # pre-set: jumps directly to upload node
        document_id=None,
        file_path=temp_path,      # upload node reads from here
        filename=file.filename,
        result={},
        error=None,
    )

    # 5. Invoke orchestrator — routes to run_upload → UploadAgent
    try:
        orchestrator = build_orchestrator(db)
        final_state: GraphState = orchestrator.invoke(state)
    except Exception as exc:
        # Clean up temp file on failure
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc

    # Clean up temp file (UploadAgent saves its own copy with the real doc UUID)
    if os.path.exists(temp_path):
        os.remove(temp_path)

    if final_state.get("error"):
        error = final_state["error"]
        if "not a valid" in error.lower() or "invalid" in error.lower():
            raise HTTPException(status_code=400, detail=error)
        raise HTTPException(status_code=500, detail=error)

    # 6. Fetch and return the created Document from DB
    result = final_state.get("result", {})
    document_id = result.get("document_id")
    if not document_id:
        raise HTTPException(status_code=500, detail="Upload succeeded but document ID not returned.")

    document = db.query(Document).filter(Document.id == uuid.UUID(document_id)).first()
    if not document:
        raise HTTPException(status_code=500, detail="Document created but not found in DB.")

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
