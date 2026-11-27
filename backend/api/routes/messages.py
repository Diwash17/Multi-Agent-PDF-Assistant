import json
import logging
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.agents.orchestrator import build_orchestrator
from backend.core.state import GraphState
from backend.db.models import Conversation, Document, Message
from backend.db.session import get_db

router = APIRouter(prefix="/conversations", tags=["Messages"])
logger = logging.getLogger(__name__)

HISTORY_WINDOW = 10


# ── Schemas ───────────────────────────────────────────────────────────────────

class MessageCreate(BaseModel):
    content: str
    document_id: str | None = None  # required for edit operations


class CitationOut(BaseModel):
    page: int
    chunk: str


class MessageOut(BaseModel):
    id: uuid.UUID
    conversation_id: uuid.UUID
    role: str
    content: str
    citations: list[CitationOut] | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatResponse(BaseModel):
    user_message: MessageOut
    assistant_message: MessageOut


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_citations(citations_json: str | None) -> list[dict[str, Any]] | None:
    if not citations_json:
        return None
    try:
        return json.loads(citations_json)
    except (json.JSONDecodeError, TypeError):
        return None


def _message_to_out(msg: Message) -> MessageOut:
    citations = _parse_citations(msg.citations)
    return MessageOut(
        id=msg.id,
        conversation_id=msg.conversation_id,
        role=msg.role,
        content=msg.content,
        citations=[CitationOut(**c) for c in citations] if citations else None,
        created_at=msg.created_at,
    )


def _build_state(
    conversation_id: uuid.UUID,
    message: str,
    document_id: str | None,
    db: Session,
) -> GraphState:
    """Build the initial GraphState from DB history."""
    recent = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(HISTORY_WINDOW)
        .all()
    )
    history = [{"role": m.role, "content": m.content} for m in reversed(recent)]

    documents = db.query(Document).filter(Document.conversation_id == conversation_id).all()
    doc_list = [{"id": str(d.id), "filename": d.filename} for d in documents]

    return GraphState(
        conversation_id=str(conversation_id),
        message=message,
        history=history,
        documents=doc_list,
        document_id=document_id,
        intent="",
        result={},
        error=None,
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "/{conversation_id}/messages",
    response_model=ChatResponse,
    status_code=status.HTTP_201_CREATED,
)
def send_message(
    conversation_id: uuid.UUID,
    body: MessageCreate,
    db: Session = Depends(get_db),
) -> ChatResponse:
    """Send a user message through the orchestrator and get an agent reply."""
    # Verify conversation exists
    convo = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Persist user message
    user_msg = Message(
        conversation_id=conversation_id,
        role="user",
        content=body.content,
    )
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)

    # Build state and invoke orchestrator
    state = _build_state(conversation_id, body.content, body.document_id, db)
    orchestrator = build_orchestrator(db)

    try:
        final_state: GraphState = orchestrator.invoke(state)
    except Exception as exc:
        logger.exception("Orchestrator invoke failed")
        raise HTTPException(status_code=500, detail=f"Orchestrator error: {exc}") from exc

    if final_state.get("error"):
        raise HTTPException(status_code=500, detail=final_state["error"])

    # Build reply content — route owns ALL message persistence
    result = final_state.get("result", {})
    intent = final_state.get("intent", "question")

    if intent == "edit":
        filename = result.get("filename", "document")
        matches = result.get("matches", 0)
        content = (
            f"✅ Done! Applied your edit to **'{filename}'** "
            f"({matches} occurrence{'s' if matches != 1 else ''} changed). "
            "Download the edited version below."
        )
        citations_json = None
    else:
        content = result.get("answer", "I could not generate a response.")
        raw_citations = result.get("citations", [])
        citations_json = json.dumps(raw_citations) if raw_citations else None

    assistant_msg = Message(
        conversation_id=conversation_id,
        role="assistant",
        content=content,
        citations=citations_json,
    )
    db.add(assistant_msg)
    db.commit()
    db.refresh(assistant_msg)

    return ChatResponse(
        user_message=_message_to_out(user_msg),
        assistant_message=_message_to_out(assistant_msg),
    )


@router.get("/{conversation_id}/messages", response_model=list[MessageOut])
def get_messages(
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> list[MessageOut]:
    """Return full message history (oldest → newest).

    Messages survive server restarts — stored in PostgreSQL.
    """
    convo = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .all()
    )
    return [_message_to_out(m) for m in messages]
