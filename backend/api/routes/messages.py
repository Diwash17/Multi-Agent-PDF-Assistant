import json
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.agents.rag_agent import RAGAgent
from backend.core.exceptions import LLMError
from backend.db.models import Conversation, Message
from backend.db.session import get_db

router = APIRouter(prefix="/conversations", tags=["Messages"])

# How many recent messages to pass as multi-turn context to the RAG agent
HISTORY_WINDOW = 10


# ── Schemas ───────────────────────────────────────────────────────────────────

class MessageCreate(BaseModel):
    content: str


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
    """Send a user message and get an agent reply.

    - Saves the user message to DB
    - Passes last 10 messages as multi-turn context to RAGAgent
    - Saves the assistant reply to DB
    - Returns both messages
    """
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

    # Fetch recent history for multi-turn context (excluding the message just saved)
    recent = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(HISTORY_WINDOW)
        .all()
    )
    history = [{"role": m.role, "content": m.content} for m in reversed(recent)]

    # Call RAGAgent
    try:
        agent = RAGAgent()
        result = agent.answer(
            question=body.content,
            conversation_id=str(conversation_id),
            history=history,
            db=db,
        )
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Fetch the assistant message RAGAgent just persisted
    assistant_msg = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id, Message.role == "assistant")
        .order_by(Message.created_at.desc())
        .first()
    )

    return ChatResponse(
        user_message=_message_to_out(user_msg),
        assistant_message=_message_to_out(assistant_msg),
    )


@router.get("/{conversation_id}/messages", response_model=list[MessageOut])
def get_messages(
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> list[MessageOut]:
    """Return full message history for a conversation (ordered oldest → newest).

    Messages survive server restarts because they are stored in PostgreSQL.
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
