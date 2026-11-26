import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.db.models import Conversation
from backend.db.session import get_db

router = APIRouter(prefix="/conversations", tags=["Conversations"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class ConversationCreate(BaseModel):
    title: str = "New Conversation"


class ConversationOut(BaseModel):
    id: uuid.UUID
    title: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("", response_model=ConversationOut, status_code=status.HTTP_201_CREATED)
def create_conversation(
    body: ConversationCreate,
    db: Session = Depends(get_db),
) -> Conversation:
    """Create a new conversation session."""
    convo = Conversation(title=body.title)
    db.add(convo)
    db.commit()
    db.refresh(convo)
    return convo


@router.get("", response_model=list[ConversationOut])
def list_conversations(db: Session = Depends(get_db)) -> list[Conversation]:
    """Return all conversations ordered by most recent first."""
    return db.query(Conversation).order_by(Conversation.created_at.desc()).all()


@router.get("/{conversation_id}", response_model=ConversationOut)
def get_conversation(
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> Conversation:
    """Return a single conversation by ID."""
    convo = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return convo
