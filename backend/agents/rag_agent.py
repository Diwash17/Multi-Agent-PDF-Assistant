import uuid
from typing import Any

from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from sqlalchemy import select
from sqlalchemy.orm import Session
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.core.config import settings
from backend.core.exceptions import LLMError
from backend.db.models import Document, DocumentChunk, Message

# Number of chunks to retrieve per query
TOP_K = 5

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about uploaded PDF documents.

You will be given relevant excerpts from the document along with their page numbers.
Always base your answer strictly on the provided context.
Cite the page number for every piece of information you use, using the format [p.N].
If the answer cannot be found in the context, say "I could not find that information in the uploaded documents."

Context:
{context}
"""


class RAGAgent:
    """Retrieval-Augmented Generation agent.

    Embeds the user question → retrieves top-K chunks from pgvector →
    builds a cited prompt → calls Gemini → returns answer + citations.
    """

    def __init__(self) -> None:
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            google_api_key=settings.gemini_api_key,
        )
        self.llm = ChatGoogleGenerativeAI(
            model=settings.gemini_chat_model,
            google_api_key=settings.gemini_api_key,
            temperature=0.2,
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{question}"),
            ]
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def answer(
        self,
        question: str,
        conversation_id: str,
        history: list[dict],
        db: Session,
    ) -> dict[str, Any]:
        """Answer a question using retrieved document chunks.

        Args:
            question:        The user's question.
            conversation_id: UUID string of the conversation.
            history:         Recent message history for multi-turn context.
            db:              SQLAlchemy session.

        Returns:
            {"answer": str, "citations": [{"page": int, "chunk": str}]}
        """
        # 1. Embed the question
        question_vector = self._embed_question(question)

        # 2. Retrieve top-K relevant chunks from pgvector
        chunks = self._retrieve_chunks(question_vector, conversation_id, db)

        if not chunks:
            answer_text = "I could not find any relevant content in the uploaded documents."
            citations: list[dict] = []
        else:
            # 3. Build context string with page citations
            context, citations = self._build_context(chunks)

            # 4. Call Gemini
            answer_text = self._generate_answer(question, context, history)

        # 5. Persist assistant message to DB
        self._save_message(
            conversation_id=conversation_id,
            content=answer_text,
            citations=citations,
            db=db,
        )

        return {"answer": answer_text, "citations": citations}

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _embed_question(self, question: str) -> list[float]:
        """Embed the question text into a vector."""
        try:
            return self.embeddings.embed_query(question)
        except Exception as exc:
            raise LLMError(f"Failed to embed question: {exc}") from exc

    def _retrieve_chunks(
        self,
        question_vector: list[float],
        conversation_id: str,
        db: Session,
    ) -> list[DocumentChunk]:
        """Retrieve top-K chunks from pgvector using cosine distance."""
        # Join DocumentChunk → Document to filter by conversation
        stmt = (
            select(DocumentChunk)
            .join(Document, DocumentChunk.document_id == Document.id)
            .filter(Document.conversation_id == uuid.UUID(conversation_id))
            .order_by(DocumentChunk.embedding.cosine_distance(question_vector))
            .limit(TOP_K)
        )
        return list(db.execute(stmt).scalars().all())

    def _build_context(
        self, chunks: list[DocumentChunk]
    ) -> tuple[str, list[dict]]:
        """Format retrieved chunks into a context string and citation list."""
        context_parts = []
        citations = []
        for chunk in chunks:
            context_parts.append(f"[p.{chunk.page_number}] {chunk.chunk_text}")
            citations.append({"page": chunk.page_number, "chunk": chunk.chunk_text})
        return "\n\n".join(context_parts), citations

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _generate_answer(
        self,
        question: str,
        context: str,
        history: list[dict],
    ) -> str:
        """Call Gemini with the retrieved context and return the answer text."""
        try:
            chain = self.prompt | self.llm
            response = chain.invoke({"context": context, "question": question})
            return response.content
        except Exception as exc:
            raise LLMError(f"Gemini call failed: {exc}") from exc

    def _save_message(
        self,
        conversation_id: str,
        content: str,
        citations: list[dict],
        db: Session,
    ) -> Message:
        """Persist the assistant reply as a Message row."""
        import json

        msg = Message(
            conversation_id=uuid.UUID(conversation_id),
            role="assistant",
            content=content,
            citations=json.dumps(citations) if citations else None,
        )
        db.add(msg)
        db.commit()
        db.refresh(msg)
        return msg
