import os
import uuid

import pymupdf as fitz  # PyMuPDF (use pymupdf import to avoid deprecation warning)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sqlalchemy.orm import Session

from backend.core.config import settings
from backend.core.exceptions import InvalidPDFError
from backend.db.models import Document, DocumentChunk


class UploadAgent:
    """Handles PDF ingestion: validate → extract → chunk → embed → store."""

    def __init__(self) -> None:
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            google_api_key=settings.gemini_api_key,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        os.makedirs(settings.storage_originals_dir, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def ingest(
        self,
        file_bytes: bytes,
        filename: str,
        conversation_id: str,
        db: Session,
    ) -> Document:
        """Ingest a PDF file and store its chunks with embeddings in the DB.

        Args:
            file_bytes:      Raw PDF bytes.
            filename:        Original filename (for display / metadata).
            conversation_id: UUID string of the owning conversation.
            db:              SQLAlchemy session.

        Returns:
            The persisted Document ORM object.

        Raises:
            InvalidPDFError: If the bytes are not a valid, readable PDF.
        """
        # 1. Validate & open
        pdf = self._open_pdf(file_bytes, filename)

        # 2. Extract text page-by-page  →  [(page_number, text), ...]
        pages = self._extract_pages(pdf)

        # 3. Chunk, preserving page number metadata
        chunks = self._chunk_pages(pages)

        # 4. Create Document row first (we need its id for the file path)
        doc_id = uuid.uuid4()
        original_path = self._save_original(file_bytes, doc_id)

        document = Document(
            id=doc_id,
            conversation_id=uuid.UUID(conversation_id),
            filename=filename,
            original_path=original_path,
        )
        db.add(document)
        db.flush()  # write doc row so FK is valid for chunks

        # 5. Embed all chunks in one batched call
        texts = [c["text"] for c in chunks]
        vectors = self.embeddings.embed_documents(texts)

        # 6. Bulk-insert DocumentChunk rows
        chunk_rows = [
            DocumentChunk(
                document_id=doc_id,
                page_number=chunks[i]["page"],
                chunk_text=chunks[i]["text"],
                embedding=vectors[i],
            )
            for i in range(len(chunks))
        ]
        db.add_all(chunk_rows)
        db.commit()
        db.refresh(document)

        return document

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _open_pdf(self, file_bytes: bytes, filename: str) -> fitz.Document:
        """Open and validate a PDF from raw bytes."""
        try:
            pdf = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as exc:
            raise InvalidPDFError(f"Could not open '{filename}' as a PDF: {exc}") from exc

        if pdf.is_encrypted:
            raise InvalidPDFError(f"'{filename}' is password-protected and cannot be processed.")

        if pdf.page_count == 0:
            raise InvalidPDFError(f"'{filename}' contains no pages.")

        return pdf

    def _extract_pages(self, pdf: fitz.Document) -> list[dict]:
        """Extract text from every page.  Returns [{"page": int, "text": str}]."""
        pages = []
        for page_num in range(pdf.page_count):
            page = pdf.load_page(page_num)
            text = page.get_text("text").strip()
            if text:  # skip blank / image-only pages
                pages.append({"page": page_num + 1, "text": text})  # 1-indexed
        return pages

    def _chunk_pages(self, pages: list[dict]) -> list[dict]:
        """Split each page's text into chunks, carrying the page number along."""
        chunks: list[dict] = []
        for page_data in pages:
            page_chunks = self.splitter.split_text(page_data["text"])
            for chunk_text in page_chunks:
                if chunk_text.strip():
                    chunks.append({"page": page_data["page"], "text": chunk_text.strip()})
        return chunks

    def _save_original(self, file_bytes: bytes, doc_id: uuid.UUID) -> str:
        """Persist raw PDF bytes to storage/originals/ and return the path."""
        path = os.path.join(settings.storage_originals_dir, f"{doc_id}.pdf")
        with open(path, "wb") as f:
            f.write(file_bytes)
        return path
