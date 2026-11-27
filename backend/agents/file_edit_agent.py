import json
import os
import uuid

import pymupdf
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy.orm import Session
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.core.config import settings, extract_llm_text
from backend.core.exceptions import DocumentNotFoundError, EditParseError, LLMError
from backend.db.models import Document

# ── Module-level singleton ────────────────────────────────────────────────────
_llm: ChatGoogleGenerativeAI | None = None


def _get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=settings.gemini_chat_model,
            google_api_key=settings.gemini_api_key,
        )
    return _llm

# ── Gemini prompt to parse a natural-language instruction ─────────────────────
PARSE_PROMPT = """You are a PDF editing assistant. Parse the user's instruction into a structured edit command.

Return a JSON object with exactly this structure (no markdown, just raw JSON):
{{
  "operation": "<replace|redact|remove_pages>",
  "target": "<text to find, OR comma-separated page numbers like '1,3,5'>",
  "replacement": "<replacement text, only required for replace operation>"
}}

Rules:
- operation must be one of: replace, redact, remove_pages
- For replace: target is the text to find, replacement is what to put instead
- For redact: target is the text to black out, replacement is not needed (use "")
- For remove_pages: target is comma-separated 1-indexed page numbers (e.g. "2,4")

User instruction: {instruction}"""


class FileEditAgent:
    """Applies natural-language edit instructions to a copy of a PDF.

    Original files are never modified. Edited copies go to storage/edited/.
    """

    def __init__(self) -> None:
        # Reuse module-level singleton — no new API client per request
        self.llm = _get_llm()
        os.makedirs(settings.storage_edited_dir, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def edit(
        self,
        instruction: str,
        document_id: str,
        db: Session,
    ) -> Document:
        """Apply a natural-language edit to a PDF document.

        Args:
            instruction: e.g. "Replace 'Confidential' with 'Public'"
            document_id: UUID string of the Document to edit.
            db:          SQLAlchemy session.

        Returns:
            Updated Document ORM object with edited_path set.

        Raises:
            DocumentNotFoundError: If the document doesn't exist.
            EditParseError:        If Gemini cannot parse the instruction.
            LLMError:              If the Gemini call fails after retries.
        """
        # 1. Load document
        doc_row = db.query(Document).filter(Document.id == uuid.UUID(document_id)).first()
        if not doc_row:
            raise DocumentNotFoundError(f"Document {document_id} not found.")

        # 2. Parse instruction via Gemini
        cmd = self._parse_instruction(instruction)

        # 3. Apply edit to a copy of the original PDF
        edited_path, matches = self._apply_edit(doc_row.original_path, document_id, cmd)

        # 4. Persist edited_path
        doc_row.edited_path = edited_path
        db.commit()
        db.refresh(doc_row)

        # Attach match count to the doc object so the route can use it
        doc_row.__dict__["_matches"] = matches
        return doc_row

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _parse_instruction(self, instruction: str) -> dict:
        """Call Gemini to parse instruction into a structured EditCommand dict."""
        try:
            response = self.llm.invoke(PARSE_PROMPT.format(instruction=instruction))
            raw = extract_llm_text(response.content).strip()

            # Strip markdown code fences if Gemini wraps it
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            cmd = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EditParseError(
                f"Could not parse edit instruction into a valid command. "
                f"Raw response: {raw!r}"
            ) from exc
        except Exception as exc:
            raise LLMError(f"Gemini call failed during instruction parsing: {exc}") from exc

        # Validate required fields
        op = cmd.get("operation", "")
        if op not in ("replace", "redact", "remove_pages"):
            raise EditParseError(
                f"Unknown operation '{op}'. Must be replace, redact, or remove_pages."
            )
        if not cmd.get("target"):
            raise EditParseError("Parsed command is missing 'target' field.")
        if op == "replace" and not cmd.get("replacement"):
            raise EditParseError("Replace operation requires a 'replacement' field.")

        return cmd

    def _apply_edit(self, original_path: str, document_id: str, cmd: dict) -> tuple[str, int]:
        """Open the original PDF, apply the edit, save to storage/edited/.
        
        Returns:
            Tuple of (edited_path, number_of_matches)
        """
        pdf = pymupdf.open(original_path)
        operation = cmd["operation"]
        matches = 0

        if operation == "replace":
            matches = self._do_replace(pdf, cmd["target"], cmd["replacement"])
        elif operation == "redact":
            matches = self._do_redact(pdf, cmd["target"])
        elif operation == "remove_pages":
            matches = self._do_remove_pages(pdf, cmd["target"])

        # Save edited copy
        edited_path = os.path.join(
            settings.storage_edited_dir, f"{document_id}_edited.pdf"
        )
        pdf.save(edited_path)
        pdf.close()
        return edited_path, matches

    # ── PyMuPDF operations ────────────────────────────────────────────────────

    def _do_replace(self, pdf: pymupdf.Document, target: str, replacement: str) -> int:
        """Replace all occurrences of target text with replacement on every page.

        Strategy:
        1. Find all rects for target text
        2. Expand each rect slightly to ensure full glyph coverage
        3. Redact with white fill (physically removes original text)
        4. Insert replacement at stored baseline position
        """
        total_found = 0
        for page in pdf:
            instances = page.search_for(target)
            if not instances:
                continue
            total_found += len(instances)

            inserts = []
            for rect in instances:
                fontsize = max(6.0, round(rect.height * 0.72, 1))
                baseline_y = rect.y1 - rect.height * 0.18
                x0 = rect.x0

                # Expand rect by 1pt on all sides to catch full glyphs
                expanded = pymupdf.Rect(
                    rect.x0 - 1,
                    rect.y0 - 1,
                    rect.x1 + 1,
                    rect.y1 + 1,
                )
                page.add_redact_annot(expanded, fill=(1, 1, 1))
                inserts.append((x0, baseline_y, fontsize))

            page.apply_redactions()

            for x0, baseline_y, fontsize in inserts:
                page.insert_text(
                    pymupdf.Point(x0, baseline_y),
                    replacement,
                    fontsize=fontsize,
                    color=(0, 0, 0),
                    overlay=True,
                )

        if total_found == 0:
            raise EditParseError(
                f"Text '{target}' was not found in any page of the document. "
                "Check the exact spelling and try again."
            )
        return total_found

    def _do_redact(self, pdf: pymupdf.Document, target: str) -> int:
        """Black out all occurrences of target text using PyMuPDF redactions."""
        total_found = 0
        for page in pdf:
            instances = page.search_for(target)
            total_found += len(instances)
            for rect in instances:
                page.add_redact_annot(rect, fill=(0, 0, 0))
            if instances:
                page.apply_redactions()
        if total_found == 0:
            raise EditParseError(
                f"Text '{target}' was not found in any page of the document. "
                "Nothing was redacted."
            )
        return total_found

    def _do_remove_pages(self, pdf: pymupdf.Document, target: str) -> int:
        """Remove the specified 1-indexed page numbers from the PDF."""
        try:
            # Parse "1,3,5" → [0, 2, 4] (0-indexed for PyMuPDF)
            page_nums = [int(p.strip()) - 1 for p in target.split(",")]
            # Validate range
            valid = [p for p in page_nums if 0 <= p < pdf.page_count]
            if not valid:
                raise EditParseError(
                    f"No valid pages to remove. Document has {pdf.page_count} pages."
                )
            pdf.delete_pages(valid)
            return len(valid)
        except ValueError as exc:
            raise EditParseError(
                f"Could not parse page numbers from '{target}'. "
                "Expected comma-separated integers like '1,3'."
            ) from exc
