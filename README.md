# Multi-Agent PDF Assistant

A production-quality agentic system where users upload PDF documents and interact with them through chat. Four specialized agents collaborate through a LangGraph `StateGraph`, with all state persisted in PostgreSQL and pgvector.

---

## Architecture Overview

```mermaid
graph TD
    UI[Streamlit UI :8501] -->|HTTP| API[FastAPI :8000]
    API --> ORC[Orchestrator\nLangGraph StateGraph]

    subgraph Graph ["LangGraph StateGraph"]
        EP[route_from_entry\nconditional entry point]
        CI[classify_intent\nGemini one-shot]
        UPA[upload node\nUploadAgent]
        RAG[question node\nRAGAgent]
        FEA[edit node\nFileEditAgent]
        EP -->|intent pre-set = upload| UPA
        EP -->|intent pre-set = question/edit| RAG
        EP -->|intent unknown| CI
        CI -->|question| RAG
        CI -->|edit| FEA
        CI -->|upload| UPA
    end

    ORC --> EP
    UPA -->|chunk + embed| PGV[(pgvector)]
    UPA -->|metadata| PG[(PostgreSQL)]
    RAG -->|cosine search| PGV
    RAG -->|generate answer| GEM[Gemini 3.5 Flash Lite]
    FEA -->|parse instruction| GEM
    FEA -->|read original| FS[storage/originals]
    FEA -->|write copy| FS2[storage/edited]
    ORC -->|persist messages| PG
```

### How the Orchestrator Routes Requests

| Trigger | Intent set by | Classifier called? | Agent node |
|---|---|---|---|
| Upload PDF (`POST /documents`) | API layer pre-sets `intent="upload"` | ❌ No — skipped | `run_upload` → UploadAgent |
| Chat question (`POST /messages`) | Gemini classifier | ✅ Yes | `run_rag` → RAGAgent |
| Edit instruction (`POST /messages`) | Gemini classifier | ✅ Yes | `run_edit` → FileEditAgent |

The classifier is only called when intent is genuinely ambiguous (chat messages). Upload always skips it — no wasted LLM call for a known intent.

---

## Agent Responsibilities

| Agent | Responsibility |
|---|---|
| **Orchestrator** | LangGraph `StateGraph` — `route_from_entry` skips classifier for known intents; `classify_intent` calls Gemini for ambiguous chat messages; routes to correct agent node |
| **Upload Agent** | Validates PDF (rejects encrypted/empty), extracts text page-by-page (PyMuPDF), chunks with `RecursiveCharacterTextSplitter`, embeds with `gemini-embedding-001` (3072-dim), stores in pgvector |
| **RAG Agent** | Embeds question, retrieves top-5 chunks by cosine similarity, builds cited prompt, calls Gemini, returns answer with `[p.N]` page citations |
| **File Edit Agent** | Parses natural-language instruction via Gemini, applies replace / redact / remove-pages to a **copy** of the original PDF. Original is never modified. Raises clear error if target text is not found |

---

## State & Persistence

All state flows through `GraphState` (a `TypedDict`) shared between agent nodes:

```python
class GraphState(TypedDict, total=False):
    conversation_id: str       # active conversation
    message: str               # user message text
    history: list[dict]        # last 10 messages (multi-turn context)
    documents: list[dict]      # documents in this conversation
    intent: str                # question | edit | upload
    document_id: str | None    # target doc for edit operations
    file_path: str | None      # saved PDF path for upload node
    filename: str | None       # original filename
    result: dict               # agent output
    error: str | None          # error description if node failed
```

PostgreSQL tables:

| Table | Stores |
|---|---|
| `conversations` | id, title, created_at |
| `messages` | id, conversation_id, role (user/assistant/tool), content, citations (JSON), created_at |
| `documents` | id, conversation_id, filename, original_path, edited_path, created_at |
| `document_chunks` | id, document_id, page_number, chunk_text, embedding Vector(3072) |

Multi-turn conversations survive server restarts — history is rebuilt from the DB on every request.

---

## Setup

### Option A — Docker Compose (recommended)

**Prerequisites:** Docker Desktop, a [Gemini API key](https://aistudio.google.com/app/apikey)

```bash
# 1. Clone
git clone https://github.com/Diwash17/Multi-Agent-PDF-Assistant.git
cd Multi-Agent-PDF-Assistant

# 2. Configure environment
cp .env.example .env
# Open .env and set GEMINI_API_KEY=your-key-here

# 3. Build and start all services (db + backend + frontend)
docker compose up --build

# 4. Services are ready
# Streamlit UI  → http://localhost:8501
# FastAPI docs  → http://localhost:8000/docs
# Health check  → http://localhost:8000/health
```

> **Note:** The backend container runs `alembic upgrade head` automatically on startup. No manual migration step needed.

---

### Option B — Local Development (without Docker)

**Prerequisites:** Python 3.11, [`uv`](https://docs.astral.sh/uv/), Docker Desktop (for the DB only)

```bash
# 1. Clone and install
git clone https://github.com/Diwash17/Multi-Agent-PDF-Assistant.git
cd Multi-Agent-PDF-Assistant
uv venv && uv sync

# 2. Configure environment
cp .env.example .env
# Set GEMINI_API_KEY and ensure DATABASE_URL points to localhost:5433
```

```bash
# 3. Start PostgreSQL with pgvector (Docker)
docker compose up db -d

# 4. Run database migrations
uv run alembic upgrade head

# 5. Start backend
uv run uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload

# 6. Start frontend (new terminal)
uv run streamlit run frontend/app.py --server.port 8501
```

---

## Environment Variables

| Variable | Description | Docker default | Local default |
|---|---|---|---|
| `DATABASE_URL` | PostgreSQL connection string — must use `postgresql+psycopg2://` driver prefix | `postgresql+psycopg2://postgres:postgres@db:5432/pdf_assistant` | `postgresql+psycopg2://postgres:postgres@localhost:5433/pdf_assistant` |
| `GEMINI_API_KEY` | Google Gemini API key — **required**. Get one at [aistudio.google.com](https://aistudio.google.com/app/apikey) | — | — |
| `BACKEND_URL` | URL the Streamlit frontend uses to reach the FastAPI backend | `http://backend:8000` | `http://localhost:8000` |
| `MAX_UPLOAD_SIZE_MB` | Maximum PDF upload size in megabytes | `20` | `20` |

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe → `{"status": "ok"}` |
| `POST` | `/conversations` | Create a new conversation |
| `GET` | `/conversations` | List all conversations (newest first) |
| `GET` | `/conversations/{id}` | Get a single conversation |
| `DELETE` | `/conversations/{id}` | Delete conversation + all messages, documents, chunks (cascade) |
| `POST` | `/conversations/{id}/documents` | Upload and index a PDF via orchestrator |
| `GET` | `/conversations/{id}/documents` | List documents in a conversation |
| `GET` | `/conversations/{id}/documents/{doc_id}/download-edited` | Download the edited PDF copy |
| `POST` | `/conversations/{id}/messages` | Send a message — routed via orchestrator (question or edit) |
| `GET` | `/conversations/{id}/messages` | Get full message history (oldest → newest) |

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend | Python 3.11, FastAPI, Uvicorn |
| Orchestration | LangGraph, LangChain |
| LLM | Google Gemini 3.5 Flash Lite (`gemini-3.5-flash-lite`) |
| Embeddings | Google Gemini Embedding 001 (`gemini-embedding-001`, 3072-dim) |
| Database | PostgreSQL 16 + pgvector extension |
| PDF processing | PyMuPDF (`pymupdf`) |
| Frontend | Streamlit |
| Package manager | `uv` (Python 3.11 pinned) |
| Containers | Docker Compose |

---

## Project Structure

```
Multi-Agent-PDF-Assistant/
├── backend/
│   ├── agents/
│   │   ├── orchestrator.py     # LangGraph StateGraph — entry routing + intent classification
│   │   ├── upload_agent.py     # PDF ingestion: extract → chunk → embed → pgvector
│   │   ├── rag_agent.py        # RAG: embed question → retrieve → Gemini → cited answer
│   │   └── file_edit_agent.py  # PDF editing: replace / redact / remove pages
│   ├── api/
│   │   ├── main.py             # FastAPI app + global exception handlers
│   │   └── routes/
│   │       ├── conversations.py  # CRUD + delete conversations
│   │       ├── documents.py      # Upload (via orchestrator), download edited PDF
│   │       └── messages.py       # Send message (via orchestrator), get history
│   ├── db/
│   │   ├── models.py           # SQLAlchemy ORM: Conversation, Message, Document, DocumentChunk
│   │   └── session.py          # Engine, SessionLocal, get_db() FastAPI dependency
│   ├── core/
│   │   ├── config.py           # Pydantic BaseSettings + extract_llm_text() helper
│   │   ├── state.py            # GraphState TypedDict — shared agent state
│   │   └── exceptions.py       # Typed exceptions: InvalidPDFError, LLMError, EditParseError, …
│   └── storage/
│       ├── originals/          # Uploaded PDFs — never modified
│       └── edited/             # Edited PDF copies only
├── frontend/
│   └── app.py                  # Streamlit UI — upload, chat with citations, edit + download
├── alembic/
│   └── versions/
│       └── 0001_initial_schema.py  # pgvector extension + all tables
├── docker-compose.yml          # db (pgvector), backend, frontend services
├── Dockerfile.backend
├── Dockerfile.frontend
├── pyproject.toml              # uv project — all dependencies declared
└── .env.example                # All required environment variables documented
```

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| Non-PDF upload | 422 — "Only PDF files are accepted" |
| Encrypted / empty PDF | 400 — `InvalidPDFError` with description |
| File too large | 422 — size limit message |
| Target text not found in edit | 400 — "Text 'X' was not found in any page" |
| Unparseable edit instruction | 400 — `EditParseError` with description |
| Gemini failure | 503 — `LLMError` with retry hint; `tenacity` retries 3× with exponential backoff |
| Missing conversation / document | 404 — resource-specific message |
| Unexpected server error | 500 — safe generic message, full traceback logged server-side |
