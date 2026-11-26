import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routes import conversations, documents, messages
from backend.core.exceptions import (
    ConversationNotFoundError,
    DocumentNotFoundError,
    EditParseError,
    InvalidPDFError,
    LLMError,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multi-Agent PDF Assistant",
    description=(
        "Agentic PDF assistant — upload, chat, and edit PDFs "
        "via multiple specialized agents."
    ),
    version="0.1.0",
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(conversations.router)
app.include_router(documents.router)
app.include_router(messages.router)


# ── Global exception handlers ─────────────────────────────────────────────────

@app.exception_handler(InvalidPDFError)
async def invalid_pdf_handler(request: Request, exc: InvalidPDFError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"error": "InvalidPDFError", "detail": str(exc)},
    )


@app.exception_handler(DocumentNotFoundError)
async def document_not_found_handler(request: Request, exc: DocumentNotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"error": "DocumentNotFoundError", "detail": str(exc)},
    )


@app.exception_handler(ConversationNotFoundError)
async def conversation_not_found_handler(
    request: Request, exc: ConversationNotFoundError
) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"error": "ConversationNotFoundError", "detail": str(exc)},
    )


@app.exception_handler(EditParseError)
async def edit_parse_handler(request: Request, exc: EditParseError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"error": "EditParseError", "detail": str(exc)},
    )


@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
    logger.error("LLM failure: %s", exc)
    return JSONResponse(
        status_code=503,
        content={
            "error": "LLMError",
            "detail": str(exc),
            "hint": "The AI service is temporarily unavailable. Please try again.",
        },
    )


@app.exception_handler(Exception)
async def generic_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content={"error": "InternalServerError", "detail": "An unexpected error occurred."},
    )


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    """Liveness probe — returns ok when the service is up."""
    return {"status": "ok"}
