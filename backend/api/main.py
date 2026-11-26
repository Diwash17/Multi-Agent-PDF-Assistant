from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import conversations, documents, messages

app = FastAPI(
    title="Multi-Agent PDF Assistant",
    description="Agentic PDF assistant — upload, chat, and edit PDFs via multiple specialized agents.",
    version="0.1.0",
)

# Allow Streamlit frontend (and local dev tools) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(conversations.router)
app.include_router(documents.router)
app.include_router(messages.router)


@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    """Liveness probe — returns ok when the service is up."""
    return {"status": "ok"}
