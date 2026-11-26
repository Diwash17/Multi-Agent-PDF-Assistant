from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "postgresql+psycopg2://postgres:postgres@localhost:5433/pdf_assistant"

    # ── Google Gemini ─────────────────────────────────────────────────────────
    gemini_api_key: str = ""

    # ── Service URLs ──────────────────────────────────────────────────────────
    backend_url: str = "http://localhost:8000"

    # ── Upload limits ─────────────────────────────────────────────────────────
    max_upload_size_mb: int = 20

    # ── Storage paths ─────────────────────────────────────────────────────────
    storage_originals_dir: str = "backend/storage/originals"
    storage_edited_dir: str = "backend/storage/edited"

    # ── LLM model names ───────────────────────────────────────────────────────
    gemini_chat_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "gemini-embedding-001"


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings instance (read once per process)."""
    return Settings()


# Module-level singleton for convenience
settings = get_settings()
