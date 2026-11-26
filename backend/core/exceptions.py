"""Custom exception hierarchy for the Multi-Agent PDF Assistant."""


class AppError(Exception):
    """Base class for all application exceptions."""


class InvalidPDFError(AppError):
    """Raised when an uploaded file is not a valid, readable PDF."""


class DocumentNotFoundError(AppError):
    """Raised when a requested document does not exist."""


class ConversationNotFoundError(AppError):
    """Raised when a requested conversation does not exist."""


class LLMError(AppError):
    """Raised when an LLM call fails after all retries."""


class EditParseError(AppError):
    """Raised when the File Edit Agent cannot parse a natural-language instruction."""
