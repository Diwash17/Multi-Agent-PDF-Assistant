from typing import Any
from typing_extensions import TypedDict


class GraphState(TypedDict, total=False):
    """Shared state passed between all nodes in the LangGraph orchestrator.

    Fields
    ------
    conversation_id : str
        UUID of the active conversation.
    message : str
        The latest user message text (empty string for upload events).
    history : list[dict]
        Recent message history [{"role": str, "content": str}, ...].
    documents : list[dict]
        Metadata of documents uploaded to this conversation.
    intent : str
        Classified intent: "question" | "edit" | "upload".
        When pre-set by the caller (e.g. upload endpoint), the classifier
        node is skipped entirely — LangGraph routes directly to the agent.
    document_id : str | None
        Target document UUID (required for edit operations).
    file_path : str | None
        Absolute path to the saved PDF file on disk.
        Set by the upload endpoint before invoking the graph so UploadAgent
        can read the file without binary bytes in state.
    filename : str | None
        Original filename of the uploaded PDF (for display / metadata).
    result : dict
        Output from the agent that handled the request.
    error : str | None
        Error description if any node failed.
    """

    conversation_id: str
    message: str
    history: list[dict[str, Any]]
    documents: list[dict[str, Any]]
    intent: str
    document_id: str | None
    file_path: str | None
    filename: str | None
    result: dict[str, Any]
    error: str | None
