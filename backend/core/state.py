from typing import Any
from typing_extensions import TypedDict


class GraphState(TypedDict, total=False):
    """Shared state passed between all nodes in the LangGraph orchestrator.

    Fields
    ------
    conversation_id : str
        UUID of the active conversation.
    message : str
        The latest user message text.
    history : list[dict]
        Recent message history [{"role": str, "content": str}, ...].
    documents : list[dict]
        Metadata of documents uploaded to this conversation.
    intent : str
        Classified intent: "question" | "edit" | "upload".
    document_id : str | None
        Target document UUID (required for edit operations).
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
    result: dict[str, Any]
    error: str | None
