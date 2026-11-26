"""LangGraph orchestrator — classifies intent and routes to the right agent."""

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from sqlalchemy.orm import Session
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.agents.file_edit_agent import FileEditAgent
from backend.agents.rag_agent import RAGAgent
from backend.core.config import settings, extract_llm_text
from backend.core.exceptions import EditParseError, LLMError
from backend.core.state import GraphState

# ── Intent classifier prompt ──────────────────────────────────────────────────
CLASSIFIER_PROMPT = """You are a routing assistant. Classify the user message into exactly one category.

Categories:
- "question"  → user is asking something about document content
- "edit"      → user wants to modify a PDF (replace text, redact, remove pages)
- "upload"    → user is uploading or referring to uploading a file

Reply with ONE word only: question | edit | upload

User message: {message}"""


# ── Node functions ────────────────────────────────────────────────────────────

def classify_intent(state: GraphState) -> GraphState:
    """Call Gemini to classify the message intent."""
    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_chat_model,
        google_api_key=settings.gemini_api_key,
        temperature=0,
    )
    try:
        response = llm.invoke(
            CLASSIFIER_PROMPT.format(message=state["message"])
        )
        intent = extract_llm_text(response.content).strip().lower()
        # Normalise — keep only valid values
        if intent not in ("question", "edit", "upload"):
            intent = "question"  # default to RAG if unsure
    except Exception:
        intent = "question"

    return {**state, "intent": intent}


def run_rag(state: GraphState, db: Session) -> GraphState:
    """Route to RAGAgent and store result."""
    try:
        agent = RAGAgent()
        result = agent.answer(
            question=state["message"],
            conversation_id=state["conversation_id"],
            history=state.get("history", []),
            db=db,
        )
        return {**state, "result": result, "error": None}
    except LLMError as exc:
        return {**state, "result": {}, "error": str(exc)}
    except Exception as exc:
        return {**state, "result": {}, "error": f"RAG agent error: {exc}"}


def run_edit(state: GraphState, db: Session) -> GraphState:
    """Route to FileEditAgent and store result."""
    document_id = state.get("document_id")
    if not document_id:
        return {
            **state,
            "result": {},
            "error": "No document_id provided for edit operation.",
        }
    try:
        agent = FileEditAgent()
        doc = agent.edit(
            instruction=state["message"],
            document_id=document_id,
            db=db,
        )
        return {
            **state,
            "result": {
                "document_id": str(doc.id),
                "edited_path": doc.edited_path,
                "filename": doc.filename,
            },
            "error": None,
        }
    except (EditParseError, LLMError) as exc:
        return {**state, "result": {}, "error": str(exc)}
    except Exception as exc:
        return {**state, "result": {}, "error": f"Edit agent error: {exc}"}


def handle_upload(state: GraphState) -> GraphState:
    """Upload is handled before graph invocation at the API layer.

    This node just acknowledges and passes through.
    """
    return {
        **state,
        "result": {"message": "File upload is handled via the /documents endpoint."},
        "error": None,
    }


def route_intent(state: GraphState) -> str:
    """Conditional edge: return the next node name based on classified intent."""
    return state.get("intent", "question")


# ── Build the graph ───────────────────────────────────────────────────────────

def build_orchestrator(db: Session) -> StateGraph:
    """Build and compile the LangGraph StateGraph with the DB session bound to nodes."""

    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("question", lambda s: run_rag(s, db))
    graph.add_node("edit", lambda s: run_edit(s, db))
    graph.add_node("upload", handle_upload)

    # Entry point
    graph.set_entry_point("classify_intent")

    # Conditional routing after classification
    graph.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "question": "question",
            "edit": "edit",
            "upload": "upload",
        },
    )

    # All agent nodes go to END
    graph.add_edge("question", END)
    graph.add_edge("edit", END)
    graph.add_edge("upload", END)

    return graph.compile()
