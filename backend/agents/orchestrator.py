"""LangGraph orchestrator — classifies intent and routes to the right agent.

Flow for chat messages (POST /messages):
    entry → classify_intent → [question | edit | upload] → END

Flow for file uploads (POST /documents):
    entry → route_from_entry  (intent already set to "upload", skip classifier)
          → upload → END

The classifier is only called when intent is not pre-set, keeping Gemini calls
to the minimum necessary.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from sqlalchemy.orm import Session

from backend.agents.file_edit_agent import FileEditAgent
from backend.agents.rag_agent import RAGAgent
from backend.agents.upload_agent import UploadAgent
from backend.core.config import settings, extract_llm_text
from backend.core.exceptions import EditParseError, InvalidPDFError, LLMError
from backend.core.state import GraphState

# ── Module-level singleton ────────────────────────────────────────────────────
_classifier_llm: ChatGoogleGenerativeAI | None = None


def _get_classifier_llm() -> ChatGoogleGenerativeAI:
    global _classifier_llm
    if _classifier_llm is None:
        _classifier_llm = ChatGoogleGenerativeAI(
            model=settings.gemini_chat_model,
            google_api_key=settings.gemini_api_key,
        )
    return _classifier_llm

# ── Intent classifier prompt ──────────────────────────────────────────────────
CLASSIFIER_PROMPT = """You are a routing assistant. Classify the user message into exactly one category.

Categories:
- "question"  → user is asking something about document content
- "edit"      → user wants to modify a PDF (replace text, redact, remove pages)
- "upload"    → user is uploading or referring to uploading a file

Reply with ONE word only: question | edit | upload

User message: {message}"""


# ── Node functions ────────────────────────────────────────────────────────────

def route_from_entry(state: GraphState) -> str:
    """Entry-point router.

    If intent is already set (e.g. upload endpoint pre-sets it to 'upload'),
    skip the LLM classifier and jump straight to the right agent node.
    Otherwise go to classify_intent.
    """
    intent = state.get("intent", "")
    if intent in ("question", "edit", "upload"):
        return intent  # skip classifier — go directly to agent node
    return "classify"  # run LLM classifier


def classify_intent(state: GraphState) -> GraphState:
    """Call Gemini to classify the message intent (only for chat messages)."""
    llm = _get_classifier_llm()  # reuse singleton — no new client per request
    try:
        response = llm.invoke(
            CLASSIFIER_PROMPT.format(message=state["message"])
        )
        intent = extract_llm_text(response.content).strip().lower()
        if intent not in ("question", "edit", "upload"):
            intent = "question"
    except Exception:
        intent = "question"

    return {**state, "intent": intent}


def route_after_classify(state: GraphState) -> str:
    """Route after classify_intent node based on classified intent."""
    return state.get("intent", "question")


def run_upload(state: GraphState, db: Session) -> GraphState:
    """Upload node — calls UploadAgent.ingest() with the pre-saved file path."""
    file_path = state.get("file_path")
    filename = state.get("filename", "document.pdf")

    if not file_path:
        return {
            **state,
            "result": {},
            "error": "No file_path in state for upload operation.",
        }
    try:
        # Read bytes from the already-saved file
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        agent = UploadAgent()
        document = agent.ingest(
            file_bytes=file_bytes,
            filename=filename,
            conversation_id=state["conversation_id"],
            db=db,
        )
        return {
            **state,
            "result": {
                "document_id": str(document.id),
                "filename": document.filename,
                "original_path": document.original_path,
            },
            "error": None,
        }
    except InvalidPDFError as exc:
        return {**state, "result": {}, "error": str(exc)}
    except Exception as exc:
        return {**state, "result": {}, "error": f"Upload agent error: {exc}"}


def run_rag(state: GraphState, db: Session) -> GraphState:
    """RAG node — answers questions with page citations."""
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
    """Edit node — applies natural-language PDF edits."""
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
        matches = doc.__dict__.get("_matches", 0)
        return {
            **state,
            "result": {
                "document_id": str(doc.id),
                "edited_path": doc.edited_path,
                "filename": doc.filename,
                "matches": matches,
            },
            "error": None,
        }
    except (EditParseError, LLMError) as exc:
        return {**state, "result": {}, "error": str(exc)}
    except Exception as exc:
        return {**state, "result": {}, "error": f"Edit agent error: {exc}"}


# ── Build the graph ───────────────────────────────────────────────────────────

def build_orchestrator(db: Session) -> StateGraph:
    """Build and compile the LangGraph StateGraph.

    Graph topology:
        entry_router (conditional)
            ├── "classify"  → classify_intent → route_after_classify
            │                     ├── "question" → run_rag  → END
            │                     ├── "edit"     → run_edit → END
            │                     └── "upload"   → run_upload → END
            ├── "question"  → run_rag    → END  (pre-set intent, skip classifier)
            ├── "edit"      → run_edit   → END  (pre-set intent, skip classifier)
            └── "upload"    → run_upload → END  (pre-set intent, skip classifier)
    """
    graph = StateGraph(GraphState)

    # Add all nodes
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("question", lambda s: run_rag(s, db))
    graph.add_node("edit", lambda s: run_edit(s, db))
    graph.add_node("upload", lambda s: run_upload(s, db))

    # Entry point: conditional router
    graph.set_conditional_entry_point(
        route_from_entry,
        {
            "classify":  "classify_intent",
            "question":  "question",
            "edit":      "edit",
            "upload":    "upload",
        },
    )

    # After classifier: route to agent
    graph.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {
            "question": "question",
            "edit":     "edit",
            "upload":   "upload",
        },
    )

    # All agent nodes terminate
    graph.add_edge("question", END)
    graph.add_edge("edit", END)
    graph.add_edge("upload", END)

    return graph.compile()
