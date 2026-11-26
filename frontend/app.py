"""Multi-Agent PDF Assistant — Streamlit frontend."""

import os

import httpx
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="PDF Assistant",
    page_icon="📄",
    layout="wide",
)

# ── Shared HTTP client ────────────────────────────────────────────────────────

def api(method: str, path: str, **kwargs):
    """Make an API call and return the parsed JSON (or raise on error)."""
    url = f"{BACKEND_URL}{path}"
    try:
        resp = httpx.request(method, url, timeout=120, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        detail = e.response.json().get("detail", str(e)) if e.response.content else str(e)
        st.error(f"API error: {detail}")
        return None
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None


# ── Session state helpers ─────────────────────────────────────────────────────

def init_state():
    defaults = {
        "conversation_id": None,
        "conversation_title": None,
        "messages": [],
        "documents": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def load_messages():
    if not st.session_state.conversation_id:
        return
    data = api("GET", f"/conversations/{st.session_state.conversation_id}/messages")
    if data is not None:
        st.session_state.messages = data


def load_documents():
    if not st.session_state.conversation_id:
        return
    data = api("GET", f"/conversations/{st.session_state.conversation_id}/documents")
    if data is not None:
        st.session_state.documents = data


# ── Sidebar — conversation management ────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.title("📄 PDF Assistant")
        st.divider()

        # New conversation
        with st.form("new_conv_form", clear_on_submit=True):
            title = st.text_input("Conversation title", placeholder="My PDF chat")
            submitted = st.form_submit_button("➕ New Conversation", use_container_width=True)
            if submitted:
                t = title.strip() or "New Conversation"
                result = api("POST", "/conversations", json={"title": t})
                if result:
                    st.session_state.conversation_id = result["id"]
                    st.session_state.conversation_title = result["title"]
                    st.session_state.messages = []
                    st.session_state.documents = []
                    st.success(f"Created: {t}")
                    st.rerun()

        st.divider()

        # List existing conversations
        convos = api("GET", "/conversations") or []
        if convos:
            st.subheader("Conversations")
            for c in convos:
                label = f"💬 {c['title']}"
                if st.button(label, key=c["id"], use_container_width=True):
                    st.session_state.conversation_id = c["id"]
                    st.session_state.conversation_title = c["title"]
                    load_messages()
                    load_documents()
                    st.rerun()
        else:
            st.info("No conversations yet. Create one above.")

        # Active conversation info
        if st.session_state.conversation_id:
            st.divider()
            st.caption(f"Active: **{st.session_state.conversation_title}**")
            st.caption(f"`{st.session_state.conversation_id}`")


# ── Main area ─────────────────────────────────────────────────────────────────

def render_main():
    if not st.session_state.conversation_id:
        st.markdown("## Welcome to PDF Assistant 👋")
        st.markdown(
            "Create or select a conversation from the sidebar to get started.\n\n"
            "**What you can do:**\n"
            "- 📤 Upload PDF documents\n"
            "- 💬 Ask questions and get answers with page citations\n"
            "- ✏️ Edit PDFs with natural language (replace, redact, remove pages)\n"
            "- 📥 Download edited documents"
        )
        return

    st.title(f"💬 {st.session_state.conversation_title}")

    tab_chat, tab_upload, tab_edit = st.tabs(["💬 Chat", "📤 Upload PDF", "✏️ Edit PDF"])

    # ── Chat tab ──────────────────────────────────────────────────────────────
    with tab_chat:
        render_chat()

    # ── Upload tab ────────────────────────────────────────────────────────────
    with tab_upload:
        render_upload()

    # ── Edit tab ──────────────────────────────────────────────────────────────
    with tab_edit:
        render_edit()


# ── Chat tab ──────────────────────────────────────────────────────────────────

def render_chat():
    # Reload messages if none loaded yet
    if not st.session_state.messages:
        load_messages()

    # Display message history
    for msg in st.session_state.messages:
        role = msg["role"]
        if role == "tool":
            continue  # skip internal tool messages
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(msg["content"])
            # Show citations if present
            if msg.get("citations"):
                with st.expander(f"📚 Sources ({len(msg['citations'])} chunks)", expanded=False):
                    for c in msg["citations"]:
                        st.markdown(f"**Page {c['page']}**")
                        st.text(c["chunk"][:300] + ("..." if len(c["chunk"]) > 300 else ""))
                        st.divider()

    # Chat input
    question = st.chat_input("Ask a question about your documents...")
    if question:
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = api(
                    "POST",
                    f"/conversations/{st.session_state.conversation_id}/messages",
                    json={"content": question},
                )
            if result:
                answer = result["assistant_message"]["content"]
                citations = result["assistant_message"].get("citations") or []
                st.markdown(answer)
                if citations:
                    with st.expander(f"📚 Sources ({len(citations)} chunks)", expanded=False):
                        for c in citations:
                            st.markdown(f"**Page {c['page']}**")
                            st.text(c["chunk"][:300] + ("..." if len(c["chunk"]) > 300 else ""))
                            st.divider()
                # Reload full history to keep state consistent
                load_messages()
                st.rerun()


# ── Upload tab ────────────────────────────────────────────────────────────────

def render_upload():
    st.subheader("Upload a PDF Document")
    st.markdown("Uploaded PDFs are chunked, embedded, and indexed for retrieval.")

    uploaded = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help=f"Maximum file size: {os.environ.get('MAX_UPLOAD_SIZE_MB', 20)} MB",
    )

    if uploaded:
        if st.button("📤 Upload & Index", use_container_width=True):
            with st.spinner(f"Uploading and indexing {uploaded.name}..."):
                result = api(
                    "POST",
                    f"/conversations/{st.session_state.conversation_id}/documents",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                )
            if result:
                st.success(f"✅ **{uploaded.name}** uploaded and indexed successfully!")
                st.json({
                    "document_id": result["id"],
                    "filename": result["filename"],
                    "created_at": result["created_at"],
                })
                load_documents()

    # List existing documents
    if st.session_state.documents:
        st.divider()
        st.subheader("Uploaded Documents")
        for doc in st.session_state.documents:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"📄 **{doc['filename']}**")
                st.caption(f"ID: `{doc['id']}` | Uploaded: {doc['created_at'][:10]}")
            with col2:
                if doc.get("edited_path"):
                    # Download edited version
                    dl = httpx.get(
                        f"{BACKEND_URL}/conversations/{st.session_state.conversation_id}/documents/{doc['id']}/download-edited",
                        timeout=30,
                    )
                    if dl.status_code == 200:
                        st.download_button(
                            "📥 Download edited",
                            data=dl.content,
                            file_name=f"edited_{doc['filename']}",
                            mime="application/pdf",
                            key=f"dl_{doc['id']}",
                        )


# ── Edit tab ──────────────────────────────────────────────────────────────────

def render_edit():
    st.subheader("Edit a PDF Document")
    st.markdown(
        "Describe the edit in plain English. Examples:\n"
        "- *Replace 'Confidential' with 'Public'*\n"
        "- *Redact all occurrences of 'John Doe'*\n"
        "- *Remove page 3*"
    )

    # Reload documents if needed
    if not st.session_state.documents:
        load_documents()

    if not st.session_state.documents:
        st.warning("No documents uploaded yet. Go to the **Upload PDF** tab first.")
        return

    # Select document
    doc_options = {d["filename"]: d["id"] for d in st.session_state.documents}
    selected_name = st.selectbox("Select document to edit", list(doc_options.keys()))
    selected_id = doc_options[selected_name]

    instruction = st.text_area(
        "Edit instruction",
        placeholder="e.g. Replace 'draft' with 'final'",
        height=100,
    )

    if st.button("✏️ Apply Edit", use_container_width=True, disabled=not instruction.strip()):
        with st.spinner("Applying edit..."):
            result = api(
                "POST",
                f"/conversations/{st.session_state.conversation_id}/messages",
                json={"content": instruction, "document_id": selected_id},
            )
        if result:
            reply = result["assistant_message"]["content"]
            st.success(reply)

            # Offer download of edited file
            dl = httpx.get(
                f"{BACKEND_URL}/conversations/{st.session_state.conversation_id}/documents/{selected_id}/download-edited",
                timeout=30,
            )
            if dl.status_code == 200:
                st.download_button(
                    "📥 Download edited PDF",
                    data=dl.content,
                    file_name=f"edited_{selected_name}",
                    mime="application/pdf",
                    use_container_width=True,
                )
            load_documents()
            st.rerun()


# ── Entry point ───────────────────────────────────────────────────────────────

init_state()
render_sidebar()
render_main()
