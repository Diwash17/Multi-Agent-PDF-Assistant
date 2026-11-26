"""Multi-Agent PDF Assistant — Streamlit frontend."""

import os

import httpx
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme CSS (light / dark) ──────────────────────────────────────────────────
DARK_CSS = """
<style>
:root {
    --bg: #0f172a;
    --sidebar-bg: #1e293b;
    --card: #1e293b;
    --border: #334155;
    --text: #f1f5f9;
    --text-muted: #94a3b8;
    --accent: #818cf8;
    --accent-hover: #6366f1;
    --user-bubble: #312e81;
    --assistant-bubble: #1e293b;
    --success: #22c55e;
    --danger: #ef4444;
    --badge: #334155;
}

[data-testid="stApp"] {
    background-color: #0f172a !important;
    color: #f1f5f9 !important;
}
[data-testid="stSidebar"] {
    background-color: #1e293b !important;
    border-right: 1px solid #334155 !important;
}
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background-color: #0f172a !important;
    color: #f1f5f9 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploader"] {
    background-color: #1e293b !important;
    border: 2px dashed #818cf8 !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span {
    color: #f1f5f9 !important;
}
[data-testid="stButton"] button {
    background-color: #6366f1 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
}
[data-testid="stSelectbox"] > div {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #f1f5f9 !important;
    border-radius: 8px !important;
}
[data-testid="stTabs"] [role="tab"] { color: #94a3b8 !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #818cf8 !important;
    border-bottom-color: #818cf8 !important;
}
[data-testid="stExpander"] {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}
[data-testid="stAlert"] {
    background-color: #1e1b4b !important;
    color: #c7d2fe !important;
    border-left: 4px solid #818cf8 !important;
    border-radius: 8px !important;
}
[data-testid="stDownloadButton"] button {
    background-color: #6366f1 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    border: none !important;
}
hr { border-color: #334155 !important; }
p, label, span { color: #f1f5f9; }
</style>
"""

LIGHT_CSS = """
<style>
:root {
    --bg: #f8fafc;
    --sidebar-bg: #ffffff;
    --card: #ffffff;
    --border: #e2e8f0;
    --text: #1e293b;
    --text-muted: #64748b;
    --accent: #6366f1;
    --accent-hover: #4f46e5;
    --user-bubble: #ede9fe;
    --assistant-bubble: #f1f5f9;
    --success: #16a34a;
    --danger: #dc2626;
    --badge: #f1f5f9;
}

/* Force Streamlit native components to light theme */
[data-testid="stApp"] {
    background-color: #f8fafc !important;
    color: #1e293b !important;
}
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}
/* Text inputs */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background-color: #ffffff !important;
    color: #1e293b !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
}
/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #ffffff !important;
    border: 2px dashed #6366f1 !important;
    border-radius: 10px !important;
    color: #1e293b !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span {
    color: #1e293b !important;
}
/* Buttons */
[data-testid="stButton"] button {
    background-color: #6366f1 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
}
[data-testid="stButton"] button:hover {
    background-color: #4f46e5 !important;
}
/* Primary buttons */
[data-testid="stFormSubmitButton"] button[kind="primary"],
[data-testid="stButton"] button[kind="primary"] {
    background-color: #6366f1 !important;
    color: #ffffff !important;
}
/* Select box */
[data-testid="stSelectbox"] > div {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    color: #1e293b !important;
    border-radius: 8px !important;
}
/* Tabs */
[data-testid="stTabs"] [role="tab"] {
    color: #64748b !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #6366f1 !important;
    border-bottom-color: #6366f1 !important;
}
/* Expander */
[data-testid="stExpander"] {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
}
/* Info box */
[data-testid="stAlert"] {
    background-color: #ede9fe !important;
    color: #3730a3 !important;
    border-left: 4px solid #6366f1 !important;
    border-radius: 8px !important;
}
/* Download button */
[data-testid="stDownloadButton"] button {
    background-color: #6366f1 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    border: none !important;
}
/* Divider */
hr { border-color: #e2e8f0 !important; }
/* General text */
p, label, span, div { color: #1e293b; }
.stMarkdown { color: #1e293b !important; }
</style>
"""

SHARED_CSS = """
<style>
/* Global */
html, body, [data-testid="stApp"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg) !important;
    border-right: 1px solid var(--border);
}

/* Cards */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}

/* Step banner */
.step-banner {
    display: flex;
    gap: 0.5rem;
    align-items: center;
    padding: 0.6rem 1rem;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 1.2rem;
    font-size: 0.85rem;
    color: var(--text-muted);
}
.step-banner .active {
    color: var(--accent);
    font-weight: 700;
}
.step-banner .sep { color: var(--border); }

/* Chat bubbles */
.bubble-user {
    background: var(--user-bubble);
    border-radius: 12px 12px 4px 12px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0;
    max-width: 80%;
    margin-left: auto;
    color: var(--text);
    border: 1px solid var(--border);
}
.bubble-assistant {
    background: var(--assistant-bubble);
    border-radius: 12px 12px 12px 4px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0;
    max-width: 85%;
    color: var(--text);
    border: 1px solid var(--border);
}
.bubble-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-bottom: 0.25rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.citation-badge {
    display: inline-block;
    background: var(--badge);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 0.75rem;
    color: var(--accent);
    margin: 0 2px;
    font-family: monospace;
}

/* Doc item */
.doc-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 0.8rem;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 0.5rem;
}
.doc-name { font-weight: 600; font-size: 0.9rem; }
.doc-meta { font-size: 0.75rem; color: var(--text-muted); }

/* Conversation button active */
.conv-active button {
    border-left: 3px solid var(--accent) !important;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: var(--text-muted);
}
.empty-state .icon { font-size: 3rem; margin-bottom: 0.5rem; }
.empty-state h3 { color: var(--text); margin-bottom: 0.5rem; }

/* Section header */
.section-header {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin-bottom: 0.8rem;
}
</style>
"""

# ── API helpers ───────────────────────────────────────────────────────────────

def api(method: str, path: str, **kwargs):
    url = f"{BACKEND_URL}{path}"
    try:
        resp = httpx.request(method, url, timeout=120, **kwargs)
        resp.raise_for_status()
        if resp.status_code == 204:
            return {}
        return resp.json()
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        st.error(f"❌ {detail}")
        return None
    except Exception as e:
        st.error(f"❌ Connection error: {e}")
        return None


def api_bytes(path: str):
    """Download binary (PDF) response."""
    try:
        resp = httpx.get(f"{BACKEND_URL}{path}", timeout=30)
        if resp.status_code == 200:
            return resp.content
        return None
    except Exception:
        return None


# ── Session state ─────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "conversation_id": None,
        "conversation_title": None,
        "messages": [],
        "documents": [],
        "dark_mode": True,
        "active_step": 1,  # 1=upload, 2=chat, 3=edit
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def refresh_messages():
    if not st.session_state.conversation_id:
        return
    data = api("GET", f"/conversations/{st.session_state.conversation_id}/messages")
    if data is not None:
        st.session_state.messages = data


def refresh_documents():
    if not st.session_state.conversation_id:
        return
    data = api("GET", f"/conversations/{st.session_state.conversation_id}/documents")
    if data is not None:
        st.session_state.documents = data
        if data:
            st.session_state.active_step = max(st.session_state.active_step, 2)


def select_conversation(conv_id: str, conv_title: str):
    st.session_state.conversation_id = conv_id
    st.session_state.conversation_title = conv_title
    st.session_state.messages = []
    st.session_state.documents = []
    st.session_state.active_step = 1
    refresh_messages()
    refresh_documents()


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        # Theme toggle
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("## 📄 PDF Assistant")
        with col2:
            icon = "☀️" if st.session_state.dark_mode else "🌙"
            if st.button(icon, key="theme_toggle", help="Toggle light/dark theme"):
                st.session_state.dark_mode = not st.session_state.dark_mode
                st.rerun()

        st.divider()

        # New conversation form
        with st.form("new_conv", clear_on_submit=True):
            title = st.text_input(
                "Title",
                placeholder="e.g. Contract Review",
                label_visibility="collapsed",
            )
            if st.form_submit_button("➕ New Conversation", use_container_width=True):
                t = title.strip() or "New Conversation"
                result = api("POST", "/conversations", json={"title": t})
                if result:
                    select_conversation(result["id"], result["title"])
                    st.rerun()

        st.divider()

        # Conversation list
        convos = api("GET", "/conversations") or []
        if convos:
            st.markdown("**Conversations**")
            for c in convos:
                is_active = c["id"] == st.session_state.conversation_id
                prefix = "▶ " if is_active else "   "
                col_btn, col_del = st.columns([5, 1])
                with col_btn:
                    label = f"{prefix}{c['title']}"
                    if st.button(label, key=f"conv_{c['id']}", use_container_width=True):
                        select_conversation(c["id"], c["title"])
                        st.rerun()
                with col_del:
                    if st.button("🗑", key=f"del_{c['id']}", help="Delete conversation"):
                        api("DELETE", f"/conversations/{c['id']}")
                        if st.session_state.conversation_id == c["id"]:
                            st.session_state.conversation_id = None
                            st.session_state.conversation_title = None
                            st.session_state.messages = []
                            st.session_state.documents = []
                        st.rerun()
        else:
            st.caption("No conversations yet.")

        # Active info
        if st.session_state.conversation_id:
            st.divider()
            st.caption(f"**Active:** {st.session_state.conversation_title}")
            st.caption(f"`{str(st.session_state.conversation_id)[:18]}...`")


# ── Step banner ───────────────────────────────────────────────────────────────

def render_step_banner():
    step = st.session_state.active_step
    s1 = "active" if step == 1 else ""
    s2 = "active" if step == 2 else ""
    s3 = "active" if step == 3 else ""
    st.markdown(
        f"""<div class="step-banner">
            <span class="{s1}">① Upload PDF</span>
            <span class="sep">›</span>
            <span class="{s2}">② Chat & Ask Questions</span>
            <span class="sep">›</span>
            <span class="{s3}">③ Edit PDF</span>
        </div>""",
        unsafe_allow_html=True,
    )


# ── Main area ─────────────────────────────────────────────────────────────────

def render_main():
    if not st.session_state.conversation_id:
        st.markdown(
            """<div class="empty-state">
                <div class="icon">📄</div>
                <h3>Welcome to PDF Assistant</h3>
                <p>Create a new conversation from the sidebar to get started.</p>
                <p style="font-size:0.85rem">Upload a PDF → Ask questions → Edit documents</p>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    st.markdown(f"### 💬 {st.session_state.conversation_title}")
    render_step_banner()

    tab_upload, tab_chat, tab_edit = st.tabs(
        ["① 📤 Upload PDF", "② 💬 Chat", "③ ✏️ Edit PDF"]
    )

    with tab_upload:
        render_upload()

    with tab_chat:
        render_chat()

    with tab_edit:
        render_edit()


# ── Upload tab ────────────────────────────────────────────────────────────────

def render_upload():
    st.markdown('<div class="section-header">Upload PDF Documents</div>', unsafe_allow_html=True)

    col_info, _ = st.columns([3, 1])
    with col_info:
        st.info(
            "**How it works:** Your PDF is extracted page-by-page, split into chunks, "
            "embedded with Gemini, and stored in pgvector for semantic search.",
            icon="ℹ️",
        )

    uploaded = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help=f"Max {os.environ.get('MAX_UPLOAD_SIZE_MB', 20)} MB",
        label_visibility="collapsed",
    )

    if uploaded:
        st.markdown(f"**Selected:** `{uploaded.name}` ({uploaded.size / 1024:.1f} KB)")
        if st.button("📤 Upload & Index", type="primary", use_container_width=True):
            with st.spinner(f"Indexing {uploaded.name}... this may take a moment"):
                result = api(
                    "POST",
                    f"/conversations/{st.session_state.conversation_id}/documents",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                )
            if result:
                st.success(f"✅ **{uploaded.name}** uploaded and indexed! Switch to the **Chat** tab to ask questions.")
                st.session_state.active_step = 2
                refresh_documents()
                st.rerun()

    # Document list
    if st.session_state.documents:
        st.divider()
        st.markdown('<div class="section-header">Indexed Documents</div>', unsafe_allow_html=True)
        for doc in st.session_state.documents:
            edited_label = " ✏️ *edited*" if doc.get("edited_path") else ""
            st.markdown(
                f"""<div class="doc-item">
                    <div>
                        <div class="doc-name">📄 {doc['filename']}{edited_label}</div>
                        <div class="doc-meta">ID: {doc['id'][:18]}... &nbsp;|&nbsp; {doc['created_at'][:10]}</div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
            if doc.get("edited_path"):
                pdf_bytes = api_bytes(
                    f"/conversations/{st.session_state.conversation_id}/documents/{doc['id']}/download-edited"
                )
                if pdf_bytes:
                    st.download_button(
                        "📥 Download edited PDF",
                        data=pdf_bytes,
                        file_name=f"edited_{doc['filename']}",
                        mime="application/pdf",
                        key=f"dl_{doc['id']}",
                    )
    else:
        st.markdown(
            '<div class="empty-state"><div class="icon">📂</div><p>No documents yet. Upload one above.</p></div>',
            unsafe_allow_html=True,
        )


# ── Chat tab ──────────────────────────────────────────────────────────────────

def render_chat():
    if not st.session_state.documents:
        st.markdown(
            """<div class="empty-state">
                <div class="icon">📤</div>
                <h3>Upload a PDF first</h3>
                <p>Go to the <strong>Upload PDF</strong> tab and index a document before chatting.</p>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    st.markdown('<div class="section-header">Ask questions about your documents</div>', unsafe_allow_html=True)
    st.caption("Answers include page citations like **[p.3]** so you can verify the source.")

    # Render message history
    if not st.session_state.messages:
        refresh_messages()

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="bubble-label">You</div>'
                f'<div class="bubble-user">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        elif msg["role"] == "assistant":
            citations = msg.get("citations") or []
            st.markdown(
                f'<div class="bubble-label">🤖 Assistant</div>'
                f'<div class="bubble-assistant">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            if citations:
                with st.expander(f"📚 {len(citations)} source chunk(s)"):
                    for c in citations:
                        st.markdown(f"**Page {c['page']}**")
                        st.text(c["chunk"][:300] + ("…" if len(c["chunk"]) > 300 else ""))
                        st.divider()

    st.divider()

    # Input
    with st.form("chat_form", clear_on_submit=True):
        question = st.text_input(
            "Your question",
            placeholder="e.g. What are the key requirements in section 3?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Send ➤", type="primary", use_container_width=True)

    if submitted and question.strip():
        with st.spinner("Thinking..."):
            result = api(
                "POST",
                f"/conversations/{st.session_state.conversation_id}/messages",
                json={"content": question},
            )
        if result:
            st.session_state.active_step = max(st.session_state.active_step, 2)
            refresh_messages()
            st.rerun()


# ── Edit tab ──────────────────────────────────────────────────────────────────

def render_edit():
    if not st.session_state.documents:
        st.markdown(
            """<div class="empty-state">
                <div class="icon">📤</div>
                <h3>Upload a PDF first</h3>
                <p>Go to the <strong>Upload PDF</strong> tab and index a document.</p>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    st.markdown('<div class="section-header">Edit a PDF with natural language</div>', unsafe_allow_html=True)

    # Examples
    with st.expander("💡 Example instructions"):
        st.markdown(
            "- `Replace 'Confidential' with 'Public'`\n"
            "- `Redact all occurrences of 'John Doe'`\n"
            "- `Remove page 3`\n"
            "- `Delete pages 1 and 5`"
        )

    # Document selector
    doc_map = {d["filename"]: d["id"] for d in st.session_state.documents}
    selected_name = st.selectbox(
        "Select document to edit",
        list(doc_map.keys()),
        label_visibility="visible",
    )
    selected_id = doc_map[selected_name]

    with st.form("edit_form", clear_on_submit=True):
        instruction = st.text_area(
            "Edit instruction",
            placeholder="e.g. Replace 'draft' with 'final'",
            height=90,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("✏️ Apply Edit", type="primary", use_container_width=True)

    if submitted and instruction.strip():
        with st.spinner("Applying edit via AI..."):
            result = api(
                "POST",
                f"/conversations/{st.session_state.conversation_id}/messages",
                json={"content": instruction, "document_id": selected_id},
            )
        if result:
            st.success(result["assistant_message"]["content"])
            st.session_state.active_step = 3
            refresh_documents()
            # Don't rerun — keep success message + download button visible

    # ── Always show download for ANY edited document ──────────────────────────
    st.divider()
    st.markdown('<div class="section-header">📥 Edited Documents</div>', unsafe_allow_html=True)

    edited_docs = [d for d in st.session_state.documents if d.get("edited_path")]
    if edited_docs:
        for doc in edited_docs:
            st.markdown(f"**📄 {doc['filename']}** ✏️")
            pdf_bytes = api_bytes(
                f"/conversations/{st.session_state.conversation_id}/documents/{doc['id']}/download-edited"
            )
            if pdf_bytes:
                st.download_button(
                    f"📥 Download edited_{doc['filename']}",
                    data=pdf_bytes,
                    file_name=f"edited_{doc['filename']}",
                    mime="application/pdf",
                    use_container_width=True,
                    key=f"edit_dl_{doc['id']}",
                )
    else:
        st.caption("No edited documents yet. Apply an edit above.")


# ── Entry point ───────────────────────────────────────────────────────────────

init_state()

# Inject theme CSS
theme_css = DARK_CSS if st.session_state.dark_mode else LIGHT_CSS
st.markdown(theme_css + SHARED_CSS, unsafe_allow_html=True)

render_sidebar()
render_main()
