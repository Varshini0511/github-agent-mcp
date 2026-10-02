"""Streamlit UI for the GitHub Agent.

    streamlit run streamlit_app.py

Streamlit reruns this entire script on every click/keystroke, so anything
that must survive across reruns -- the running agent, chat history -- is
kept in ``st.session_state`` / ``st.cache_resource``. The agent itself
runs on a dedicated background thread (``runner.AsyncRunner``) inside one
continuous task (``runner.AgentWorker``): its MCP subprocess connection
and SQLite checkpointer both require the SAME asyncio task to stay alive
for as long as they're used, which a UI's "build once, call many times
from many separate browser events" pattern would otherwise violate.
"""

import base64
import os
import uuid

import streamlit as st

# On Streamlit Community Cloud there is no .env file -- secrets come from the
# app's dashboard and are exposed via st.secrets. Bridge them into the
# environment BEFORE config is read, so get_settings() (pydantic-settings)
# and the MCP subprocess it spawns both pick them up. setdefault keeps a
# local .env authoritative when running on your own machine.
try:
    for _k, _v in st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass  # no secrets.toml locally -> fall back to .env

from agent_core import build_agent, resume, send
from config import get_settings
from multi_agent import build_supervisor
from runner import AgentWorker, AsyncRunner

st.set_page_config(page_title="GitHub Agent", page_icon="🐙", layout="centered")

MODE_LABELS = {"single": "Single agent", "multi": "Supervisor team"}


# -- shared resources (built once for the whole app, not per session) --
#
# st.cache_resource is Streamlit's purpose-built tool for this: an
# internal lock guarantees the wrapped function's body runs exactly
# once even if two script reruns race for it. One AgentWorker per mode
# is shared across every visitor; conversations stay isolated because
# each browser session gets its own thread_id.


@st.cache_resource
def get_runner() -> AsyncRunner:
    return AsyncRunner()


@st.cache_resource
def get_worker(mode: str) -> AgentWorker:
    """One AgentWorker per mode, shared by every visitor and every rerun.

    See runner.AgentWorker for why this can't just be
    ``runner.run(build_agent().__aenter__())`` -- the MCP subprocess and
    the SQLite checkpointer both require the SAME asyncio Task to stay
    running for as long as they're used, and a UI naturally calls in from
    many separate events over time.
    """
    runner = get_runner()
    builder = build_supervisor if mode == "multi" else build_agent
    worker = AgentWorker(runner, builder)
    worker.wait_ready(timeout=60)
    return worker


def new_thread_id(mode: str) -> str:
    return f"streamlit-{mode}-{uuid.uuid4().hex[:8]}"


def get_session(mode: str) -> dict:
    """Per-browser-session chat state for `mode` (messages, thread id,
    any pending approval). The underlying agent is shared (see above);
    only this bookkeeping is session-local."""
    sessions = st.session_state.setdefault("sessions", {})
    if mode not in sessions:
        sessions[mode] = {
            "worker": get_worker(mode),
            "thread_id": new_thread_id(mode),
            "messages": [],  # [{"role": "user"|"assistant", "content": str}]
            "pending": None,  # list[ActionRequest] while awaiting approval
        }
    return sessions[mode]


def apply_step(session: dict, step: dict) -> None:
    """Fold a `send`/`resume` result into the session's chat state."""
    if step["type"] == "interrupt":
        session["pending"] = step["action_requests"]
    else:
        session["pending"] = None
        session["messages"].append({"role": "assistant", "content": step["content"]})


# -- sidebar -------------------------------------------------------------

with st.sidebar:
    st.title("🐙 GitHub Agent")

    mode = st.radio(
        "Mode",
        list(MODE_LABELS),
        format_func=lambda m: MODE_LABELS[m],
        index=0,
        help="Single agent: one model with every tool. "
        "Supervisor team: routes to repo_qa / issue_triage / code_writer specialists.",
    )
    session = get_session(mode)

    st.caption(f"thread: `{session['thread_id']}`")
    if st.button("🔄 New conversation", use_container_width=True):
        session["thread_id"] = new_thread_id(mode)
        session["messages"] = []
        session["pending"] = None
        st.rerun()

    st.divider()
    settings = get_settings()
    st.caption("Configuration")
    st.write(f"chat model — `{settings.chat_model}`")
    st.write("writes — " + ("🟢 enabled" if settings.allow_writes else "🔴 disabled"))
    if not settings.allow_writes:
        st.caption("Set ALLOW_WRITES=true in .env to let approved writes execute.")

    with st.expander("Available tools"):
        st.markdown(
            "- `list_repo_files`, `read_file`\n"
            "- `list_issues`, `get_issue`\n"
            "- `index_repo`, `search_code` (RAG)\n"
            "- `create_issue`, `comment_on_issue`, `add_labels` 🔒\n"
            "- `create_branch`, `write_file`, `open_pull_request` 🔒\n\n"
            "🔒 = requires your approval before it runs"
        )

# -- main chat area -------------------------------------------------

st.title("GitHub Agent")
st.caption(
    "Ask about a repository, search its code, triage issues, or "
    "(with your approval) open a pull request."
)

for msg in session["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

config = {"configurable": {"thread_id": session["thread_id"]}}
worker: AgentWorker = session["worker"]

# -- pending write approval ------------------------------------------

if session["pending"]:
    with st.chat_message("assistant"):
        st.warning("⚠️ Approval required before the agent continues:")
        for action in session["pending"]:
            st.markdown(f"**{action['name']}**")
            st.json(action.get("args", {}))
        col_approve, col_reject = st.columns(2)
        approve_clicked = col_approve.button("✅ Approve", use_container_width=True)
        reject_clicked = col_reject.button("❌ Reject", use_container_width=True)

    if approve_clicked or reject_clicked:
        decisions = [
            {"type": "approve"}
            if approve_clicked
            else {"type": "reject", "message": "Rejected by the user via the UI."}
            for _ in session["pending"]
        ]
        with st.spinner("Working..."):
            try:
                step = worker.call(lambda agent: resume(agent, config, decisions))
            except Exception as exc:  # e.g. Groq rate limit
                step = {"type": "final", "content": f"⚠️ Error: {exc}"}
        apply_step(session, step)
        st.rerun()

# -- new message (text and/or screenshots) ---------------------------

# accept_file lets the user attach screenshots right in the chat box. The
# model is multimodal, so an error screenshot is read like typed text.
chat_val = st.chat_input(
    "Message the agent…  (📎 attach a screenshot of an error)",
    accept_file="multiple",
    file_type=["png", "jpg", "jpeg", "webp"],
    disabled=bool(session["pending"]),
)

if chat_val:
    text = (getattr(chat_val, "text", None) or "").strip()
    files = list(getattr(chat_val, "files", None) or [])

    if text or files:
        # Build the message: plain string if text-only, else multimodal blocks.
        if files:
            content = [{
                "type": "text",
                "text": text or "Here is a screenshot. Read any error in it and help me fix it.",
            }]
            for f in files:
                raw = f.getvalue()
                b64 = base64.b64encode(raw).decode()
                mime = getattr(f, "type", None) or "image/png"
                content.append({
                    "type": "image_url",
                    "image_url": f"data:{mime};base64,{b64}",
                })
        else:
            content = text

        # Record for history: show the text plus a note that images were attached.
        shown = text or "_(screenshot only)_"
        if files:
            shown += f"\n\n📎 {len(files)} image(s) attached"
        session["messages"].append({"role": "user", "content": shown})

        with st.spinner("Thinking…"):
            try:
                step = worker.call(lambda agent: send(agent, config, content))
            except Exception as exc:
                step = {"type": "final", "content": f"⚠️ Error: {exc}"}
        apply_step(session, step)
        st.rerun()
