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
# app's dashboard and are exposed via st.secrets. Two things need them:
#   1. this process (config.get_settings) -> bridge into os.environ
#   2. the MCP server, which runs as a CHILD process that the MCP stdio
#      client spawns with a minimal environment (it does NOT inherit our
#      os.environ). Locally that child reads the .env file directly; on the
#      cloud there is none, so we write one from the secrets so the child
#      can load the keys the same way. (A local .env is never overwritten.)
try:
    _secrets = dict(st.secrets)
except Exception:
    _secrets = {}  # no secrets.toml locally -> fall back to the real .env

if _secrets:
    for _k, _v in _secrets.items():
        os.environ.setdefault(_k, str(_v))
    from pathlib import Path as _Path

    _envf = _Path(__file__).with_name(".env")
    if not _envf.exists():
        try:
            _envf.write_text(
                "\n".join(f"{k}={v}" for k, v in _secrets.items()),
                encoding="utf-8",
            )
        except Exception:
            pass

from agent_core import build_agent, stream_turn
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
            "usage": empty_usage(),
        }
    return sessions[mode]


def empty_usage() -> dict:
    return {"model_calls": 0, "tokens": 0, "cost": 0.0, "latency": 0.0, "tool_calls": 0}


def _notes_for(step: dict) -> list[str]:
    """Human-readable footnotes for a finished answer."""
    notes: list[str] = []
    if step.get("input_guardrails"):
        notes.append(f"🛡️ Your message: {step['input_guardrails']} before it reached the model.")
    if step.get("guardrails"):
        notes.append(f"🛡️ Answer filtered: {step['guardrails']}.")
    refl = step.get("reflection")
    if refl:
        if refl.get("revised"):
            notes.append("🔍 Self-check found unsupported claims — answer was revised.")
        else:
            notes.append("🔍 Self-check: answer is supported by the tool output.")
    return notes


def apply_step(session: dict, step: dict) -> None:
    """Fold a terminal stream event (final / interrupt) into chat state."""
    u = step.get("usage")
    if u:
        t = session["usage"]
        t["model_calls"] += u.get("model_calls", 0)
        t["tokens"] += u.get("total_tokens", 0)
        t["cost"] += u.get("cost_usd") or 0.0
        t["latency"] += u.get("latency_s", 0.0)
        t["tool_calls"] += u.get("tool_calls", 0)

    if step["type"] == "interrupt":
        session["pending"] = step["action_requests"]
    else:
        session["pending"] = None
        session["messages"].append(
            {"role": "assistant", "content": step["content"], "notes": _notes_for(step)}
        )


def run_stream(session: dict, worker, config: dict, *, content=None, decisions=None) -> None:
    """Run a turn, rendering tool activity and tokens live, then fold the
    terminal event into the session and rerun to draw it permanently."""
    final: dict | None = None
    buffer = ""
    with st.chat_message("assistant"):
        status = st.status("Working…", expanded=True)
        answer = st.empty()
        try:
            events = worker.stream(
                lambda agent: stream_turn(agent, config, content, decisions=decisions)
            )
            for ev in events:
                kind = ev["type"]
                if kind == "tool_start":
                    status.write(f"🔧 calling `{ev['name']}`…")
                elif kind == "tool_end":
                    status.write(f"✅ `{ev['name']}` returned")
                elif kind == "token":
                    buffer += ev["text"]
                    answer.markdown(buffer + " ▌")
                else:  # "final" | "interrupt"
                    final = ev
        except Exception as exc:  # e.g. provider outage after retries
            final = {"type": "final", "content": f"⚠️ Error: {exc}"}
        status.update(label="Done", state="complete", expanded=False)

    apply_step(session, final or {"type": "final", "content": buffer or "(no response)"})
    st.rerun()


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
        session["usage"] = empty_usage()
        st.rerun()

    st.divider()
    usage = session["usage"]
    st.caption("This conversation")
    m1, m2 = st.columns(2)
    m1.metric("Model calls", usage["model_calls"])
    m2.metric("Tokens", f"{usage['tokens']:,}")
    st.caption(
        f"{usage['tool_calls']} tool calls · {usage['latency']:.0f}s · "
        f"≈ ${usage['cost']:.4f} at list price (free tier bills $0)"
    )

    st.divider()
    settings = get_settings()
    st.caption("Configuration")
    st.write(f"chat model — `{settings.chat_model}`")
    st.write("writes — " + ("🟢 enabled" if settings.allow_writes else "🔴 disabled"))
    st.write("self-check — " + ("🟢 on" if settings.enable_reflection else "⚪ off"))
    st.write("PII redaction — " + ("🟢 on" if settings.guardrails_redact_pii else "⚪ off"))
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
        for note in msg.get("notes", []):
            st.caption(note)

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
        run_stream(session, worker, config, decisions=decisions)

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
            content = [
                {
                    "type": "text",
                    "text": text
                    or "Here is a screenshot. Read any error in it and help me fix it.",
                }
            ]
            for f in files:
                raw = f.getvalue()
                b64 = base64.b64encode(raw).decode()
                mime = getattr(f, "type", None) or "image/png"
                content.append(
                    {
                        "type": "image_url",
                        "image_url": f"data:{mime};base64,{b64}",
                    }
                )
        else:
            content = text

        # Record for history: show the text plus a note that images were attached.
        shown = text or "_(screenshot only)_"
        if files:
            shown += f"\n\n📎 {len(files)} image(s) attached"
        session["messages"].append({"role": "user", "content": shown})

        run_stream(session, worker, config, content=content)
