"""Assembles the conversational GitHub agent.

This module answers "how is the agent built"; the callers
(`chat.py`, `agent.py`, and later the multi-agent supervisor) answer
"how is it driven". Keeping those apart means the wiring below is written
once and reused everywhere.

Pieces:
* model      -- one shared chat model, used by the agent and the summariser
* tools      -- discovered from the MCP server over stdio (MCPAdapter)
* memory     -- an AsyncSqliteSaver checkpointer, so a conversation
                (identified by thread_id) is remembered across turns and
                across process restarts
* middleware -- SummarizationMiddleware (context-window management) and
                ModelCallLimitMiddleware (runaway-loop safety valve)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import sys
from pathlib import Path

from fastmcp.client.transports import PythonStdioTransport
from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    SummarizationMiddleware,
    TodoListMiddleware,
    wrap_model_call,
    wrap_tool_call,
)
from langchain.chat_models import init_chat_model
from langchain.mcp import MCPAdapter
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore
from langgraph.types import Command

from config import configure_logging, get_settings
from guardrails import (
    GuardrailReport,
    sanitize_agent_output,
    sanitize_tool_output,
    sanitize_user_input,
)
from memory import build_memory_tools
from observability import TRACKER, UsageCallback, log_turn, resolve_prices
from ratelimit import is_daily_quota, retry_delay_seconds, should_retry
from reflection import critique, revision_prompt

logger = logging.getLogger("github-agent.core")

# Tools that change GitHub state. Every one is gated behind a human
# approval interrupt (HumanInTheLoopMiddleware) AND the ALLOW_WRITES
# switch checked inside the tool itself.
WRITE_TOOLS = (
    "create_issue",
    "comment_on_issue",
    "add_labels",
    "create_branch",
    "write_file",
    "edit_file",
    "open_pull_request",
)

# The MCP server lives next to this file; MCPAdapter launches it over stdio.
SERVER_PATH = Path(__file__).with_name("server.py")


def server_transport() -> PythonStdioTransport:
    """How to launch the bundled MCP server: this interpreter, this environment.

    The MCP stdio client otherwise starts the server with only a small
    allow-list of variables (PATH, HOME, ...). Settings supplied through the
    environment -- ``docker run --env-file``, CI secrets, a per-run
    ``CHAT_MODEL`` or ``RAG_HYBRID`` -- then never reach the tools, which fall
    back to whatever ``.env`` sits next to server.py. Inside the Docker image
    there is none, so the server could not even find ``GITHUB_TOKEN``.
    """
    return PythonStdioTransport(SERVER_PATH, python_cmd=sys.executable, env=dict(os.environ))


def mcp_server() -> MCPAdapter:
    """Connect to the bundled MCP server (use as ``async with mcp_server() as mcp``)."""
    return MCPAdapter(server_transport())


def _is_malformed_tool_call(exc: BaseException) -> bool:
    """True for Groq's strict-schema rejection of a malformed tool call.

    The gpt-oss models occasionally emit a tool call with a missing,
    extra, or misnamed argument -- most often when one argument (e.g. a
    file's full ``content``) is large enough to crowd out a smaller
    required field like ``repo``. Groq returns HTTP 400 'tool_use_failed'
    for these before the tool is ever invoked.
    """
    s = str(exc).lower()
    return any(
        marker in s
        for marker in ("tool_use_failed", "tool call validation failed", "did not match schema")
    )


_TOOL_NAME_RE = re.compile(r"parameters for tool (\S+) did not match schema")
_MISSING_PROPS_RE = re.compile(r"missing properties: ((?:'[^']+'(?:,\s*)?)+)")
_EXTRA_PROP_RE = re.compile(r"additionalProperties ((?:'[^']+'(?:,\s*)?)+) not allowed")


def _explain_tool_call_error(exc: BaseException) -> str:
    """Turn Groq's schema-validation error into a plain-English correction
    to hand back to the model -- a blind resample tends to repeat the
    exact same mistake (see ``retry_malformed_tool_call`` below); pointing
    at exactly what was wrong fixes it far more reliably."""
    text = str(exc)
    tool_match = _TOOL_NAME_RE.search(text)
    tool_name = tool_match.group(1) if tool_match else "the tool you just called"

    parts = [
        f"Your last call to `{tool_name}` was rejected: its arguments didn't match its schema."
    ]

    missing = _MISSING_PROPS_RE.search(text)
    if missing:
        parts.append(
            f"You left out required argument(s): {missing.group(1)}. "
            f"Call `{tool_name}` again with EVERY required argument included."
        )
    extra = _EXTRA_PROP_RE.search(text)
    if extra:
        parts.append(f"You also included {extra.group(1)}, which isn't a real argument. Remove it.")
    if not missing and not extra:
        parts.append(
            f"Re-check `{tool_name}`'s exact parameter names and call it again with nothing "
            f"missing and nothing extra."
        )
    return " ".join(parts)


@wrap_model_call
async def retry_malformed_tool_call(request, handler):
    """Recover from Groq's strict-schema rejection of a malformed tool
    call by retrying with a corrective note, not a blind resample.

    ``ModelRetryMiddleware`` (langchain's built-in) just calls the model
    again with the identical prompt -- if the mistake was systematic (a
    large ``content`` argument crowding out a smaller required field,
    say) that reliably reproduces the exact same failure every attempt.
    This hook instead appends a message naming precisely what was wrong
    before each retry, which the model can actually act on.
    """
    messages = request.messages
    attempts, max_attempts = 0, 3
    while True:
        try:
            return await handler(request.override(messages=messages))
        except Exception as exc:
            attempts += 1
            if not _is_malformed_tool_call(exc) or attempts >= max_attempts:
                raise
            messages = [*messages, HumanMessage(content=_explain_tool_call_error(exc))]


@wrap_model_call
async def retry_on_rate_limit(request, handler):
    """Wait out a provider rate-limit (HTTP 429 / RESOURCE_EXHAUSTED) and
    retry, instead of failing the whole run.

    Groq's SDK backs off internally, but the Gemini integration surfaces a
    429 straight to the caller -- so on a free tier the agent dies mid-task
    the moment it exceeds the per-minute request quota. This honours the
    provider's suggested retry delay (parsed from the error) and resumes,
    which is what makes free-tier multi-step runs actually complete (just
    slowly).
    """
    attempts, max_attempts = 0, 6
    while True:
        try:
            return await handler(request)
        except Exception as exc:
            attempts += 1
            if is_daily_quota(exc):
                logger.error(
                    "Daily quota for the model is used up; not retrying: %s", str(exc)[:300]
                )
            if not should_retry(exc) or attempts >= max_attempts:
                raise
            wait = retry_delay_seconds(exc)
            logger.warning(
                "Rate limited by the model provider; waiting %.0fs then retrying (attempt %d/%d).",
                wait,
                attempts,
                max_attempts,
            )
            await asyncio.sleep(wait)


SYSTEM_PROMPT = """\
You are GitHub Agent, an assistant that answers questions about GitHub
repositories and, when explicitly asked, acts on them.

Capabilities:
- list_repo_files / read_file / read_file_lines: browse and read files.
  For a LARGE file, use read_file_lines to load only the section you need
  instead of the whole file.
- index_repo / search_code: for "where / how does this work" questions
  over a large codebase, index the repo once, then search_code with a
  plain-English query to retrieve the most relevant chunks (each comes
  back with a path:line citation).
- edit_file / write_file: to change a file, PREFER edit_file -- it
  replaces one exact snippet (old -> new) and you send only that snippet,
  which keeps requests small. Use write_file only to create a new file or
  replace a tiny one wholesale.
- Use tools whenever an answer depends on the real state of a repository.
  Never guess a file's contents or a project's layout from its name.

Working style:
- Pick the right tool: a specific known path -> read_file (or
  read_file_lines for a big file); a broad "where is X handled" ->
  search_code (index_repo first if needed).
- To modify a large file: read the relevant lines with read_file_lines,
  then edit_file with an old_string copied verbatim from what you read
  (exact indentation) that is unique in the file. Never load a big file
  in full just to change a few lines.
- Aim for the fewest tool calls that fully answer the question. Read any
  given file at most once.
- When a tool result starts with "Error:", read it -- it explains what
  went wrong and what to try instead. Relay the useful part to the user
  instead of retrying blindly.

Implementing a code change (branch + edits + PR) -- follow this exactly:
- Work the WHOLE requirement autonomously in one go: figure out every
  file that must change, change them all, and open the PR -- without
  stopping to ask the user to confirm files or steps. Decide and act.
  Only pause for the write-approval gate (that is automatic).
- If the requirement doesn't name the files, work out the full set
  yourself first: a full-stack feature usually spans data model ->
  database/service -> API endpoint -> API client -> UI. Briefly explore
  (list_repo_files + read the few relevant files) to confirm the real
  paths, then change ALL of them. Do not do just one layer.
- FIRST call write_todos with ONE item per file you must change, plus a
  final "open pull request" item. This checklist is how you track
  progress -- keep it accurate.
- Create ONE feature branch. Commit every file to THAT branch. Open ONE
  pull request into the default branch as the final step.
- Then work the checklist top to bottom: mark an item in_progress, make
  its edit, mark it completed, move to the next. Mark completed the
  moment a file's edit_file succeeds.
- A file marked completed is DONE. Never edit it again. Never re-commit
  the same change. If you just successfully edited a file, move to the
  NEXT unchecked item -- do not revisit it.
- NEVER search for an identifier you are about to ADD (a new column,
  field, or function) -- it doesn't exist yet, so search_code returns
  nothing. Read the existing code where it belongs (read_file /
  read_file_lines) and edit_file to insert it.
- NEVER repeat a tool call with the same arguments. One edit_file per
  file is usually enough. When every file item is completed, open the
  pull request, then stop.
- Cite the files you actually looked at (path, plus line numbers when
  relevant).
- If the retrieved material doesn't contain the answer, say so plainly
  rather than speculating.

Tone: concise and technical. Assume the user is a developer.
"""


def content_to_text(content) -> str:
    """Flatten a message's ``content`` to plain text.

    Providers disagree on the shape: OpenAI-style models return a ``str``,
    Gemini returns a list of blocks such as
    ``[{"type": "text", "text": "...", "extras": {...}}]``. Rendering the
    raw list in a UI shows Python repr garbage, so everything user-facing
    goes through this.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type", "text") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


# --- guardrail middleware ----------------------------------------------------


def make_tool_output_guard(redact_pii: bool = True):
    """Middleware that sanitises every tool result BEFORE the model sees it.

    This is the indirect-prompt-injection defence: a file in a repository is
    untrusted input, so anything the tools return is scrubbed of credentials
    (and optionally PII) and, if it reads like instructions, wrapped in an
    explicit "treat as data" notice.
    """

    @wrap_tool_call
    async def guard_tool_output(request, handler):
        result = await handler(request)
        if isinstance(result, ToolMessage):
            raw = content_to_text(result.content)
            clean, report = sanitize_tool_output(raw, redact_pii_values=redact_pii)
            if not report.clean:
                logger.warning(
                    "guardrail triggered on tool=%s: %s",
                    request.tool_call.get("name"),
                    report.summary(),
                )
                return result.model_copy(update={"content": clean})
        return result

    return guard_tool_output


# --- assembly ---------------------------------------------------------------


def assemble_agent(model, tools, checkpointer=None, *, settings=None, system_prompt=SYSTEM_PROMPT):
    """Build the compiled agent graph from its parts.

    Kept separate from ``build_agent`` (which also opens the MCP subprocess
    and SQLite connections) so the *same* graph -- middleware stack and all --
    can be constructed in tests around a scripted fake model and local tools.
    """
    settings = settings or get_settings()
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        middleware=[
            # Progress tracking: gives the agent a `write_todos` tool so a
            # multi-file change becomes an explicit checklist, marked done as
            # it goes. Stops a weaker model re-doing a file it already did.
            TodoListMiddleware(),
            # Context-window management: past ~40 messages, fold older turns
            # into a summary, keeping the latest 12 verbatim.
            SummarizationMiddleware(model=model, trigger=("messages", 40), keep=("messages", 12)),
            # Approval gate: pause and hand control to a human before any
            # state-changing GitHub call executes.
            HumanInTheLoopMiddleware(
                interrupt_on={
                    name: {"allowed_decisions": ["approve", "reject"]} for name in WRITE_TOOLS
                },
                description_prefix="Approval required -- this changes GitHub",
            ),
            # Wait out provider rate limits instead of failing the run.
            retry_on_rate_limit,
            # Recover from malformed tool calls with a corrective note.
            retry_malformed_tool_call,
            # Treat tool output as untrusted data: redact secrets/PII and
            # quarantine anything that reads like instructions.
            make_tool_output_guard(redact_pii=settings.guardrails_redact_pii),
            # Safety valve: cap model calls per turn.
            ModelCallLimitMiddleware(run_limit=settings.agent_run_limit, exit_behavior="end"),
        ],
    )
    # Kept for the optional reflection step (see send()).
    try:
        agent._reflection_model = model
    except Exception:  # pragma: no cover - attribute assignment is best-effort
        pass
    return agent


@contextlib.asynccontextmanager
async def build_agent():
    """Yield a compiled, memory-backed agent.

        async with build_agent() as agent:
            cfg = {"configurable": {"thread_id": "abc123", "user_id": "me"}}
            await agent.ainvoke({"messages": [("user", "hi")]}, cfg)

    ``thread_id`` selects which conversation's SHORT-term memory to load;
    ``user_id`` selects whose LONG-term memory ``remember``/``recall`` use.
    """
    configure_logging()
    settings = get_settings()

    # A single model object, shared so the summariser doesn't spin up a
    # second client/connection pool.
    model = init_chat_model(settings.chat_model)

    db_path = Path(settings.checkpoint_db).resolve()
    store_path = Path(settings.memory_db).resolve()

    async with (
        mcp_server() as mcp,
        AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer,
        AsyncSqliteStore.from_conn_string(str(store_path)) as store,
    ):
        await store.setup()
        tools = [*(await mcp.list_tools()), *build_memory_tools(store)]
        yield assemble_agent(model, tools, checkpointer, settings=settings)


# --- running a turn -----------------------------------------------------------


def _thread_id(config: dict) -> str:
    return str((config.get("configurable") or {}).get("thread_id", "unknown"))


def _with_callbacks(config: dict, callback) -> dict:
    merged = dict(config)
    merged["callbacks"] = [*(config.get("callbacks") or []), callback]
    return merged


def _sanitize_user_content(content) -> tuple[object, GuardrailReport]:
    """Redact credentials from what the human typed, text or multimodal."""
    if isinstance(content, str):
        return sanitize_user_input(content)
    report = GuardrailReport()
    out = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            clean, r = sanitize_user_input(str(block.get("text", "")))
            report = report.merge(r)
            out.append({**block, "text": clean})
        else:
            out.append(block)
    return out, report


def _step_from_result(result: dict, *, redact_pii: bool = True) -> dict:
    """Turn a raw graph result into one of the two states any caller needs:

    {"type": "final", "content": "..."}
    {"type": "interrupt", "action_requests": [{"name": ..., "args": ...}]}
    """
    if result.get("__interrupt__"):
        interrupt = result["__interrupt__"][0]
        payload = getattr(interrupt, "value", interrupt)  # HITLRequest (a dict)
        return {"type": "interrupt", "action_requests": payload["action_requests"]}
    text = content_to_text(result["messages"][-1].content)
    clean, report = sanitize_agent_output(text, redact_pii_values=redact_pii)
    step: dict = {"type": "final", "content": clean}
    if not report.clean:
        step["guardrails"] = report.summary()
    return step


def _evidence_since_last_human(messages) -> list[str]:
    """Tool outputs produced in the current turn (the claims' ground truth)."""
    out: list[str] = []
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            break
        if isinstance(m, ToolMessage):
            out.append(content_to_text(m.content))
    return list(reversed(out))


async def _run_graph(agent, payload, config: dict) -> tuple[dict, list]:
    """Invoke the graph once, attaching usage tracking. Returns (step, messages)."""
    settings = get_settings()
    cb = UsageCallback(resolve_prices(settings.model_prices_json))
    result = await agent.ainvoke(payload, _with_callbacks(config, cb))
    metrics = cb.snapshot()
    TRACKER.record(_thread_id(config), metrics)
    log_turn(_thread_id(config), metrics)
    step = _step_from_result(result, redact_pii=settings.guardrails_redact_pii)
    step["usage"] = metrics.as_dict()
    return step, result.get("messages", [])


async def send(agent, config: dict, content) -> dict:
    """Send one user message. Returns a "final" or "interrupt" step.

    ``content`` is a plain string, or a list of content blocks for a
    multimodal message (text plus images). Credentials in the user's text are
    redacted before they reach the model or the saved conversation.
    """
    settings = get_settings()
    content, input_report = _sanitize_user_content(content)
    step, messages = await _run_graph(agent, {"messages": [HumanMessage(content=content)]}, config)

    if not input_report.clean:
        step["input_guardrails"] = input_report.summary()

    # Optional self-critique: check the answer against the tool evidence and
    # allow one revision. Off by default -- it costs extra model calls.
    model = getattr(agent, "_reflection_model", None)
    evidence = _evidence_since_last_human(messages)
    if settings.enable_reflection and step["type"] == "final" and model is not None and evidence:
        question = content if isinstance(content, str) else content_to_text(content)
        verdict = await critique(model, question, step["content"], evidence)
        step["reflection"] = {
            "verdict": verdict.verdict,
            "issues": verdict.issues,
            "revised": False,
        }
        if verdict.verdict == "revise":
            note = {"messages": [HumanMessage(content=revision_prompt(verdict.issues))]}
            revised, _ = await _run_graph(agent, note, config)
            if revised["type"] == "final":
                revised["reflection"] = {**step["reflection"], "revised": True}
                if step.get("usage") and revised.get("usage"):
                    for k in ("model_calls", "input_tokens", "output_tokens", "tool_calls"):
                        revised["usage"][k] += step["usage"].get(k, 0)
                    revised["usage"]["total_tokens"] = (
                        revised["usage"]["input_tokens"] + revised["usage"]["output_tokens"]
                    )
                step = revised
    return step


async def resume(agent, config: dict, decisions: list[dict]) -> dict:
    """Resume a run paused on an approval request, one decision per pending
    tool call (``{"type": "approve"}`` or ``{"type": "reject", "message": ...}``)."""
    step, _ = await _run_graph(agent, Command(resume={"decisions": decisions}), config)
    return step


async def stream_turn(agent, config: dict, content=None, *, decisions: list[dict] | None = None):
    """Run a turn and yield events as they happen (async generator).

    Events::

        {"type": "token",      "text": "..."}        answer text, incrementally
        {"type": "tool_start", "name": "read_file"}  agent decided to call a tool
        {"type": "tool_end",   "name": "read_file"}  the tool returned
        {"type": "final",      "content": "...", "usage": {...}}   terminal
        {"type": "interrupt",  "action_requests": [...]}           terminal

    Pass ``content`` for a new message, or ``decisions`` to resume after an
    approval. The terminal event has the same shape as ``send``/``resume``.
    Note: tokens stream raw; the terminal ``final`` carries the sanitised text
    and callers should replace what they displayed with it.
    """
    settings = get_settings()
    input_report = GuardrailReport()
    if decisions is not None:
        payload = Command(resume={"decisions": decisions})
    else:
        content, input_report = _sanitize_user_content(content)
        payload = {"messages": [HumanMessage(content=content)]}

    cb = UsageCallback(resolve_prices(settings.model_prices_json))
    cfg = _with_callbacks(config, cb)
    interrupted = None

    async for mode, data in agent.astream(payload, cfg, stream_mode=["messages", "updates"]):
        if mode == "messages":
            chunk, _meta = data
            if isinstance(chunk, AIMessageChunk):
                for tcc in chunk.tool_call_chunks or []:
                    if tcc.get("name"):
                        yield {"type": "tool_start", "name": tcc["name"]}
                text = content_to_text(chunk.content)
                if text:
                    yield {"type": "token", "text": text}
            elif isinstance(chunk, ToolMessage):
                yield {"type": "tool_end", "name": chunk.name}
        elif mode == "updates" and isinstance(data, dict) and "__interrupt__" in data:
            interrupted = data["__interrupt__"]

    metrics = cb.snapshot()
    TRACKER.record(_thread_id(config), metrics)
    log_turn(_thread_id(config), metrics, streamed=True)

    if interrupted:
        first = interrupted[0]
        payload_v = getattr(first, "value", first)
        yield {
            "type": "interrupt",
            "action_requests": payload_v["action_requests"],
            "usage": metrics.as_dict(),
        }
        return

    state = await agent.aget_state(cfg)
    last = (state.values.get("messages") or [None])[-1]
    text = content_to_text(getattr(last, "content", ""))
    clean, report = sanitize_agent_output(text, redact_pii_values=settings.guardrails_redact_pii)
    final: dict = {"type": "final", "content": clean, "usage": metrics.as_dict()}
    if not report.clean:
        final["guardrails"] = report.summary()
    if not input_report.clean:
        final["input_guardrails"] = input_report.summary()
    yield final


async def run_turn(agent, config: dict, text: str, approve) -> str:
    """Run one user turn end to end and return the assistant's reply.

    A thin blocking loop over ``send``/``resume`` for callers (the CLIs)
    that want one call in, one string out. ``approve(name, args) -> bool``
    is asked for every pending write tool call.
    """
    step = await send(agent, config, text)

    while step["type"] == "interrupt":
        decisions = [
            {"type": "approve"}
            if await approve(action["name"], action.get("args", {}))
            else {"type": "reject", "message": "Rejected by the user."}
            for action in step["action_requests"]
        ]
        step = await resume(agent, config, decisions)

    return step["content"]
