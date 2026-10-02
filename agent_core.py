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
import re
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    SummarizationMiddleware,
    TodoListMiddleware,
    wrap_model_call,
)
from langchain.chat_models import init_chat_model
from langchain.mcp import MCPAdapter
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from config import configure_logging, get_settings

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

    parts = [f"Your last call to `{tool_name}` was rejected: its arguments didn't match its schema."]

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


_RATE_LIMIT_MARKERS = (
    "resource_exhausted", "rate limit", "rate_limit", "quota",
    "too many requests", " 429", "'code': 429",
)
# Pull the provider's suggested wait out of the error (e.g. "retry in 51.2s"
# or "'retryDelay': '51s'").
_RETRY_DELAY_RE = re.compile(r"retry[^0-9]{0,25}?(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)


def _is_rate_limited(exc: BaseException) -> bool:
    s = str(exc).lower()
    return any(m in s for m in _RATE_LIMIT_MARKERS)


def _retry_delay_seconds(exc: BaseException, default: float = 30.0) -> float:
    m = _RETRY_DELAY_RE.search(str(exc))
    if m:
        return min(float(m.group(1)) + 1.0, 65.0)  # +1s cushion, capped
    return default


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
            if not _is_rate_limited(exc) or attempts >= max_attempts:
                raise
            wait = _retry_delay_seconds(exc)
            logger.warning(
                "Rate limited by the model provider; waiting %.0fs then "
                "retrying (attempt %d/%d).", wait, attempts, max_attempts,
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


@contextlib.asynccontextmanager
async def build_agent():
    """Yield a compiled, memory-backed agent.

        async with build_agent() as agent:
            cfg = {"configurable": {"thread_id": "abc123"}}
            await agent.ainvoke({"messages": [("user", "hi")]}, cfg)

    The `thread_id` in the config selects which conversation's memory to
    load and append to. Reuse it for a continuing chat; change it to
    start fresh.
    """
    configure_logging()
    settings = get_settings()

    # A single model object, shared so the summariser doesn't spin up a
    # second client/connection pool.
    model = init_chat_model(settings.chat_model)

    db_path = Path(settings.checkpoint_db).resolve()

    async with (
        MCPAdapter(SERVER_PATH) as mcp,
        AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer,
    ):
        tools = await mcp.list_tools()

        agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=checkpointer,
            middleware=[
                # Progress tracking: gives the agent a `write_todos` tool so
                # a multi-file change becomes an explicit checklist (one item
                # per file), marked done as it goes. This is what stops a
                # weaker model from re-doing a file it already finished --
                # the failure mode that made it commit one file five times.
                TodoListMiddleware(),
                # Context-window management: once the history passes ~40
                # messages, fold the older ones into a running summary,
                # always keeping the most recent 12 verbatim. Without
                # this a long chat eventually overflows the model's
                # context and every turn keeps getting more expensive.
                SummarizationMiddleware(
                    model=model,
                    trigger=("messages", 40),
                    keep=("messages", 12),
                ),
                # Approval gate: pause the run and hand control back to a
                # human before any state-changing GitHub call executes.
                HumanInTheLoopMiddleware(
                    interrupt_on={
                        name: {"allowed_decisions": ["approve", "reject"]}
                        for name in WRITE_TOOLS
                    },
                    description_prefix="Approval required -- this changes GitHub",
                ),
                # Recover from Groq's strict-schema rejection of a
                # malformed tool call by retrying with a corrective note
                # (not a blind resample -- see the hook's docstring).
                retry_on_rate_limit,
                retry_malformed_tool_call,
                # Safety valve: cap model calls per user turn so a confused
                # tool loop can't run up an unbounded bill. Configurable
                # (AGENT_RUN_LIMIT) because a multi-file code change needs
                # many legitimate read+write steps.
                ModelCallLimitMiddleware(
                    run_limit=settings.agent_run_limit, exit_behavior="end"
                ),
            ],
        )
        yield agent


def _step_from_result(result: dict) -> dict:
    """Turn a raw graph result into either a final answer or a pending
    approval request -- the two states any caller (CLI or UI) needs to
    handle.

        {"type": "final", "content": "..."}
        {"type": "interrupt", "action_requests": [{"name": ..., "args": ...}]}
    """
    if result.get("__interrupt__"):
        interrupt = result["__interrupt__"][0]
        payload = getattr(interrupt, "value", interrupt)  # HITLRequest (a dict)
        return {"type": "interrupt", "action_requests": payload["action_requests"]}
    return {"type": "final", "content": result["messages"][-1].content}


async def send(agent, config: dict, content) -> dict:
    """Send one user message. Returns a "final" or "interrupt" step (see
    ``_step_from_result``).

    ``content`` is either a plain string, or a list of content blocks for a
    multimodal message, e.g. text plus one or more images::

        [{"type": "text", "text": "fix this error"},
         {"type": "image_url", "image_url": "data:image/png;base64,..."}]

    HumanMessage accepts both forms, so the agent can read a screenshot the
    same way it reads typed text (works because the model is multimodal).
    """
    result = await agent.ainvoke({"messages": [HumanMessage(content=content)]}, config)
    return _step_from_result(result)


async def resume(agent, config: dict, decisions: list[dict]) -> dict:
    """Resume a run paused on an approval request, one decision per
    pending tool call (``{"type": "approve"}`` or
    ``{"type": "reject", "message": "..."}``)."""
    result = await agent.ainvoke(Command(resume={"decisions": decisions}), config)
    return _step_from_result(result)


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
