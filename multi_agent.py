"""Block D -- a supervisor that routes work to specialist sub-agents.

Why not one agent with every tool? Two reasons:
* A single 12-tool list makes every model call bigger and gives smaller
  models more ways to pick the wrong tool.
* Different jobs want different instructions. A code reviewer and an
  issue triager think differently; one system prompt can't be great at
  both.

So we build three focused specialists and a supervisor that delegates.
Each specialist is handed to the supervisor as ONE tool ("agent as
tool"): the supervisor calls it with a task string and gets back a
result string. The supervisor owns the conversation memory; specialists
are stateless workers.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    SummarizationMiddleware,
)
from langchain.chat_models import init_chat_model
from langchain.mcp import MCPAdapter
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent_core import retry_malformed_tool_call, retry_on_rate_limit
from config import configure_logging, get_settings

SERVER_PATH = Path(__file__).with_name("server.py")

# Which MCP tools each specialist is allowed to see.
_SPECIALIST_TOOLS = {
    "repo_qa": {
        "list_repo_files", "read_file", "read_file_lines",
        "index_repo", "search_code",
    },
    "issue_triage": {
        "list_issues", "get_issue", "search_code", "read_file",
        "add_labels", "comment_on_issue",
    },
    "code_writer": {
        "read_file", "read_file_lines", "search_code", "list_repo_files",
        "create_branch", "write_file", "edit_file",
        "open_pull_request", "create_issue",
    },
}

_SPECIALIST_PROMPTS = {
    "repo_qa": (
        "You answer questions about a GitHub repository's code and layout. "
        "Prefer search_code for 'where/how' questions (index_repo first if "
        "needed); use read_file for a known path. Always cite path:line."
    ),
    "issue_triage": (
        "You triage GitHub issues. For each issue in scope: read it, decide "
        "a type label (bug/enhancement/question/docs) and a rough priority, "
        "note likely duplicates, and - only if asked to act - apply labels "
        "or comment. Be concise and consistent."
    ),
    "code_writer": (
        "You make small, safe code changes. Never commit to the default "
        "branch: create_branch, then write_file on that branch, then "
        "open_pull_request. Keep diffs minimal and explain what you changed."
    ),
}

SUPERVISOR_PROMPT = """\
You are the coordinator of a GitHub assistant team. You do not use GitHub
tools yourself -- you delegate to a specialist and then relay/synthesise
their answer.

Specialists (each takes a single self-contained task string):
- repo_qa: questions about code, structure, "where/how does X work".
- issue_triage: reading, labelling, prioritising, de-duplicating issues.
- code_writer: creating branches, editing files, opening pull requests.

Rules:
- Put everything the specialist needs INTO the task string (repo in
  owner/name form, issue numbers, file paths). They share no memory with
  you or each other.
- One specialist is usually enough. Only chain them when the task
  genuinely has two phases (e.g. triage THEN comment).
- For anything that changes the repo, delegate to code_writer and let the
  human approve.
"""


def _make_specialist(name: str, model, all_tools: list):
    wanted = _SPECIALIST_TOOLS[name]
    tools = [t for t in all_tools if t.name in wanted]
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=_SPECIALIST_PROMPTS[name],
        middleware=[
            retry_on_rate_limit,
            retry_malformed_tool_call,
            ModelCallLimitMiddleware(run_limit=6, exit_behavior="end"),
        ],
    )


@contextlib.asynccontextmanager
async def build_supervisor():
    """Yield a compiled supervisor agent (same call shape as build_agent)."""
    configure_logging()
    settings = get_settings()
    model = init_chat_model(settings.chat_model)
    db_path = Path(settings.checkpoint_db).resolve()

    async with (
        MCPAdapter(SERVER_PATH) as mcp,
        AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer,
    ):
        all_tools = await mcp.list_tools()
        specialists = {
            name: _make_specialist(name, model, all_tools)
            for name in _SPECIALIST_TOOLS
        }

        def _wrap(name: str):
            agent = specialists[name]

            @tool(name, description=_SPECIALIST_PROMPTS[name])
            async def _delegate(task: str) -> str:
                result = await agent.ainvoke(
                    {"messages": [("user", task)]}
                )
                return result["messages"][-1].content

            return _delegate

        supervisor = create_agent(
            model=model,
            tools=[_wrap(n) for n in _SPECIALIST_TOOLS],
            system_prompt=SUPERVISOR_PROMPT,
            checkpointer=checkpointer,
            middleware=[
                SummarizationMiddleware(
                    model=model, trigger=("messages", 40), keep=("messages", 12)
                ),
                # Coarse-grained approval: the human okays the delegation
                # to code_writer; ALLOW_WRITES still gates each real call.
                HumanInTheLoopMiddleware(
                    interrupt_on={
                        "code_writer": {"allowed_decisions": ["approve", "reject"]}
                    },
                    description_prefix="Approve delegation to code_writer",
                ),
                retry_on_rate_limit,
                retry_malformed_tool_call,
                ModelCallLimitMiddleware(run_limit=10, exit_behavior="end"),
            ],
        )
        yield supervisor
