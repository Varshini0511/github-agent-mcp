"""Block D -- a supervisor that routes work to specialist sub-agents.

Why not one agent with every tool? Two reasons:
* A single 12-tool list makes every model call bigger and gives smaller
  models more ways to pick the wrong tool.
* Different jobs want different instructions. A code reviewer and an
  issue triager think differently; one system prompt can't be great at
  both.

So we build five focused specialists (repo Q&A, issue triage, code writing,
CI/CD triage, PR review) and a supervisor that delegates.
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
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain.mcp import MCPAdapter
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore

from agent_core import (
    content_to_text,
    make_tool_output_guard,
    retry_malformed_tool_call,
    retry_on_rate_limit,
)
from config import configure_logging, get_settings
from devops_schemas import CIReport, PRReview
from memory import build_memory_tools

SERVER_PATH = Path(__file__).with_name("server.py")

# Which MCP tools each specialist is allowed to see.
_SPECIALIST_TOOLS = {
    "repo_qa": {
        "list_repo_files",
        "read_file",
        "read_file_lines",
        "index_repo",
        "search_code",
    },
    "issue_triage": {
        "list_issues",
        "get_issue",
        "search_code",
        "read_file",
        "add_labels",
        "comment_on_issue",
    },
    "code_writer": {
        "read_file",
        "read_file_lines",
        "search_code",
        "list_repo_files",
        "create_branch",
        "write_file",
        "edit_file",
        "open_pull_request",
        "create_issue",
    },
    # --- DevOps specialists: READ-ONLY by construction (no write tool in the set) ---
    "ci_triage": {
        "get_ci_failure",
        "list_workflow_runs",
        "get_workflow_run",
        "get_job_log",
        "list_check_runs",
        "read_file_lines",
        "list_repo_files",
    },
    "pr_reviewer": {
        "list_pull_requests",
        "get_pull_request",
        "get_pr_diff",
        "list_check_runs",
        "read_file_lines",
        "search_code",
    },
}

# These two return a validated report object instead of free text.
_SPECIALIST_SCHEMAS = {"ci_triage": CIReport, "pr_reviewer": PRReview}

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
    "ci_triage": (
        "You diagnose failed CI/CD runs (GitHub Actions) for a repository.\n"
        "1. Call get_ci_failure(repo[, run_id, branch]) FIRST. It already returns the "
        "failing jobs, a shrunk log excerpt, heuristic category hints and the "
        "workflow's recent history. Never ask for raw logs.\n"
        "2. Only if the excerpt is not enough, use get_job_log on a failing job id "
        "shown in the report, or get_workflow_run.\n"
        "3. Then submit your CIReport.\n"
        "Rules:\n"
        "- Base every claim on the report. Quote evidence VERBATIM from the log "
        "excerpt; never paraphrase and never invent a log line.\n"
        "- The category hints are heuristics: confirm or overrule them.\n"
        "- If there is no failure, or logs are unavailable, say so: category "
        "'unknown', confidence 'low'.\n"
        "- Recommend a re-run only for transient causes (flaky, infra, timeout), "
        "e.g. when the report shows a FLAKINESS SIGNAL; never for test, build or "
        "lint failures.\n"
        "- Be concise."
    ),
    "pr_reviewer": (
        "You review pull requests like a careful senior engineer.\n"
        "1. get_pull_request(repo, number): description, size, CI status.\n"
        "2. get_pr_diff(repo, number): the changes. The left column is the line number "
        "in the NEW file; cite those numbers. If the output says files were "
        "truncated or not shown, call get_pr_diff with path=<file> for the "
        "important ones.\n"
        "3. Use read_file_lines or search_code only for context on a specific concern.\n"
        "4. Submit your PRReview.\n"
        "Rules:\n"
        "- Report only real problems you can point to: bugs, security issues "
        "(secrets, injection, authorisation), missing tests for new behaviour, "
        "breaking changes, serious performance or design problems. No style "
        "nitpicks. Fewer, better findings beat many weak ones.\n"
        "- Every finding needs a file from the PR, the line number from the diff and "
        "a VERBATIM quote of the changed line.\n"
        "- If a CI diagnosis is included in your task, use it: connect the failure "
        "to the code that likely caused it.\n"
        "- Use not_reviewed for anything you could not see (skipped or truncated files).\n"
        "- You only advise. You never approve, merge or post comments."
    ),
}

# What the SUPERVISOR sees in its tool menu: short, "when to use me". (The long
# prompts above are for the specialist itself; sending them to the manager on
# every call would waste tokens.)
_SPECIALIST_DESCRIPTIONS = {
    "repo_qa": "Answers questions about a repository's code and structure ('where/how does "
    "X work'), citing file:line. Give it the repo (owner/name) and the question.",
    "issue_triage": "Reads, labels, prioritises and de-duplicates GitHub issues.",
    "code_writer": "Makes code changes: creates a branch, edits files, opens a pull request. "
    "Needs human approval.",
    "ci_triage": "Diagnoses a FAILED CI/CD (GitHub Actions) run: finds the failing job, reads "
    "the log excerpt and returns a structured diagnosis (category, root cause, evidence, fix, "
    "re-run advice). Give it the repo and optionally a run id or branch. Read-only.",
    "pr_reviewer": "Reviews a pull request's diff and returns structured findings (severity, "
    "file:line, issue, suggestion) plus a risk rating. Give it the repo and PR number, and "
    "paste in any CI diagnosis you already have so it can connect failures to the code. "
    "Read-only.",
}

SUPERVISOR_PROMPT = """\
You are the coordinator of a GitHub assistant team. You do not use GitHub
tools yourself -- you delegate to a specialist and then relay/synthesise
their answer.

Specialists (each takes a single self-contained task string):
- repo_qa: questions about code, structure, "where/how does X work".
- issue_triage: reading, labelling, prioritising, de-duplicating issues.
- code_writer: creating branches, editing files, opening pull requests.
- ci_triage: diagnosing a failed CI/CD run ("why did the build fail?").
- pr_reviewer: reviewing the changes in a pull request.

Rules:
- Put everything the specialist needs INTO the task string (repo in
  owner/name form, issue/PR numbers, run ids, file paths). They share no
  memory with you or each other.
- One specialist is usually enough. Only chain them when the task
  genuinely has two phases (e.g. triage THEN comment).
- HANDOFF for "is PR #N failing, and is it risky?": call ci_triage first, then
  call pr_reviewer and PASTE the diagnosis into its task string so the reviewer
  can connect the failure to the code. If CI is green, skip ci_triage. Finish
  with ONE combined answer: why it fails, the review findings, your overall
  recommendation.
- ci_triage and pr_reviewer are read-only; they never change anything.
- For anything that changes the repo, delegate to code_writer and let the
  human approve.
- You also have `remember` / `recall` for long-term facts about the user
  (preferred repo, conventions). Use `remember` for durable preferences
  only, and `recall` when personalising an answer.
"""


def _make_specialist(name: str, model, all_tools: list, *, redact_pii: bool = True):
    wanted = _SPECIALIST_TOOLS[name]
    tools = [t for t in all_tools if t.name in wanted]
    schema = _SPECIALIST_SCHEMAS.get(name)
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=_SPECIALIST_PROMPTS[name],
        # ToolStrategy: the final answer is a call to a tool named after the
        # schema, validated by pydantic. Provider-agnostic (works anywhere tool
        # calling does) and, unlike native JSON-mode, safe to combine with tools.
        response_format=ToolStrategy(schema) if schema else None,
        middleware=[
            retry_on_rate_limit,
            retry_malformed_tool_call,
            # Specialists are the ones actually running the GitHub tools, so
            # THEY must treat tool output as untrusted data too.
            make_tool_output_guard(redact_pii=redact_pii),
            ModelCallLimitMiddleware(run_limit=8 if schema else 6, exit_behavior="end"),
        ],
    )


def assemble_supervisor(model, all_tools, checkpointer=None, *, extra_tools=(), settings=None):
    """Build the supervisor + specialists from their parts.

    Separate from ``build_supervisor`` (which opens the MCP subprocess and
    databases) so the whole team can be constructed in tests around a
    scripted fake model and local tools.
    """
    settings = settings or get_settings()
    redact_pii = settings.guardrails_redact_pii
    specialists = {
        name: _make_specialist(name, model, all_tools, redact_pii=redact_pii)
        for name in _SPECIALIST_TOOLS
    }

    def _wrap(name: str):
        agent = specialists[name]

        schema = _SPECIALIST_SCHEMAS.get(name)

        @tool(name, description=_SPECIALIST_DESCRIPTIONS[name])
        async def _delegate(task: str) -> str:
            result = await agent.ainvoke({"messages": [("user", task)]})
            report = result.get("structured_response")
            if report is None:
                return content_to_text(result["messages"][-1].content)
            # Grounding corpus = what the specialist's tools actually returned.
            # The framework's own "structured response" ToolMessage echoes the
            # report back, so it MUST be excluded or every quote would verify
            # against itself.
            corpus = "\n".join(
                content_to_text(m.content)
                for m in result["messages"]
                if isinstance(m, ToolMessage) and m.name != schema.__name__
            )
            return report.render(corpus)

        return _delegate

    supervisor = create_agent(
        model=model,
        tools=[*[_wrap(n) for n in _SPECIALIST_TOOLS], *extra_tools],
        system_prompt=SUPERVISOR_PROMPT,
        checkpointer=checkpointer,
        middleware=[
            SummarizationMiddleware(model=model, trigger=("messages", 40), keep=("messages", 12)),
            # Coarse-grained approval: the human okays the delegation to
            # code_writer; ALLOW_WRITES still gates each real call.
            HumanInTheLoopMiddleware(
                interrupt_on={"code_writer": {"allowed_decisions": ["approve", "reject"]}},
                description_prefix="Approve delegation to code_writer",
            ),
            retry_on_rate_limit,
            retry_malformed_tool_call,
            ModelCallLimitMiddleware(run_limit=10, exit_behavior="end"),
        ],
    )
    try:
        supervisor._reflection_model = model
    except Exception:  # pragma: no cover
        pass
    return supervisor


@contextlib.asynccontextmanager
async def build_supervisor():
    """Yield a compiled supervisor agent (same call shape as build_agent)."""
    configure_logging()
    settings = get_settings()
    model = init_chat_model(settings.chat_model)
    db_path = Path(settings.checkpoint_db).resolve()
    store_path = Path(settings.memory_db).resolve()

    async with (
        MCPAdapter(SERVER_PATH) as mcp,
        AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer,
        AsyncSqliteStore.from_conn_string(str(store_path)) as store,
    ):
        await store.setup()
        all_tools = await mcp.list_tools()
        yield assemble_supervisor(
            model,
            all_tools,
            checkpointer,
            extra_tools=build_memory_tools(store),
            settings=settings,
        )
