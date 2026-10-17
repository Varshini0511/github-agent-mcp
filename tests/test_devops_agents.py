"""The DevOps specialists: structured reports, grounding, and the triage -> review handoff.

Real supervisor + specialist graphs, scripted model, fake GitHub tools.
"""

from fakes import ScriptedChatModel, ai_text, ai_tool, tool_call
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from agent_core import WRITE_TOOLS, content_to_text, send
from config import get_settings
from devops_schemas import CIReport, Finding, PRReview, quote_is_grounded
from multi_agent import _SPECIALIST_SCHEMAS, _SPECIALIST_TOOLS, assemble_supervisor

LOG_LINE = "FAILED tests/test_api.py::test_chat_returns - AssertionError"
DIFF = "=== app/service.py (modified, +1 -1) ===\n   41 +total = price * qty"


def cfg(thread: str) -> dict:
    return {"configurable": {"thread_id": thread}}


def fake_tools():
    ran: list[str] = []

    @tool
    def get_ci_failure(repo: str, run_id: int = 0, branch: str = "") -> str:
        """CI failure report."""
        ran.append("get_ci_failure")
        return f"CI FAILURE REPORT - run 900\n{LOG_LINE}\nPERSISTENT: last 3 runs failed"

    @tool
    def get_pr_diff(repo: str, number: int, path: str = "") -> str:
        """PR diff."""
        ran.append("get_pr_diff")
        return DIFF

    @tool
    def get_pull_request(repo: str, number: int) -> str:
        """PR overview."""
        ran.append("get_pull_request")
        return "PR #12: Fix totals\nCI: 2 passed, 1 failed: test"

    @tool
    def create_branch(repo: str, new_branch: str) -> str:
        """Create a branch."""
        ran.append("create_branch")
        return "created"

    return [get_ci_failure, get_pr_diff, get_pull_request, create_branch], ran


def build(script):
    tools, ran = fake_tools()
    model = ScriptedChatModel(script=script)
    sup = assemble_supervisor(model, tools, InMemorySaver(), settings=get_settings())
    return sup, model, ran


def ci_report(evidence=(LOG_LINE,), **over) -> dict:
    return {
        "run_id": 900,
        "workflow": "CI",
        "branch": "main",
        "commit": "abc1234",
        "category": "test_failure",
        "root_cause": "test_chat_returns asserts the wrong casing.",
        "evidence": list(evidence),
        "suggested_fix": "Fix the expected string in tests/test_api.py.",
        "rerun_recommended": False,
        "confidence": "high",
        **over,
    }


def tool_texts(messages) -> str:
    return "\n".join(content_to_text(m.content) for m in messages if isinstance(m, ToolMessage))


# --- schemas and grounding -------------------------------------------------------------


def test_quote_grounding_ignores_whitespace_and_diff_prefixes():
    corpus = "   41 +total  =  price * qty\n   42  return total"
    assert quote_is_grounded("total = price * qty", corpus)
    assert quote_is_grounded("+total = price * qty", corpus)
    assert not quote_is_grounded("total = price * quantity", corpus)
    assert not quote_is_grounded("", corpus)


def test_ci_report_marks_each_quote_verified_or_not():
    report = CIReport(**ci_report(evidence=[LOG_LINE, "SegmentationFault in libfoo"]))
    out = report.render(corpus=LOG_LINE)
    assert f"`{LOG_LINE}` [verified]" in out
    assert "`SegmentationFault in libfoo` [UNVERIFIED" in out
    assert "**Category:** test_failure" in out and "**Re-run:** not recommended" in out


def test_ci_report_without_a_corpus_makes_no_verification_claims():
    assert "verified" not in CIReport(**ci_report()).render().lower()


def test_pr_review_verifies_file_and_quote_against_the_diff():
    good = Finding(
        severity="high",
        kind="bug",
        file="app/service.py",
        line=41,
        issue="No guard for qty < 0.",
        suggestion="Validate qty.",
        quote="total = price * qty",
    )
    wrong_file = good.model_copy(update={"file": "app/other.py"})
    made_up = good.model_copy(update={"quote": "total = price ** qty"})
    review = PRReview(
        pr_number=12,
        summary="Fixes totals.",
        risk="medium",
        findings=[good, wrong_file, made_up],
        recommendation="changes_requested",
    )
    out = review.render(corpus=DIFF)
    verdicts = [line for line in out.splitlines() if line.startswith("   > ")]
    assert verdicts[0].endswith("[verified]")
    assert "UNVERIFIED" in verdicts[1]  # the file is not part of the PR
    assert "UNVERIFIED" in verdicts[2]  # the quote is not in the diff


def test_review_with_no_findings_says_so():
    out = PRReview(pr_number=3, summary="Docs.", risk="low", recommendation="looks_good").render(
        DIFF
    )
    assert "No issues found" in out


# --- team wiring ---------------------------------------------------------------------------


def test_devops_specialists_are_read_only_and_structured():
    for name in ("ci_triage", "pr_reviewer"):
        assert not set(_SPECIALIST_TOOLS[name]) & set(WRITE_TOOLS), f"{name} holds write tools"
    assert _SPECIALIST_SCHEMAS == {"ci_triage": CIReport, "pr_reviewer": PRReview}


async def test_ci_triage_returns_a_rendered_verified_diagnosis_without_any_approval():
    sup, model, ran = build(
        [
            ai_tool([tool_call("ci_triage", {"task": "why did the build fail in o/r?"}, "m1")]),
            ai_tool([tool_call("get_ci_failure", {"repo": "o/r"}, "s1")]),
            ai_tool([tool_call("CIReport", ci_report(), "s2")]),
            ai_text("The build failed on a test assertion."),
        ]
    )
    step = await send(sup, cfg("ci-1"), "why did the build fail in o/r?")

    assert step["type"] == "final"  # read-only: no interrupt
    assert ran == ["get_ci_failure"]
    delegated = tool_texts(model.seen_messages[-1])  # what the manager got back
    assert "### CI diagnosis: CI (run 900)" in delegated
    assert f"`{LOG_LINE}` [verified]" in delegated


async def test_a_fabricated_quote_reaches_the_manager_flagged_as_unverified():
    sup, model, _ = build(
        [
            ai_tool([tool_call("ci_triage", {"task": "diagnose"}, "m1")]),
            ai_tool([tool_call("get_ci_failure", {"repo": "o/r"}, "s1")]),
            ai_tool([tool_call("CIReport", ci_report(evidence=["OutOfMemoryError: heap"]), "s2")]),
            ai_text("Looks like an OOM."),
        ]
    )
    await send(sup, cfg("ci-2"), "diagnose")
    assert "[UNVERIFIED" in tool_texts(model.seen_messages[-1])


async def test_handoff_triage_diagnosis_reaches_the_reviewer():
    diagnosis = "Root cause: test_chat_returns asserts the wrong casing"
    review = {
        "pr_number": 12,
        "title": "Fix totals",
        "summary": "Changes how totals are computed.",
        "ci_status": "2 passed, 1 failed (test)",
        "risk": "medium",
        "findings": [
            {
                "severity": "medium",
                "kind": "missing_test",
                "file": "app/service.py",
                "line": 41,
                "issue": "No test covers the new total.",
                "suggestion": "Add a unit test.",
                "quote": "total = price * qty",
            }
        ],
        "recommendation": "changes_requested",
    }
    sup, model, ran = build(
        [
            ai_tool([tool_call("ci_triage", {"task": "why is PR 12 failing in o/r?"}, "m1")]),  # 0
            ai_tool([tool_call("get_ci_failure", {"repo": "o/r"}, "s1")]),  # 1
            ai_tool([tool_call("CIReport", ci_report(), "s2")]),  # 2
            ai_tool(  # 3: manager hands the diagnosis to the reviewer
                [
                    tool_call(
                        "pr_reviewer", {"task": f"Review PR 12 in o/r. CI said: {diagnosis}"}, "m2"
                    )
                ]
            ),
            ai_tool([tool_call("get_pr_diff", {"repo": "o/r", "number": 12}, "s3")]),  # 4
            ai_tool([tool_call("PRReview", review, "s4")]),  # 5
            ai_text("The failure is a test typo; the change itself needs a test."),  # 6
        ]
    )
    step = await send(sup, cfg("ho-1"), "PR 12 is failing - why, and is the change risky?")

    assert step["type"] == "final" and "needs a test" in step["content"]
    assert ran == ["get_ci_failure", "get_pr_diff"]  # no writes, no extra tools
    assert model.cursor == 7
    # The reviewer's first call (index 4) was shown the triage diagnosis as its task.
    reviewer_task = " ".join(
        content_to_text(m.content) for m in model.seen_messages[4] if isinstance(m, HumanMessage)
    )
    assert diagnosis in reviewer_task
    # And the final manager call saw a verified review.
    final_view = tool_texts(model.seen_messages[6])
    assert "### PR review: #12" in final_view and "[verified]" in final_view


async def test_code_writer_is_still_the_only_path_to_a_write():
    sup, model, ran = build(
        [
            ai_tool([tool_call("code_writer", {"task": "create branch fix"}, "m1")]),
            ai_text("not run"),
        ]
    )
    step = await send(sup, cfg("w-1"), "fix it")
    assert step["type"] == "interrupt" and ran == []
