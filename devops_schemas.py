"""Structured reports for the DevOps specialists, with grounding checks.

Why structured? Free-form prose is hard to test, hard to post as a PR comment
and impossible to aggregate. A schema forces the model to commit to a category,
a confidence and evidence -- and makes the output machine-readable for the
webhook / eval work that builds on it.

Why grounding? The worst failure of a review/triage agent is a confident claim
that is not in the data: a log line nobody printed, a diff line that does not
exist. The model is told to quote VERBATIM; ``render`` then checks each quote
against the real tool output and flags what it cannot find. That turns
"trust me" into "here is what I could and could not verify".

The field ``description``s double as instructions: they are part of the tool
schema the model sees when it submits its report.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# grounding helpers (pure)
# --------------------------------------------------------------------------

_DIFF_PREFIX = re.compile(r"^\s*\d*\s*[+-]?\s?")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def quote_is_grounded(quote: str, corpus: str) -> bool:
    """Is ``quote`` (whitespace-insensitive) present in ``corpus``?

    Diff markers / line numbers the model may have copied along with the code
    are tolerated by also trying the quote with that prefix stripped.
    """
    if not quote or not quote.strip():
        return False
    haystack = _norm(corpus)
    candidates = {_norm(quote), _norm(_DIFF_PREFIX.sub("", quote, count=1))}
    return any(len(c) >= 3 and c in haystack for c in candidates)


def _mark(ok: bool) -> str:
    return "[verified]" if ok else "[UNVERIFIED - not found in the tool output]"


# --------------------------------------------------------------------------
# CI triage
# --------------------------------------------------------------------------


class CIReport(BaseModel):
    """Diagnosis of a failed CI/CD run. Submit this as your final answer."""

    run_id: int | None = Field(default=None, description="Workflow run id, if known.")
    workflow: str = Field(description="Workflow name, e.g. 'CI'.")
    branch: str = Field(default="", description="Branch the run was on.")
    commit: str = Field(default="", description="Short commit SHA of the run.")
    category: Literal[
        "test_failure",
        "lint_or_format",
        "build_or_compile",
        "dependency_install",
        "timeout_or_cancelled",
        "infra_or_network",
        "auth_or_permissions",
        "flaky",
        "unknown",
    ] = Field(description="Best-fit failure category. Use 'unknown' rather than guessing.")
    root_cause: str = Field(description="1-3 plain sentences: WHY the run failed.")
    evidence: list[str] = Field(
        default_factory=list,
        max_length=4,
        description="Up to 4 short quotes COPIED VERBATIM from the log excerpt that prove "
        "the root cause. Never paraphrase and never invent.",
    )
    suggested_fix: str = Field(description="Concrete next step for the developer.")
    rerun_recommended: bool = Field(
        description="True ONLY if the failure looks transient (flaky/infra) and the code is "
        "not at fault."
    )
    confidence: Literal["low", "medium", "high"]

    def render(self, corpus: str | None = None) -> str:
        """Markdown. If ``corpus`` (the tool output the model saw) is given,
        each evidence quote is verified against it."""
        lines = [
            f"### CI diagnosis: {self.workflow}" + (f" (run {self.run_id})" if self.run_id else ""),
            f"**Category:** {self.category} | **Confidence:** {self.confidence} | "
            f"**Re-run:** {'recommended' if self.rerun_recommended else 'not recommended'}",
        ]
        where = " @ ".join(x for x in (self.branch, self.commit) if x)
        if where:
            lines.append(f"**Where:** {where}")
        lines += ["", f"**Root cause:** {self.root_cause}"]
        if self.evidence:
            lines += ["", "**Evidence:**"]
            for q in self.evidence:
                tag = f" {_mark(quote_is_grounded(q, corpus))}" if corpus is not None else ""
                lines.append(f"- `{q}`{tag}")
        lines += ["", f"**Suggested fix:** {self.suggested_fix}"]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# PR review
# --------------------------------------------------------------------------


class Finding(BaseModel):
    """One concrete issue in the changed code."""

    severity: Literal["high", "medium", "low"]
    kind: Literal["bug", "security", "missing_test", "design", "performance", "maintainability"]
    file: str = Field(description="Path of a file that IS changed in this PR.")
    line: int | None = Field(
        default=None,
        description="Line number in the NEW file, taken from the left column of the diff.",
    )
    issue: str = Field(description="What is wrong, and why it matters. One or two sentences.")
    suggestion: str = Field(description="How to fix it.")
    quote: str = Field(
        default="",
        description="The exact changed line this refers to, copied VERBATIM from the diff "
        "(without the line number).",
    )


class PRReview(BaseModel):
    """A code review of a pull request. Submit this as your final answer."""

    pr_number: int
    title: str = ""
    summary: str = Field(description="2-3 sentences: what this PR does.")
    ci_status: str = Field(
        default="unknown", description="CI state, e.g. '3 passed, 1 failed (test)'."
    )
    risk: Literal["low", "medium", "high"]
    findings: list[Finding] = Field(
        default_factory=list,
        max_length=8,
        description="Real issues only, most severe first. Do NOT pad with style nitpicks.",
    )
    not_reviewed: str = Field(
        default="",
        description="Anything you could NOT review (skipped, truncated or unseen files).",
    )
    recommendation: Literal["looks_good", "changes_requested", "discuss"]

    def render(self, corpus: str | None = None) -> str:
        """Markdown. If ``corpus`` (the diff the model saw) is given, every
        finding's file and quote are verified against it."""
        head = f"### PR review: #{self.pr_number}" + (f" - {self.title}" if self.title else "")
        lines = [
            head,
            self.summary,
            "",
            f"**CI:** {self.ci_status} | **Risk:** {self.risk} | "
            f"**Recommendation:** {self.recommendation}",
        ]
        if not self.findings:
            lines += ["", "No issues found in the reviewed diff."]
        else:
            lines += ["", f"**Findings ({len(self.findings)}):**"]
            for n, f in enumerate(self.findings, 1):
                loc = f"{f.file}:{f.line}" if f.line else f.file
                lines.append(f"{n}. **[{f.severity.upper()}][{f.kind}]** `{loc}` - {f.issue}")
                lines.append(f"   Suggestion: {f.suggestion}")
                if f.quote:
                    tag = ""
                    if corpus is not None:
                        in_pr = f"=== {f.file} (" in corpus
                        ok = in_pr and quote_is_grounded(f.quote, corpus)
                        tag = f" {_mark(ok)}"
                    lines.append(f"   > `{f.quote}`{tag}")
                elif corpus is not None:
                    lines.append("   > [UNVERIFIED - no quote given]")
        if self.not_reviewed:
            lines += ["", f"**Not reviewed:** {self.not_reviewed}"]
        return "\n".join(lines)
