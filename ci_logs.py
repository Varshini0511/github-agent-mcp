"""Shrink a raw CI job log to the lines that explain the failure.

A GitHub Actions job log is routinely megabytes of timestamps, ANSI colour
codes, dependency installs and progress bars. Feeding that to a model is slow,
expensive, and -- on a small context window -- simply impossible. But the
*cause* of a failure is almost always a handful of lines near the end.

``extract_failure`` finds them. ``classify`` adds cheap, deterministic category
*hints* (test failure, lint, build error, ...) so even a weak model starts from
the right place -- the model still decides; hints are labelled as such.

Pure functions, no network: every behaviour here is unit-tested against
realistic log fixtures (tests/test_ci_logs.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
# GitHub prefixes every line with an ISO timestamp (the first also has a BOM).
_TIMESTAMP = re.compile(r"^﻿?\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z ?")
_GROUP = "##[group]"
_ERROR_MARK = "##[error]"

# Lines that very likely ARE the failure (as opposed to merely mentioning
# the word "error"). Deliberately specific to keep false positives low.
_STRONG = re.compile(
    r"##\[error\]"
    r"|\bFAILED\b"
    r"|Traceback \(most recent call last\)"
    r"|\b[A-Z][A-Za-z]*(?:Error|Exception)\b"
    r"|npm ERR!"
    r"|\bfatal:"
    r"|error TS\d+|error CS\d+|error\[E\d+\]"
    r"|Process completed with exit code [1-9]\d*"
    r"|command not found"
    r"|^E {3}"
)

# category -> pattern. Order = precedence in the output (specific first).
_CATEGORIES: list[tuple[str, re.Pattern[str]]] = [
    (
        "dependency_install",
        re.compile(
            r"ERESOLVE|ResolutionImpossible|No matching distribution found"
            r"|Could not find a version that satisfies|npm ERR! 404"
            r"|Failed to resolve dependencies|Unable to resolve dependency"
            r"|Could not resolve dependencies",
        ),
    ),
    (
        "build_or_compile",
        re.compile(
            r"error TS\d+|error CS\d+|error\[E\d+\]|SyntaxError|Module not found"
            r"|Cannot find module|cannot find symbol|[Cc]ompilation failed|Build failed"
            r"|ModuleNotFoundError|ImportError|Failed to compile",
        ),
    ),
    (
        "lint_or_format",
        re.compile(
            r"would reformat|would be reformatted|Found \d+ errors?\b"
            r"|\d+ problems? \(\d+ errors?|[\w/.\-]+:\d+:\d+: [A-Z]+\d{3,4}\b",
        ),
    ),
    (
        "test_failure",
        re.compile(
            r"FAILED\s+\S+::|\b\d+ failed\b|AssertionError|Tests?:\s+\d+ failed"
            r"|✕|●\s.+›|\bFAIL\s+\S+\.(?:test|spec)\.|Test Run Failed",
        ),
    ),
    (
        "timeout_or_cancelled",
        re.compile(
            r"timed? ?out|exceeded the maximum execution time|operation was canceled"
            r"|deadline exceeded",
            re.IGNORECASE,
        ),
    ),
    (
        "infra_or_network",
        re.compile(
            r"ECONNRESET|ETIMEDOUT|EAI_AGAIN|503 Service Unavailable|502 Bad Gateway"
            r"|rate limit exceeded|No space left on device"
            r"|Temporary failure in name resolution|connection reset by peer"
            r"|runner has received a shutdown signal|lost communication with the server",
            re.IGNORECASE,
        ),
    ),
    (
        "auth_or_permissions",
        re.compile(
            r"permission denied|Bad credentials|authentication failed"
            r"|Resource not accessible by integration|could not read Username"
            r"|secret .* (?:is not set|not found)",
            re.IGNORECASE,
        ),
    ),
]


@dataclass
class FailureExcerpt:
    step: str | None  # the workflow step the failure happened in
    excerpt: str  # the shrunk, human/LLM-readable text
    total_lines: int  # size of the original log
    found_error: bool  # False => no error marker; excerpt is just the log tail


def clean_lines(log_text: str) -> list[str]:
    """Strip ANSI colours, per-line timestamps and carriage returns."""
    out = []
    for raw in log_text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = _TIMESTAMP.sub("", _ANSI.sub("", raw)).rstrip()
        out.append(line)
    return out


def _collapse_repeats(lines: list[str]) -> list[str]:
    """``x, x, x`` -> ``x  (repeated 3x)`` so spinners/retries don't eat the budget."""
    out: list[str] = []
    i = 0
    while i < len(lines):
        j = i
        while j + 1 < len(lines) and lines[j + 1] == lines[i]:
            j += 1
        out.append(lines[i] if j == i else f"{lines[i]}  (repeated {j - i + 1}x)")
        i = j + 1
    return out


def _step_of(lines: list[str], index: int) -> str | None:
    """Name of the step the line at ``index`` belongs to (nearest preceding
    ``##[group]``; GitHub echoes the command in that group, then the output)."""
    for i in range(index, -1, -1):
        if lines[i].startswith(_GROUP):
            return lines[i][len(_GROUP) :].strip() or None
    return None


def extract_failure(
    log_text: str,
    *,
    context_before: int = 10,
    context_after: int = 4,
    max_lines: int = 60,
    max_chars: int = 4500,
) -> FailureExcerpt:
    """Return the part of ``log_text`` that explains why the job failed.

    Strategy: a failure is the LAST cluster of error-looking lines (the job
    stopped there). Take that cluster plus some lead-in context, always append
    the final ``##[error]`` line (the exit code), and, if still too long, keep
    the head and tail of the window rather than the middle.
    """
    raw = clean_lines(log_text)
    total = len(raw)
    # Drop blank lines, then collapse repeats over the WHOLE log (not just the
    # excerpt), so "retrying..." x50 is reported as 50x -- not as however many
    # happened to fall inside the window -- and spinner-heavy logs shrink hard.
    lines = _collapse_repeats([line for line in raw if line.strip()])

    strong = [i for i, line in enumerate(lines) if _STRONG.search(line)]
    markers = [i for i, line in enumerate(lines) if _ERROR_MARK in line]
    anchors = strong or markers

    if not anchors:
        tail = _collapse_repeats([line for line in lines[-40:] if line.strip()])
        return FailureExcerpt(
            step=None,
            excerpt="[no explicit error marker found; showing the end of the log]\n"
            + "\n".join(tail)[-max_chars:],
            total_lines=total,
            found_error=False,
        )

    # Walk back from the last anchor to the start of its cluster.
    last = anchors[-1]
    cluster_start = last
    for idx in reversed(anchors[:-1]):
        if cluster_start - idx <= 25:
            cluster_start = idx
        else:
            break

    start = max(0, cluster_start - context_before)
    end = min(total, last + context_after + 1)
    window = [line for line in lines[start:end] if line.strip()]

    final_marker = next((lines[i] for i in reversed(markers) if i >= end), None)
    if final_marker:
        window.append(final_marker)

    window = _collapse_repeats(window)
    if len(window) > max_lines:
        half = max_lines // 2
        omitted = len(window) - 2 * half
        window = [*window[:half], f"... ({omitted} lines omitted) ...", *window[-half:]]

    text = "\n".join(window)
    if len(text) > max_chars:  # last-resort cap, keep the end (where the failure is)
        text = "... (truncated) ...\n" + text[-max_chars:]

    return FailureExcerpt(
        step=_step_of(lines, cluster_start),
        excerpt=text,
        total_lines=total,
        found_error=True,
    )


def classify(log_text: str, *, scan_last_lines: int = 600, limit: int = 3) -> list[tuple[str, str]]:
    """Heuristic category hints: ``[(category, evidence_line), ...]``.

    These are HINTS for the model, not verdicts: they are cheap and
    deterministic, which makes them testable and gives a weak model a head start.
    """
    lines = [line for line in clean_lines(log_text)[-scan_last_lines:] if line.strip()]
    hints: list[tuple[str, str]] = []
    for category, pattern in _CATEGORIES:
        for line in lines:
            if pattern.search(line):
                hints.append((category, line.strip()[:160]))
                break
        if len(hints) >= limit:
            break
    return hints
