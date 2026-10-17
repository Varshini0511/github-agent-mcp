"""ci_logs: shrinking raw GitHub Actions logs to the lines that explain the failure.

Fixtures imitate real logs: per-line timestamps (first line has a BOM), ANSI
colour codes, ``##[group]`` step markers and plenty of irrelevant noise.
"""

from ci_logs import classify, clean_lines, extract_failure


def gh_log(*lines: str) -> str:
    """Prefix every line the way GitHub does."""
    return "\n".join(
        ("﻿" if i == 0 else "") + f"2026-10-16T05:41:{i % 60:02d}.1234567Z {line}"
        for i, line in enumerate(lines)
    )


PYTEST_FAILURE = gh_log(
    "##[group]Run actions/checkout@v4",
    "Syncing repository: o/r",
    "##[endgroup]",
    '##[group]Run pip install -e ".[dev]"',
    "##[endgroup]",
    *[f"Collecting package-{i}==1.{i}.0" for i in range(300)],
    "Successfully installed everything",
    "##[group]Run pytest -q",
    "pytest -q",
    "shell: /usr/bin/bash -e {0}",
    "##[endgroup]",
    "============================= test session starts ==============================",
    "collected 110 items",
    "tests/test_api.py ........F..",
    "=================================== FAILURES ===================================",
    "_______________________________ test_chat_returns ______________________________",
    "    def test_chat_returns():",
    '>       assert body["content"] == "It has a README."',
    "E       AssertionError: assert 'It has a readme' == 'It has a README.'",
    "E         - It has a README.",
    "tests/test_api.py:41: AssertionError",
    "=========================== short test summary info ============================",
    "\x1b[31mFAILED\x1b[0m tests/test_api.py::test_chat_returns - AssertionError: assert 'It has a readme' == ...",
    "========================= 1 failed, 109 passed in 9.84s ========================",
    "##[error]Process completed with exit code 1.",
)


def test_clean_lines_strips_timestamps_bom_and_ansi():
    lines = clean_lines(PYTEST_FAILURE)
    assert lines[0].startswith("##[group]Run actions/checkout")  # BOM + timestamp gone
    assert all("\x1b" not in line for line in lines)
    assert all(not line.startswith("2026-10-16T") for line in lines)


def test_pytest_failure_excerpt_keeps_the_cause_and_drops_the_noise():
    ex = extract_failure(PYTEST_FAILURE)

    assert ex.found_error
    assert ex.step == "Run pytest -q"
    assert "FAILED tests/test_api.py::test_chat_returns" in ex.excerpt
    assert "AssertionError" in ex.excerpt
    assert "Process completed with exit code 1" in ex.excerpt
    assert "Collecting package-" not in ex.excerpt  # 300 lines of pip noise
    assert "\x1b" not in ex.excerpt and "2026-10-16T" not in ex.excerpt
    assert len(ex.excerpt.splitlines()) <= 62
    assert ex.total_lines > 300  # it did see the whole log


def test_classification_hints_the_right_category():
    hints = dict(classify(PYTEST_FAILURE))
    assert "test_failure" in hints
    # The hint carries a real line from the log as evidence (the model decides).
    assert any(k in hints["test_failure"] for k in ("AssertionError", "FAILED tests/test_api.py"))


def test_build_error_is_classified_as_build():
    log = gh_log(
        "##[group]Run npm run build",
        "> next build",
        "src/app/page.tsx(12,5): error TS2339: Property 'foo' does not exist on type 'Props'.",
        "##[error]Process completed with exit code 2.",
    )
    assert classify(log)[0][0] == "build_or_compile"
    assert "error TS2339" in extract_failure(log).excerpt


def test_dependency_resolution_beats_other_categories():
    log = gh_log(
        "##[group]Run npm ci",
        "npm ERR! code ERESOLVE",
        "npm ERR! ERESOLVE could not resolve",
        "npm ERR! Found: react@19.0.0",
        "##[error]Process completed with exit code 1.",
    )
    assert classify(log)[0][0] == "dependency_install"


def test_lint_failure_is_recognised():
    log = gh_log(
        "##[group]Run ruff check .",
        "tests/test_x.py:3:1: F401 `os` imported but unused",
        "Found 1 error.",
        "##[error]Process completed with exit code 1.",
    )
    assert classify(log)[0][0] == "lint_or_format"


def test_timeout_and_cancellation_are_recognised():
    log = gh_log(
        "##[group]Run pytest",
        "tests/test_slow.py ....",
        "##[error]The operation was canceled.",
        "The job running on runner GitHub Actions 3 has exceeded the maximum execution time of 360 minutes.",
    )
    assert "timeout_or_cancelled" in dict(classify(log))


def test_network_flake_is_recognised_as_infra():
    log = gh_log(
        "##[group]Run npm install",
        "npm ERR! code ECONNRESET",
        "npm ERR! network request to https://registry.npmjs.org failed, reason: read ECONNRESET",
        "##[error]Process completed with exit code 1.",
    )
    assert "infra_or_network" in dict(classify(log))


def test_permission_errors_are_recognised():
    log = gh_log(
        "##[group]Run git push",
        "remote: Permission denied to github-actions[bot].",
        "fatal: unable to access 'https://github.com/o/r/': The requested URL returned error: 403",
        "##[error]Process completed with exit code 128.",
    )
    assert "auth_or_permissions" in dict(classify(log))


def test_a_huge_log_is_bounded_and_keeps_the_error():
    noise = [f"downloading dependency {i}" for i in range(30_000)]
    log = gh_log(
        "##[group]Run make test",
        *noise,
        "Traceback (most recent call last):",
        '  File "app.py", line 9, in <module>',
        "ZeroDivisionError: division by zero",
        "##[error]Process completed with exit code 1.",
    )
    ex = extract_failure(log)
    assert "ZeroDivisionError: division by zero" in ex.excerpt
    assert len(ex.excerpt) <= 4600
    assert ex.excerpt.count("downloading dependency") <= 12  # only the lead-in context
    assert ex.total_lines > 30_000


def test_oversized_error_window_keeps_head_and_tail_not_the_middle():
    body = [f"E   detail line {i}" for i in range(200)]
    log = gh_log("##[group]Run pytest", *body, "##[error]Process completed with exit code 1.")
    ex = extract_failure(log, max_lines=20)
    assert "lines omitted" in ex.excerpt
    assert "detail line 0" in ex.excerpt and "detail line 199" in ex.excerpt
    assert "detail line 100" not in ex.excerpt


def test_repeated_lines_are_collapsed():
    log = gh_log(
        "##[group]Run flaky-script",
        *["retrying connection..."] * 50,
        "fatal: gave up",
        "##[error]Process completed with exit code 1.",
    )
    ex = extract_failure(log)
    assert "retrying connection...  (repeated 50x)" in ex.excerpt
    assert ex.excerpt.count("retrying connection") == 1


def test_log_without_an_error_falls_back_to_the_tail_and_says_so():
    log = gh_log("##[group]Run build", "compiling...", "done", "all good")
    ex = extract_failure(log)
    assert not ex.found_error
    assert ex.excerpt.startswith("[no explicit error marker found")
    assert "all good" in ex.excerpt


def test_classify_returns_nothing_for_a_clean_log():
    assert classify(gh_log("##[group]Run build", "compiling...", "done")) == []
