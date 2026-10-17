"""devops_tools against faked GitHub API responses (no network, no token)."""

import json
from urllib.parse import parse_qs, urlparse

import responses
from test_ci_logs import PYTEST_FAILURE

from devops_tools import (
    checks_overview,
    ci_failure_report,
    job_log_tail,
    pr_diff_text,
    pr_list,
    pr_overview,
    run_detail,
    runs_overview,
)
from github_client import GitHubClient

API = "https://api.github.com"
R = "o/r"


def gh() -> GitHubClient:
    return GitHubClient(backoff_initial=0, backoff_max=0)


def run(run_id, conclusion="failure", sha="abc1234def00", name="CI", branch="main"):
    return {
        "id": run_id,
        "name": name,
        "conclusion": conclusion,
        "status": "completed",
        "head_branch": branch,
        "head_sha": sha,
        "display_title": "Fix policy filter",
        "html_url": f"https://github.com/o/r/actions/runs/{run_id}",
        "run_attempt": 1,
        "event": "push",
    }


JOBS = {
    "jobs": [
        {
            "id": 11,
            "name": "test (3.11)",
            "conclusion": "failure",
            "status": "completed",
            "steps": [
                {"name": "Set up job", "conclusion": "success"},
                {"name": "Run pytest", "conclusion": "failure"},
            ],
        },
        {"id": 12, "name": "lint", "conclusion": "success", "status": "completed", "steps": []},
    ]
}


def mock_runs(chooser):
    """Serve /actions/runs, choosing the payload from the query parameters."""

    def callback(request):
        params = {k: v[0] for k, v in parse_qs(urlparse(request.url).query).items()}
        return 200, {}, json.dumps({"workflow_runs": chooser(params)})

    responses.add_callback(
        responses.GET,
        f"{API}/repos/{R}/actions/runs",
        callback=callback,
        content_type="application/json",
    )


def mock_failed_run(history):
    mock_runs(lambda params: [run(900)] if params.get("status") == "failure" else history)
    responses.get(f"{API}/repos/{R}/actions/runs/900/jobs", json=JOBS)


# --- CI failure report ------------------------------------------------------------


@responses.activate
def test_failure_report_is_compact_and_diagnostic():
    mock_failed_run(history=[run(900), run(899, sha="d0"), run(898, sha="e0")])
    responses.get(
        f"{API}/repos/{R}/actions/jobs/11/logs", body=PYTEST_FAILURE, content_type="text/plain"
    )

    out = ci_failure_report(gh(), R)

    assert "CI FAILURE REPORT - run 900" in out
    assert "failed job 11: test (3.11)" in out and "failed step: Run pytest" in out
    assert "FAILED tests/test_api.py::test_chat_returns" in out  # the actual cause
    assert "category hints" in out and "test_failure" in out
    assert "PERSISTENT" in out  # last 3 runs all failed -> real regression
    assert "Collecting package-" not in out  # the 300 lines of pip noise are gone
    assert len(out) < 6500  # bounded, whatever the log size
    # the passing job's log is never fetched
    assert not [c for c in responses.calls if "/jobs/12/logs" in c.request.url]


@responses.activate
def test_a_passing_rerun_of_the_same_commit_raises_a_flakiness_signal():
    mock_failed_run(history=[run(900), run(901, conclusion="success", sha="abc1234def00")])
    responses.get(f"{API}/repos/{R}/actions/jobs/11/logs", body=PYTEST_FAILURE)

    out = ci_failure_report(gh(), R)

    assert "FLAKINESS SIGNAL" in out
    assert "PERSISTENT" not in out


@responses.activate
def test_no_failed_runs_says_so_instead_of_inventing_one():
    mock_runs(lambda params: [])
    out = ci_failure_report(gh(), R, branch="dev")
    assert "No failed workflow runs found on branch 'dev'" in out


@responses.activate
def test_asking_about_a_green_run_is_not_a_failure():
    responses.get(f"{API}/repos/{R}/actions/runs/777", json=run(777, conclusion="success"))
    out = ci_failure_report(gh(), R, run_id=777)
    assert "not a failure" in out


@responses.activate
def test_unavailable_logs_degrade_gracefully():
    mock_failed_run(history=[run(900)])
    responses.get(
        f"{API}/repos/{R}/actions/jobs/11/logs", json={"message": "Not Found"}, status=404
    )

    out = ci_failure_report(gh(), R)

    assert "failed job 11" in out
    assert "[logs unavailable" in out
    assert "history of 'CI'" in out  # the rest of the report still arrives


@responses.activate
def test_job_log_tail_returns_cleaned_last_lines():
    responses.get(f"{API}/repos/{R}/actions/jobs/11/logs", body=PYTEST_FAILURE)
    out = job_log_tail(gh(), R, 11, last_lines=10)
    assert out.startswith("Last 10 of ")
    assert "##[error]Process completed with exit code 1." in out
    assert "2026-10-16T" not in out and "\x1b" not in out


# --- listings ---------------------------------------------------------------------


@responses.activate
def test_runs_overview_lists_state_workflow_branch_and_sha():
    mock_runs(lambda params: [run(900), run(899, conclusion="success", sha="def5678")])
    out = runs_overview(gh(), R)
    assert "run 900 | failure" in out and "run 899 | success" in out
    assert "main @ abc1234" in out


@responses.activate
def test_runs_overview_empty_mentions_the_filters():
    mock_runs(lambda params: [])
    assert "No workflow runs found for status=failure branch=dev" in runs_overview(
        gh(), R, status="failure", branch="dev"
    )


@responses.activate
def test_run_detail_names_the_failed_steps():
    responses.get(f"{API}/repos/{R}/actions/runs/900", json=run(900))
    responses.get(f"{API}/repos/{R}/actions/runs/900/jobs", json=JOBS)
    out = run_detail(gh(), R, 900)
    assert "job 11 | failure | test (3.11) (failed step: Run pytest)" in out
    assert "job 12 | success | lint" in out


CHECKS = {
    "check_runs": [
        {"name": "test", "conclusion": "failure", "status": "completed"},
        {"name": "lint", "conclusion": "success", "status": "completed"},
        {"name": "build", "conclusion": "success", "status": "completed"},
    ]
}


@responses.activate
def test_checks_overview_summarises_pass_and_fail():
    responses.get(f"{API}/repos/{R}/commits/abc1234/check-runs", json=CHECKS)
    out = checks_overview(gh(), R, "abc1234")
    assert "2 passed, 1 failed: test" in out


# --- pull requests ------------------------------------------------------------------

PR = {
    "number": 12,
    "title": "Add policy filter",
    "state": "open",
    "draft": False,
    "merged": False,
    "user": {"login": "dev"},
    "head": {"ref": "feat/filter", "sha": "abc1234def00"},
    "base": {"ref": "main"},
    "changed_files": 3,
    "additions": 40,
    "deletions": 5,
    "commits": 2,
    "labels": [{"name": "enhancement"}],
    "body": "Adds a filter.",
}


@responses.activate
def test_pr_overview_includes_ci_status_for_the_head_commit():
    responses.get(f"{API}/repos/{R}/pulls/12", json=PR)
    responses.get(f"{API}/repos/{R}/commits/abc1234def00/check-runs", json=CHECKS)

    out = pr_overview(gh(), R, 12)

    assert "PR #12: Add policy filter" in out
    assert "feat/filter -> main" in out and "author: dev" in out
    assert "3 files, +40 -5" in out
    assert "CI: 2 passed, 1 failed: test" in out
    assert "labels: enhancement" in out


@responses.activate
def test_pr_overview_survives_a_token_without_checks_permission():
    responses.get(f"{API}/repos/{R}/pulls/12", json=PR)
    responses.get(
        f"{API}/repos/{R}/commits/abc1234def00/check-runs", json={"message": "no"}, status=403
    )

    out = pr_overview(gh(), R, 12)

    assert "PR #12" in out  # the overview still works
    assert "CI: unavailable" in out and "Forbidden" in out


@responses.activate
def test_pr_list_shows_branches_and_author():
    responses.get(f"{API}/repos/{R}/pulls", json=[PR])
    out = pr_list(gh(), R)
    assert "#12 | open | Add policy filter | feat/filter -> main | by dev" in out


@responses.activate
def test_pr_diff_text_annotates_lines_and_skips_lockfiles():
    files = [
        {
            "filename": "app/service.py",
            "status": "modified",
            "additions": 1,
            "deletions": 1,
            "changes": 2,
            "patch": "@@ -40,2 +40,2 @@\n ctx\n-old = 1\n+new = 2",
        },
        {
            "filename": "package-lock.json",
            "status": "modified",
            "additions": 900,
            "deletions": 3,
            "changes": 903,
            "patch": "@@ -1 +1 @@\n+x",
        },
    ]
    responses.get(f"{API}/repos/{R}/pulls/12/files", json=files)

    out = pr_diff_text(gh(), R, 12)

    assert "   41 +new = 2" in out  # real new-file line number
    assert "package-lock.json (lock/generated file)" in out
    assert "=== package-lock.json" not in out


@responses.activate
def test_pr_files_are_paginated():
    all_files = [
        {
            "filename": f"f{i}.py",
            "additions": 1,
            "deletions": 0,
            "changes": 1,
            "patch": "@@ -1 +1 @@\n+x",
        }
        for i in range(105)
    ]

    def callback(request):
        page = int(parse_qs(urlparse(request.url).query).get("page", ["1"])[0])
        return 200, {}, json.dumps(all_files[(page - 1) * 100 : page * 100])

    responses.add_callback(
        responses.GET,
        f"{API}/repos/{R}/pulls/12/files",
        callback=callback,
        content_type="application/json",
    )
    assert len(gh().list_pr_files(R, 12)) == 105
    assert len(responses.calls) == 2  # two pages, not an endless loop


@responses.activate
def test_a_pr_with_no_files_is_reported_plainly():
    responses.get(f"{API}/repos/{R}/pulls/12/files", json=[])
    assert "has no changed files" in pr_diff_text(gh(), R, 12)
