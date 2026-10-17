"""CI/CD-triage and PR-review tool logic (read-only), as plain functions.

``server.py`` exposes these as MCP tools; keeping the logic here means it can be
unit-tested against faked GitHub responses without importing the MCP server
(which pulls in the heavy RAG stack). Each function takes a ``GitHubClient`` and
returns a compact text report written for a model to read -- never raw API JSON,
never a raw log.

Design rule: ONE call should return what the model needs. ``get_ci_failure``
alone finds the failed run, its failing jobs, the shrunk log excerpts, category
hints and the workflow's recent history -- so a triage costs one tool call, not
six, which matters a lot when every model call is rate-limited.
"""

from __future__ import annotations

from ci_logs import classify, extract_failure
from github_client import GitHubClient, GitHubError
from pr_diff import render_pr_diff

_BAD = {"failure", "timed_out", "cancelled", "startup_failure", "action_required"}


def _sha(value: str | None) -> str:
    return (value or "")[:7]


def _state(run: dict) -> str:
    return run.get("conclusion") or run.get("status") or "unknown"


# --- workflow runs -----------------------------------------------------------------


def runs_overview(
    gh: GitHubClient, repo: str, status: str = "", branch: str = "", limit: int = 10
) -> str:
    runs = gh.list_workflow_runs(repo, status=status, branch=branch, limit=limit)
    if not runs:
        scope = " ".join(
            x
            for x in (f"status={status}" if status else "", f"branch={branch}" if branch else "")
            if x
        )
        return f"No workflow runs found{(' for ' + scope) if scope else ''}."
    rows = [
        f"run {r['id']} | {_state(r):<11} | {r.get('name', '?')} | {r.get('head_branch', '?')} "
        f"@ {_sha(r.get('head_sha'))} | {r.get('display_title', '')[:60]} | {r.get('html_url', '')}"
        for r in runs
    ]
    return "Recent workflow runs (newest first):\n" + "\n".join(rows)


def run_detail(gh: GitHubClient, repo: str, run_id: int) -> str:
    run = gh.get_workflow_run(repo, run_id)
    jobs = gh.list_run_jobs(repo, run_id)
    lines = [
        f"Run {run['id']}: {run.get('name', '?')} -> {_state(run)}",
        f"branch {run.get('head_branch', '?')} @ {_sha(run.get('head_sha'))}, event={run.get('event', '?')}, "
        f"attempt {run.get('run_attempt', 1)}",
        "Jobs:",
    ]
    for j in jobs:
        failed_steps = [s["name"] for s in j.get("steps", []) if s.get("conclusion") in _BAD]
        extra = f" (failed step: {', '.join(failed_steps)})" if failed_steps else ""
        lines.append(
            f"  job {j['id']} | {j.get('conclusion') or j.get('status')} | {j['name']}{extra}"
        )
    return "\n".join(lines)


def ci_failure_report(
    gh: GitHubClient, repo: str, run_id: int = 0, branch: str = "", max_jobs: int = 2
) -> str:
    """Everything needed to diagnose a failed run, already shrunk for a model."""
    if run_id:
        run = gh.get_workflow_run(repo, run_id)
    else:
        failed = gh.list_workflow_runs(repo, status="failure", branch=branch, limit=1)
        if not failed:
            where = f" on branch '{branch}'" if branch else ""
            return f"No failed workflow runs found{where} in {repo}. Nothing to triage."
        run = failed[0]
        run_id = run["id"]

    if _state(run) not in _BAD:
        return (
            f"Run {run_id} ({run.get('name', '?')}) is '{_state(run)}', not a failure. "
            "There is no failure to diagnose."
        )

    jobs = gh.list_run_jobs(repo, run_id)
    failed_jobs = [j for j in jobs if j.get("conclusion") in _BAD][:max_jobs]

    out = [
        f"CI FAILURE REPORT - run {run_id}: {run.get('name', '?')} ({_state(run)})",
        f"branch {run.get('head_branch', '?')} @ {_sha(run.get('head_sha'))} | "
        f"commit: {run.get('display_title', '')[:80]} | attempt {run.get('run_attempt', 1)}",
        f"url: {run.get('html_url', '')}",
        f"jobs: {len(jobs)} total, {sum(1 for j in jobs if j.get('conclusion') in _BAD)} failed",
    ]

    for j in failed_jobs:
        steps = [s["name"] for s in j.get("steps", []) if s.get("conclusion") in _BAD]
        out += [
            "",
            f"--- failed job {j['id']}: {j['name']} (failed step: {', '.join(steps) or 'unknown'}) ---",
        ]
        try:
            log = gh.get_job_logs(repo, j["id"])
        except GitHubError as exc:
            out.append(f"[logs unavailable: {exc}]")
            continue
        ex = extract_failure(log)
        out.append(
            f"[log excerpt - step: {ex.step or 'unknown'}; {ex.total_lines} log lines in total, shrunk]"
        )
        out.append(ex.excerpt)
        hints = classify(log)
        if hints:
            out.append("category hints (heuristic - verify against the excerpt above):")
            out += [f"  - {cat}: `{evidence}`" for cat, evidence in hints]

    out += ["", _history_section(gh, repo, run)]
    return "\n".join(out)


def _history_section(gh: GitHubClient, repo: str, run: dict) -> str:
    """Recent results of the same workflow on the same branch + a flakiness signal."""
    try:
        recent = gh.list_workflow_runs(repo, branch=run.get("head_branch", ""), limit=20)
    except GitHubError as exc:
        return f"[run history unavailable: {exc}]"
    same = [r for r in recent if r.get("name") == run.get("name")]
    trail = ", ".join(_state(r) for r in same[:6]) or "n/a"
    lines = [f"history of '{run.get('name', '?')}' on this branch (newest first): {trail}"]
    sha = run.get("head_sha")
    same_commit = [r for r in same if r.get("head_sha") == sha and r["id"] != run["id"]]
    if any(_state(r) == "success" for r in same_commit):
        lines.append(
            "FLAKINESS SIGNAL: this exact commit also has a PASSING run of the same workflow "
            "-> the failure may be transient."
        )
    elif same and all(_state(r) in _BAD for r in same[:3]) and len(same) >= 3:
        lines.append(
            "PERSISTENT: the last 3+ runs of this workflow all failed -> likely a real regression."
        )
    return "\n".join(lines)


def job_log_tail(gh: GitHubClient, repo: str, job_id: int, last_lines: int = 120) -> str:
    """Last N cleaned lines of one job's log -- for a closer look after the report."""
    from ci_logs import clean_lines

    log = gh.get_job_logs(repo, job_id)
    lines = [ln for ln in clean_lines(log) if ln.strip()]
    tail = lines[-max(10, min(last_lines, 400)) :]
    return f"Last {len(tail)} of {len(lines)} lines of job {job_id}:\n" + "\n".join(tail)


def checks_overview(gh: GitHubClient, repo: str, ref: str) -> str:
    checks = gh.list_check_runs(repo, ref)
    if not checks:
        return f"No check runs found for '{ref}'."
    lines = [f"Check runs for {ref} ({_summarise_checks(checks)}):"]
    lines += [f"  {c.get('conclusion') or c.get('status')} | {c['name']}" for c in checks]
    return "\n".join(lines)


def _summarise_checks(checks: list[dict]) -> str:
    bad = [c["name"] for c in checks if c.get("conclusion") in _BAD]
    ok = sum(1 for c in checks if c.get("conclusion") == "success")
    pending = sum(1 for c in checks if c.get("status") != "completed")
    parts = [f"{ok} passed"]
    if bad:
        parts.append(f"{len(bad)} failed: {', '.join(bad)}")
    if pending:
        parts.append(f"{pending} pending")
    return ", ".join(parts)


# --- pull requests -----------------------------------------------------------------


def pr_list(gh: GitHubClient, repo: str, state: str = "open", limit: int = 15) -> str:
    prs = gh.list_pull_requests(repo, state=state, limit=limit)
    if not prs:
        return f"No {state} pull requests in {repo}."
    rows = [
        f"#{p['number']} | {'draft ' if p.get('draft') else ''}{p['state']} | {p['title'][:70]} | "
        f"{p['head']['ref']} -> {p['base']['ref']} | by {p['user']['login']}"
        for p in prs
    ]
    return f"{state.capitalize()} pull requests:\n" + "\n".join(rows)


def pr_overview(gh: GitHubClient, repo: str, number: int) -> str:
    pr = gh.get_pull_request(repo, number)
    body = (pr.get("body") or "(no description)").strip()
    out = [
        f"PR #{pr['number']}: {pr['title']}",
        f"state: {pr['state']}{' (draft)' if pr.get('draft') else ''}{', MERGED' if pr.get('merged') else ''} | "
        f"{pr['head']['ref']} -> {pr['base']['ref']} | author: {pr['user']['login']}",
        f"size: {pr.get('changed_files', '?')} files, +{pr.get('additions', '?')} -{pr.get('deletions', '?')}, "
        f"{pr.get('commits', '?')} commit(s)",
        f"head commit: {_sha(pr['head'].get('sha'))}",
    ]
    labels = [lb["name"] for lb in pr.get("labels", [])]
    if labels:
        out.append("labels: " + ", ".join(labels))
    # CI state alongside the PR, so reviewer and triager share one picture.
    try:
        checks = gh.list_check_runs(repo, pr["head"]["sha"])
        out.append(f"CI: {_summarise_checks(checks) if checks else 'no check runs'}")
    except GitHubError as exc:
        out.append(f"CI: unavailable ({exc})")
    out += ["", "description:", body[:3000] + ("..." if len(body) > 3000 else "")]
    return "\n".join(out)


def pr_diff_text(
    gh: GitHubClient, repo: str, number: int, path: str = "", max_chars: int = 14000
) -> str:
    files = gh.list_pr_files(repo, number)
    if not files:
        return f"PR #{number} has no changed files."
    return render_pr_diff(files, max_chars=max_chars, only_path=path)
