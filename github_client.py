"""A thin, resilient GitHub REST client.

Knows nothing about MCP, agents, or tools. Its job:

* make authenticated GitHub API calls over a pooled HTTP session,
* automatically retry the failures that are worth retrying
  (network blips, 5xx, rate limits) with exponential backoff,
* translate the failures that are NOT worth retrying (404, 401, 403)
  into a single ``GitHubError`` carrying a message a human -- or an LLM --
  can act on.

Anything that imports this module gets a clean Python API and never has
to think about status codes.
"""

from __future__ import annotations

import base64
import logging
import time

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from config import get_settings

logger = logging.getLogger(__name__)

# Transient HTTP statuses: the same request may well succeed if retried.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class GitHubError(RuntimeError):
    """A GitHub API call failed in a way the caller should surface.

    The message is written to be read by an LLM: it says what went wrong
    AND what a sensible next step is.
    """


class _Retry(Exception):
    """Internal signal: tell the tenacity decorator to try again."""


class GitHubClient:
    """Synchronous GitHub REST client with retries and error mapping."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        backoff_initial: float | None = None,
        backoff_max: float | None = None,
    ) -> None:
        s = get_settings()
        self._base = (base_url or s.github_api_url).rstrip("/")
        self._timeout = timeout if timeout is not None else s.request_timeout
        self._max_retries = max_retries if max_retries is not None else s.max_retries
        self._backoff_initial = (
            backoff_initial if backoff_initial is not None else s.retry_backoff_initial
        )
        self._backoff_max = (
            backoff_max if backoff_max is not None else s.retry_backoff_max
        )

        # A Session pools TCP connections and reuses them across calls
        # (keep-alive), which is noticeably faster than a fresh connection
        # per request.
        self._session = requests.Session()
        self._session.headers.update(s.github_headers)

    # -- public API ----------------------------------------------------

    def get(self, path: str, **params) -> requests.Response:
        """GET ``path`` (absolute, or relative to the API base)."""
        return self._request("GET", path, params=params or None)

    def post(self, path: str, json: dict) -> requests.Response:
        return self._request("POST", path, json=json)

    def put(self, path: str, json: dict) -> requests.Response:
        return self._request("PUT", path, json=json)

    # -- higher-level helpers the MCP tools call ---------------------

    def list_contents(self, repo: str, path: str = "") -> list[dict]:
        """Return the directory listing at ``repo`` / ``path``.

        GitHub returns a JSON array for a directory and a single JSON
        object for a file; this normalises both to a list.
        """
        resp = self.get(f"/repos/{repo}/contents/{path}".rstrip("/"))
        data = resp.json()
        return data if isinstance(data, list) else [data]

    # -- write operations (Block C) ---------------------------------
    # Each returns a short human-readable confirmation string including
    # the URL of whatever was created.

    def get_default_branch(self, repo: str) -> str:
        return self.get(f"/repos/{repo}").json()["default_branch"]

    def create_issue(self, repo: str, title: str, body: str = "") -> str:
        data = self.post(f"/repos/{repo}/issues", {"title": title, "body": body}).json()
        return f"Opened issue #{data['number']}: {data['html_url']}"

    def comment_on_issue(self, repo: str, number: int, body: str) -> str:
        data = self.post(
            f"/repos/{repo}/issues/{number}/comments", {"body": body}
        ).json()
        return f"Commented on #{number}: {data['html_url']}"

    def add_labels(self, repo: str, number: int, labels: list[str]) -> str:
        self.post(f"/repos/{repo}/issues/{number}/labels", {"labels": labels})
        return f"Added labels to #{number}: {', '.join(labels)}"

    def create_branch(self, repo: str, new_branch: str, from_branch: str = "") -> str:
        base = from_branch or self.get_default_branch(repo)
        sha = self.get(f"/repos/{repo}/git/ref/heads/{base}").json()["object"]["sha"]
        self.post(
            f"/repos/{repo}/git/refs",
            {"ref": f"refs/heads/{new_branch}", "sha": sha},
        )
        return f"Created branch '{new_branch}' from '{base}' ({sha[:7]})."

    def put_file(
        self, repo: str, path: str, content: str, message: str, branch: str
    ) -> str:
        import base64 as _b64

        body: dict = {
            "message": message,
            "content": _b64.b64encode(content.encode("utf-8")).decode("ascii"),
            "branch": branch,
        }
        # If the file already exists on that branch we must pass its blob sha.
        try:
            existing = self.get(f"/repos/{repo}/contents/{path}", ref=branch).json()
            if isinstance(existing, dict) and existing.get("sha"):
                body["sha"] = existing["sha"]
        except GitHubError:
            pass  # file doesn't exist yet -> a plain create
        data = self.put(f"/repos/{repo}/contents/{path}", body).json()
        return f"Wrote {path} on '{branch}': {data['commit']['html_url']}"

    def edit_file(
        self, repo: str, path: str, old: str, new: str, message: str, branch: str
    ) -> str:
        """Replace ``old`` with ``new`` in a file and commit -- a surgical
        edit. The full file is fetched and rewritten HERE, on the server,
        so the caller only ever handles the small changed snippet (this is
        what keeps big-file edits under a tight token budget). ``old`` must
        match exactly once."""
        import base64 as _b64

        data = self.get(f"/repos/{repo}/contents/{path}", ref=branch).json()
        if isinstance(data, list):
            raise GitHubError(f"'{path}' is a directory, not a file.")
        content = _b64.b64decode(data["content"]).decode("utf-8")

        count = content.count(old)
        if count == 0:
            raise GitHubError(
                f"old_string was not found in {path}. Read the exact text with "
                f"read_file_lines and copy it verbatim, including indentation."
            )
        if count > 1:
            raise GitHubError(
                f"old_string appears {count} times in {path}; it must be unique. "
                f"Include more surrounding lines so it matches exactly once."
            )

        updated = content.replace(old, new)
        body = {
            "message": message,
            "content": _b64.b64encode(updated.encode("utf-8")).decode("ascii"),
            "branch": branch,
            "sha": data["sha"],
        }
        resp = self.put(f"/repos/{repo}/contents/{path}", body).json()
        return f"Edited {path} on '{branch}': {resp['commit']['html_url']}"

    def open_pull_request(
        self, repo: str, title: str, head: str, base: str = "", body: str = ""
    ) -> str:
        base = base or self.get_default_branch(repo)
        data = self.post(
            f"/repos/{repo}/pulls",
            {"title": title, "head": head, "base": base, "body": body},
        ).json()
        return f"Opened PR #{data['number']}: {data['html_url']}"

    def list_issues(self, repo: str, state: str = "open", limit: int = 20) -> list[dict]:
        resp = self.get(f"/repos/{repo}/issues", state=state, per_page=limit)
        # The issues endpoint also returns PRs; drop those.
        return [i for i in resp.json() if "pull_request" not in i]

    def get_issue(self, repo: str, number: int) -> dict:
        return self.get(f"/repos/{repo}/issues/{number}").json()

    def download_tarball(self, repo: str, ref: str = "") -> bytes:
        """Return the gzipped-tar bytes of a whole repository.

        One request instead of hundreds of per-file calls -- the right
        way to pull a repo for indexing. ``ref`` may be a branch, tag or
        commit SHA; empty means the default branch.
        """
        path = f"/repos/{repo}/tarball/{ref}".rstrip("/")
        # GitHub 302-redirects this to codeload; the Session follows it.
        return self.get(path).content

    def read_text_file(self, repo: str, path: str) -> str:
        """Return the decoded UTF-8 text of a file in ``repo``."""
        resp = self.get(f"/repos/{repo}/contents/{path}")
        data = resp.json()
        if isinstance(data, list):
            raise GitHubError(
                f"'{path}' is a directory, not a file. Use list_repo_files "
                f"to see what's inside it."
            )
        if data.get("encoding") != "base64":
            raise GitHubError(
                f"'{path}' isn't a text file the API can return inline "
                f"(encoding: {data.get('encoding')!r}, size: {data.get('size')} "
                f"bytes). Files over ~1 MB need the blobs API."
            )
        return base64.b64decode(data["content"]).decode("utf-8", errors="replace")

    def read_lines(self, repo: str, path: str, start: int, end: int) -> str:
        """Return lines ``start``..``end`` (1-based, inclusive) of a file.

        For large files, reading a slice keeps the response small enough to
        fit a tight token budget. No line-number prefixes: the text comes
        back verbatim so it can be copied straight into ``edit_file`` as the
        ``old`` string. A header names the range so the caller knows where
        it is.
        """
        lines = self.read_text_file(repo, path).splitlines()
        total = len(lines)
        start = max(1, start)
        end = min(total, end) if end > 0 else total
        if start > total:
            return f"--- {path}: only {total} lines; {start} is past the end ---"
        body = "\n".join(lines[start - 1 : end])
        return f"--- {path} lines {start}-{end} of {total} ---\n{body}"

    # -- internals -----------------------------------------------------

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        url = path if path.startswith("http") else f"{self._base}/{path.lstrip('/')}"

        # The retry policy is built here (not as a module-level decorator)
        # so it can read this instance's configuration -- and so tests can
        # dial the backoff down to zero.
        @retry(
            retry=retry_if_exception_type(
                (_Retry, requests.ConnectionError, requests.Timeout)
            ),
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential_jitter(
                initial=self._backoff_initial, max=self._backoff_max
            ),
            reraise=True,
        )
        def _attempt() -> requests.Response:
            resp = self._session.request(method, url, timeout=self._timeout, **kwargs)

            if self._is_rate_limited(resp):
                self._sleep_off_rate_limit(resp)
                raise _Retry("primary rate limit")

            if resp.status_code in _RETRYABLE_STATUS:
                logger.warning("Transient %s from %s; will retry", resp.status_code, url)
                raise _Retry(f"HTTP {resp.status_code}")

            return resp

        try:
            resp = _attempt()
        except _Retry as exc:
            raise GitHubError(
                f"GitHub kept failing after {self._max_retries} attempts "
                f"({exc}). Try again in a minute."
            ) from exc

        return self._raise_for_status(resp)

    @staticmethod
    def _is_rate_limited(resp: requests.Response) -> bool:
        return (
            resp.status_code in (403, 429)
            and resp.headers.get("X-RateLimit-Remaining") == "0"
        )

    def _sleep_off_rate_limit(self, resp: requests.Response) -> None:
        """Wait until the rate-limit window resets (bounded, so the agent
        never hangs for the full hour a GitHub reset can be away)."""
        retry_after = resp.headers.get("Retry-After", "")
        if retry_after.isdigit():
            wait = int(retry_after)
        else:
            reset = int(resp.headers.get("X-RateLimit-Reset", "0"))
            wait = reset - int(time.time())
        wait = max(0, min(wait, 60))
        logger.warning("Rate limited by GitHub; sleeping %ss before retry", wait)
        time.sleep(wait)

    @staticmethod
    def _raise_for_status(resp: requests.Response) -> requests.Response:
        if resp.ok:
            return resp

        code = resp.status_code
        if code == 401:
            raise GitHubError(
                "Authentication failed (401). GITHUB_TOKEN is missing, "
                "malformed, or expired."
            )
        if code == 403:
            raise GitHubError(
                "Forbidden (403). The token is valid but not allowed to touch "
                "this resource -- check its scopes / repository permissions."
            )
        if code == 404:
            raise GitHubError(
                "Not found (404). Double-check the owner/repo and the path. "
                "Private repositories also need a token with 'repo' scope."
            )
        if code == 422:
            raise GitHubError(f"GitHub rejected the request (422): {resp.text[:300]}")
        raise GitHubError(f"Unexpected GitHub API error {code}: {resp.text[:200]}")
