"""GitHub webhook handling: verify, parse, de-duplicate and rate-cap CI failures.

The HTTP wiring lives in ``api.py``. Everything here is pure and unit-tested so
the signature check, event parsing, redelivery de-duplication and the
per-commit action cap can all be verified without a running server or a real
GitHub delivery.

Why a cap? The whole point of a webhook is that a failing build triggers the
agent *by itself*. That is also the danger: a commit that keeps failing (or a
delivery GitHub keeps retrying) would otherwise fire the model on every event
and drain a free-tier quota. ``AttemptLimiter`` is the brake.
"""

from __future__ import annotations

import hashlib
import hmac
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field

# GitHub marks these as "the run did not succeed". A plain success or a run
# that is still in progress is not something to triage.
_FAILED_CONCLUSIONS = {"failure", "timed_out", "startup_failure"}


def verify_signature(secret: str, body: bytes, header: str | None) -> bool:
    """Is ``header`` a valid ``sha256=...`` HMAC of ``body`` under ``secret``?

    GitHub signs every delivery with the webhook secret. An unsigned or
    wrongly-signed request must be rejected -- the endpoint acts on GitHub with
    the owner's token, so the signature is the only thing proving the request
    really came from GitHub. The compare is constant-time so a timing
    side-channel cannot be used to forge the digest byte by byte.
    """
    if not secret or not header or not header.startswith("sha256="):
        return False
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest("sha256=" + digest, header)


@dataclass(frozen=True)
class WorkflowFailure:
    """The few facts about a failed run the triage step actually needs."""

    repo: str
    run_id: int
    workflow: str
    branch: str
    head_sha: str
    html_url: str = ""

    @property
    def commit_key(self) -> str:
        """Identity for the per-commit cap: one commit, however many re-runs."""
        return f"{self.repo}@{self.head_sha}"


def parse_workflow_failure(event: str, payload: dict) -> WorkflowFailure | None:
    """Return a ``WorkflowFailure`` iff ``payload`` is a COMPLETED, FAILED
    ``workflow_run`` event.

    Anything else -- a different event, a success, a run still in progress, a
    malformed body -- returns ``None`` so the caller ignores it. GitHub sends
    many event types to one endpoint, so this filter is doing real work.
    """
    if event != "workflow_run":
        return None
    run = payload.get("workflow_run")
    if not isinstance(run, dict):
        return None
    if run.get("status") != "completed" or run.get("conclusion") not in _FAILED_CONCLUSIONS:
        return None
    repo = (payload.get("repository") or {}).get("full_name") or ""
    run_id = run.get("id")
    if not repo or not isinstance(run_id, int):
        return None
    return WorkflowFailure(
        repo=repo,
        run_id=run_id,
        workflow=run.get("name") or "",
        branch=run.get("head_branch") or "",
        head_sha=(run.get("head_sha") or "")[:40],
        html_url=run.get("html_url") or "",
    )


class DeliveryLog:
    """Remembers recently-seen delivery ids so GitHub's redeliveries are handled
    once. GitHub retries until it gets a 2xx and also lets you replay a delivery
    by hand, so the same ``X-GitHub-Delivery`` id can arrive several times."""

    def __init__(self, capacity: int = 4096):
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._capacity = capacity
        self._lock = threading.Lock()

    def seen_before(self, delivery_id: str) -> bool:
        """Record ``delivery_id`` and report whether it had been seen already."""
        if not delivery_id:
            return False
        with self._lock:
            if delivery_id in self._seen:
                self._seen.move_to_end(delivery_id)
                return True
            self._seen[delivery_id] = None
            while len(self._seen) > self._capacity:
                self._seen.popitem(last=False)
            return False


class AttemptLimiter:
    """Caps automatic actions per key (here: per commit). The brake on an
    auto-triage / auto-fix loop."""

    def __init__(self, max_attempts: int):
        self.max_attempts = max_attempts
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def take(self, key: str) -> bool:
        """Consume one attempt for ``key``; return True if within the cap."""
        with self._lock:
            used = self._counts.get(key, 0)
            if used >= self.max_attempts:
                return False
            self._counts[key] = used + 1
            return True

    def count(self, key: str) -> int:
        with self._lock:
            return self._counts.get(key, 0)


@dataclass
class Diagnosis:
    """One record of what auto-triage did with a failure (for the API / UI)."""

    failure: WorkflowFailure
    status: str  # diagnosed | error | skipped-cap | skipped-disabled
    detail: str = ""
    at: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {
            "repo": self.failure.repo,
            "run_id": self.failure.run_id,
            "workflow": self.failure.workflow,
            "branch": self.failure.branch,
            "commit": self.failure.head_sha[:7],
            "url": self.failure.html_url,
            "status": self.status,
            "detail": self.detail,
            "at": self.at,
        }


class DiagnosisStore:
    """Bounded, newest-first log of auto-triage outcomes. In-memory on purpose:
    it is an activity feed, not a system of record, and the real diagnosis also
    lives in the agent's own conversation thread."""

    def __init__(self, capacity: int = 50):
        self._items: deque[Diagnosis] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def add(self, diagnosis: Diagnosis) -> None:
        with self._lock:
            self._items.appendleft(diagnosis)

    def recent(self, limit: int = 20) -> list[dict]:
        with self._lock:
            return [d.as_dict() for d in list(self._items)[: max(0, limit)]]
