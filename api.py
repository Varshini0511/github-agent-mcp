"""HTTP API for the GitHub Agent (FastAPI + Server-Sent Events).

    uvicorn api:app --port 8000        # then open http://localhost:8000/docs

Endpoints
---------
GET  /health              liveness + effective config (no secrets)
POST /chat                one turn, JSON in / JSON out
POST /chat/stream         one turn as an SSE stream (tokens, tool events, final)
POST /resume              answer a pending approval (approve / reject)
GET  /usage[/{thread}]    token / cost / latency totals
POST /webhooks/github     HMAC-signed CI webhook; a failed run can auto-triage
GET  /webhooks/ci         recent auto-triage activity feed

Security model (this API can act on GitHub with YOUR token -- so it is
locked down by default):
* Set ``API_KEY`` and every request must send ``X-API-Key``.
* With no ``API_KEY`` set, only loopback clients are accepted -- convenient
  for local development, refused for anything remote.
* Input is length- and charset-validated; a per-client sliding-window rate
  limit returns 429; credentials in prompts are redacted before the model.
* Writes still require the ``ALLOW_WRITES`` switch AND an explicit approval
  round-trip (``/resume``) -- an API call alone can never mutate a repo.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from agent_core import build_agent, resume, send, stream_turn
from config import get_settings
from multi_agent import build_supervisor
from observability import TRACKER
from runner import AgentWorker, AsyncRunner
from webhook import (
    AttemptLimiter,
    DeliveryLog,
    Diagnosis,
    DiagnosisStore,
    WorkflowFailure,
    parse_workflow_failure,
    verify_signature,
)

logger = logging.getLogger("github-agent.api")

ID_PATTERN = r"^[A-Za-z0-9_.:\-]{1,64}$"
LOOPBACK = {"127.0.0.1", "::1", "localhost", "testclient"}
# GitHub's workflow_run payloads are a few KB; cap well above that and reject
# anything larger unread rather than buffering it.
WEBHOOK_MAX_BYTES = 2_000_000


# --- request / response models -----------------------------------------------


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    thread_id: str | None = Field(default=None, pattern=ID_PATTERN)
    user_id: str | None = Field(default=None, pattern=ID_PATTERN)
    mode: Literal["single", "multi"] = "single"


class Decision(BaseModel):
    type: Literal["approve", "reject"]
    message: str | None = Field(default=None, max_length=500)


class ResumeRequest(BaseModel):
    thread_id: str = Field(pattern=ID_PATTERN)
    decisions: list[Decision] = Field(min_length=1, max_length=20)
    user_id: str | None = Field(default=None, pattern=ID_PATTERN)
    mode: Literal["single", "multi"] = "single"


# --- rate limiting -------------------------------------------------------------


class SlidingWindowLimiter:
    """Allow ``limit`` events per ``window`` seconds per key (in-memory)."""

    def __init__(self, limit: int, window: float = 60.0):
        self.limit, self.window = limit, window
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            q = self._hits[key]
            while q and q[0] <= now - self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            return True


# --- app factory -----------------------------------------------------------------


def create_app(
    builders: dict[str, Callable] | None = None,
    *,
    api_key: str | None = None,
    rate_limit_per_min: int | None = None,
    webhook_secret: str | None = None,
    auto_triage: bool | None = None,
    max_attempts_per_commit: int | None = None,
    triage_runner: Callable | None = None,
    await_triage: bool = False,
) -> FastAPI:
    """Build the FastAPI app.

    ``builders`` maps mode -> async-context-manager factory yielding an agent;
    it defaults to the real ``build_agent`` / ``build_supervisor`` and exists
    so tests can inject a fake-model agent without touching the network.

    The webhook knobs (``webhook_secret``, ``auto_triage``,
    ``max_attempts_per_commit``) default to the configured settings. A test
    passes ``triage_runner`` (an ``async (WorkflowFailure) -> str``) to stand in
    for a live model run, and ``await_triage=True`` to run that triage inline so
    the result is observable without racing a background task; in production the
    triage runs as a fire-and-forget task so the webhook answers GitHub fast.
    """
    settings = get_settings()
    builders = builders or {"single": build_agent, "multi": build_supervisor}
    key = (
        api_key
        if api_key is not None
        else (settings.api_key.get_secret_value() if settings.api_key else None)
    )
    limiter = SlidingWindowLimiter(rate_limit_per_min or settings.api_rate_limit_per_min)

    wh_secret = (
        webhook_secret
        if webhook_secret is not None
        else (
            settings.github_webhook_secret.get_secret_value()
            if settings.github_webhook_secret
            else None
        )
    )
    auto_triage = settings.ci_auto_triage if auto_triage is None else auto_triage
    cap = (
        settings.ci_auto_triage_max_per_commit
        if max_attempts_per_commit is None
        else max_attempts_per_commit
    )
    deliveries = DeliveryLog()
    attempts = AttemptLimiter(cap)
    diagnoses = DiagnosisStore()

    state: dict = {"runner": None, "workers": {}}
    worker_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        state["runner"] = AsyncRunner()
        try:
            yield
        finally:
            for worker in list(state["workers"].values()):
                await run_in_threadpool(worker.close)
            state["workers"].clear()
            state["runner"].stop()

    app = FastAPI(
        title="GitHub Agent API",
        version="1.0.0",
        description="Tool-using GitHub assistant (MCP + LangGraph) with RAG, "
        "approval-gated writes, guardrails and usage metering.",
        lifespan=lifespan,
    )

    # -- dependencies ---------------------------------------------------------

    async def guard(request: Request) -> str:
        """Authenticate + rate-limit. Returns the client identity string."""
        host = request.client.host if request.client else "unknown"
        if key:
            supplied = request.headers.get("x-api-key", "")
            if not secrets.compare_digest(supplied.encode(), key.encode()):
                raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key.")
            identity = "key:" + key[-4:]
        else:
            if host not in LOOPBACK:
                raise HTTPException(
                    status_code=403,
                    detail="No API_KEY configured: only loopback clients are accepted.",
                )
            identity = "host:" + host
        if not limiter.allow(identity):
            raise HTTPException(status_code=429, detail="Rate limit exceeded; slow down.")
        return identity

    def get_worker(mode: str) -> AgentWorker:
        with worker_lock:
            worker = state["workers"].get(mode)
            if worker is None:
                worker = AgentWorker(state["runner"], builders[mode])
                state["workers"][mode] = worker
        return worker

    async def ready_worker(mode: str) -> AgentWorker:
        worker = get_worker(mode)
        try:
            await run_in_threadpool(worker.wait_ready, 90)
        except Exception as exc:
            logger.exception("agent failed to start")
            raise HTTPException(
                status_code=503, detail=f"Agent unavailable: {type(exc).__name__}"
            ) from exc
        return worker

    def _config(thread_id: str, user_id: str | None) -> dict:
        return {"configurable": {"thread_id": thread_id, "user_id": user_id or "default"}}

    # -- routes -------------------------------------------------------------------

    @app.get("/health")
    async def health():
        s = get_settings()
        return {
            "status": "ok",
            "model": s.chat_model,
            "writes_enabled": s.allow_writes,
            "reflection": s.enable_reflection,
            "hybrid_retrieval": s.rag_hybrid,
            "auth": "api-key" if key else "loopback-only",
            "webhook": "enabled" if wh_secret else "disabled",
            "auto_triage": auto_triage,
        }

    @app.post("/chat")
    async def chat(body: ChatRequest, _who: str = Depends(guard)):
        thread_id = body.thread_id or f"api-{uuid.uuid4().hex[:12]}"
        worker = await ready_worker(body.mode)
        cfg = _config(thread_id, body.user_id)
        try:
            step = await run_in_threadpool(
                worker.call, lambda agent: send(agent, cfg, body.message)
            )
        except Exception as exc:
            logger.exception("chat failed")
            raise HTTPException(
                status_code=502, detail=f"Model call failed: {type(exc).__name__}"
            ) from exc
        return {"thread_id": thread_id, **step}

    @app.post("/resume")
    async def resume_run(body: ResumeRequest, _who: str = Depends(guard)):
        worker = await ready_worker(body.mode)
        cfg = _config(body.thread_id, body.user_id)
        decisions = [d.model_dump(exclude_none=True) for d in body.decisions]
        try:
            step = await run_in_threadpool(worker.call, lambda agent: resume(agent, cfg, decisions))
        except Exception as exc:
            logger.exception("resume failed")
            raise HTTPException(
                status_code=502, detail=f"Resume failed: {type(exc).__name__}"
            ) from exc
        return {"thread_id": body.thread_id, **step}

    @app.post("/chat/stream")
    async def chat_stream(body: ChatRequest, _who: str = Depends(guard)):
        thread_id = body.thread_id or f"api-{uuid.uuid4().hex[:12]}"
        worker = await ready_worker(body.mode)
        cfg = _config(thread_id, body.user_id)

        async def events():
            yield {"event": "thread", "data": json.dumps({"thread_id": thread_id})}
            try:
                sync_iter = worker.stream(lambda agent: stream_turn(agent, cfg, body.message))
                async for ev in iterate_in_threadpool(sync_iter):
                    yield {"event": ev["type"], "data": json.dumps(ev, default=str)}
            except asyncio.CancelledError:  # client went away
                raise
            except Exception as exc:
                logger.exception("stream failed")
                yield {"event": "error", "data": json.dumps({"detail": type(exc).__name__})}

        return EventSourceResponse(events())

    # -- CI webhook: a failed run can trigger the agent by itself -----------

    async def _default_triage(failure: WorkflowFailure) -> str:
        """Ask the DevOps team (ci_triage) to diagnose the failed run.

        Read-only: this reaches get_ci_failure and reports a cause. It never
        reaches a write tool, so nothing is changed on the repo without the
        separate, human-approved code_writer path.
        """
        worker = await ready_worker("multi")
        thread = ("ci-" + failure.repo.replace("/", "-") + "-" + str(failure.run_id))[:64]
        cfg = _config(thread, "ci-webhook")
        prompt = (
            f"The CI workflow '{failure.workflow}' failed for {failure.repo} "
            f"(run {failure.run_id}, branch {failure.branch}). Use get_ci_failure "
            f"to find why it failed, then summarise the root cause and a suggested fix."
        )
        step = await run_in_threadpool(worker.call, lambda agent: send(agent, cfg, prompt))
        return step.get("content", "")

    run_triage = triage_runner or _default_triage

    @app.post("/webhooks/github")
    async def github_webhook(request: Request):
        # No X-API-Key here: GitHub can't send one. The HMAC signature IS the
        # authentication, so the endpoint is disabled until a secret is set.
        if not wh_secret:
            raise HTTPException(
                status_code=503,
                detail="Webhook endpoint is disabled: set GITHUB_WEBHOOK_SECRET.",
            )
        body = await request.body()
        if len(body) > WEBHOOK_MAX_BYTES:
            raise HTTPException(status_code=413, detail="Payload too large.")
        if not verify_signature(wh_secret, body, request.headers.get("x-hub-signature-256")):
            raise HTTPException(status_code=401, detail="Invalid or missing signature.")

        event = request.headers.get("x-github-event", "")
        if event == "ping":  # GitHub sends this once when the hook is created
            return {"status": "pong"}
        if deliveries.seen_before(request.headers.get("x-github-delivery", "")):
            return {"status": "duplicate"}
        try:
            payload = json.loads(body)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Body is not valid JSON.") from exc

        failure = parse_workflow_failure(event, payload if isinstance(payload, dict) else {})
        if failure is None:
            return {"status": "ignored", "event": event}
        if not auto_triage:
            diagnoses.add(Diagnosis(failure, "skipped-disabled", "CI_AUTO_TRIAGE is off."))
            return {"status": "noted", "detail": "auto-triage disabled"}
        if not attempts.take(failure.commit_key):
            diagnoses.add(
                Diagnosis(failure, "skipped-cap", f"Reached {cap} triage attempts for this commit.")
            )
            return {"status": "attempt-limit-reached", "commit": failure.head_sha[:7]}

        async def _job() -> None:
            try:
                text = await run_triage(failure)
                diagnoses.add(Diagnosis(failure, "diagnosed", (text or "")[:4000]))
            except Exception as exc:  # a failed triage must not crash the webhook
                logger.exception("auto-triage failed for %s run %s", failure.repo, failure.run_id)
                diagnoses.add(Diagnosis(failure, "error", type(exc).__name__))

        if await_triage:
            await _job()
        else:
            asyncio.create_task(_job())
        return {"status": "accepted", "repo": failure.repo, "run_id": failure.run_id}

    @app.get("/webhooks/ci")
    async def webhook_diagnoses(_who: str = Depends(guard), limit: int = 20):
        return {"diagnoses": diagnoses.recent(limit)}

    @app.get("/usage")
    async def usage_overall(_who: str = Depends(guard)):
        return TRACKER.overall()

    @app.get("/usage/{thread_id}")
    async def usage_thread(thread_id: str, _who: str = Depends(guard)):
        return TRACKER.for_thread(thread_id)

    return app


# Module-level app for `uvicorn api:app`. Cheap to import: agents are built
# lazily on the first request for each mode.
app = create_app()
