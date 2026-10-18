"""The CI webhook: signature, parsing, de-dup and the per-commit cap (pure),
plus the /webhooks/github endpoint end to end (real FastAPI, fake triage)."""

import hashlib
import hmac
import json

from fakes import ai_text
from fastapi.testclient import TestClient
from test_api import make_builders

from api import create_app
from webhook import (
    AttemptLimiter,
    DeliveryLog,
    WorkflowFailure,
    parse_workflow_failure,
    verify_signature,
)

SECRET = "whsec_test_123"


def sign(body: bytes, secret: str = SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def workflow_event(conclusion="failure", status="completed", run_id=900, sha="abc1234def"):
    return {
        "action": "completed",
        "workflow_run": {
            "id": run_id,
            "name": "CI",
            "status": status,
            "conclusion": conclusion,
            "head_branch": "main",
            "head_sha": sha,
            "html_url": f"https://github.com/o/r/actions/runs/{run_id}",
        },
        "repository": {"full_name": "o/r"},
    }


# --- signature -------------------------------------------------------------------


def test_signature_accepts_a_correct_hmac_and_rejects_everything_else():
    body = b'{"hello":"world"}'
    assert verify_signature(SECRET, body, sign(body))
    assert not verify_signature(SECRET, body, sign(body, "wrong-secret"))
    assert not verify_signature(SECRET, body + b"x", sign(body))  # body tampered
    assert not verify_signature(SECRET, body, None)  # missing header
    assert not verify_signature(SECRET, body, "md5=deadbeef")  # wrong algo prefix
    assert not verify_signature("", body, sign(body))  # no secret configured


# --- parsing ---------------------------------------------------------------------


def test_parse_extracts_a_failed_workflow_run():
    f = parse_workflow_failure("workflow_run", workflow_event())
    assert f == WorkflowFailure(
        "o/r", 900, "CI", "main", "abc1234def", "https://github.com/o/r/actions/runs/900"
    )
    assert f.commit_key == "o/r@abc1234def"


def test_parse_ignores_success_in_progress_other_events_and_junk():
    assert parse_workflow_failure("workflow_run", workflow_event(conclusion="success")) is None
    assert parse_workflow_failure("workflow_run", workflow_event(status="in_progress")) is None
    assert parse_workflow_failure("push", workflow_event()) is None
    assert parse_workflow_failure("workflow_run", {}) is None
    assert parse_workflow_failure("workflow_run", {"workflow_run": "nope"}) is None


def test_parse_treats_timeout_and_startup_failure_as_failures():
    assert parse_workflow_failure("workflow_run", workflow_event(conclusion="timed_out"))
    assert parse_workflow_failure("workflow_run", workflow_event(conclusion="startup_failure"))


# --- de-dup and cap --------------------------------------------------------------


def test_delivery_log_reports_repeats_and_bounds_its_size():
    log = DeliveryLog(capacity=3)
    assert not log.seen_before("d1")
    assert log.seen_before("d1")  # same delivery again
    assert not log.seen_before("")  # empty id is never "seen"
    for d in ("d2", "d3", "d4"):  # evicts d1 (oldest)
        log.seen_before(d)
    assert not log.seen_before("d1")  # aged out, treated as new


def test_attempt_limiter_caps_per_key_independently():
    lim = AttemptLimiter(max_attempts=2)
    assert lim.take("o/r@sha1") and lim.take("o/r@sha1")
    assert not lim.take("o/r@sha1")  # third is over the cap
    assert lim.take("o/r@sha2")  # a different commit is unaffected
    assert lim.count("o/r@sha1") == 2


# --- endpoint --------------------------------------------------------------------


def webhook_client(**kwargs):
    builders, _ = make_builders([ai_text("unused")])
    app = create_app(builders, webhook_secret=SECRET, await_triage=True, **kwargs)
    return TestClient(app)


def post_event(client, payload, *, event="workflow_run", delivery="d-1", secret=SECRET):
    body = json.dumps(payload).encode()
    headers = {
        "X-GitHub-Event": event,
        "X-GitHub-Delivery": delivery,
        "X-Hub-Signature-256": sign(body, secret),
        "Content-Type": "application/json",
    }
    return client.post("/webhooks/github", content=body, headers=headers)


def test_endpoint_is_disabled_without_a_secret():
    builders, _ = make_builders([ai_text("x")])
    with TestClient(create_app(builders)) as c:  # no webhook_secret
        r = c.post("/webhooks/github", content=b"{}", headers={"X-GitHub-Event": "ping"})
    assert r.status_code == 503


def test_a_wrong_signature_is_rejected_and_never_triages():
    seen = []
    with webhook_client(auto_triage=True, triage_runner=_recording_runner(seen)) as c:
        r = post_event(c, workflow_event(), secret="attacker-secret")
    assert r.status_code == 401
    assert seen == []  # the model was never run


def test_ping_is_acknowledged():
    with webhook_client() as c:
        r = post_event(c, {"zen": "hi"}, event="ping")
    assert r.status_code == 200 and r.json()["status"] == "pong"


def _recording_runner(sink):
    async def runner(failure):
        sink.append(failure)
        return f"Root cause for run {failure.run_id}: a failing assertion."

    return runner


def test_a_failed_run_triggers_triage_and_is_recorded():
    seen = []
    with webhook_client(auto_triage=True, triage_runner=_recording_runner(seen)) as c:
        r = post_event(c, workflow_event())
        assert r.status_code == 200 and r.json()["status"] == "accepted"
        assert [f.run_id for f in seen] == [900]  # the model ran once

        feed = c.get("/webhooks/ci").json()["diagnoses"]
    assert feed[0]["status"] == "diagnosed"
    assert feed[0]["repo"] == "o/r" and feed[0]["run_id"] == 900
    assert "failing assertion" in feed[0]["detail"]


def test_a_success_event_is_ignored_and_not_triaged():
    seen = []
    with webhook_client(auto_triage=True, triage_runner=_recording_runner(seen)) as c:
        r = post_event(c, workflow_event(conclusion="success"))
    assert r.status_code == 200 and r.json()["status"] == "ignored"
    assert seen == []


def test_auto_triage_off_records_the_failure_but_does_not_run_the_model():
    seen = []
    with webhook_client(auto_triage=False, triage_runner=_recording_runner(seen)) as c:
        r = post_event(c, workflow_event())
        assert r.json()["status"] == "noted"
        assert seen == []
        assert c.get("/webhooks/ci").json()["diagnoses"][0]["status"] == "skipped-disabled"


def test_redelivery_is_processed_once():
    seen = []
    with webhook_client(auto_triage=True, triage_runner=_recording_runner(seen)) as c:
        first = post_event(c, workflow_event(), delivery="dup-1")
        again = post_event(c, workflow_event(), delivery="dup-1")
    assert first.json()["status"] == "accepted"
    assert again.json()["status"] == "duplicate"
    assert len(seen) == 1  # the retry did NOT trigger a second model run


def test_the_per_commit_cap_stops_a_failure_storm():
    seen = []
    runner = _recording_runner(seen)
    with webhook_client(auto_triage=True, max_attempts_per_commit=2, triage_runner=runner) as c:
        # Same commit, three distinct runs (e.g. re-runs) => three deliveries.
        statuses = [
            post_event(c, workflow_event(run_id=r), delivery=f"d-{r}").json()["status"]
            for r in (901, 902, 903)
        ]
    assert statuses == ["accepted", "accepted", "attempt-limit-reached"]
    assert len(seen) == 2  # triage ran twice, then the brake held


def test_a_triage_error_is_captured_not_crashed():
    async def boom(failure):
        raise RuntimeError("provider down")

    with webhook_client(auto_triage=True, triage_runner=boom) as c:
        r = post_event(c, workflow_event())
        assert r.status_code == 200 and r.json()["status"] == "accepted"
        feed = c.get("/webhooks/ci").json()["diagnoses"]
    assert feed[0]["status"] == "error" and feed[0]["detail"] == "RuntimeError"
