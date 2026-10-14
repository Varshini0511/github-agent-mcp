"""HTTP API tests: real FastAPI app + real agent graph + scripted fake model."""

import json
from contextlib import asynccontextmanager

import pytest
from fakes import ScriptedChatModel, ai_text, ai_tool, tool_call
from fastapi.testclient import TestClient
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from agent_core import assemble_agent
from api import SlidingWindowLimiter, create_app
from config import get_settings


def make_builders(script, ran=None):
    ran = ran if ran is not None else []

    @tool
    def list_repo_files(repo: str, path: str = "") -> str:
        """List files."""
        return "file: README"

    @tool
    def create_branch(repo: str, new_branch: str) -> str:
        """Create a branch."""
        ran.append(new_branch)
        return "created"

    @asynccontextmanager
    async def builder():
        model = ScriptedChatModel(script=list(script))
        yield assemble_agent(
            model, [list_repo_files, create_branch], InMemorySaver(), settings=get_settings()
        )

    return {"single": builder, "multi": builder}, ran


def client_for(script, **kwargs):
    builders, ran = make_builders(script)
    app = create_app(builders, **kwargs)
    return TestClient(app), ran


def test_health_reports_config_without_secrets():
    c, _ = client_for([])
    with c:
        r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and "model" in body
    assert "token" not in json.dumps(body).lower() or body["auth"] in ("loopback-only", "api-key")


def test_chat_returns_answer_thread_id_and_usage():
    c, _ = client_for(
        [
            ai_tool([tool_call("list_repo_files", {"repo": "o/r"}, "c1")]),
            ai_text("It has a README."),
        ]
    )
    with c:
        r = c.post("/chat", json={"message": "files in o/r?"})
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "final" and body["content"] == "It has a README."
    assert body["thread_id"].startswith("api-")
    assert body["usage"]["model_calls"] == 2 and body["usage"]["tool_calls"] == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"message": ""},
        {"message": "x" * 8001},
        {"message": "hi", "thread_id": "../../etc/passwd"},
        {"message": "hi", "mode": "nope"},
    ],
)
def test_invalid_requests_are_rejected_with_422(payload):
    c, _ = client_for([ai_text("never reached")])
    with c:
        assert c.post("/chat", json=payload).status_code == 422


def test_api_key_is_enforced_when_configured():
    c, _ = client_for([ai_text("hi")], api_key="s3cret-key-1234")
    with c:
        assert c.post("/chat", json={"message": "hi"}).status_code == 401
        bad = c.post("/chat", json={"message": "hi"}, headers={"X-API-Key": "wrong"})
        assert bad.status_code == 401
        ok = c.post("/chat", json={"message": "hi"}, headers={"X-API-Key": "s3cret-key-1234"})
        assert ok.status_code == 200


def test_remote_clients_are_refused_when_no_api_key_is_set():
    builders, _ = make_builders([ai_text("hi")])
    app = create_app(builders)
    with TestClient(app, client=("203.0.113.9", 50000)) as c:
        r = c.post("/chat", json={"message": "hi"})
    assert r.status_code == 403


def test_rate_limit_returns_429():
    c, _ = client_for([ai_text("a"), ai_text("b"), ai_text("c")], rate_limit_per_min=2)
    with c:
        codes = [c.post("/chat", json={"message": "hi"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_write_requires_an_explicit_resume_round_trip():
    builders, ran = make_builders(
        [
            ai_tool([tool_call("create_branch", {"repo": "o/r", "new_branch": "feat"}, "c1")]),
            ai_text("Branch created."),
        ]
    )
    with TestClient(create_app(builders)) as c:
        first = c.post("/chat", json={"message": "make branch feat", "thread_id": "t-api-1"})
        assert first.json()["type"] == "interrupt"
        assert first.json()["action_requests"][0]["name"] == "create_branch"
        assert ran == []  # an API call alone did NOT change anything

        done = c.post("/resume", json={"thread_id": "t-api-1", "decisions": [{"type": "approve"}]})
    assert done.status_code == 200 and done.json()["type"] == "final"
    assert ran == ["feat"]


def test_rejecting_via_the_api_prevents_the_write():
    builders, ran = make_builders(
        [
            ai_tool([tool_call("create_branch", {"repo": "o/r", "new_branch": "feat"}, "c1")]),
            ai_text("Okay, not doing that."),
        ]
    )
    with TestClient(create_app(builders)) as c:
        c.post("/chat", json={"message": "make branch", "thread_id": "t-api-2"})
        done = c.post("/resume", json={"thread_id": "t-api-2", "decisions": [{"type": "reject"}]})
    assert done.json()["type"] == "final" and ran == []


def test_sse_stream_emits_tool_token_and_final_events():
    c, _ = client_for(
        [
            ai_tool([tool_call("list_repo_files", {"repo": "o/r"}, "c1")]),
            ai_text("It has a README."),
        ]
    )
    events: list[tuple[str, dict]] = []
    with c, c.stream("POST", "/chat/stream", json={"message": "files?"}) as r:
        assert r.status_code == 200
        current = None
        for line in r.iter_lines():
            if line.startswith("event:"):
                current = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and current:
                events.append((current, json.loads(line.split(":", 1)[1])))
    names = [n for n, _ in events]
    assert names[0] == "thread"
    assert "tool_start" in names and "token" in names and names[-1] == "final"
    assert events[-1][1]["content"] == "It has a README."


def test_usage_endpoint_reflects_traffic():
    c, _ = client_for([ai_text("hello")])
    with c:
        r = c.post("/chat", json={"message": "hi", "thread_id": "t-usage-1"})
        assert r.status_code == 200
        u = c.get("/usage/t-usage-1").json()
    assert u["turns"] == 1 and u["model_calls"] == 1


def test_sliding_window_limiter_expires_old_hits():
    lim = SlidingWindowLimiter(limit=2, window=10)
    assert lim.allow("k", now=0) and lim.allow("k", now=1)
    assert not lim.allow("k", now=2)
    assert lim.allow("k", now=11.5)  # first hit aged out of the window
    assert lim.allow("other", now=2)  # independent key
