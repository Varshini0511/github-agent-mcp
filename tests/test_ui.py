"""Headless UI tests: drive the real streamlit_app.py with Streamlit's AppTest.

The agent builders are swapped for a fake-model agent, so the script runs its
true code path (worker thread, streaming, approval card, usage panel) with no
network and no API keys.
"""

from contextlib import asynccontextmanager
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from fakes import ScriptedChatModel, ai_text, ai_tool, tool_call  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from agent_core import assemble_agent  # noqa: E402
from config import get_settings  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


def install_fake_agent(monkeypatch, script, ran=None):
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

    monkeypatch.setattr("agent_core.build_agent", builder)
    monkeypatch.setattr("multi_agent.build_supervisor", builder)

    import streamlit as st

    st.cache_resource.clear()  # get_worker is cached process-wide
    return ran


@pytest.fixture(autouse=True)
def _reap_workers():
    yield
    from runner import AgentWorker

    AgentWorker.close_all()


def new_app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=60)


def test_app_renders_without_exceptions(monkeypatch):
    install_fake_agent(monkeypatch, [])
    at = new_app().run()
    assert not at.exception, at.exception
    # sidebar shows the mode switch and the usage meter
    assert any("Model calls" in m.label for m in at.metric)


def test_a_message_flows_through_and_updates_the_usage_meter(monkeypatch):
    install_fake_agent(
        monkeypatch,
        [
            ai_tool([tool_call("list_repo_files", {"repo": "o/r"}, "c1")]),
            ai_text("The repo contains a README."),
        ],
    )
    at = new_app().run()
    at.chat_input[0].set_value("what files are in o/r?").run()

    assert not at.exception, at.exception
    rendered = " ".join(m.value for m in at.markdown)
    assert "The repo contains a README." in rendered
    calls = next(m for m in at.metric if m.label == "Model calls")
    assert int(calls.value) == 2


def test_write_request_shows_an_approval_card_and_blocks_input(monkeypatch):
    ran = install_fake_agent(
        monkeypatch,
        [
            ai_tool([tool_call("create_branch", {"repo": "o/r", "new_branch": "feat"}, "c1")]),
            ai_text("Branch created."),
        ],
    )
    at = new_app().run()
    at.chat_input[0].set_value("create branch feat in o/r").run()

    assert not at.exception, at.exception
    labels = [b.label for b in at.button]
    assert any("Approve" in label for label in labels)
    assert any("Reject" in label for label in labels)
    assert ran == []  # nothing executed before approval

    approve = next(b for b in at.button if "Approve" in b.label)
    approve.click().run()
    assert not at.exception, at.exception
    assert ran == ["feat"]
