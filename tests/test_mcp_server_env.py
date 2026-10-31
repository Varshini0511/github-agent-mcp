"""The MCP server subprocess must see the same settings as the agent process.

Regression: the stdio client used to start server.py with only a minimal
allow-list of environment variables. Settings passed through the environment
(``docker run --env-file``, CI secrets, a per-run CHAT_MODEL / RAG_HYBRID)
never reached the tools; inside the Docker image -- which excludes .env --
the server could not even find GITHUB_TOKEN and offered no tools at all.
"""

import sys

from agent_core import SERVER_PATH, WRITE_TOOLS, mcp_server, server_transport


def test_transport_forwards_the_full_environment(monkeypatch):
    monkeypatch.setenv("AGENT_ENV_PROBE", "forwarded")
    t = server_transport()
    assert t.env["AGENT_ENV_PROBE"] == "forwarded"
    assert str(t.script_path) == str(SERVER_PATH)
    assert t.command == sys.executable


async def test_server_starts_from_environment_only_settings(monkeypatch):
    # In CI (and in Docker) these exist ONLY as environment variables -- there
    # is no .env next to server.py -- so this fails without the forwarding.
    monkeypatch.setenv("GITHUB_TOKEN", "test-token-from-env")
    monkeypatch.setenv("GROQ_API_KEY", "test-key-from-env")
    # GOOGLE_API_KEY is optional (Groq-only setups); the server must still
    # start -- only the RAG tools need it, and only when called.
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    async with mcp_server() as mcp:
        names = {t.name for t in await mcp.list_tools()}
    assert {"list_repo_files", "read_file", "search_code"} <= names
    assert set(WRITE_TOOLS) <= names


async def test_server_starts_without_the_optional_rag_extra(monkeypatch, tmp_path):
    # Shadow the RAG stack with a package that refuses to import, as if
    # `pip install -e ".[rag]"` had never been run.
    blocker = tmp_path / "langchain_community"
    blocker.mkdir()
    (blocker / "__init__.py").write_text("raise ImportError('rag extra not installed')\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("GITHUB_TOKEN", "test-token-from-env")
    monkeypatch.setenv("GROQ_API_KEY", "test-key-from-env")

    async with mcp_server() as mcp:
        tools = {t.name: t for t in await mcp.list_tools()}
        assert "read_file" in tools
        reply = await tools["search_code"].ainvoke({"repo": "o/r", "query": "x"})
    assert "pip install" in str(reply) and "[rag]" in str(reply)
