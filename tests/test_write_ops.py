"""Offline tests for the write half of GitHubClient (Block C).

Still no network: `responses` fakes each GitHub endpoint. These prove the
client builds the right requests and returns a useful confirmation
string; the human-approval gate is exercised separately in agent_core.
"""

import base64
import json

import pytest
import responses

from github_client import GitHubClient, GitHubError

API = "https://api.github.com"


@pytest.fixture
def gh():
    return GitHubClient(backoff_initial=0, backoff_max=0)


@responses.activate
def test_create_issue(gh):
    responses.post(
        f"{API}/repos/o/r/issues",
        json={"number": 7, "html_url": "https://github.com/o/r/issues/7"},
        status=201,
    )
    out = gh.create_issue("o/r", "Bug: crash on start", "steps...")
    assert "issue #7" in out and "issues/7" in out
    assert json.loads(responses.calls[0].request.body) == {
        "title": "Bug: crash on start",
        "body": "steps...",
    }


@responses.activate
def test_comment_on_issue(gh):
    responses.post(
        f"{API}/repos/o/r/issues/7/comments",
        json={"html_url": "https://github.com/o/r/issues/7#issuecomment-1"},
        status=201,
    )
    assert "Commented on #7" in gh.comment_on_issue("o/r", 7, "thanks")


@responses.activate
def test_add_labels(gh):
    responses.post(f"{API}/repos/o/r/issues/7/labels", json=[{"name": "bug"}], status=200)
    assert "bug" in gh.add_labels("o/r", 7, ["bug", "P1"])
    assert json.loads(responses.calls[0].request.body) == {"labels": ["bug", "P1"]}


@responses.activate
def test_create_branch_reads_base_sha_then_creates_ref(gh):
    responses.get(f"{API}/repos/o/r", json={"default_branch": "main"}, status=200)
    responses.get(
        f"{API}/repos/o/r/git/ref/heads/main",
        json={"object": {"sha": "abc1234def"}},
        status=200,
    )
    responses.post(f"{API}/repos/o/r/git/refs", json={}, status=201)

    out = gh.create_branch("o/r", "fix/typo")
    assert "fix/typo" in out and "abc1234" in out
    assert json.loads(responses.calls[2].request.body) == {
        "ref": "refs/heads/fix/typo",
        "sha": "abc1234def",
    }


@responses.activate
def test_put_file_creates_when_absent(gh):
    responses.get(f"{API}/repos/o/r/contents/docs/x.md", json={"message": "Not Found"}, status=404)
    responses.put(
        f"{API}/repos/o/r/contents/docs/x.md",
        json={"commit": {"html_url": "https://github.com/o/r/commit/deadbeef"}},
        status=201,
    )
    out = gh.put_file("o/r", "docs/x.md", "# Hi", "add docs", "fix/docs")
    assert "docs/x.md" in out
    sent = json.loads(responses.calls[1].request.body)
    assert base64.b64decode(sent["content"]).decode() == "# Hi"
    assert sent["branch"] == "fix/docs"
    assert "sha" not in sent  # brand-new file


@responses.activate
def test_put_file_updates_when_present(gh):
    responses.get(
        f"{API}/repos/o/r/contents/README.md",
        json={"sha": "oldsha123"},
        status=200,
    )
    responses.put(
        f"{API}/repos/o/r/contents/README.md",
        json={"commit": {"html_url": "https://github.com/o/r/commit/c0ffee"}},
        status=200,
    )
    gh.put_file("o/r", "README.md", "new", "update", "main")
    assert json.loads(responses.calls[1].request.body)["sha"] == "oldsha123"


@responses.activate
def test_open_pull_request(gh):
    responses.get(f"{API}/repos/o/r", json={"default_branch": "main"}, status=200)
    responses.post(
        f"{API}/repos/o/r/pulls",
        json={"number": 42, "html_url": "https://github.com/o/r/pull/42"},
        status=201,
    )
    out = gh.open_pull_request("o/r", "Fix typo", head="fix/typo")
    assert "PR #42" in out
    assert json.loads(responses.calls[1].request.body)["base"] == "main"


@responses.activate
def test_read_lines_returns_slice(gh):
    text = "\n".join(f"line{i}" for i in range(1, 21))
    responses.get(
        f"{API}/repos/o/r/contents/f.txt",
        json={"encoding": "base64", "content": base64.b64encode(text.encode()).decode()},
        status=200,
    )
    out = gh.read_lines("o/r", "f.txt", 5, 8)
    assert "line5" in out and "line8" in out
    assert "line9" not in out and "line4" not in out
    assert "lines 5-8 of 20" in out  # range header


@responses.activate
def test_edit_file_replaces_unique_snippet(gh):
    content = "hello\nworld\nfoo\n"
    responses.get(
        f"{API}/repos/o/r/contents/f.txt",
        json={"encoding": "base64", "content": base64.b64encode(content.encode()).decode(), "sha": "abc"},
        status=200,
    )
    responses.put(
        f"{API}/repos/o/r/contents/f.txt",
        json={"commit": {"html_url": "https://github.com/o/r/commit/x"}},
        status=200,
    )
    gh.edit_file("o/r", "f.txt", "world", "WORLD", "msg", "br")
    sent = json.loads(responses.calls[1].request.body)
    assert base64.b64decode(sent["content"]).decode() == "hello\nWORLD\nfoo\n"
    assert sent["sha"] == "abc" and sent["branch"] == "br"


@responses.activate
def test_edit_file_errors_when_snippet_absent(gh):
    responses.get(
        f"{API}/repos/o/r/contents/f.txt",
        json={"encoding": "base64", "content": base64.b64encode(b"abc").decode(), "sha": "s"},
        status=200,
    )
    with pytest.raises(GitHubError, match="was not found"):
        gh.edit_file("o/r", "f.txt", "ZZZ", "x", "m", "br")


@responses.activate
def test_edit_file_errors_when_snippet_ambiguous(gh):
    responses.get(
        f"{API}/repos/o/r/contents/f.txt",
        json={"encoding": "base64", "content": base64.b64encode(b"x x x").decode(), "sha": "s"},
        status=200,
    )
    with pytest.raises(GitHubError, match="appears 3 times"):
        gh.edit_file("o/r", "f.txt", "x", "y", "m", "br")
