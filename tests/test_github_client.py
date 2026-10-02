"""Offline tests for GitHubClient.

`responses` intercepts every call the `requests` library makes, so these
tests never touch the network and never need a real token. Each test
registers the fake HTTP responses it expects, then asserts on the
client's behaviour.
"""

import base64

import pytest
import responses

from github_client import GitHubClient, GitHubError

API = "https://api.github.com"


@pytest.fixture
def gh():
    # backoff_* = 0 so retry tests don't actually sleep.
    return GitHubClient(backoff_initial=0, backoff_max=0, max_retries=3)


@responses.activate
def test_list_contents_returns_items(gh):
    responses.get(
        f"{API}/repos/o/r/contents",
        json=[
            {"type": "file", "name": "README.md"},
            {"type": "dir", "name": "src"},
        ],
        status=200,
    )
    items = gh.list_contents("o/r")
    assert {i["name"] for i in items} == {"README.md", "src"}


@responses.activate
def test_single_file_response_is_normalised_to_a_list(gh):
    responses.get(
        f"{API}/repos/o/r/contents/setup.py",
        json={"type": "file", "name": "setup.py"},  # object, not array
        status=200,
    )
    items = gh.list_contents("o/r", "setup.py")
    assert isinstance(items, list) and items[0]["name"] == "setup.py"


@responses.activate
def test_read_text_file_decodes_base64(gh):
    encoded = base64.b64encode(b"hello world").decode()
    responses.get(
        f"{API}/repos/o/r/contents/a.txt",
        json={"encoding": "base64", "content": encoded},
        status=200,
    )
    assert gh.read_text_file("o/r", "a.txt") == "hello world"


@responses.activate
def test_reading_a_directory_raises_a_helpful_error(gh):
    responses.get(
        f"{API}/repos/o/r/contents/src",
        json=[{"type": "file", "name": "main.py"}],  # array => it's a dir
        status=200,
    )
    with pytest.raises(GitHubError, match="directory, not a file"):
        gh.read_text_file("o/r", "src")


@responses.activate
def test_404_maps_to_friendly_message(gh):
    responses.get(f"{API}/repos/o/missing/contents", json={"message": "Not Found"}, status=404)
    with pytest.raises(GitHubError, match=r"Not found \(404\)"):
        gh.list_contents("o/missing")


@responses.activate
def test_401_maps_to_auth_message(gh):
    responses.get(f"{API}/repos/o/r/contents", json={"message": "Bad creds"}, status=401)
    with pytest.raises(GitHubError, match=r"Authentication failed \(401\)"):
        gh.list_contents("o/r")


@responses.activate
def test_transient_503_is_retried_then_succeeds(gh):
    # First call 503 (retryable), second call 200.
    responses.get(f"{API}/repos/o/r/contents", json={"message": "unavailable"}, status=503)
    responses.get(f"{API}/repos/o/r/contents", json=[{"type": "file", "name": "ok"}], status=200)

    items = gh.list_contents("o/r")
    assert items[0]["name"] == "ok"
    assert len(responses.calls) == 2  # proves it retried exactly once


@responses.activate
def test_gives_up_after_max_retries(gh):
    for _ in range(3):
        responses.get(f"{API}/repos/o/r/contents", json={"message": "boom"}, status=502)
    with pytest.raises(GitHubError, match="kept failing after 3 attempts"):
        gh.list_contents("o/r")
    assert len(responses.calls) == 3


@responses.activate
def test_primary_rate_limit_waits_then_retries(gh, monkeypatch):
    slept = []
    monkeypatch.setattr("github_client.time.sleep", lambda s: slept.append(s))

    responses.get(
        f"{API}/repos/o/r/contents",
        json={"message": "rate limited"},
        status=403,
        headers={"X-RateLimit-Remaining": "0", "Retry-After": "1"},
    )
    responses.get(f"{API}/repos/o/r/contents", json=[{"type": "file", "name": "ok"}], status=200)

    items = gh.list_contents("o/r")
    assert items[0]["name"] == "ok"
    # 1s came from the Retry-After header; any extra 0.0 is tenacity's
    # (zeroed) backoff nap, which shares the same time.sleep.
    assert 1 in slept
