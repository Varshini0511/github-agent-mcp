# server.py -- an MCP server exposing GitHub operations as tools.
#
# Each @mcp.tool function is deliberately thin: it validates nothing and
# does no HTTP itself. It calls github_client (which handles auth,
# retries and error mapping) and converts a GitHubError into a short
# string, because that string is what the LLM reads and reasons about.

import logging

from fastmcp import FastMCP

from config import configure_logging, get_settings
from github_client import GitHubClient, GitHubError
from rag import RepoIndex

configure_logging()  # logs go to STDERR; STDOUT is the MCP protocol channel
logger = logging.getLogger("github-agent.server")

settings = get_settings()  # fails fast if required secrets are missing
gh = GitHubClient()
index = RepoIndex()

mcp = FastMCP("github-agent")


@mcp.tool
def list_repo_files(repo: str, path: str = "") -> str:
    """List files and folders in a GitHub repository.

    repo: the repository in 'owner/name' format, e.g. 'octocat/Hello-World'.
    path: optional subfolder path; empty means the repository root.
    """
    logger.info("list_repo_files repo=%s path=%r", repo, path)
    try:
        items = gh.list_contents(repo, path)
    except GitHubError as exc:
        return f"Error: {exc}"

    if not items:
        return "(empty directory)"

    # Directories first, then files, each alphabetical -- a stable,
    # readable ordering for the model.
    items.sort(key=lambda it: (it.get("type") != "dir", it.get("name", "")))
    return "\n".join(f"{it['type']}: {it['name']}" for it in items)


@mcp.tool
def read_file(repo: str, path: str) -> str:
    """Read the contents of a specific text file in a GitHub repository.

    repo: the repository in 'owner/name' format.
    path: the full path to the file, e.g. 'src/main.py' or 'README.md'.
    """
    logger.info("read_file repo=%s path=%s", repo, path)
    try:
        return gh.read_text_file(repo, path)
    except GitHubError as exc:
        return f"Error: {exc}"


@mcp.tool
def read_file_lines(repo: str, path: str, start_line: int, end_line: int) -> str:
    """Read only lines start_line..end_line of a file (1-based, inclusive).

    Use this instead of read_file for LARGE files, so you load only the
    section you need. The text comes back verbatim (no line-number
    prefixes) so you can copy a chunk of it straight into edit_file.

    repo: 'owner/name'. path: full file path.
    """
    logger.info("read_file_lines repo=%s path=%s %s-%s", repo, path, start_line, end_line)
    try:
        return gh.read_lines(repo, path, start_line, end_line)
    except GitHubError as exc:
        return f"Error: {exc}"


@mcp.tool
def list_issues(repo: str, state: str = "open") -> str:
    """List issues in a repository. state is 'open', 'closed' or 'all'."""
    logger.info("list_issues repo=%s state=%s", repo, state)
    try:
        issues = gh.list_issues(repo, state)
    except GitHubError as exc:
        return f"Error: {exc}"
    if not issues:
        return f"No {state} issues."
    return "\n".join(
        f"#{i['number']} [{', '.join(l['name'] for l in i.get('labels', [])) or 'no labels'}] "
        f"{i['title']}"
        for i in issues
    )


@mcp.tool
def get_issue(repo: str, number: int) -> str:
    """Get the title, state, labels and full body of one issue."""
    logger.info("get_issue repo=%s number=%s", repo, number)
    try:
        i = gh.get_issue(repo, number)
    except GitHubError as exc:
        return f"Error: {exc}"
    labels = ", ".join(l["name"] for l in i.get("labels", [])) or "none"
    return (
        f"#{i['number']} {i['title']}\n"
        f"state: {i['state']}   labels: {labels}\n\n"
        f"{i.get('body') or '(no description)'}"
    )


@mcp.tool
def index_repo(repo: str, ref: str = "") -> str:
    """Build a semantic search index over an entire repository.

    Run this once before using search_code on a repo. Downloads the repo,
    splits its text files into chunks and embeds them. May take 10-60s
    for a medium repo.

    repo: the repository in 'owner/name' format.
    ref:  optional branch, tag or commit SHA; empty means the default branch.
    """
    logger.info("index_repo repo=%s ref=%r", repo, ref)
    try:
        return index.index(repo, ref)
    except GitHubError as exc:
        return f"Error: {exc}"
    except Exception as exc:  # embedding / tar / disk problems
        logger.exception("index_repo failed")
        return f"Error: indexing failed ({exc})."


@mcp.tool
def search_code(repo: str, query: str) -> str:
    """Find the code most relevant to a natural-language query.

    Returns the top matching chunks with their file path and starting
    line number, best match first. Requires index_repo to have been run
    for this repo.

    repo:  the repository in 'owner/name' format.
    query: what you're looking for, in plain English
           (e.g. "where are JWT tokens verified").
    """
    logger.info("search_code repo=%s query=%r", repo, query)
    try:
        hits = index.search(repo, query, k=6)
    except FileNotFoundError as exc:
        return f"Error: {exc}"
    except Exception as exc:
        logger.exception("search_code failed")
        return f"Error: search failed ({exc})."

    if not hits:
        return "No matches."

    blocks = [
        f"--- {h.path}:{h.start_line} ---\n{h.text}".rstrip() for h in hits
    ]
    return "\n\n".join(blocks)


# --- Write tools (Block C) -------------------------------------------
# Two layers of protection sit in front of these:
#   1. settings.allow_writes -- a hard off switch (default off).
#   2. HumanInTheLoopMiddleware in agent_core -- the agent must get
#      explicit human approval before any of these actually run.

_WRITE_DISABLED = (
    "Error: write operations are disabled. Set ALLOW_WRITES=true in .env "
    "and use a GITHUB_TOKEN that has write access to this repository."
)


def _guard() -> str | None:
    return None if settings.allow_writes else _WRITE_DISABLED


@mcp.tool
def create_issue(repo: str, title: str, body: str = "") -> str:
    """Open a new issue on a repository. repo is 'owner/name'."""
    logger.info("create_issue repo=%s title=%r", repo, title)
    if msg := _guard():
        return msg
    try:
        return gh.create_issue(repo, title, body)
    except GitHubError as exc:
        return f"Error: {exc}"


@mcp.tool
def comment_on_issue(repo: str, number: int, body: str) -> str:
    """Add a comment to an existing issue or pull request by its number."""
    logger.info("comment_on_issue repo=%s number=%s", repo, number)
    if msg := _guard():
        return msg
    try:
        return gh.comment_on_issue(repo, number, body)
    except GitHubError as exc:
        return f"Error: {exc}"


@mcp.tool
def add_labels(repo: str, number: int, labels: list[str]) -> str:
    """Add one or more labels to an issue or pull request by its number."""
    logger.info("add_labels repo=%s number=%s labels=%s", repo, number, labels)
    if msg := _guard():
        return msg
    try:
        return gh.add_labels(repo, number, labels)
    except GitHubError as exc:
        return f"Error: {exc}"


@mcp.tool
def create_branch(repo: str, new_branch: str, from_branch: str = "") -> str:
    """Create a new branch. from_branch defaults to the repo's default branch."""
    logger.info("create_branch repo=%s new_branch=%s", repo, new_branch)
    if msg := _guard():
        return msg
    try:
        return gh.create_branch(repo, new_branch, from_branch)
    except GitHubError as exc:
        return f"Error: {exc}"


@mcp.tool
def write_file(repo: str, path: str, content: str, message: str, branch: str) -> str:
    """Create or overwrite a file on a branch with a commit.

    Never commit to the default branch directly -- create_branch first,
    write here, then open_pull_request.
    """
    logger.info("write_file repo=%s path=%s branch=%s", repo, path, branch)
    if msg := _guard():
        return msg
    try:
        return gh.put_file(repo, path, content, message, branch)
    except GitHubError as exc:
        return f"Error: {exc}"


@mcp.tool
def edit_file(
    repo: str, path: str, old_string: str, new_string: str, message: str, branch: str
) -> str:
    """Make a surgical edit to a file: replace old_string with new_string
    and commit. Prefer this over write_file for anything but a brand-new
    or tiny file -- you send only the snippet that changes, not the whole
    file, which keeps the request small.

    old_string must appear EXACTLY ONCE in the file; include enough
    surrounding lines (with exact indentation) to make it unique. Read the
    exact text first with read_file_lines. Commit to a feature branch, not
    the default branch.
    """
    logger.info("edit_file repo=%s path=%s branch=%s", repo, path, branch)
    if msg := _guard():
        return msg
    try:
        return gh.edit_file(repo, path, old_string, new_string, message, branch)
    except GitHubError as exc:
        return f"Error: {exc}"


@mcp.tool
def open_pull_request(
    repo: str, title: str, head: str, base: str = "", body: str = ""
) -> str:
    """Open a pull request from branch `head` into `base` (default branch if empty)."""
    logger.info("open_pull_request repo=%s head=%s", repo, head)
    if msg := _guard():
        return msg
    try:
        return gh.open_pull_request(repo, title, head, base, body)
    except GitHubError as exc:
        return f"Error: {exc}"


if __name__ == "__main__":
    mcp.run()
