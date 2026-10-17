"""Which files are noise for code search and code review.

Shared by the repository indexer (``rag.py``) and the PR-review tooling
(``pr_diff.py``) so both skip the same junk: dependency lock files and
generated / minified output. They are text, but carry no signal -- a single
``package-lock.json`` can add thousands of lines of hashes to a diff or
hundreds of useless chunks to an index.

Pure Python, no third-party imports, so it is cheap to import anywhere.
"""

from __future__ import annotations

SKIP_NAMES = frozenset(
    {
        "package-lock.json",
        "npm-shrinkwrap.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "Pipfile.lock",
        "uv.lock",
        "Cargo.lock",
        "composer.lock",
        "Gemfile.lock",
        "go.sum",
    }
)
SKIP_SUFFIXES = (".min.js", ".min.css", ".map", ".lock")


def is_noise_file(path: str) -> bool:
    """True for lock files and generated/minified output (path or bare name)."""
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    return name in SKIP_NAMES or name.lower().endswith(SKIP_SUFFIXES)
