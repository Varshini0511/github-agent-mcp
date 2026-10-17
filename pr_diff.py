"""Turn a pull request's changed files into a diff a model can review.

Two problems with feeding a raw GitHub diff to an LLM:

1. **It can't cite lines.** A unified diff only numbers hunks (``@@ -10,6
   +12,8 @@``); the model must count lines itself and gets it wrong. We rewrite each
   patch so every added/context line carries its true NEW-file line number.
2. **It's too big.** Lock files and generated output can dwarf the real change.
   We drop those, order what's left by how much review it deserves (source,
   then tests, then docs), cap each file, and cap the total -- and say plainly
   what was left out, so the reviewer knows what it did NOT see.

Pure functions; tested in tests/test_pr_diff.py.
"""

from __future__ import annotations

import re

from filetypes import is_noise_file

_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
_TEST_PATH = re.compile(r"(^|/)(tests?|__tests__|spec)(/|_)|\.(test|spec)\.|_test\.", re.IGNORECASE)
_DOC_SUFFIXES = (".md", ".rst", ".txt", ".adoc")


def annotate_patch(patch: str) -> str:
    """Prefix each added / context line with its new-file line number.

    ``+`` lines and unchanged context lines get a number; ``-`` lines (which
    no longer exist in the new file) get blank padding. Hunk headers are kept.
    """
    out: list[str] = []
    new_line: int | None = None
    lines = patch.split("\n")
    if lines and lines[-1] == "":
        lines.pop()  # trailing newline, not a blank context line
    for line in lines:
        hunk = _HUNK.match(line)
        if hunk:
            new_line = int(hunk.group(1))
            out.append(line)
        elif new_line is None:
            out.append(line)
        elif line.startswith("+"):
            out.append(f"{new_line:>5} +{line[1:]}")
            new_line += 1
        elif line.startswith("-"):
            out.append(f"      -{line[1:]}")
        elif line.startswith("\\"):  # "\ No newline at end of file"
            out.append(f"      {line}")
        else:  # context line (leading space)
            out.append(f"{new_line:>5}  {line[1:] if line.startswith(' ') else line}")
            new_line += 1
    return "\n".join(out)


def _priority(filename: str) -> int:
    """Review order: source code (0) before tests (1) before docs/config text (2)."""
    lower = filename.lower()
    if _TEST_PATH.search(lower):
        return 1
    if lower.endswith(_DOC_SUFFIXES) or "/docs/" in lower:
        return 2
    return 0


def _truncate_lines(text: str, limit: int) -> tuple[str, int]:
    """Cut ``text`` at a line boundary under ``limit`` chars; return (text, lines_dropped)."""
    if len(text) <= limit:
        return text, 0
    kept: list[str] = []
    size = 0
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if size + len(line) + 1 > limit and kept:
            return "\n".join(kept), len(lines) - i
        kept.append(line)
        size += len(line) + 1
    return text, 0


def render_pr_diff(
    files: list[dict],
    *,
    max_chars: int = 14000,
    per_file_chars: int = 4000,
    only_path: str = "",
) -> str:
    """Render PR files (GitHub ``/pulls/{n}/files`` objects) for review.

    ``only_path`` returns just that one file with the whole budget -- the
    "zoom in" call after a first pass told the reviewer a file was truncated.
    """
    total_add = sum(f.get("additions", 0) for f in files)
    total_del = sum(f.get("deletions", 0) for f in files)

    if only_path:
        match = [f for f in files if f["filename"] == only_path] or [
            f for f in files if f["filename"].endswith(only_path)
        ]
        if not match:
            names = ", ".join(f["filename"] for f in files[:30])
            return f"File '{only_path}' is not part of this PR. Changed files: {names}"
        files, per_file_chars = match[:1], max_chars

    reviewable, skipped = [], []
    for f in files:
        name = f["filename"]
        if is_noise_file(name):
            skipped.append(f"{name} (lock/generated file)")
        elif not f.get("patch"):
            skipped.append(
                f"{name} (binary, renamed without changes, or too large for GitHub to show)"
            )
        else:
            reviewable.append(f)
    reviewable.sort(key=lambda f: (_priority(f["filename"]), -f.get("changes", 0)))

    sections: list[str] = []
    not_shown: list[str] = []
    used = 0
    for f in reviewable:
        patch = annotate_patch(f["patch"])
        patch, dropped = _truncate_lines(patch, per_file_chars)
        header = f"=== {f['filename']} ({f.get('status', 'modified')}, +{f.get('additions', 0)} -{f.get('deletions', 0)}) ==="
        note = (
            f"\n[... {dropped} more diff lines omitted; call get_pr_diff with path='{f['filename']}' for the rest]"
            if dropped
            else ""
        )
        section = f"{header}\n{patch}{note}"
        if sections and used + len(section) > max_chars:
            not_shown.append(f["filename"])
            continue
        if not sections and len(section) > max_chars:  # always show at least one file
            patch, dropped = _truncate_lines(patch, max_chars - len(header) - 120)
            section = (
                f"{header}\n{patch}\n[... {dropped} more diff lines omitted; "
                f"call get_pr_diff with path='{f['filename']}']"
            )
        sections.append(section)
        used += len(section)

    head = (
        f"PR diff: {len(files)} file(s) changed (+{total_add} -{total_del}); "
        f"showing {len(sections)}.\n"
        "Left column = line number in the NEW version of the file ('+' added, '-' removed, "
        "removed lines have no number). Cite lines using that number."
    )
    parts = [head, *sections]
    if skipped:
        parts.append("Skipped: " + "; ".join(skipped))
    if not_shown:
        parts.append(
            "Not shown (size budget): "
            + ", ".join(not_shown)
            + " -- call get_pr_diff with path=<file> to read one."
        )
    return "\n\n".join(parts)
