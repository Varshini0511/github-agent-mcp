"""Self-reflection: have a critic check an answer against the evidence.

After the agent drafts a final answer, a second model call -- the *critic* --
is shown (1) the question, (2) the draft, and (3) the raw tool outputs the
draft should be grounded in. It answers one question: *does the draft claim
anything the evidence does not support?* If so, the agent gets one chance to
revise.

This targets the failure mode that matters most for a code agent -- stating a
file, line number or behaviour that was never actually retrieved -- and it
complements the offline eval suite: evals catch it in CI, reflection can catch
it at run time.

Design rules
------------
* OPT-IN (ENABLE_REFLECTION): it costs an extra model call per answer.
* FAIL OPEN: if the critic errors or returns junk, the original answer stands.
  A safety net must never become a new way for a request to fail.
* BOUNDED: at most one revision, evidence truncated to a fixed budget.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

MAX_EVIDENCE_CHARS = 6000
MAX_ISSUES = 5

CRITIC_PROMPT = """\
You are a strict fact-checker for an AI coding assistant. Decide whether the
DRAFT ANSWER makes any claim that is NOT supported by the TOOL EVIDENCE.

Flag as problems ONLY:
- file names, paths, line numbers, function or field names that do not
  appear in the evidence;
- statements of behaviour contradicted by, or absent from, the evidence;
- a claim that an action succeeded when the evidence shows an error.

Do NOT flag style, brevity, or reasonable inference from the evidence.

<question>
{question}
</question>

<draft_answer>
{answer}
</draft_answer>

<tool_evidence>
{evidence}
</tool_evidence>

Reply with ONLY a JSON object:
{{"verdict": "ok" | "revise", "issues": ["<short problem>", ...]}}
Use "ok" with an empty list when every claim is supported.
"""

_JSON = re.compile(r"\{[\s\S]*\}")


@dataclass
class Critique:
    verdict: str = "ok"  # "ok" | "revise"
    issues: list[str] = field(default_factory=list)


def _truncate_evidence(evidence: list[str], budget: int = MAX_EVIDENCE_CHARS) -> str:
    """Fit evidence into a fixed budget, favouring the most recent outputs."""
    kept: list[str] = []
    remaining = budget
    for item in reversed(evidence):
        if remaining <= 0:
            break
        chunk = item[:remaining]
        kept.append(chunk + ("...[truncated]" if len(item) > len(chunk) else ""))
        remaining -= len(chunk)
    return "\n---\n".join(reversed(kept)) or "(no tool output)"


def parse_critique(text: str) -> Critique:
    """Parse the critic's reply; anything unparseable means 'ok' (fail open)."""
    match = _JSON.search(text or "")
    if not match:
        return Critique()
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return Critique()
    verdict = str(data.get("verdict", "ok")).lower()
    issues = [str(i).strip() for i in (data.get("issues") or []) if str(i).strip()][:MAX_ISSUES]
    if verdict == "revise" and issues:
        return Critique("revise", issues)
    return Critique()


async def critique(model, question: str, answer: str, evidence: list[str]) -> Critique:
    """Ask the critic model to verify ``answer`` against ``evidence``."""
    prompt = CRITIC_PROMPT.format(
        question=question[:2000],
        answer=answer[:4000],
        evidence=_truncate_evidence(evidence),
    )
    try:
        resp = await model.ainvoke(prompt)
    except Exception:  # fail open -- never let the safety net break the answer
        return Critique()
    content = getattr(resp, "content", resp)
    if isinstance(content, list):
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return parse_critique(str(content))


def revision_prompt(issues: list[str]) -> str:
    bullets = "\n".join(f"- {i}" for i in issues)
    return (
        "A fact-checker reviewed your previous answer against the tool results "
        "and found claims that are not supported:\n"
        f"{bullets}\n\n"
        "Rewrite your answer using ONLY information present in the tool "
        "results. Remove or correct the unsupported claims; if something could "
        "not be verified, say so plainly. If you need more evidence, call a "
        "tool first."
    )
