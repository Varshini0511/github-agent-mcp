"""LLM-as-judge: score an agent's answer against a rubric.

Deterministic string/tool checks catch structural mistakes, but they
can't tell "correct and well-grounded" from "technically contains the
right word but reasons wrong." A separate model, given the question, the
answer, and the rubric, scores 1-5 and returns a pass/fail verdict.

Using a model to grade a model is standard practice for agent
evaluation; the judge is deliberately given a strict rubric and asked for
structured JSON so its verdict is itself checkable.
"""

from __future__ import annotations

import json
import re

JUDGE_PROMPT = """\
You are a strict evaluator grading an AI agent's answer. Be skeptical:
reward only answers that are correct AND grounded in real data, and
punish any fabrication.

<question>
{question}
</question>

<agent_answer>
{answer}
</agent_answer>

<rubric>
{rubric}
</rubric>

Score the answer:
5 = fully correct and grounded; satisfies the rubric completely
4 = correct with only minor omissions
3 = partially correct
2 = mostly incorrect
1 = wrong, or fabricates facts not supported by real data

Respond with ONLY a JSON object and nothing else:
{{"score": <integer 1-5>, "verdict": "pass" | "fail", "reasoning": "<one short sentence>"}}

Rules for the verdict: "pass" REQUIRES a score of 3 or higher AND no
fabricated facts. Any invented file contents, paths, or results are an
automatic "fail".
"""

_JSON_RE = re.compile(r"\{.*\}", re.S)


async def judge_answer(model, question: str, answer: str, rubric: str) -> dict:
    """Return {'score': int, 'verdict': 'pass'|'fail', 'reasoning': str}."""
    prompt = JUDGE_PROMPT.format(question=question, answer=answer, rubric=rubric)
    resp = await model.ainvoke(prompt)
    text = getattr(resp, "content", None) or str(resp)
    if isinstance(text, list):  # some providers return content parts
        text = " ".join(str(p) for p in text)

    match = _JSON_RE.search(text)
    if not match:
        return {"score": 0, "verdict": "fail", "reasoning": "judge output was not JSON"}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"score": 0, "verdict": "fail", "reasoning": "judge JSON did not parse"}

    # Normalise / defend against a sloppy judge.
    score = int(data.get("score", 0) or 0)
    verdict = str(data.get("verdict", "fail")).lower()
    if verdict not in ("pass", "fail"):
        verdict = "pass" if score >= 3 else "fail"
    return {
        "score": score,
        "verdict": verdict,
        "reasoning": str(data.get("reasoning", "")).strip(),
    }
