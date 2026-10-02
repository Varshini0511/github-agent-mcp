"""Block E -- a scripted tour of what the agent can do.

Runs each scenario end to end and prints the result. Designed to be
friendly to Groq's free tier: scenarios are independent, a failure in one
does not stop the rest, and there is a pause between them.

    python demo.py            # run all scenarios
    python demo.py 3          # run only scenario 3
    python demo.py --list     # show the scenario list
"""

import asyncio
import sys
import time
import uuid

from agent_core import build_agent, run_turn
from multi_agent import build_supervisor

# A small public repo with real (if tiny) Python code, used throughout.
REPO = "Varshini0511/multi-agent-travel-planner"

SCENARIOS = [
    (
        "Repo explainer (single agent)",
        "single",
        f"Give me a 3-sentence overview of what {REPO} does and its main "
        f"directories.",
    ),
    (
        "Code locator via RAG (single agent)",
        "single",
        f"In {REPO}, where is the language model configured and where is "
        f"conversation memory / checkpointing set up? Index it if needed and "
        f"cite path:line.",
    ),
    (
        "Read a specific file (single agent)",
        "single",
        f"Show the first 15 lines of README.md in {REPO} and summarise the "
        f"setup steps.",
    ),
    (
        "Issue triage (supervisor -> issue_triage)",
        "multi",
        f"List the open issues in {REPO}. For each, suggest a type label and "
        f"a priority. Do not change anything.",
    ),
    (
        "Write flow with approval gate (single agent)",
        "single",
        f"Fix a typo in the README of {REPO} by opening a pull request from a "
        f"new branch.",
    ),
]


async def _decline(name, args):
    print(f"    [approval requested for {name} -> auto-declined in demo]")
    return False


async def _run_single(question):
    async with build_agent() as agent:
        cfg = {"configurable": {"thread_id": f"demo-{uuid.uuid4().hex[:8]}"}}
        return await run_turn(agent, cfg, question, _decline)


async def _run_multi(question):
    async with build_supervisor() as sup:
        cfg = {"configurable": {"thread_id": f"demo-{uuid.uuid4().hex[:8]}"}}
        return await run_turn(sup, cfg, question, _decline)


async def _run_one(idx: int) -> None:
    title, mode, question = SCENARIOS[idx]
    print("\n" + "=" * 72)
    print(f"[{idx + 1}] {title}")
    print("-" * 72)
    print(f"Q: {question}\n")
    try:
        answer = await (_run_multi if mode == "multi" else _run_single)(question)
        print(f"A: {answer}")
    except Exception as exc:  # keep the tour going
        print(f"!! scenario failed: {type(exc).__name__}: {exc}")


async def main(which: list[str]) -> None:
    if which == ["--list"]:
        for i, (title, mode, _) in enumerate(SCENARIOS, 1):
            print(f"{i}. [{mode:6}] {title}")
        return

    indices = (
        [int(which[0]) - 1] if which and which[0].isdigit() else range(len(SCENARIOS))
    )
    for n, i in enumerate(indices):
        await _run_one(i)
        if n + 1 < len(list(indices)):
            time.sleep(5)  # be gentle with the rate limit


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main(sys.argv[1:]))
