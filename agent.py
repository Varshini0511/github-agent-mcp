"""One-shot query against the GitHub agent.

Each run uses a throwaway thread_id, so there is no memory carried
between runs. For a stateful conversation use `chat.py`.

    python agent.py "What files are in octocat/Hello-World?"
"""

import asyncio
import sys
import uuid

from agent_core import build_agent, run_turn

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_QUESTION = "What files are in the octocat/Hello-World repository?"


async def _auto_reject(name: str, args: dict) -> bool:
    # No human is present in one-shot mode, so any write is declined.
    print(f"[write tool '{name}' auto-declined: run chat.py to approve writes]")
    return False


async def main(question: str) -> None:
    async with build_agent() as agent:
        config = {"configurable": {"thread_id": f"oneshot-{uuid.uuid4().hex[:8]}"}}
        reply = await run_turn(agent, config, question, _auto_reject)
        print(reply)


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
    asyncio.run(main(question))
