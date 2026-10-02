"""Interactive chat with the GitHub agent, with per-conversation memory.

Every turn is saved to the SQLite checkpointer under the current
thread_id, so the agent remembers what was said earlier in this
conversation -- and still remembers it if you stop the process and start
`chat.py` again with the same thread.
"""

import asyncio
import sys
import uuid

from agent_core import build_agent, run_turn
from multi_agent import build_supervisor

# LLMs emit plenty of non-ASCII (curly quotes, en-dashes, narrow spaces).
# The Windows console defaults to cp1252 and raises on those; force UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BANNER = """\
GitHub Agent -- interactive chat
  /new     start a fresh conversation (new memory thread)
  /thread  show the current thread id
  /exit    quit

  python chat.py [thread_id]      single agent (default)
  python chat.py --multi [id]     supervisor + specialist team
"""


async def _read(prompt: str) -> str:
    """input() without blocking the event loop (the MCP stdio client
    needs the loop to keep running)."""
    loop = asyncio.get_running_loop()
    return (await loop.run_in_executor(None, input, prompt)).strip()


async def _approve(name: str, args: dict) -> bool:
    """Interactive approval prompt for a write tool."""
    print(f"\n  ⚠  the agent wants to call: {name}")
    for key, value in args.items():
        shown = str(value).replace("\n", " ")
        print(f"       {key}: {shown[:160]}")
    answer = await _read("     approve? [y/N] ")
    return answer.lower() in ("y", "yes")


async def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--multi"]
    multi = "--multi" in sys.argv
    thread_id = args[0] if args else uuid.uuid4().hex[:12]

    print(BANNER)
    print(f"({'supervisor' if multi else 'single agent'} -- thread {thread_id})\n")

    builder = build_supervisor if multi else build_agent
    async with builder() as agent:
        while True:
            try:
                text = await _read("you> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not text:
                continue
            if text == "/exit":
                break
            if text == "/new":
                thread_id = uuid.uuid4().hex[:12]
                print(f"(new thread {thread_id})\n")
                continue
            if text == "/thread":
                print(f"(thread {thread_id})\n")
                continue

            config = {"configurable": {"thread_id": thread_id}}
            reply = await run_turn(agent, config, text, _approve)
            print(f"\nagent> {reply}\n")


if __name__ == "__main__":
    asyncio.run(main())
