"""Long-term (cross-conversation) memory.

Two different kinds of memory exist in this project, and it is worth being
precise about the difference:

* SHORT-TERM  -- the checkpointer (agent_core): the message history of ONE
  conversation, keyed by ``thread_id``. Start a new chat and it is gone.

* LONG-TERM   -- this module: durable facts the agent chooses to keep about
  the USER ("prefers snake_case branch names", "main repo is
  Varshini0511/expense-tracker"), keyed by ``user_id`` and shared across
  every conversation. Backed by LangGraph's ``Store`` abstraction (SQLite
  here; Postgres/Redis in production -- same interface).

The model reads and writes it through two ordinary tools, ``remember`` and
``recall``. That keeps memory *explicit and auditable*: nothing is stored
unless the agent calls a tool the user can see in the trace.
"""

from __future__ import annotations

import time
import uuid

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool

NAMESPACE = "memories"
DEFAULT_USER = "default"
MAX_FACT_CHARS = 500


def user_id_from(config: RunnableConfig | None) -> str:
    """Whose memory this is. Pass ``configurable={"user_id": ...}`` per call."""
    return str(((config or {}).get("configurable") or {}).get("user_id") or DEFAULT_USER)


def _normalise(fact: str) -> str:
    return " ".join(fact.lower().split())


def build_memory_tools(store) -> list[BaseTool]:
    """Create ``remember`` / ``recall`` tools bound to ``store``.

    Built as closures (not MCP tools) because they are *agent-local*: they
    touch the agent's own store, not GitHub, so they have no business living
    in the GitHub tool server.
    """

    @tool
    async def remember(fact: str, config: RunnableConfig) -> str:
        """Save a durable fact or preference about the user for FUTURE
        conversations (e.g. their preferred repo, branch-naming style, coding
        conventions). Only store stable, useful facts -- not one-off requests.
        fact: one short self-contained sentence."""
        fact = " ".join(fact.split())[:MAX_FACT_CHARS]
        if not fact:
            return "Nothing to remember."
        ns = (NAMESPACE, user_id_from(config))
        existing = await store.asearch(ns, limit=200)
        if any(_normalise(i.value.get("fact", "")) == _normalise(fact) for i in existing):
            return "Already remembered."
        await store.aput(ns, uuid.uuid4().hex, {"fact": fact, "saved_at": time.time()})
        return f"Remembered: {fact}"

    @tool
    async def recall(query: str, config: RunnableConfig) -> str:
        """Look up what you previously remembered about the user. Use it when
        personalising an answer or when the user refers to "my usual ..." .
        query: keywords to match, or an empty string to list everything."""
        ns = (NAMESPACE, user_id_from(config))
        items = await store.asearch(ns, limit=200)
        facts = [
            i.value.get("fact", "") for i in sorted(items, key=lambda i: i.value.get("saved_at", 0))
        ]
        words = [w for w in _normalise(query).split() if len(w) > 2]
        if words:
            facts = [f for f in facts if any(w in _normalise(f) for w in words)]
        if not facts:
            return "No matching memories."
        return "\n".join(f"- {f}" for f in facts[:20])

    return [remember, recall]
