"""A persistent background event loop for driving async code from sync callers.

Streamlit reruns the whole script on every interaction and does not
guarantee the same OS thread executes each rerun. The agent, however,
holds thread-affine async resources -- an MCP subprocess talked to over
stdio, an async SQLite checkpointer -- that must always be driven from
the SAME event loop they were created on.

``AsyncRunner`` owns one event loop in one dedicated background thread
for as long as the object lives. Sync code calls ``.run(coro)`` and
blocks until the coroutine completes, no matter which thread the call
came from.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


class AsyncRunner:
    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run ``coro`` on the background loop; block for its result.

        Only safe for coroutines that are self-contained -- do not use
        this to enter an async context manager in one call and use it in
        a later, separate call. Each call here becomes its own top-level
        asyncio Task, and several async resources this project depends on
        (aiosqlite's connection, the MCP client's background reader) are
        only valid for as long as the ONE task that opened them is still
        running; a later, unrelated task trying to reuse them fails in
        confusing ways (a Thread that refuses to restart, "no active
        connection"). For anything that must be entered once and used
        repeatedly, use ``AgentWorker`` instead.
        """
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def spawn(self, coro: Coroutine[Any, Any, Any]) -> None:
        """Schedule ``coro`` on the background loop and return immediately
        (fire and forget) -- used to start a long-lived task."""
        asyncio.run_coroutine_threadsafe(coro, self._loop)


class AgentWorker:
    """Owns one agent for one continuous asyncio Task, fed through a queue.

    ``build_agent()``/``build_supervisor()`` open resources -- an MCP
    stdio subprocess, an async SQLite connection -- that assume the SAME
    task which entered them keeps running for as long as they're used
    (exactly how the CLI (`chat.py`) uses them: one
    ``async with build_agent() as agent:`` wraps the whole multi-turn
    session). A UI, in contrast, naturally wants to "build once, call
    many times from many separate events." AgentWorker reconciles the
    two: one dedicated coroutine holds the ``async with`` open for the
    worker's entire life and pulls requests off a queue; callers (from
    any thread) submit work and block for the result via a
    ``concurrent.futures.Future``.
    """

    def __init__(self, runner: AsyncRunner, builder: Callable[[], Any]) -> None:
        self._runner = runner
        self._queue: asyncio.Queue | None = None
        self._ready: concurrent.futures.Future = concurrent.futures.Future()
        runner.spawn(self._serve(builder))

    async def _serve(self, builder: Callable[[], Any]) -> None:
        try:
            self._queue = asyncio.Queue()
            async with builder() as agent:
                self._ready.set_result(agent)
                while True:
                    fut, coro_fn = await self._queue.get()
                    try:
                        fut.set_result(await coro_fn(agent))
                    except Exception as exc:  # noqa: BLE001 - relayed to the caller
                        fut.set_exception(exc)
        except Exception as exc:  # build itself failed
            if not self._ready.done():
                self._ready.set_exception(exc)

    def wait_ready(self, timeout: float | None = None) -> None:
        """Block until the agent is built (raises if construction failed)."""
        self._ready.result(timeout=timeout)

    def call(self, coro_fn: Callable[[Any], Coroutine[Any, Any, T]]) -> T:
        """Run ``await coro_fn(agent)`` inside the worker's task; block
        for the result. Safe to call from any thread."""
        self.wait_ready()
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._runner.run(self._queue.put((fut, coro_fn)))  # type: ignore[union-attr]
        return fut.result()
