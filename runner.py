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
import queue
import threading
import weakref
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator
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

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the background loop and join its thread (idempotent)."""
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout)

    def spawn(self, coro: Coroutine[Any, Any, Any]) -> None:
        """Schedule ``coro`` on the background loop and return immediately
        (fire and forget) -- used to start a long-lived task."""
        asyncio.run_coroutine_threadsafe(coro, self._loop)


# Every AgentWorker that has not been closed (so shutdown/tests can reap them).
_LIVE_WORKERS: weakref.WeakSet = weakref.WeakSet()


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
        self._closed: concurrent.futures.Future = concurrent.futures.Future()
        _LIVE_WORKERS.add(self)
        runner.spawn(self._serve(builder))

    async def _serve(self, builder: Callable[[], Any]) -> None:
        """The owning task: opens the agent's resources and keeps them open.

        Jobs are run as *child tasks on this same event loop* so several
        requests can be in flight at once (an API serving many users must not
        queue behind one slow turn), while THIS task -- the one that entered
        ``async with builder()`` -- stays alive for the worker's whole life,
        which is the invariant the MCP client and SQLite connection need.
        """
        jobs: set[asyncio.Task] = set()

        async def _run(fut: concurrent.futures.Future, coro_fn, agent) -> None:
            try:
                fut.set_result(await coro_fn(agent))
            except Exception as exc:  # noqa: BLE001 - relayed to the caller
                fut.set_exception(exc)

        try:
            self._queue = asyncio.Queue()
            async with builder() as agent:
                self._ready.set_result(agent)
                while True:
                    item = await self._queue.get()
                    if item is None:  # close() sentinel
                        break
                    fut, coro_fn = item
                    task = asyncio.create_task(_run(fut, coro_fn, agent))
                    jobs.add(task)
                    task.add_done_callback(jobs.discard)
                # Let in-flight turns finish before the resources close.
                if jobs:
                    await asyncio.gather(*jobs, return_exceptions=True)
        except Exception as exc:  # build itself failed
            if not self._ready.done():
                self._ready.set_exception(exc)
        finally:
            if not self._closed.done():
                self._closed.set_result(None)

    def close(self, timeout: float = 10.0) -> None:
        """Gracefully shut down: finish in-flight turns, then close the MCP
        subprocess and database connections the worker owns."""
        if self._queue is not None and not self._closed.done():
            try:
                self._runner.run(self._queue.put(None))
                self._closed.result(timeout=timeout)
            except Exception:  # best effort during interpreter/test teardown
                pass
        _LIVE_WORKERS.discard(self)

    @staticmethod
    def close_all(timeout: float = 10.0) -> None:
        """Close every live worker (used by tests and app shutdown)."""
        for worker in list(_LIVE_WORKERS):
            worker.close(timeout)

    def wait_ready(self, timeout: float | None = None) -> None:
        """Block until the agent is built (raises if construction failed)."""
        self._ready.result(timeout=timeout)

    def submit(self, coro_fn: Callable[[Any], Coroutine[Any, Any, T]]) -> concurrent.futures.Future:
        """Queue ``await coro_fn(agent)`` and return a Future for its result
        without waiting. Safe to call from any thread."""
        self.wait_ready()
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._runner.run(self._queue.put((fut, coro_fn)))  # type: ignore[union-attr]
        return fut

    def call(self, coro_fn: Callable[[Any], Coroutine[Any, Any, T]]) -> T:
        """Run ``await coro_fn(agent)`` inside the worker; block for the result."""
        return self.submit(coro_fn).result()

    def stream(self, agen_fn: Callable[[Any], AsyncIterator[Any]]) -> Iterator[Any]:
        """Consume an async generator that runs inside the worker as a plain
        synchronous iterator (events arrive as they are produced).

            for event in worker.stream(lambda agent: stream_turn(agent, cfg, "hi")):
                ...
        """
        events: queue.Queue = queue.Queue()
        end = object()

        async def pump(agent) -> None:
            try:
                async for event in agen_fn(agent):
                    events.put(event)
            except Exception as exc:  # noqa: BLE001 - re-raised in the consumer
                events.put(exc)
            finally:
                events.put(end)

        self.submit(pump)
        while True:
            item = events.get()
            if item is end:
                return
            if isinstance(item, Exception):
                raise item
            yield item
