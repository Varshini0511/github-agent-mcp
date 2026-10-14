"""Observability: per-turn token usage, latency, tool calls and estimated cost.

An agent turn is not one model call -- it is a loop (decide -> tool -> decide
-> ... -> answer), plus the summariser and any sub-agents. The only honest way
to see what a turn *cost* is to listen at the model boundary, which is what a
LangChain callback handler does: it is told about every LLM and tool call
that happens underneath the run, however deeply nested.

Pieces
------
``UsageCallback``   pass in ``config={"callbacks": [cb]}``; collects counts.
``TurnMetrics``     the immutable result for one send()/resume().
``UsageTracker``    thread-safe running totals per conversation.
``estimate_cost``   token counts x a price table (overridable).

Optional tracing: LangChain auto-ships traces to LangSmith when
``LANGSMITH_TRACING=true`` and ``LANGSMITH_API_KEY`` are set -- no code
required, see README. This module is the always-on, dependency-free layer.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

metrics_logger = logging.getLogger("github-agent.metrics")

# USD per 1,000,000 tokens: (input, output). These are LIST-PRICE ESTIMATES
# used to give a sense of scale -- on a free tier the real bill is zero, and
# vendors change prices, so treat the output as "roughly". Override with the
# MODEL_PRICES_JSON setting, e.g. {"gemini": [0.3, 2.5]}.
# Matched by case-insensitive substring, first match wins -> keep specific
# patterns before general ones.
DEFAULT_PRICES: list[tuple[str, tuple[float, float]]] = [
    ("flash-lite", (0.10, 0.40)),
    ("gemini", (0.30, 2.50)),
    ("gpt-oss-120b", (0.15, 0.60)),
    ("gpt-oss-20b", (0.075, 0.30)),
    ("claude", (3.00, 15.00)),
]


def resolve_prices(overrides_json: str | None = None) -> list[tuple[str, tuple[float, float]]]:
    """Defaults, with any user overrides taking precedence."""
    table: list[tuple[str, tuple[float, float]]] = []
    if overrides_json:
        try:
            raw = json.loads(overrides_json)
            table.extend((str(k).lower(), (float(v[0]), float(v[1]))) for k, v in raw.items())
        except (ValueError, TypeError, IndexError, KeyError) as exc:
            metrics_logger.warning("ignoring invalid MODEL_PRICES_JSON: %s", exc)
    return table + DEFAULT_PRICES


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    prices: list[tuple[str, tuple[float, float]]] | None = None,
) -> float | None:
    """Estimated USD for one model's usage, or None if the model is unknown."""
    name = model.lower()
    for pattern, (p_in, p_out) in prices or DEFAULT_PRICES:
        if pattern in name:
            return (input_tokens * p_in + output_tokens * p_out) / 1_000_000
    return None


@dataclass(frozen=True)
class TurnMetrics:
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    tools: tuple[str, ...] = ()
    latency_s: float = 0.0
    cost_usd: float | None = None
    models: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["total_tokens"] = self.total_tokens
        d["tools"] = list(self.tools)
        return d


class UsageCallback(BaseCallbackHandler):
    """Collects model/tool activity for ONE turn. Create a fresh one per turn.

    LangChain may invoke sync handlers from worker threads even in async
    code, so every mutation is lock-protected.
    """

    def __init__(self, prices: list[tuple[str, tuple[float, float]]] | None = None):
        self._lock = threading.Lock()
        self._prices = prices
        self._started = time.perf_counter()
        self._calls = 0
        self._tools: list[str] = []
        self._by_model: dict[str, dict[str, int]] = defaultdict(
            lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        )

    # Chat models fire this before on_llm_start; defining it avoids
    # LangChain's "falling back to on_llm_start" warning path.
    def on_chat_model_start(self, serialized, messages, **kwargs):  # noqa: D401
        return None

    def on_llm_end(self, response, **kwargs):
        with self._lock:
            self._calls += 1
            for generations in getattr(response, "generations", []) or []:
                for gen in generations:
                    msg = getattr(gen, "message", None)
                    usage = getattr(msg, "usage_metadata", None) or {}
                    meta = getattr(msg, "response_metadata", None) or {}
                    model = meta.get("model_name") or meta.get("model") or "unknown"
                    bucket = self._by_model[model]
                    bucket["calls"] += 1
                    bucket["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
                    bucket["output_tokens"] += int(usage.get("output_tokens", 0) or 0)

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = (serialized or {}).get("name") or kwargs.get("name") or "unknown_tool"
        with self._lock:
            self._tools.append(name)

    def snapshot(self) -> TurnMetrics:
        with self._lock:
            models = {m: dict(v) for m, v in self._by_model.items()}
            tin = sum(v["input_tokens"] for v in models.values())
            tout = sum(v["output_tokens"] for v in models.values())
            costs = [
                estimate_cost(m, v["input_tokens"], v["output_tokens"], self._prices)
                for m, v in models.items()
            ]
            known = [c for c in costs if c is not None]
            return TurnMetrics(
                model_calls=self._calls,
                input_tokens=tin,
                output_tokens=tout,
                tool_calls=len(self._tools),
                tools=tuple(self._tools),
                latency_s=round(time.perf_counter() - self._started, 3),
                cost_usd=round(sum(known), 6) if known else None,
                models=models,
            )


class UsageTracker:
    """Thread-safe running totals, per conversation and overall."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_thread: dict[str, dict[str, float]] = defaultdict(self._empty)
        self._total: dict[str, float] = self._empty()

    @staticmethod
    def _empty() -> dict[str, float]:
        return {
            "turns": 0,
            "model_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "tool_calls": 0,
            "cost_usd": 0.0,
            "latency_s": 0.0,
        }

    def record(self, thread_id: str, m: TurnMetrics) -> None:
        with self._lock:
            for bucket in (self._by_thread[thread_id], self._total):
                bucket["turns"] += 1
                bucket["model_calls"] += m.model_calls
                bucket["input_tokens"] += m.input_tokens
                bucket["output_tokens"] += m.output_tokens
                bucket["tool_calls"] += m.tool_calls
                bucket["cost_usd"] += m.cost_usd or 0.0
                bucket["latency_s"] += m.latency_s

    def for_thread(self, thread_id: str) -> dict[str, float]:
        with self._lock:
            return dict(self._by_thread[thread_id])

    def overall(self) -> dict[str, float]:
        with self._lock:
            return dict(self._total)


# Process-wide tracker; the UI and API both read from it.
TRACKER = UsageTracker()


def log_turn(thread_id: str, m: TurnMetrics, **extra: Any) -> None:
    """One structured JSON line per turn -- easy to grep, ship or chart."""
    payload = {"event": "turn", "thread_id": thread_id, **m.as_dict(), **extra}
    metrics_logger.info(json.dumps(payload, default=str))
