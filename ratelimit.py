"""Helpers for recognising provider rate limits and honouring their retry hints.

Shared by the chat-model retry middleware (agent_core) and the embedding
batcher (rag) so the same 429 / RESOURCE_EXHAUSTED handling lives in one place.
"""

from __future__ import annotations

import re

RATE_LIMIT_MARKERS = (
    "resource_exhausted",
    "rate limit",
    "rate_limit",
    "quota",
    "too many requests",
    " 429",
    "'code': 429",
)
# Pull the provider's suggested wait out of the error, e.g.
# "Please retry in 51.2s" or "'retryDelay': '51s'".
_RETRY_DELAY_RE = re.compile(r"retry[^0-9]{0,25}?(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)


def is_rate_limited(exc: BaseException) -> bool:
    s = str(exc).lower()
    return any(m in s for m in RATE_LIMIT_MARKERS)


def retry_delay_seconds(exc: BaseException, default: float = 30.0, cap: float = 65.0) -> float:
    """Seconds to wait before retrying (provider hint + 1s cushion, capped)."""
    m = _RETRY_DELAY_RE.search(str(exc))
    if m:
        return min(float(m.group(1)) + 1.0, cap)
    return default
