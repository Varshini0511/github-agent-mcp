"""Hybrid retrieval primitives: BM25 keyword search + rank fusion.

Why hybrid? Dense (embedding) search matches *meaning* but can be weak on
exact tokens -- an error code, a function name like ``get_policies_async``, an
issue number. Keyword search (BM25) is the opposite. Fusing the two rankings is
a cheap hedge: it never needs to be told which kind of query it got.

Measured, not assumed: on one small repo (157 chunks, 8 identifier queries,
ground truth = "chunk literally contains the token") hybrid recovered 21/28
relevant chunks in the top 5 vs 19/28 for dense alone, with the same mean rank
of the first hit. A modest, never-worse gain -- strong embeddings already do
well on small repos. Reproduce with ``scripts/measure_retrieval.py``.

This module is deliberately dependency-free (pure Python) so it is fast to
import and trivially unit-testable; ``rag.py`` supplies the dense results and
calls into here.

BM25 here is the standard Okapi formulation with an identifier-aware
tokenizer (``getPolicies`` / ``get_policies`` both split into ``get``,
``policies``).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Hashable, Iterable, Sequence

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_WORD = re.compile(r"[A-Za-z0-9]+")

# Words that carry no signal for code search.
_STOP = frozenset(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "is",
        "it",
        "for",
        "on",
        "with",
        "as",
        "at",
        "by",
        "be",
        "this",
        "that",
        "from",
        "are",
        "was",
    ]
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; splits snake_case, camelCase and punctuation."""
    tokens: list[str] = []
    for raw in _WORD.findall(text.replace("_", " ")):
        for part in _CAMEL.split(raw):
            p = part.lower()
            if p and p not in _STOP:
                tokens.append(p)
    return tokens


class BM25:
    """Okapi BM25 over a fixed corpus of documents."""

    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._docs = [Counter(tokenize(d)) for d in documents]
        self._len = [sum(c.values()) for c in self._docs]
        self._avg = (sum(self._len) / len(self._len)) if self._len else 0.0
        df: Counter[str] = Counter()
        for c in self._docs:
            df.update(c.keys())
        n = len(self._docs)
        # +1 inside the log keeps IDF positive for very common terms.
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = [0.0] * len(self._docs)
        if not q or not self._avg:
            return out
        for i, tf in enumerate(self._docs):
            norm = self.k1 * (1 - self.b + self.b * self._len[i] / self._avg)
            s = 0.0
            for term in q:
                f = tf.get(term)
                if f:
                    s += self._idf.get(term, 0.0) * (f * (self.k1 + 1)) / (f + norm)
            out[i] = s
        return out

    def top(self, query: str, k: int) -> list[int]:
        """Indices of the best-matching documents (zero-score docs excluded)."""
        scored = [(s, i) for i, s in enumerate(self.scores(query)) if s > 0]
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [i for _, i in scored[:k]]


_IDENTIFIER_HINTS = re.compile(
    r"[A-Za-z0-9]_[A-Za-z0-9]"  # snake_case
    r"|[a-z][A-Z]"  # camelCase
    r"|[A-Za-z0-9][./\\][A-Za-z0-9]"  # paths and dotted names (a/b, a.b, a\b)
    r"|\w+\(\)"  # call syntax: init()
    r"|\b[A-Z]+[0-9]{2,}\b"  # error codes: E1042
    r"|\b[A-Z]{2,}(?:_[A-Z0-9]+)+\b"  # CONSTANT_NAMES
)


def looks_like_identifier(query: str) -> bool:
    """Heuristic: does the query contain code-shaped tokens?

    ``reconcile_ledger_entries`` or ``Policies.tsx`` or ``E1042`` are lookups
    where the user wants THAT exact token; prose like "how is auth handled"
    is a meaning query. Used to decide how much to trust keyword matches.
    """
    return bool(_IDENTIFIER_HINTS.search(query))


def reciprocal_rank_fusion(
    rankings: Iterable[Sequence[Hashable]],
    k: int = 60,
    top_n: int | None = None,
    weights: Sequence[float] | None = None,
) -> list[Hashable]:
    """Fuse several ranked lists into one (Cormack et al., 2009).

    Each item scores ``sum(weight / (k + rank))`` over the lists it appears
    in. It needs only RANKS, never raw scores -- which is exactly why it works
    for combining BM25 scores and cosine distances that live on incomparable
    scales. ``k=60`` is the value from the original paper.

    ``weights`` lets one retriever count for more than another (default: all
    equal); it also breaks the otherwise-common tie between the #1 of each
    list in favour of the retriever you trust for this query.
    """
    rankings = list(rankings)
    w = list(weights) if weights is not None else [1.0] * len(rankings)
    fused: dict[Hashable, float] = {}
    first_seen: dict[Hashable, int] = {}
    for ranking, weight in zip(rankings, w, strict=False):
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + weight / (k + rank)
            first_seen.setdefault(item, len(first_seen))
    ordered = sorted(fused, key=lambda it: (-fused[it], first_seen[it]))
    return ordered[:top_n] if top_n else ordered
