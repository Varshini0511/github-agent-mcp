"""Compare dense-only vs hybrid (dense + BM25 + RRF) retrieval on a real repo.

    python scripts/compare_search.py                       # defaults below
    python scripts/compare_search.py owner/name "query 1" "query 2"

It indexes the repo if needed (embedding calls use your GOOGLE_API_KEY), then
runs each query twice -- once with ``rag_hybrid`` off, once on -- and prints
the top hits side by side. Results marked ``+`` are ones hybrid surfaced that
dense-only did not.

What to look for
----------------
* Identifier queries (``GetPoliciesAsync``, ``NEXT_PUBLIC_API_URL``): hybrid
  should put the file that literally contains the token first.
* Meaning queries ("how does X decide Y"): the two should mostly agree --
  hybrid must not make semantic search worse.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import RepoIndex  # noqa: E402

DEFAULT_REPO = "Varshini0511/expense-tracker"
DEFAULT_QUERIES = [
    "GetPoliciesAsync",  # identifier
    "FindRelevantPolicyAsync",  # identifier
    "NEXT_PUBLIC_API_URL",  # constant
    "how does the app decide whether an expense follows company policy",  # meaning
]
K = 5


def label(hit) -> str:
    """path:line, shortened from the LEFT so the file name stays visible."""
    full = f"{hit.path}:{hit.start_line}"
    return full if len(full) <= 36 else "..." + full[-33:]


def main(argv: list[str]) -> None:
    repo = argv[0] if argv else DEFAULT_REPO
    queries = argv[1:] or DEFAULT_QUERIES

    idx = RepoIndex()
    if not idx.is_indexed(repo):
        t = time.perf_counter()
        print(f"indexing {repo} ...")
        print(" ", idx.index(repo), f"({time.perf_counter() - t:.0f}s)\n")
    else:
        print(f"using existing index for {repo}\n")

    for q in queries:
        idx._hybrid = False
        dense = [label(h) for h in idx.search(repo, q, K)]
        idx._hybrid = True
        hybrid = [label(h) for h in idx.search(repo, q, K)]

        print("=" * 78)
        print(f"QUERY: {q}")
        print("-" * 78)
        print(f"   {'DENSE ONLY':<37}  {'HYBRID (dense + BM25 + RRF)'}")
        for i in range(K):
            d = dense[i] if i < len(dense) else ""
            h = hybrid[i] if i < len(hybrid) else ""
            mark = "+" if h and h not in dense else " "
            print(f"{i + 1}. {d:<37}{mark} {h}")
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
