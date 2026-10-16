"""Measure retrieval quality: dense-only vs hybrid, with objective ground truth.

    python scripts/measure_retrieval.py                       # defaults below
    python scripts/measure_retrieval.py owner/name Token1 Token2 ...

For each token (an identifier / constant / route), the "relevant" chunks are
those whose text literally contains it. We then ask each retrieval mode for its
top-K and count how many relevant chunks it recovered and how early the first
one appeared. The repo must already be indexed (run compare_search.py or the
agent's index_repo first).

Caveat: "contains the token" is a crude ground truth and small repos flatter
dense search -- treat the numbers as a regression check, not a benchmark.
"""

import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from langchain_community.vectorstores import FAISS  # noqa: E402

from rag import RepoIndex  # noqa: E402

REPO = sys.argv[1] if len(sys.argv) > 1 else "Varshini0511/expense-tracker"
K = 5
TOKENS = sys.argv[2:] or [
    "GetPoliciesAsync",
    "FindRelevantPolicyAsync",
    "AskGroqAboutPolicyAsync",
    "EmbedPoliciesAsync",
    "NEXT_PUBLIC_API_URL",
    "SubmitExpenseAsync",
    "AddPolicyRequest",
    "embed-policies",
]

idx = RepoIndex()
store = FAISS.load_local(
    str(idx._dir_for(REPO)), idx._embeddings, allow_dangerous_deserialization=True
)
chunks = list(store.docstore._dict.values())
key = lambda d: (d.metadata["path"], int(d.metadata["start_line"]))  # noqa: E731

print(f"index: {len(chunks)} chunks. Ground truth = chunks that literally contain the token.\n")
print(f"{'token':26}{'#rel':>5}  {'DENSE found/rank':>18}  {'HYBRID found/rank':>19}")
tot = {"rel": 0, "dense": 0, "hybrid": 0}
first = {"dense": [], "hybrid": []}
for tok in TOKENS:
    rel = {key(d) for d in chunks if tok in d.page_content}
    if not rel:
        continue
    out = {}
    for mode, flag in (("dense", False), ("hybrid", True)):
        idx._hybrid = flag
        hits = [(h.path, h.start_line) for h in idx.search(REPO, tok, K)]
        found = [i for i, h in enumerate(hits, 1) if h in rel]
        out[mode] = (len(found), found[0] if found else None)
        first[mode].append(found[0] if found else K + 1)
    exp = min(len(rel), K)  # best achievable in top-K
    tot["rel"] += exp
    tot["dense"] += out["dense"][0]
    tot["hybrid"] += out["hybrid"][0]

    def fmt(t, exp=exp):
        return f"{t[0]}/{exp}  first@{t[1] if t[1] else '-'}"

    print(f"{tok:26}{len(rel):>5}  {fmt(out['dense']):>18}  {fmt(out['hybrid']):>19}")

print("-" * 74)
print(
    f"relevant chunks recovered in top-{K}:  dense {tot['dense']}/{tot['rel']}   hybrid {tot['hybrid']}/{tot['rel']}"
)
print(
    f"mean rank of first relevant hit:      dense {sum(first['dense']) / len(first['dense']):.2f}   hybrid {sum(first['hybrid']) / len(first['hybrid']):.2f}  (lower is better)"
)
