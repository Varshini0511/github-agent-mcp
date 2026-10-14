"""RepoIndex.search with hybrid retrieval, using fake embeddings (offline).

DeterministicFakeEmbedding maps text to pseudo-random vectors, so DENSE
similarity here carries no semantic signal -- exactly the situation where an
exact-identifier query needs the BM25 leg to be found. That isolates the thing
we want to prove: the fusion wiring in rag.py actually works.
"""

import pytest

pytest.importorskip("faiss")
pytest.importorskip("langchain_community.vectorstores")
pytest.importorskip("langchain_google_genai")

from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402

from rag import RepoIndex  # noqa: E402

REPO = "o/r"


def _index(tmp_path, hybrid: bool) -> RepoIndex:
    emb = DeterministicFakeEmbedding(size=64)
    docs = [
        Document(
            page_content=f"filler module number {i} with unrelated content",
            metadata={"path": f"f{i}.py", "start_line": 1},
        )
        for i in range(30)
    ]
    docs.append(
        Document(
            page_content="async def reconcile_ledger_entries(batch): ...",
            metadata={"path": "billing/ledger.py", "start_line": 42},
        )
    )
    idx = object.__new__(RepoIndex)  # skip __init__ (it builds a real Gemini client)
    idx._root = tmp_path
    idx._embeddings = emb
    idx._hybrid = hybrid
    idx._candidates = 10
    store = FAISS.from_documents(docs, emb)
    store.save_local(str(idx._dir_for(REPO)))
    return idx


def test_hybrid_finds_an_exact_identifier_that_dense_alone_cannot(tmp_path):
    idx = _index(tmp_path, hybrid=True)
    hits = idx.search(REPO, "reconcile_ledger_entries", k=3)
    assert hits[0].path == "billing/ledger.py"
    assert hits[0].start_line == 42


def test_results_are_capped_at_k_and_carry_citations(tmp_path):
    idx = _index(tmp_path, hybrid=True)
    hits = idx.search(REPO, "reconcile_ledger_entries", k=4)
    assert 1 <= len(hits) <= 4
    assert all(h.path and h.start_line >= 1 for h in hits)


def test_dense_only_mode_still_works(tmp_path):
    idx = _index(tmp_path, hybrid=False)
    hits = idx.search(REPO, "reconcile_ledger_entries", k=3)
    assert len(hits) == 3  # returns something; ranking is arbitrary w/ fake vectors


def test_searching_an_unindexed_repo_raises(tmp_path):
    idx = _index(tmp_path, hybrid=True)
    with pytest.raises(FileNotFoundError):
        idx.search("never/indexed", "anything")
