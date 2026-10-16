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


# --- indexing hygiene + rate-limit-aware embedding ---------------------------


def test_lockfiles_and_generated_files_are_not_indexed():
    import io
    import tarfile

    from rag import _iter_repo_files

    files = {
        "repo-abc/src/app.py": b"print('hello')",
        "repo-abc/package-lock.json": b'{"lockfileVersion": 3}',
        "repo-abc/yarn.lock": b"# yarn lockfile",
        "repo-abc/dist/bundle.min.js": b"var a=1;",
        "repo-abc/static/app.js.map": b"{}",
        "repo-abc/README.md": b"# Title",
    }
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

    indexed = {rel for rel, _ in _iter_repo_files(buf.getvalue(), max_bytes=10_000)}
    assert indexed == {"src/app.py", "README.md"}


class _FlakyEmbeddings:
    """Raises a provider-style 429 on its first call, then works."""

    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 0.01s.")
        return [[float(len(t))] for t in texts]


def test_embedding_waits_out_a_rate_limit_and_keeps_going(monkeypatch):
    idx = object.__new__(RepoIndex)
    idx._embeddings = _FlakyEmbeddings()
    slept = []
    monkeypatch.setattr("rag.time.sleep", lambda s: slept.append(s))

    vectors = idx._embed_batched(["a", "bb", "ccc"], batch_size=2)

    assert vectors == [[1.0], [2.0], [3.0]]  # nothing lost or reordered
    assert len(slept) == 1 and slept[0] >= 1.0  # honoured the provider's hint
    assert idx._embeddings.calls == 3  # 1 failed + 2 batches


def test_embedding_does_not_swallow_non_rate_limit_errors():
    class Broken:
        def embed_documents(self, texts):
            raise ValueError("bad api key")

    idx = object.__new__(RepoIndex)
    idx._embeddings = Broken()
    with pytest.raises(ValueError, match="bad api key"):
        idx._embed_batched(["x"])
