"""Repository indexing + semantic code search -- the RAG half of the agent.

Two operations, exposed to the agent as MCP tools in ``server.py``:

``index_repo(repo)``
    Download the repo tarball, split every text file into overlapping
    chunks, embed the chunks with Gemini, and save the vectors to a
    per-repo FAISS index on disk.

``search_code(repo, query)``
    Embed the query, pull the nearest chunks from that repo's index, and
    return them with ``path:line`` so the agent can cite exactly where an
    answer came from.

Why RAG here: a large repo has far more code than fits in a model's
context window. Instead of dumping files in, we retrieve only the handful
of chunks semantically closest to the question.
"""

from __future__ import annotations

import io
import logging
import re
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_settings
from filetypes import is_noise_file
from github_client import GitHubClient
from ratelimit import is_rate_limited, retry_delay_seconds
from retrieval import BM25, looks_like_identifier, reciprocal_rank_fusion

logger = logging.getLogger(__name__)

# Directories that are never worth indexing (dependencies, build output).
_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "venv",
    ".venv",
    "env",
    "dist",
    "build",
    "__pycache__",
    ".next",
    ".nuxt",
    "target",
    "vendor",
    ".mypy_cache",
    ".pytest_cache",
    ".idea",
    ".vscode",
    "site-packages",
}
# File extensions we treat as indexable text.
_TEXT_EXT = {
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".go",
    ".rb",
    ".rs",
    ".c",
    ".h",
    ".cc",
    ".cpp",
    ".hpp",
    ".cs",
    ".php",
    ".swift",
    ".kt",
    ".scala",
    ".m",
    ".mm",
    ".sh",
    ".bash",
    ".zsh",
    ".ps1",
    ".sql",
    ".md",
    ".mdx",
    ".rst",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".ini",
    ".cfg",
    ".conf",
    ".xml",
    ".html",
    ".css",
    ".scss",
    ".less",
    ".gradle",
    ".tf",
    ".proto",
    ".graphql",
    ".vue",
    ".svelte",
}
# Extension-less files worth keeping when the basename matches.
_TEXT_NAMES = {"Dockerfile", "Makefile", "README", "LICENSE", ".gitignore"}


@dataclass
class Hit:
    path: str
    start_line: int
    text: str


def _is_text(name: str) -> bool:
    if is_noise_file(name):  # lock files, minified/generated output
        return False
    return Path(name).suffix.lower() in _TEXT_EXT or name in _TEXT_NAMES


def _iter_repo_files(tar_bytes: bytes, max_bytes: int):
    """Yield ``(relative_path, text)`` for each indexable file in the tar.

    GitHub tarballs wrap everything in a top-level ``owner-repo-<sha>/``
    directory, which we strip.
    """
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        for member in tar:
            if not member.isfile() or member.size > max_bytes:
                continue
            parts = Path(member.name).parts[1:]  # drop the wrapper dir
            if not parts or any(p in _SKIP_DIRS for p in parts):
                continue
            rel = "/".join(parts)
            if not _is_text(parts[-1]):
                continue
            fh = tar.extractfile(member)
            if fh is None:
                continue
            raw = fh.read()
            try:
                yield rel, raw.decode("utf-8")
            except UnicodeDecodeError:
                continue  # not really text


class RepoIndex:
    """Builds and queries a FAISS index per repository."""

    def __init__(self) -> None:
        s = get_settings()
        self._root = Path(s.vector_dir).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._max_chunks = s.rag_max_chunks
        self._max_file_bytes = s.rag_max_file_bytes
        self._hybrid = s.rag_hybrid
        self._candidates = s.rag_candidates
        self._embedding_model = s.embedding_model
        self._embeddings_client = None  # built on first use, see _embeddings
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=s.rag_chunk_size,
            chunk_overlap=s.rag_chunk_overlap,
            add_start_index=True,  # puts a char offset in each chunk's metadata
        )
        self._gh = GitHubClient()

    @property
    def _embeddings(self):
        """The embedding client, created on first use.

        Only index_repo / search_code need it. Building it eagerly made the
        whole MCP server -- all 22 tools -- refuse to start without
        GOOGLE_API_KEY, although the setting is optional.
        """
        if self._embeddings_client is None:
            self._embeddings_client = GoogleGenerativeAIEmbeddings(model=self._embedding_model)
        return self._embeddings_client

    @_embeddings.setter
    def _embeddings(self, client) -> None:
        self._embeddings_client = client

    # -- paths --------------------------------------------------------

    def _dir_for(self, repo: str) -> Path:
        slug = re.sub(r"[^A-Za-z0-9_.-]", "__", repo)
        return self._root / slug

    def is_indexed(self, repo: str) -> bool:
        return (self._dir_for(repo) / "index.faiss").exists()

    # -- build ------------------------------------------------------

    def _embed_batched(self, texts: list[str], batch_size: int = 32, max_attempts: int = 6):
        """Embed ``texts`` in small batches, waiting out provider rate limits.

        Free-tier embedding quotas are per-minute (e.g. 100 texts/min), so a
        repo with a few hundred chunks cannot be embedded in one burst. Each
        batch is retried after the provider's suggested delay; any other error
        propagates immediately.
        """
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            for attempt in range(1, max_attempts + 1):
                try:
                    vectors.extend(self._embeddings.embed_documents(chunk))
                    break
                except Exception as exc:
                    if not is_rate_limited(exc) or attempt == max_attempts:
                        raise
                    wait = retry_delay_seconds(exc)
                    logger.warning(
                        "embedding rate-limited; waiting %.0fs (batch %d/%d, attempt %d/%d)",
                        wait,
                        i // batch_size + 1,
                        -(-len(texts) // batch_size),
                        attempt,
                        max_attempts,
                    )
                    time.sleep(wait)
        return vectors

    def index(self, repo: str, ref: str = "") -> str:
        logger.info("indexing %s@%s", repo, ref or "default")
        tar = self._gh.download_tarball(repo, ref)

        docs: list[Document] = []
        n_files = 0
        for rel, text in _iter_repo_files(tar, self._max_file_bytes):
            n_files += 1
            for chunk in self._splitter.split_text(text):
                start = text.find(chunk[:80])
                start_line = text.count("\n", 0, max(start, 0)) + 1 if start >= 0 else 1
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={"path": rel, "start_line": start_line},
                    )
                )
                if len(docs) >= self._max_chunks:
                    break
            if len(docs) >= self._max_chunks:
                logger.warning("hit rag_max_chunks=%d; index is partial", self._max_chunks)
                break

        if not docs:
            return f"No indexable text files found in {repo}."

        vectors = self._embed_batched([d.page_content for d in docs])
        store = FAISS.from_embeddings(
            list(zip([d.page_content for d in docs], vectors, strict=True)),
            self._embeddings,
            metadatas=[d.metadata for d in docs],
        )
        store.save_local(str(self._dir_for(repo)))
        return (
            f"Indexed {repo}: {n_files} files -> {len(docs)} chunks. "
            f"search_code is now available for this repo."
        )

    # -- query ----------------------------------------------------

    def search(self, repo: str, query: str, k: int = 6) -> list[Hit]:
        """Top-``k`` chunks for ``query``.

        With hybrid on (default) this runs BOTH retrievers over the same
        index and fuses their rankings with Reciprocal Rank Fusion:

        * dense  -- embedding similarity: finds code by *meaning*
        * BM25   -- keyword scoring: finds exact identifiers / error codes

        A query like "where is get_policies_async defined" is a keyword
        problem; "how are rate limits handled" is a meaning problem. Fusing
        gets both right without choosing in advance.
        """
        if not self.is_indexed(repo):
            raise FileNotFoundError(f"{repo} has not been indexed yet. Call index_repo first.")
        store = FAISS.load_local(
            str(self._dir_for(repo)),
            self._embeddings,
            allow_dangerous_deserialization=True,  # our own local file
        )

        if not self._hybrid:
            docs = store.similarity_search(query, k=k)
        else:
            # Pull a wider candidate pool from each retriever, then fuse.
            pool = max(self._candidates, k)
            dense = store.similarity_search(query, k=pool)
            all_docs = list(store.docstore._dict.values())  # every indexed chunk
            bm25 = BM25([d.page_content for d in all_docs])
            lexical = [all_docs[i] for i in bm25.top(query, pool)]

            def key(d: Document) -> tuple[str, int]:
                return (d.metadata.get("path", "?"), int(d.metadata.get("start_line", 1)))

            by_key = {key(d): d for d in [*dense, *lexical]}
            # For code-shaped queries ("get_policies_async", "Policies.tsx")
            # the exact-match leg is far more trustworthy than embeddings.
            lexical_weight = 2.0 if looks_like_identifier(query) else 1.0
            fused = reciprocal_rank_fusion(
                [[key(d) for d in dense], [key(d) for d in lexical]],
                top_n=k,
                weights=[1.0, lexical_weight],
            )
            docs = [by_key[kk] for kk in fused]

        return [
            Hit(
                path=d.metadata.get("path", "?"),
                start_line=int(d.metadata.get("start_line", 1)),
                text=d.page_content,
            )
            for d in docs
        ]
