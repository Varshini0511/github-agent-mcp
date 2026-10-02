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
from dataclasses import dataclass
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_settings
from github_client import GitHubClient

logger = logging.getLogger(__name__)

# Directories that are never worth indexing (dependencies, build output).
_SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "venv", ".venv", "env",
    "dist", "build", "__pycache__", ".next", ".nuxt", "target", "vendor",
    ".mypy_cache", ".pytest_cache", ".idea", ".vscode", "site-packages",
}
# File extensions we treat as indexable text.
_TEXT_EXT = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rb", ".rs",
    ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".php", ".swift", ".kt",
    ".scala", ".m", ".mm", ".sh", ".bash", ".zsh", ".ps1", ".sql",
    ".md", ".mdx", ".rst", ".txt", ".toml", ".yaml", ".yml", ".json",
    ".ini", ".cfg", ".conf", ".xml", ".html", ".css", ".scss", ".less",
    ".gradle", ".tf", ".proto", ".graphql", ".vue", ".svelte",
}
# Extension-less files worth keeping when the basename matches.
_TEXT_NAMES = {"Dockerfile", "Makefile", "README", "LICENSE", ".gitignore"}


@dataclass
class Hit:
    path: str
    start_line: int
    text: str


def _is_text(name: str) -> bool:
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
        self._embeddings = GoogleGenerativeAIEmbeddings(model=s.embedding_model)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=s.rag_chunk_size,
            chunk_overlap=s.rag_chunk_overlap,
            add_start_index=True,  # puts a char offset in each chunk's metadata
        )
        self._gh = GitHubClient()

    # -- paths --------------------------------------------------------

    def _dir_for(self, repo: str) -> Path:
        slug = re.sub(r"[^A-Za-z0-9_.-]", "__", repo)
        return self._root / slug

    def is_indexed(self, repo: str) -> bool:
        return (self._dir_for(repo) / "index.faiss").exists()

    # -- build ------------------------------------------------------

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

        store = FAISS.from_documents(docs, self._embeddings)
        store.save_local(str(self._dir_for(repo)))
        return (
            f"Indexed {repo}: {n_files} files -> {len(docs)} chunks. "
            f"search_code is now available for this repo."
        )

    # -- query ----------------------------------------------------

    def search(self, repo: str, query: str, k: int = 6) -> list[Hit]:
        if not self.is_indexed(repo):
            raise FileNotFoundError(
                f"{repo} has not been indexed yet. Call index_repo first."
            )
        store = FAISS.load_local(
            str(self._dir_for(repo)),
            self._embeddings,
            allow_dangerous_deserialization=True,  # our own local file
        )
        results = store.similarity_search(query, k=k)
        return [
            Hit(
                path=d.metadata.get("path", "?"),
                start_line=int(d.metadata.get("start_line", 1)),
                text=d.page_content,
            )
            for d in results
        ]
