"""Centralised, validated configuration for the GitHub agent.

Every runtime setting lives here as ONE validated object instead of
scattered ``os.getenv()`` calls. ``pydantic-settings`` reads values from
the process environment (and the local ``.env`` file), coerces them to
the declared types, and raises at startup if a required value is missing
-- the "fail fast" principle.

Usage
-----
    from config import get_settings

    settings = get_settings()
    token = settings.github_token.get_secret_value()
"""

import logging
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env into os.environ once, at import time, for the benefit of
# THIRD-PARTY libraries (langchain, the groq/google SDKs) that read their
# keys straight from the environment. Resolve it by ABSOLUTE path next to
# this file, not via the working directory: the MCP server runs as a child
# process whose cwd may differ, and it must still find the same .env.
load_dotenv(Path(__file__).with_name(".env"))


class Settings(BaseSettings):
    """All configuration for the server and the agent.

    Fields with ``...`` (Ellipsis) as the first argument to ``Field`` are
    REQUIRED: if the matching environment variable is absent, constructing
    ``Settings()`` raises ``ValidationError`` immediately.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # tolerate unrelated variables in the environment
        case_sensitive=False,
    )

    # --- Secrets ---------------------------------------------------------
    # SecretStr keeps the value out of logs: repr() shows '**********'.
    # Call .get_secret_value() at the point of use.
    github_token: SecretStr = Field(..., alias="GITHUB_TOKEN")
    groq_api_key: SecretStr = Field(..., alias="GROQ_API_KEY")
    # Optional until Block B (RAG). None is a valid value here.
    google_api_key: SecretStr | None = Field(None, alias="GOOGLE_API_KEY")

    # --- Model choices -------------------------------------------------
    chat_model: str = Field("groq:openai/gpt-oss-120b", alias="CHAT_MODEL")
    embedding_model: str = Field(
        "models/gemini-embedding-001", alias="EMBEDDING_MODEL"
    )

    # --- Write safety (Block C) ------------------------------------
    # When False (the default) the write tools refuse and explain how to
    # enable them. Flip to True only with a token that has write scope
    # and on repositories you own.
    allow_writes: bool = Field(False, alias="ALLOW_WRITES")

    # --- HTTP / retry behaviour -------------------------------------
    github_api_url: str = Field("https://api.github.com", alias="GITHUB_API_URL")
    request_timeout: float = Field(15.0, alias="REQUEST_TIMEOUT")
    max_retries: int = Field(3, alias="MAX_RETRIES")
    # Exponential backoff between retries: wait grows 1s, 2s, 4s ... capped.
    # Tests set these to 0 so the suite doesn't actually sleep.
    retry_backoff_initial: float = Field(1.0, alias="RETRY_BACKOFF_INITIAL")
    retry_backoff_max: float = Field(20.0, alias="RETRY_BACKOFF_MAX")

    # --- Agent limits ---------------------------------------------------
    # Max model calls per user turn (the runaway-loop safety valve). A
    # multi-file code change legitimately needs many read+write steps, so
    # this is generous; lower it if you want a tighter guard.
    agent_run_limit: int = Field(25, alias="AGENT_RUN_LIMIT")

    # --- Memory ---------------------------------------------------------
    # SQLite file where the agent persists conversation state, keyed by
    # thread_id. Survives process restarts (unlike an in-memory saver).
    checkpoint_db: str = Field("agent_memory.sqlite", alias="CHECKPOINT_DB")

    # --- RAG (Block B) ------------------------------------------------
    # Directory holding one FAISS index per indexed repository.
    vector_dir: str = Field("vector_store", alias="VECTOR_DIR")
    rag_chunk_size: int = Field(1200, alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(200, alias="RAG_CHUNK_OVERLAP")
    rag_max_chunks: int = Field(3000, alias="RAG_MAX_CHUNKS")
    rag_max_file_bytes: int = Field(200_000, alias="RAG_MAX_FILE_BYTES")

    # --- Observability ----------------------------------------------
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # Convenience: ready-to-use GitHub REST headers.
    @property
    def github_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.github_token.get_secret_value()}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide singleton ``Settings`` instance.

    ``@lru_cache`` means the ``.env`` file is parsed and validated exactly
    once, the first time this function is called, no matter how many
    modules import it.
    """
    return Settings()


def configure_logging(level: str | None = None) -> None:
    """Set up root logging once, writing to STDERR.

    Important for the MCP server: it speaks JSON-RPC over STDOUT, so logs
    must never go there. ``logging.basicConfig`` uses stderr by default,
    which is safe.
    """
    logging.basicConfig(
        level=(level or get_settings().log_level).upper(),
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
