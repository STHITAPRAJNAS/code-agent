"""
config.py — Application configuration loaded from environment variables.

All settings can be overridden via environment variables or a .env file
in the working directory. A singleton accessor ``get_settings()`` is
provided so the rest of the codebase never instantiates Settings directly.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env from the project root (wherever the process is started from).
# ``override=False`` means real environment variables always take precedence.
load_dotenv(dotenv_path=Path(".env"), override=False)


def _env(key: str, default: str | None = None) -> str | None:
    """Return an environment variable, falling back to *default*."""
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    """Return an integer environment variable, falling back to *default*."""
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {key!r} must be an integer, got {raw!r}"
        ) from exc


class Settings:
    """
    Centralised application configuration.

    All fields are populated from environment variables at construction time.
    Use :func:`get_settings` to obtain the shared singleton.
    """

    # ------------------------------------------------------------------
    # Google / Gemini
    # ------------------------------------------------------------------
    GOOGLE_API_KEY: str | None
    GEMINI_MODEL: str

    # ------------------------------------------------------------------
    # GitHub
    # ------------------------------------------------------------------
    GITHUB_TOKEN: str | None

    # ------------------------------------------------------------------
    # Bitbucket Cloud
    # ------------------------------------------------------------------
    BITBUCKET_USERNAME: str | None
    BITBUCKET_APP_PASSWORD: str | None
    BITBUCKET_WORKSPACE: str | None

    # ------------------------------------------------------------------
    # Bitbucket Server (Data Center)
    # ------------------------------------------------------------------
    BITBUCKET_SERVER_URL: str | None
    BITBUCKET_SERVER_TOKEN: str | None
    BITBUCKET_SERVER_PROJECT: str | None

    # ------------------------------------------------------------------
    # Local workspace / file system
    # ------------------------------------------------------------------
    WORKSPACE_DIR: str
    CHROMA_PATH: str
    CHROMA_COLLECTION: str
    EMBEDDING_BATCH_SIZE: int
    MAX_CLONE_WORKERS: int
    SHALLOW_CLONE_DEPTH: int

    # ------------------------------------------------------------------
    # Jira
    # ------------------------------------------------------------------
    JIRA_SERVER_URL: str | None
    JIRA_USERNAME: str | None
    JIRA_API_TOKEN: str | None

    # ------------------------------------------------------------------
    # Confluence
    # ------------------------------------------------------------------
    CONFLUENCE_SERVER_URL: str | None
    CONFLUENCE_USERNAME: str | None
    CONFLUENCE_API_TOKEN: str | None

    # ------------------------------------------------------------------
    # HTTP server
    # ------------------------------------------------------------------
    APP_HOST: str
    APP_PORT: int
    LOG_LEVEL: str

    # ------------------------------------------------------------------
    # Deployment
    # ------------------------------------------------------------------
    DEPLOYMENT_MODE: str
    DATABASE_URL: str
    PGVECTOR_TABLE: str
    VERTEX_RAG_CORPUS: str
    RAG_BACKEND: str

    def __init__(self) -> None:
        # Google / Gemini
        self.GOOGLE_API_KEY = _env("GOOGLE_API_KEY")
        self.GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-2.0-flash") or "gemini-2.0-flash"

        # GitHub
        self.GITHUB_TOKEN = _env("GITHUB_TOKEN")

        # Bitbucket Cloud
        self.BITBUCKET_USERNAME = _env("BITBUCKET_USERNAME")
        self.BITBUCKET_APP_PASSWORD = _env("BITBUCKET_APP_PASSWORD")
        self.BITBUCKET_WORKSPACE = _env("BITBUCKET_WORKSPACE")

        # Bitbucket Server
        self.BITBUCKET_SERVER_URL = _env("BITBUCKET_SERVER_URL")
        self.BITBUCKET_SERVER_TOKEN = _env("BITBUCKET_SERVER_TOKEN")
        self.BITBUCKET_SERVER_PROJECT = _env("BITBUCKET_SERVER_PROJECT")

        # Local workspace / file system
        self.WORKSPACE_DIR = _env("WORKSPACE_DIR", "/tmp/code_agent_workspaces") or "/tmp/code_agent_workspaces"
        self.CHROMA_PATH = _env("CHROMA_PATH", "./chroma_db") or "./chroma_db"
        self.CHROMA_COLLECTION = _env("CHROMA_COLLECTION", "code_knowledge") or "code_knowledge"
        self.EMBEDDING_BATCH_SIZE = _env_int("EMBEDDING_BATCH_SIZE", 50)
        self.MAX_CLONE_WORKERS = _env_int("MAX_CLONE_WORKERS", 4)
        self.SHALLOW_CLONE_DEPTH = _env_int("SHALLOW_CLONE_DEPTH", 1)

        # Jira
        self.JIRA_SERVER_URL = _env("JIRA_SERVER_URL")
        self.JIRA_USERNAME = _env("JIRA_USERNAME")
        self.JIRA_API_TOKEN = _env("JIRA_API_TOKEN")

        # Confluence
        self.CONFLUENCE_SERVER_URL = _env("CONFLUENCE_SERVER_URL")
        self.CONFLUENCE_USERNAME = _env("CONFLUENCE_USERNAME")
        self.CONFLUENCE_API_TOKEN = _env("CONFLUENCE_API_TOKEN")

        # HTTP server
        self.APP_HOST = _env("APP_HOST", "0.0.0.0") or "0.0.0.0"
        self.APP_PORT = _env_int("APP_PORT", 8000)
        self.LOG_LEVEL = _env("LOG_LEVEL", "INFO") or "INFO"

        # Deployment
        self.DEPLOYMENT_MODE = _env("DEPLOYMENT_MODE", "local") or "local"
        self.DATABASE_URL = _env("DATABASE_URL", "") or ""
        self.PGVECTOR_TABLE = _env("PGVECTOR_TABLE", "code_knowledge") or "code_knowledge"
        self.VERTEX_RAG_CORPUS = _env("VERTEX_RAG_CORPUS", "") or ""
        self.RAG_BACKEND = _env("RAG_BACKEND", "llamaindex") or "llamaindex"

    def __repr__(self) -> str:  # pragma: no cover
        safe_fields = {
            "GEMINI_MODEL": self.GEMINI_MODEL,
            "DEPLOYMENT_MODE": self.DEPLOYMENT_MODE,
            "RAG_BACKEND": self.RAG_BACKEND,
            "WORKSPACE_DIR": self.WORKSPACE_DIR,
            "CHROMA_PATH": self.CHROMA_PATH,
            "CHROMA_COLLECTION": self.CHROMA_COLLECTION,
            "APP_HOST": self.APP_HOST,
            "APP_PORT": self.APP_PORT,
            "LOG_LEVEL": self.LOG_LEVEL,
        }
        return f"Settings({safe_fields})"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the application-wide :class:`Settings` singleton.

    The first call constructs and caches the instance; subsequent calls
    return the cached object without re-reading the environment.

    Example::

        from code_agent.config import get_settings

        cfg = get_settings()
        print(cfg.GEMINI_MODEL)
    """
    return Settings()
