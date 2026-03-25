"""Main orchestrator for indexing a local code repository.

Walks the directory tree, chunks every source file with :class:`CodeChunker`,
generates embeddings with :class:`CodeEmbedder`, and upserts the results into
a :class:`VectorStore`.  Supports both full and incremental (git-diff) modes.
"""

from __future__ import annotations

import fnmatch
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from code_agent.indexer.chunker import CodeChunker

if TYPE_CHECKING:
    from code_agent.indexer.embedder import CodeEmbedder
    from code_agent.search.vector_store import VectorStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Skip rules
# ---------------------------------------------------------------------------
_SKIP_DIRS: frozenset[str] = frozenset(
    [".git", "__pycache__", "node_modules", ".venv", "venv", "build", "dist", ".tox",
     ".mypy_cache", ".pytest_cache", "coverage", ".coverage", ".eggs"]
)

_BINARY_EXTENSIONS: frozenset[str] = frozenset(
    [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico",
     ".pdf", ".doc", ".docx", ".xls", ".xlsx",
     ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
     ".exe", ".dll", ".so", ".dylib", ".a", ".lib",
     ".woff", ".woff2", ".ttf", ".otf",
     ".mp3", ".mp4", ".wav", ".ogg",
     ".pyc", ".pyo", ".class", ".jar",
     ".bin", ".dat", ".db", ".sqlite",
     ".ico", ".cur",
     ]
)

_SKIP_PATTERNS: list[str] = ["*.min.js", "*.lock", "package-lock.json", "yarn.lock",
                               "Cargo.lock", "poetry.lock", "*.map", "*.min.css"]


@dataclass
class IndexingResult:
    """Summary of a repository indexing run."""

    repo_id: str
    total_files: int = 0
    indexed_files: int = 0
    total_chunks: int = 0
    skipped_files: int = 0
    errors: list[str] = field(default_factory=list)
    last_commit_sha: str = ""
    duration_seconds: float = 0.0


class RepositoryIndexer:
    """Indexes a local git repository into a vector store.

    Example::

        indexer = RepositoryIndexer(vector_store=store, embedder=embedder)
        result = indexer.index_repository(
            local_path="/repos/myapp",
            repo_id="github/acme/myapp",
            metadata={"vcs": "github", "workspace": "acme"},
        )
    """

    def __init__(self, vector_store: VectorStore, embedder: CodeEmbedder) -> None:
        """Initialise with an already-configured store and embedder.

        Args:
            vector_store: Destination for chunk embeddings.
            embedder:     Embedding model client.
        """
        self._store = vector_store
        self._embedder = embedder
        self._chunker = CodeChunker()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_repository(
        self,
        local_path: str,
        repo_id: str,
        metadata: dict,
        mode: str = "full",
        languages: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> IndexingResult:
        """Index all source files in *local_path*.

        Args:
            local_path:       Absolute path to the cloned repository root.
            repo_id:          Stable identifier, e.g. ``"github/acme/api"``.
            metadata:         Arbitrary key/value pairs stored alongside every
                              chunk (vcs, workspace, repo, branch, etc.).
            mode:             ``"full"`` re-indexes everything; ``"incremental"``
                              only indexes files changed since the last run.
            languages:        If provided, only files in these languages are indexed.
            exclude_patterns: Glob patterns for files to skip (e.g. ``["*_test.go"]``).

        Returns:
            :class:`IndexingResult` with counts and any error messages.
        """
        start_time = time.monotonic()
        result = IndexingResult(repo_id=repo_id)

        last_sha = self._store.get_last_indexed_sha(repo_id) if mode == "incremental" else None
        current_sha = _get_head_sha(local_path)
        result.last_commit_sha = current_sha

        try:
            files = self._get_files_to_index(
                local_path, languages, exclude_patterns, mode, last_sha
            )
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"File discovery failed: {exc}")
            result.duration_seconds = time.monotonic() - start_time
            return result

        result.total_files = len(files)
        logger.info("Indexing %d files for %s (mode=%s)", len(files), repo_id, mode)

        indexing_meta = dict(metadata)
        indexing_meta["repo_id"] = repo_id

        self._index_files(files, local_path, repo_id, indexing_meta, result)

        if current_sha:
            try:
                self._store.save_last_indexed_sha(repo_id, current_sha)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not save commit SHA: %s", exc)

        result.duration_seconds = time.monotonic() - start_time
        logger.info(
            "Finished indexing %s: %d files indexed, %d chunks, %d skipped, %.1fs",
            repo_id,
            result.indexed_files,
            result.total_chunks,
            result.skipped_files,
            result.duration_seconds,
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_files_to_index(
        self,
        local_path: str,
        languages: list[str] | None,
        exclude_patterns: list[str] | None,
        mode: str,
        last_sha: str | None,
    ) -> list[str]:
        """Return absolute paths of files that should be indexed.

        For ``incremental`` mode with a known *last_sha*, only files
        changed since that commit are returned.  Falls back to full scan
        if git diff fails.
        """
        if mode == "incremental" and last_sha:
            try:
                return self._get_changed_files(local_path, last_sha, languages, exclude_patterns)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "git diff failed (%s) — falling back to full scan for %s",
                    exc, local_path,
                )

        return self._walk_directory(local_path, languages, exclude_patterns)

    def _walk_directory(
        self,
        local_path: str,
        languages: list[str] | None,
        exclude_patterns: list[str] | None,
    ) -> list[str]:
        """Recursively walk *local_path* and return indexable file paths."""
        root = Path(local_path)
        all_patterns = list(_SKIP_PATTERNS) + (exclude_patterns or [])
        found: list[str] = []

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            # Skip hidden files
            if any(part.startswith(".") and part != "." for part in path.relative_to(root).parts):
                continue

            # Skip directories
            if any(part in _SKIP_DIRS for part in path.parts):
                continue

            # Skip binary extensions
            if path.suffix.lower() in _BINARY_EXTENSIONS:
                continue

            # Skip glob patterns
            rel = str(path.relative_to(root))
            if any(fnmatch.fnmatch(path.name, pat) or fnmatch.fnmatch(rel, pat)
                   for pat in all_patterns):
                continue

            # Language filter
            if languages:
                lang = self._chunker.detect_language(str(path))
                if lang not in languages:
                    continue

            found.append(str(path))

        return found

    def _get_changed_files(
        self,
        local_path: str,
        last_sha: str,
        languages: list[str] | None,
        exclude_patterns: list[str] | None,
    ) -> list[str]:
        """Return files changed since *last_sha* via ``git diff``."""
        res = subprocess.run(
            ["git", "diff", "--name-only", last_sha, "HEAD"],
            cwd=local_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if res.returncode != 0:
            raise RuntimeError(f"git diff failed: {res.stderr.strip()}")

        root = Path(local_path)
        all_patterns = list(_SKIP_PATTERNS) + (exclude_patterns or [])
        found: list[str] = []

        for rel in res.stdout.splitlines():
            rel = rel.strip()
            if not rel:
                continue
            path = root / rel
            if not path.is_file():
                continue
            if path.suffix.lower() in _BINARY_EXTENSIONS:
                continue
            if any(fnmatch.fnmatch(path.name, pat) or fnmatch.fnmatch(rel, pat)
                   for pat in all_patterns):
                continue
            if languages:
                lang = self._chunker.detect_language(str(path))
                if lang not in languages:
                    continue
            found.append(str(path))

        return found

    def _index_files(
        self,
        files: list[str],
        local_path: str,
        repo_id: str,
        metadata: dict,
        result: IndexingResult,
    ) -> None:
        """Chunk, embed, and upsert each file.  Mutates *result* in-place."""
        root = Path(local_path)

        for idx, abs_path in enumerate(files, start=1):
            if idx % 10 == 0:
                logger.info(
                    "[%s] Progress: %d/%d files (%d chunks so far)",
                    repo_id, idx, len(files), result.total_chunks,
                )

            try:
                content = Path(abs_path).read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                result.skipped_files += 1
                result.errors.append(f"Read error {abs_path}: {exc}")
                continue

            try:
                rel_path = str(Path(abs_path).relative_to(root))
            except ValueError:
                rel_path = abs_path

            try:
                chunks = self._chunker.chunk_file(rel_path, content)
            except Exception as exc:  # noqa: BLE001
                result.skipped_files += 1
                result.errors.append(f"Chunk error {rel_path}: {exc}")
                continue

            if not chunks:
                result.skipped_files += 1
                continue

            try:
                embeddings = self._embedder.embed_chunks(chunks)
            except Exception as exc:  # noqa: BLE001
                result.skipped_files += 1
                result.errors.append(f"Embed error {rel_path}: {exc}")
                continue

            try:
                self._store.add(chunks, embeddings, metadata_extra=metadata)
            except Exception as exc:  # noqa: BLE001
                result.skipped_files += 1
                result.errors.append(f"Store error {rel_path}: {exc}")
                continue

            result.indexed_files += 1
            result.total_chunks += len(chunks)


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _get_head_sha(local_path: str) -> str:
    """Return the HEAD commit SHA, or empty string if unavailable."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=local_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return ""
