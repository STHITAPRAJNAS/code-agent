"""ChromaDB vector store wrapper for code embeddings.

Supports both persistent (on-disk) and ephemeral (in-memory) modes.
Provides upsert, query, metadata filtering, and per-repo SHA tracking.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from code_agent.indexer.chunker import CodeChunk

logger = logging.getLogger(__name__)

_INDEX_STATE_COLLECTION = "index_state"


@dataclass
class SearchResult:
    """A single result returned by :meth:`VectorStore.query`."""

    content: str
    file_path: str
    language: str
    symbol_name: str
    start_line: int
    end_line: int
    score: float
    """Similarity score in [0, 1]; higher means more similar (1 − distance)."""
    metadata: dict = field(default_factory=dict)


class VectorStore:
    """ChromaDB-backed store for code chunk embeddings.

    Example::

        store = VectorStore(mode="persistent", path="./chroma_db")
        store.add(chunks, embeddings, {"repo_id": "github/acme/api"})
        results = store.query(query_vec, n_results=5)
    """

    def __init__(
        self,
        mode: str = "persistent",
        path: str | None = None,
        collection_name: str = "code_knowledge",
    ) -> None:
        """Initialise the vector store.

        Args:
            mode:            ``"persistent"`` (on-disk) or ``"memory"`` (ephemeral).
            path:            Directory path required when *mode* is ``"persistent"``.
            collection_name: Name of the ChromaDB collection for code chunks.

        Raises:
            ValueError: If *mode* is ``"persistent"`` but *path* is not provided.
        """
        import chromadb  # type: ignore[import]

        if mode == "persistent":
            if not path:
                raise ValueError("path is required for persistent VectorStore mode")
            self._client = chromadb.PersistentClient(path=path)
        elif mode == "memory":
            self._client = chromadb.EphemeralClient()
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'persistent' or 'memory'.")

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._state_collection = self._client.get_or_create_collection(
            name=_INDEX_STATE_COLLECTION,
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add(
        self,
        chunks: list[CodeChunk],
        embeddings: list[list[float]],
        metadata_extra: dict[str, Any] | None = None,
    ) -> None:
        """Upsert *chunks* and their *embeddings* into the collection.

        IDs are derived as ``{repo_id}::{file_path}::{start_line}`` using
        values from *metadata_extra* (``repo_id`` key).  Re-indexing the
        same chunks is safe; existing documents are replaced.

        Args:
            chunks:         Code chunks to store.
            embeddings:     One embedding vector per chunk (same order).
            metadata_extra: Additional metadata merged into every chunk's
                            metadata dict (e.g. ``{"repo_id": "github/org/repo"}``).

        Raises:
            ValueError: If *chunks* and *embeddings* lengths differ.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must have equal length"
            )
        if not chunks:
            return

        extra = metadata_extra or {}
        repo_id = extra.get("repo_id", "unknown")

        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict] = []

        for chunk in chunks:
            chunk_id = f"{repo_id}::{chunk.file_path}::{chunk.start_line}"
            ids.append(chunk_id)
            docs.append(chunk.content)

            meta: dict[str, Any] = {
                "file_path": chunk.file_path,
                "language": chunk.language,
                "chunk_type": chunk.chunk_type,
                "symbol_name": chunk.symbol_name,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "signature": chunk.signature,
            }
            meta.update(extra)
            metas.append(meta)

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=docs,
            metadatas=metas,
        )
        logger.debug("Upserted %d chunks for repo %s", len(chunks), repo_id)

    def delete_by_metadata(self, where: dict) -> int:
        """Delete all documents matching the *where* filter.

        Args:
            where: ChromaDB metadata filter (e.g. ``{"repo_id": "github/org/repo"}``).

        Returns:
            Number of documents deleted.
        """
        existing = self._collection.get(where=where, include=[])
        ids = existing.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
            logger.debug("Deleted %d chunks matching %s", len(ids), where)
        return len(ids)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict | None = None,
    ) -> list[SearchResult]:
        """Semantic nearest-neighbour search.

        Args:
            query_embedding: Query vector produced by :class:`~code_agent.indexer.embedder.CodeEmbedder`.
            n_results:       Maximum number of results to return.
            where:           Optional ChromaDB metadata filter.

        Returns:
            Results sorted by descending similarity score.
        """
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, max(1, self._collection.count())),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        try:
            raw = self._collection.query(**kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.error("VectorStore query failed: %s", exc)
            return []

        results: list[SearchResult] = []
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, distances):
            score = max(0.0, 1.0 - float(dist))
            results.append(
                SearchResult(
                    content=doc or "",
                    file_path=meta.get("file_path", ""),
                    language=meta.get("language", ""),
                    symbol_name=meta.get("symbol_name", ""),
                    start_line=int(meta.get("start_line", 0)),
                    end_line=int(meta.get("end_line", 0)),
                    score=score,
                    metadata=dict(meta),
                )
            )

        return results

    def get_collection_stats(self) -> dict:
        """Return a summary of the collection contents.

        Returns:
            Dict with ``count`` (total documents) and ``metadata`` sample.
        """
        count = self._collection.count()
        sample: list[dict] = []
        if count > 0:
            raw = self._collection.peek(limit=min(5, count))
            sample = raw.get("metadatas", [])
        return {"count": count, "metadata_sample": sample}

    # ------------------------------------------------------------------
    # Index state (per-repo last-commit SHA)
    # ------------------------------------------------------------------

    def get_last_indexed_sha(self, repo_id: str) -> str | None:
        """Return the last commit SHA recorded for *repo_id*, or ``None``.

        Args:
            repo_id: Repository identifier (e.g. ``"github/org/repo"``).
        """
        try:
            result = self._state_collection.get(ids=[_state_id(repo_id)], include=["documents"])
            docs = result.get("documents", [])
            if docs and docs[0]:
                data = json.loads(docs[0])
                return data.get("sha")
        except Exception:  # noqa: BLE001
            pass
        return None

    def save_last_indexed_sha(self, repo_id: str, sha: str) -> None:
        """Persist the last-indexed commit *sha* for *repo_id*.

        Args:
            repo_id: Repository identifier.
            sha:     Full or short commit SHA.
        """
        doc = json.dumps({"sha": sha, "repo_id": repo_id})
        self._state_collection.upsert(
            ids=[_state_id(repo_id)],
            documents=[doc],
            metadatas=[{"repo_id": repo_id}],
        )
        logger.debug("Saved last indexed SHA %s for %s", sha, repo_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state_id(repo_id: str) -> str:
    """Return a ChromaDB document ID for index-state storage."""
    return f"state::{repo_id}"
