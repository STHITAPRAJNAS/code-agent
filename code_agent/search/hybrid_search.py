"""Hybrid semantic + lexical search using Reciprocal Rank Fusion (RRF).

Combines vector nearest-neighbour results from :class:`VectorStore` with
literal pattern matches from :class:`LexicalSearcher`.  Results are fused
and re-ranked using the standard RRF formula.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from code_agent.search.lexical_search import LexicalSearcher
from code_agent.search.vector_store import VectorStore

if TYPE_CHECKING:
    from code_agent.indexer.embedder import CodeEmbedder

logger = logging.getLogger(__name__)

# RRF smoothing constant — standard value from the original paper
_RRF_K = 60


@dataclass
class HybridResult:
    """A search result produced by :class:`HybridSearcher`."""

    content: str
    file_path: str
    symbol_name: str
    start_line: int
    end_line: int
    language: str
    rrf_score: float
    """Combined RRF score; higher means more relevant."""
    source: str
    """``"semantic"``, ``"lexical"``, or ``"both"``."""
    metadata: dict = field(default_factory=dict)


class HybridSearcher:
    """Combines semantic (vector) and lexical (text) search results.

    Semantic search captures conceptual similarity; lexical search catches
    exact identifier matches.  Reciprocal Rank Fusion merges the two ranked
    lists without requiring score normalisation.

    Example::

        searcher = HybridSearcher(vector_store, embedder, lexical_searcher)
        results = searcher.search(
            "JWT authentication middleware",
            local_path="/repos/myapp",
            n_results=10,
        )
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: CodeEmbedder,
        lexical_searcher: LexicalSearcher,
    ) -> None:
        """Initialise the hybrid searcher.

        Args:
            vector_store:     Indexed code chunk store.
            embedder:         Embedding model used to vectorise the query.
            lexical_searcher: Text-pattern search engine.
        """
        self._vector_store = vector_store
        self._embedder = embedder
        self._lexical = lexical_searcher

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        local_path: str | None = None,
        n_results: int = 10,
        where: dict | None = None,
        semantic_weight: float = 0.7,
    ) -> list[HybridResult]:
        """Search for code relevant to *query*.

        Args:
            query:           Natural-language or code search query.
            local_path:      Local repo path for lexical search.  If ``None``,
                             only semantic results are returned.
            n_results:       Number of results to return.
            where:           ChromaDB metadata filter applied to semantic search.
            semantic_weight: Weight for semantic results in RRF (0–1).
                             Lexical weight = ``1 - semantic_weight``.

        Returns:
            :class:`HybridResult` list sorted by RRF score descending.
        """
        semantic_weight = max(0.0, min(1.0, semantic_weight))
        lexical_weight = 1.0 - semantic_weight

        # ---- Semantic search ----------------------------------------
        semantic_results = []
        try:
            query_vec = self._embedder.embed_query(query)
            semantic_results = self._vector_store.query(
                query_embedding=query_vec,
                n_results=max(n_results * 2, 20),
                where=where,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Semantic search failed: %s", exc)

        # ---- Lexical search -----------------------------------------
        lexical_results = []
        if local_path:
            try:
                lexical_results = self._lexical.search(
                    pattern=query,
                    path=local_path,
                    max_results=max(n_results * 2, 20),
                    context_lines=2,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Lexical search failed: %s", exc)

        # ---- RRF fusion ---------------------------------------------
        return self._fuse(
            semantic_results, lexical_results, n_results, semantic_weight, lexical_weight
        )

    # ------------------------------------------------------------------
    # RRF implementation
    # ------------------------------------------------------------------

    def _fuse(
        self,
        semantic_results,
        lexical_results,
        n_results: int,
        semantic_weight: float,
        lexical_weight: float,
    ) -> list[HybridResult]:
        """Merge and rank results using Reciprocal Rank Fusion.

        RRF score formula::

            score(d) = sum( weight / (k + rank(d, list)) )

        where *k* = 60 and the sum is over each ranked list in which
        document *d* appears.
        """
        # Key: (file_path, start_line) → accumulated RRF score + best result info
        scores: dict[tuple[str, int], dict] = {}

        # --- Semantic results ---
        for rank, sr in enumerate(semantic_results, start=1):
            key = (sr.file_path, sr.start_line)
            rrf = semantic_weight / (_RRF_K + rank)
            if key not in scores:
                scores[key] = {
                    "rrf_score": 0.0,
                    "source": "semantic",
                    "content": sr.content,
                    "file_path": sr.file_path,
                    "symbol_name": sr.symbol_name,
                    "start_line": sr.start_line,
                    "end_line": sr.end_line,
                    "language": sr.language,
                    "metadata": sr.metadata,
                }
            scores[key]["rrf_score"] += rrf

        # --- Lexical results ---
        for rank, lr in enumerate(lexical_results, start=1):
            # Use the exact match line as the anchor; start_line == end_line
            key = (lr.file_path, lr.line_number)
            rrf = lexical_weight / (_RRF_K + rank)
            if key not in scores:
                context_content = (
                    "\n".join(lr.context_before)
                    + ("\n" if lr.context_before else "")
                    + lr.line_content
                    + ("\n" if lr.context_after else "")
                    + "\n".join(lr.context_after)
                )
                scores[key] = {
                    "rrf_score": 0.0,
                    "source": "lexical",
                    "content": context_content,
                    "file_path": lr.file_path,
                    "symbol_name": "",
                    "start_line": lr.line_number,
                    "end_line": lr.line_number,
                    "language": "",
                    "metadata": {},
                }
            else:
                # Result appeared in both lists
                scores[key]["source"] = "both"
            scores[key]["rrf_score"] += rrf

        # --- Sort and trim ---
        ranked = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)

        return [
            HybridResult(
                content=r["content"],
                file_path=r["file_path"],
                symbol_name=r["symbol_name"],
                start_line=r["start_line"],
                end_line=r["end_line"],
                language=r["language"],
                rrf_score=r["rrf_score"],
                source=r["source"],
                metadata=r["metadata"],
            )
            for r in ranked[:n_results]
        ]
