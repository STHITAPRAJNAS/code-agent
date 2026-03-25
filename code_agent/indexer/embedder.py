"""Google Gemini embedding client for code chunks.

Wraps ``google.genai`` (the new unified SDK bundled with google-adk) to
produce embeddings suitable for semantic code search, with exponential-
backoff retry on rate-limit errors.  Uses the Gemini API key
(GOOGLE_API_KEY) — no GCP services.
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from code_agent.indexer.chunker import CodeChunk

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "models/gemini-embedding-001"
_DEFAULT_BATCH_SIZE = 50
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds


class CodeEmbedder:
    """Generates vector embeddings for code chunks using Google Gemini.

    Example::

        embedder = CodeEmbedder(api_key="AIza...")
        vectors = embedder.embed_chunks(chunks)
        query_vec = embedder.embed_query("find authentication functions")
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        """Initialise the embedder.

        Args:
            api_key:    Google API key. Falls back to ``GOOGLE_API_KEY``
                        environment variable if not provided.
            model:      Gemini embedding model name.
            batch_size: Number of chunks to embed per API call.

        Raises:
            RuntimeError: If no API key is available.
        """
        resolved_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "Google API key is required. Set GOOGLE_API_KEY or pass api_key=."
            )

        import google.genai as genai  # bundled by google-adk

        self._client = genai.Client(api_key=resolved_key)
        self.model = model
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_chunks(self, chunks: list[CodeChunk]) -> list[list[float]]:
        """Embed a list of :class:`~code_agent.indexer.chunker.CodeChunk` objects.

        Chunks are batched to stay within API request limits. Each chunk
        is prefixed with ``file_path::symbol_name\\n`` so the model has
        location context during indexing.

        Args:
            chunks: Source code chunks to embed.

        Returns:
            A list of float vectors, one per chunk, in the same order.
        """
        if not chunks:
            return []

        texts = [
            f"{chunk.file_path}::{chunk.symbol_name}\n{chunk.content}"
            for chunk in chunks
        ]

        all_embeddings: list[list[float]] = []

        for batch_start in range(0, len(texts), self.batch_size):
            batch = texts[batch_start: batch_start + self.batch_size]
            batch_embeddings = self._embed_with_retry(batch, task_type="RETRIEVAL_DOCUMENT")
            all_embeddings.extend(batch_embeddings)
            logger.debug(
                "Embedded batch %d–%d of %d chunks",
                batch_start,
                batch_start + len(batch) - 1,
                len(texts),
            )

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query string.

        Uses ``RETRIEVAL_QUERY`` task type so the model optimises for
        asymmetric retrieval (query vs. document).

        Args:
            query: Natural-language or code search query.

        Returns:
            A single embedding vector as a list of floats.
        """
        results = self._embed_with_retry([query], task_type="RETRIEVAL_QUERY")
        return results[0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_with_retry(
        self, texts: list[str], task_type: str
    ) -> list[list[float]]:
        """Call the Gemini embedding API with exponential-backoff retry.

        Args:
            texts:     Texts to embed (one API call per text).
            task_type: ``"RETRIEVAL_DOCUMENT"`` or ``"RETRIEVAL_QUERY"``.

        Returns:
            List of embedding vectors in the same order as *texts*.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        import google.genai.types as genai_types

        all_vectors: list[list[float]] = []

        for text in texts:
            last_error: Exception | None = None
            for attempt in range(_MAX_RETRIES):
                try:
                    response = self._client.models.embed_content(
                        model=self.model,
                        contents=text,
                        config=genai_types.EmbedContentConfig(task_type=task_type),
                    )
                    all_vectors.append(list(response.embeddings[0].values))
                    break

                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    err_str = str(exc).lower()
                    is_rate_limit = (
                        "429" in err_str
                        or "rate" in err_str
                        or "quota" in err_str
                        or "resource_exhausted" in err_str
                    )
                    if is_rate_limit and attempt < _MAX_RETRIES - 1:
                        delay = _RETRY_BASE_DELAY * (2 ** attempt)
                        logger.warning(
                            "Rate limit hit (attempt %d/%d) — sleeping %.1fs: %s",
                            attempt + 1, _MAX_RETRIES, delay, exc,
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "Embedding error (attempt %d/%d): %s",
                            attempt + 1, _MAX_RETRIES, exc,
                        )
                        if not is_rate_limit:
                            raise RuntimeError(
                                f"Failed to embed after {attempt + 1} attempts: {last_error}"
                            )
            else:
                raise RuntimeError(
                    f"Failed to embed after {_MAX_RETRIES} attempts: {last_error}"
                )

        return all_vectors
