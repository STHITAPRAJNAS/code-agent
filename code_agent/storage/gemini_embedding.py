"""Custom LlamaIndex embedding using google-genai (ADK-compatible).

Replaces llama-index-embeddings-google which pins google-generativeai<0.6
(incompatible with google-adk>=1.27). Uses the google-genai SDK that ADK
itself bundles, so no extra dependency is needed.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gemini-embedding-001"


class GeminiEmbedding(BaseEmbedding):
    """LlamaIndex BaseEmbedding backed by the google-genai SDK.

    Uses the same ``google.genai`` package that google-adk installs, so
    there is no conflict with the old ``google-generativeai`` package.

    Args:
        api_key:         Google API key (falls back to GOOGLE_API_KEY env var).
        model_name:      Gemini embedding model (default: gemini-embedding-001).
        embed_batch_size: Max texts per API call; default 50.
    """

    _client: Any = PrivateAttr()

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = _DEFAULT_MODEL,
        embed_batch_size: int = 50,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            **kwargs,
        )
        import google.genai as genai  # bundled by google-adk

        resolved_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self._client = genai.Client(api_key=resolved_key)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _embed_texts(self, texts: List[str], task_type: str) -> List[List[float]]:
        """Embed a list of texts with the given task type."""
        import google.genai.types as genai_types

        vectors: List[List[float]] = []
        for text in texts:
            response = self._client.models.embed_content(
                model=self.model_name,
                contents=text,
                config=genai_types.EmbedContentConfig(task_type=task_type),
            )
            vectors.append(list(response.embeddings[0].values))
        return vectors

    # ── BaseEmbedding interface ────────────────────────────────────────────────

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed_texts([query], "RETRIEVAL_QUERY")[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed_texts([text], "RETRIEVAL_DOCUMENT")[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding for indexing — uses RETRIEVAL_DOCUMENT task type."""
        return self._embed_texts(texts, "RETRIEVAL_DOCUMENT")

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)
