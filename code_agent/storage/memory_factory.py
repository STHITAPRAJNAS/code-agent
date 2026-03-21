import os
import logging
from google.adk.memory import BaseMemoryService, InMemoryMemoryService

logger = logging.getLogger(__name__)

def create_memory_service() -> BaseMemoryService:
    """
    Returns the appropriate ADK MemoryService based on DEPLOYMENT_MODE.

    Local: InMemoryMemoryService (in-process, lost on restart)
    EKS:   VertexAiRagMemoryService if VERTEX_RAG_CORPUS set,
           otherwise falls back to InMemoryMemoryService.

    In EKS, long-term semantic memory is primarily handled by the
    RAGStore (LlamaIndex + pgvector). This service handles ADK's
    own session memory abstraction.
    """
    mode = os.getenv("DEPLOYMENT_MODE", "local").lower()

    if mode == "local":
        logger.info("Memory: using InMemoryMemoryService (local mode)")
        return InMemoryMemoryService()

    vertex_corpus = os.getenv("VERTEX_RAG_CORPUS")
    if vertex_corpus:
        try:
            from google.adk.memory import VertexAiRagMemoryService
            logger.info("Memory: using VertexAiRagMemoryService corpus=%s", vertex_corpus)
            return VertexAiRagMemoryService(rag_corpus=vertex_corpus)
        except Exception as e:
            logger.warning("VertexAiRagMemoryService failed: %s — falling back", e)

    logger.info("Memory: using InMemoryMemoryService (EKS fallback)")
    return InMemoryMemoryService()
