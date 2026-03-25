import logging
from google.adk.memory import BaseMemoryService, InMemoryMemoryService

logger = logging.getLogger(__name__)

def create_memory_service() -> BaseMemoryService:
    """
    Returns the ADK MemoryService.

    Both local and EKS use InMemoryMemoryService for ADK's session memory
    abstraction. Long-term semantic code memory is handled by RAGStore
    (LlamaIndex + pgvector on EKS, SimpleVectorStore locally).
    """
    logger.info("Memory: using InMemoryMemoryService")
    return InMemoryMemoryService()
