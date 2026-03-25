"""Deployment-aware storage layer.

DEPLOYMENT_MODE=local  → InMemorySessionService + SimpleVectorStore (RAM)
DEPLOYMENT_MODE=eks    → DatabaseSessionService (Aurora PG) + PGVectorStore
"""
from code_agent.storage.session_factory import create_session_service
from code_agent.storage.rag_store import RAGStore
from code_agent.storage.memory_factory import create_memory_service
