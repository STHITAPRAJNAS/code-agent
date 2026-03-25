import os
import logging
from google.adk.sessions import BaseSessionService, InMemorySessionService

logger = logging.getLogger(__name__)

def create_session_service() -> BaseSessionService:
    """
    Returns the appropriate ADK SessionService based on DEPLOYMENT_MODE.

    Local (default):
        InMemorySessionService — zero deps, instant startup.

    EKS:
        DatabaseSessionService backed by Aurora PostgreSQL via asyncpg.
        Requires DATABASE_URL env var:
          postgresql+asyncpg://user:pass@aurora-host:5432/codeagent

    The DatabaseSessionService uses ADK's built-in SQLAlchemy-based
    implementation which stores sessions in a `sessions` table.
    """
    mode = os.getenv("DEPLOYMENT_MODE", "local").lower()

    if mode == "local":
        logger.info("Storage: using InMemorySessionService (local mode)")
        return InMemorySessionService()

    # EKS / cloud mode
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.warning(
            "DEPLOYMENT_MODE=eks but DATABASE_URL is not set — "
            "falling back to InMemorySessionService"
        )
        return InMemorySessionService()

    try:
        from google.adk.sessions import DatabaseSessionService
        logger.info("Storage: using DatabaseSessionService (EKS mode) → %s",
                    db_url.split("@")[-1])  # log host only, not credentials
        return DatabaseSessionService(db_url=db_url)
    except Exception as e:
        logger.error("Failed to create DatabaseSessionService: %s — falling back to InMemory", e)
        return InMemorySessionService()
