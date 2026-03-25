"""Factory functions for A2A task and push-notification stores.

Returns database-backed stores when DATABASE_URL is set, falling back to
in-memory stores for local development and CI.

The stores are passed directly to ``get_fast_api_app`` via the native
``a2a_task_store`` / ``a2a_push_config_store`` parameters, which were added
by patching the installed google-adk package (see ``scripts/patch_adk.py``).

Upstream tracking
-----------------
- PR #3839 adds ``task_store`` to ``to_a2a()``:
  https://github.com/google/adk-python/pull/3839
- ``get_fast_api_app`` extension follows the same pattern and is applied via
  ``scripts/patch_adk.py`` until ADK ships it natively.

Once google-adk >= the version that includes both params, delete
``scripts/patch_adk.py`` and remove the patch step from the dev setup.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _make_engine():
    """Create a shared SQLAlchemy AsyncEngine from DATABASE_URL."""
    from sqlalchemy.ext.asyncio import create_async_engine

    url = os.environ["DATABASE_URL"]
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgresql://") and "+asyncpg" not in url:
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return create_async_engine(url, pool_pre_ping=True)


def build_task_store():
    """Return a DatabaseTaskStore when DATABASE_URL is set, else InMemoryTaskStore."""
    from a2a.server.tasks import InMemoryTaskStore

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.info("A2A task store: InMemoryTaskStore (no DATABASE_URL)")
        return InMemoryTaskStore()

    from a2a.server.tasks import DatabaseTaskStore

    engine = _make_engine()
    logger.info("A2A task store: DatabaseTaskStore (Aurora / PostgreSQL)")
    return DatabaseTaskStore(engine, table_name="a2a_tasks")


def build_push_config_store():
    """Return a DatabasePushNotificationConfigStore when DATABASE_URL is set."""
    from a2a.server.tasks import InMemoryPushNotificationConfigStore

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return InMemoryPushNotificationConfigStore()

    from a2a.server.tasks import DatabasePushNotificationConfigStore

    engine = _make_engine()
    logger.info("A2A push-config store: DatabasePushNotificationConfigStore")
    return DatabasePushNotificationConfigStore(engine, table_name="a2a_push_configs")
