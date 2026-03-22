"""Persistent A2A task and push-notification stores.

ADK's ``get_fast_api_app`` hardcodes ``InMemoryTaskStore()`` with no injection
point (as of v1.27.x).  PR #3839 on google/adk-python adds a ``task_store``
parameter to ``to_a2a()`` but not yet to ``get_fast_api_app``.

Interim workaround
------------------
``patch_adk_stores()`` replaces the ``InMemoryTaskStore`` and
``InMemoryPushNotificationConfigStore`` symbols **inside the
``a2a.server.tasks`` module namespace** before ``get_fast_api_app`` is called.
ADK's import statement ``from a2a.server.tasks import InMemoryTaskStore``
executes at function-call time (inside ``if a2a:`` block), so it picks up
our replacement transparently.

Remove this module and the ``patch_adk_stores()`` call in ``main.py`` once
ADK exposes a native ``a2a_task_store`` parameter in ``get_fast_api_app``.
Track: https://github.com/google/adk-python/pull/3839
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _make_engine():
    """Create a single shared SQLAlchemy AsyncEngine from DATABASE_URL."""
    from sqlalchemy.ext.asyncio import create_async_engine

    url = os.environ["DATABASE_URL"]
    # asyncpg driver requires postgresql+asyncpg:// scheme
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgresql://") and "+asyncpg" not in url:
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)

    return create_async_engine(url, pool_pre_ping=True)


def patch_adk_stores() -> None:
    """Monkey-patch ADK's in-memory A2A stores with database-backed ones.

    When ``DATABASE_URL`` is set the following substitutions are made inside
    the ``a2a.server.tasks`` module namespace so that ``get_fast_api_app``
    picks them up transparently:

    - ``InMemoryTaskStore``              → ``DatabaseTaskStore``       (table: a2a_tasks)
    - ``InMemoryPushNotificationConfigStore`` → ``DatabasePushNotificationConfigStore``
                                                               (table: a2a_push_configs)

    When ``DATABASE_URL`` is *not* set (local / CI) the defaults are left
    unchanged and tasks stay in-memory.
    """
    import a2a.server.tasks as _tasks

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.info("A2A stores: InMemory (DATABASE_URL not set)")
        return

    try:
        from a2a.server.tasks import DatabaseTaskStore
        from a2a.server.tasks import DatabasePushNotificationConfigStore
    except ImportError as exc:
        logger.warning(
            "DatabaseTaskStore unavailable (%s) — falling back to InMemory stores", exc
        )
        return

    # One shared engine for both stores (connection pool is shared).
    engine = _make_engine()

    # Replace class references so ADK's no-arg InMemoryTaskStore() call works.
    _tasks.InMemoryTaskStore = lambda: DatabaseTaskStore(  # type: ignore[attr-defined]
        engine, table_name="a2a_tasks"
    )
    _tasks.InMemoryPushNotificationConfigStore = (  # type: ignore[attr-defined]
        lambda: DatabasePushNotificationConfigStore(engine, table_name="a2a_push_configs")
    )

    logger.info(
        "A2A stores patched: DatabaseTaskStore + DatabasePushNotificationConfigStore "
        "(Aurora / PostgreSQL)"
    )
