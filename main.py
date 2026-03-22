"""Entry point for the Code Agent server.

Uses ADK's ``get_fast_api_app`` to create a FastAPI application that
provides both:
  - ADK dev UI  at  /dev-ui/
  - A2A protocol at /a2a/code_agent/  (agent card at
                    /a2a/code_agent/.well-known/agent.json)

Services are selected by DEPLOYMENT_MODE env var:
  local (default) — InMemorySessionService + InMemoryTaskStore, no external deps
  eks             — DatabaseSessionService + DatabaseTaskStore backed by Aurora
                    (requires DATABASE_URL)

A2A task/push-notification stores
----------------------------------
ADK v1.27.x hardcodes ``InMemoryTaskStore`` in ``get_fast_api_app`` with no
injection point.  ``patch_adk_stores()`` (see ``code_agent/a2a/stores.py``)
monkey-patches the ``a2a.server.tasks`` module namespace before
``get_fast_api_app`` is called so that DatabaseTaskStore and
DatabasePushNotificationConfigStore are used when DATABASE_URL is set.

Remove the patch once ADK exposes a native parameter.
Track: https://github.com/google/adk-python/pull/3839

Start the server:
  uv run python main.py
  # or
  uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).parent
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv(_PROJECT_ROOT / ".env.local", override=True)

# Patch A2A task/push-notification stores BEFORE importing get_fast_api_app so
# that the monkey-patch is in place when ADK's `from a2a.server.tasks import
# InMemoryTaskStore` executes inside get_fast_api_app at call time.
from code_agent.a2a.stores import patch_adk_stores  # noqa: E402
patch_adk_stores()

from google.adk.cli.fast_api import get_fast_api_app  # noqa: E402

logger = logging.getLogger(__name__)

# ── Service URIs ───────────────────────────────────────────────────────────────

def _session_uri() -> str | None:
    """Return the DB URI for DatabaseSessionService, or None for InMemory."""
    mode = os.getenv("DEPLOYMENT_MODE", "local").lower()
    if mode != "local":
        uri = os.getenv("DATABASE_URL")
        if not uri:
            logger.warning(
                "DEPLOYMENT_MODE=%s but DATABASE_URL is not set — "
                "falling back to InMemorySessionService",
                mode,
            )
        return uri
    return None  # ADK will use InMemorySessionService


# ── Application ────────────────────────────────────────────────────────────────

_AGENTS_DIR = str(Path(__file__).parent)   # project root — contains code_agent/
_HOST = os.getenv("APP_HOST", "0.0.0.0")
_PORT = int(os.getenv("APP_PORT", "8000"))

app = get_fast_api_app(
    agents_dir=_AGENTS_DIR,
    session_service_uri=_session_uri(),
    web=True,          # serve ADK dev UI at /dev-ui/
    a2a=True,          # enable A2A protocol at /a2a/{agent_name}/
    host=_HOST,
    port=_PORT,
    allow_origins=["*"],
)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    uvicorn.run(
        "main:app",
        host=_HOST,
        port=_PORT,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()
