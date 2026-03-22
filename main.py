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
``get_fast_api_app`` now accepts ``a2a_task_store`` and ``a2a_push_config_store``
parameters (added via ``scripts/patch_adk.py``; follows ADK PR #3839).
``build_task_store()`` / ``build_push_config_store()`` in ``stores.py`` return
the right backend based on DATABASE_URL.

Cleanup: once google-adk ships these params natively, delete
``scripts/patch_adk.py`` and remove the patch step from dev setup.

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

from google.adk.cli.fast_api import get_fast_api_app  # noqa: E402
from code_agent.a2a.stores import build_push_config_store, build_task_store  # noqa: E402

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
    a2a_task_store=build_task_store(),
    a2a_push_config_store=build_push_config_store(),
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
