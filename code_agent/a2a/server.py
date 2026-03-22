"""
DEPRECATED — This module is no longer used and will be deleted.
===============================================================

All functionality has been migrated to ADK's ``get_fast_api_app`` in
``main.py``, extended with persistent stores via ``code_agent/a2a/stores.py``.

Why it was kept originally:
  • ADK v1.x hardcoded ``InMemoryTaskStore`` with no injection point.
  • Custom tasks/submit (fire-and-forget), tasks/resume (HITL), and
    API-key auth middleware were not available in get_fast_api_app.

Current state (see main.py + code_agent/a2a/stores.py):
  • Persistent tasks/push-notification stores via patch_adk_stores(),
    which monkey-patches the a2a.server.tasks module namespace before
    get_fast_api_app is called so DatabaseTaskStore is used when
    DATABASE_URL is set.
  • Session persistence via session_service_uri=DATABASE_URL.
  • ADK's built-in A2A handler covers tasks/send + tasks/sendSubscribe.

Upstream tracking:
  ADK PR #3839 — adds task_store param to to_a2a() (open, unmerged as of
  2026-03).  Once get_fast_api_app exposes a native a2a_task_store param,
  code_agent/a2a/stores.py can also be removed.

TODO: delete this file once the migration is confirmed stable in EKS.
----------------------------------------------------------------------

FastAPI A2A server — JSON-RPC 2.0 dispatcher with SSE streaming support.

Improvements over v0.2.0
─────────────────────────
• tasks/submit  — fire-and-forget endpoint backed by the TaskQueue worker
  pool; returns immediately with a SUBMITTED task so HTTP requests never
  block for the duration of an LLM call.
• tasks/resume  — resumes a paused INPUT_REQUIRED task (human-in-the-loop).
  Send the human's approval/rejection payload; the agent continues from
  exactly where it paused using the original invocation_id.
• API-key auth  — optional X-API-Key middleware; enable by setting API_KEY
  in the environment.  Safe default: auth is disabled if the key is unset.
• TaskQueue lifespan — workers are started at app startup and drained at
  shutdown via FastAPI's lifespan context manager.
• tasks/send    — still available for synchronous (blocking) invocation.
• tasks/sendSubscribe — SSE stream; now yields granular tool-progress events.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from code_agent.a2a.agent_card import build_agent_card
from code_agent.a2a.models import (
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcResponse,
    RpcError,
)
from code_agent.a2a.runner import AgentRunner
from code_agent.a2a.task_queue import TaskQueue

logger = logging.getLogger(__name__)

load_dotenv()


# ── Application factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    from code_agent.agent import root_agent  # lazy import to avoid circular deps

    runner = AgentRunner(root_agent)

    num_workers = int(os.getenv("TASK_QUEUE_WORKERS", "4"))
    task_queue = TaskQueue(runner, num_workers=num_workers)

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # noqa: ANN001
        """Start background workers on startup; drain them on shutdown."""
        await task_queue.start()
        logger.info("TaskQueue started (%d workers)", num_workers)
        yield
        await task_queue.stop()
        logger.info("TaskQueue stopped")

    app = FastAPI(
        title="Code Agent",
        description="A Staff Software Engineer AI agent (A2A compatible)",
        version="0.2.1",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Middleware ──────────────────────────────────────────────────────────────

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _register_auth_middleware(app)

    # ── Routes ──────────────────────────────────────────────────────────────────

    @app.get("/.well-known/agent.json", tags=["A2A"])
    async def agent_card(request: Request):
        """Return the A2A agent card for discovery."""
        base_url = str(request.base_url).rstrip("/")
        card = build_agent_card(base_url=base_url)
        return JSONResponse(card.model_dump(mode="json"))

    @app.get("/health", tags=["Ops"])
    async def health():
        """Health check."""
        return {"status": "ok", "version": "0.2.1", "agent": "code_agent"}

    @app.post("/", tags=["A2A"])
    async def rpc_handler(request: Request):
        """Main JSON-RPC 2.0 dispatcher for A2A methods."""
        try:
            body = await request.json()
        except Exception:
            resp = JsonRpcResponse(error=RpcError.PARSE_ERROR)
            return JSONResponse(resp.model_dump(mode="json"), status_code=400)

        try:
            rpc_req = JsonRpcRequest.model_validate(body)
        except Exception:
            resp = JsonRpcResponse(error=RpcError.INVALID_REQUEST)
            return JSONResponse(resp.model_dump(mode="json"), status_code=400)

        method = rpc_req.method
        params = rpc_req.params or {}

        # ── tasks/submit (async, non-blocking) ──────────────────────────────
        if method == "tasks/submit":
            try:
                user_text, session_id = _extract_message(params)
            except ValueError as e:
                return JSONResponse(
                    JsonRpcResponse(
                        id=rpc_req.id,
                        error=JsonRpcError(code=-32602, message=str(e)),
                    ).model_dump(mode="json"),
                    status_code=400,
                )
            task = await task_queue.submit(user_text, session_id)
            return JSONResponse(
                JsonRpcResponse(
                    id=rpc_req.id,
                    result={"task": task.model_dump(mode="json")},
                ).model_dump(mode="json")
            )

        # ── tasks/send (blocking, synchronous) ──────────────────────────────
        if method == "tasks/send":
            try:
                user_text, session_id = _extract_message(params)
            except ValueError as e:
                return JSONResponse(
                    JsonRpcResponse(
                        id=rpc_req.id,
                        error=JsonRpcError(code=-32602, message=str(e)),
                    ).model_dump(mode="json"),
                    status_code=400,
                )
            task = await runner.invoke(user_text, session_id)
            return JSONResponse(
                JsonRpcResponse(
                    id=rpc_req.id,
                    result={"task": task.model_dump(mode="json")},
                ).model_dump(mode="json")
            )

        # ── tasks/sendSubscribe (SSE streaming) ─────────────────────────────
        if method == "tasks/sendSubscribe":
            try:
                user_text, session_id = _extract_message(params)
            except ValueError as e:
                return JSONResponse(
                    JsonRpcResponse(
                        id=rpc_req.id,
                        error=JsonRpcError(code=-32602, message=str(e)),
                    ).model_dump(mode="json"),
                    status_code=400,
                )

            async def sse_generator() -> AsyncGenerator[dict, None]:
                async for task_snapshot in runner.stream(user_text, session_id):
                    payload = JsonRpcResponse(
                        id=rpc_req.id,
                        result={"task": task_snapshot.model_dump(mode="json")},
                    )
                    yield {"data": payload.model_dump_json()}

            return EventSourceResponse(sse_generator())

        # ── tasks/get ────────────────────────────────────────────────────────
        if method == "tasks/get":
            task_id = params.get("id") or params.get("taskId", "")
            if not task_id:
                return JSONResponse(
                    JsonRpcResponse(
                        id=rpc_req.id,
                        error=JsonRpcError(code=-32602, message="Missing task id"),
                    ).model_dump(mode="json"),
                    status_code=400,
                )
            task = runner.get_task(task_id)
            if task is None:
                return JSONResponse(
                    JsonRpcResponse(id=rpc_req.id, error=RpcError.TASK_NOT_FOUND).model_dump(mode="json"),
                    status_code=404,
                )
            return JSONResponse(
                JsonRpcResponse(
                    id=rpc_req.id,
                    result={"task": task.model_dump(mode="json")},
                ).model_dump(mode="json")
            )

        # ── tasks/cancel ─────────────────────────────────────────────────────
        if method == "tasks/cancel":
            task_id = params.get("id") or params.get("taskId", "")
            if not task_id:
                return JSONResponse(
                    JsonRpcResponse(
                        id=rpc_req.id,
                        error=JsonRpcError(code=-32602, message="Missing task id"),
                    ).model_dump(mode="json"),
                    status_code=400,
                )
            found = runner.cancel_task(task_id)
            if not found:
                return JSONResponse(
                    JsonRpcResponse(id=rpc_req.id, error=RpcError.TASK_NOT_FOUND).model_dump(mode="json"),
                    status_code=404,
                )
            task = runner.get_task(task_id)
            return JSONResponse(
                JsonRpcResponse(
                    id=rpc_req.id,
                    result={"task": task.model_dump(mode="json")},
                ).model_dump(mode="json")
            )

        # ── tasks/resume (human-in-the-loop) ────────────────────────────────
        if method == "tasks/resume":
            # Required: taskId, invocationId, functionCallId, toolName, response
            task_id = params.get("taskId") or params.get("id", "")
            invocation_id = params.get("invocationId") or params.get("invocation_id", "")
            function_call_id = params.get("functionCallId") or params.get("function_call_id", "")
            tool_name = params.get("toolName") or params.get("tool_name", "")
            response_payload = params.get("response") or {}

            missing = [
                f for f, v in [
                    ("taskId", task_id),
                    ("invocationId", invocation_id),
                    ("functionCallId", function_call_id),
                ]
                if not v
            ]
            if missing:
                return JSONResponse(
                    JsonRpcResponse(
                        id=rpc_req.id,
                        error=JsonRpcError(
                            code=-32602,
                            message=f"Missing required params: {', '.join(missing)}",
                        ),
                    ).model_dump(mode="json"),
                    status_code=400,
                )

            task = runner.get_task(task_id)
            if task is None:
                return JSONResponse(
                    JsonRpcResponse(id=rpc_req.id, error=RpcError.TASK_NOT_FOUND).model_dump(mode="json"),
                    status_code=404,
                )

            from code_agent.a2a.models import TaskState
            if task.status.state != TaskState.INPUT_REQUIRED:
                return JSONResponse(
                    JsonRpcResponse(
                        id=rpc_req.id,
                        error=JsonRpcError(
                            code=-32003,
                            message=f"Task is in state '{task.status.state}', not 'input-required'",
                        ),
                    ).model_dump(mode="json"),
                    status_code=400,
                )

            try:
                resumed_task = await runner.resume(
                    task_id=task_id,
                    invocation_id=invocation_id,
                    function_call_id=function_call_id,
                    tool_name=tool_name,
                    response_payload=response_payload,
                )
            except Exception as exc:
                logger.exception("Resume failed for task %s: %s", task_id, exc)
                return JSONResponse(
                    JsonRpcResponse(
                        id=rpc_req.id,
                        error=JsonRpcError(code=-32603, message=f"Resume error: {exc}"),
                    ).model_dump(mode="json"),
                    status_code=500,
                )

            return JSONResponse(
                JsonRpcResponse(
                    id=rpc_req.id,
                    result={"task": resumed_task.model_dump(mode="json")},
                ).model_dump(mode="json")
            )

        # ── Unknown method ───────────────────────────────────────────────────
        return JSONResponse(
            JsonRpcResponse(id=rpc_req.id, error=RpcError.METHOD_NOT_FOUND).model_dump(mode="json"),
            status_code=404,
        )

    return app


# ── Auth middleware ────────────────────────────────────────────────────────────

def _register_auth_middleware(app: FastAPI) -> None:
    """Register optional API-key middleware.

    If API_KEY env var is set, every request must carry:
        X-API-Key: <value>

    Health check and agent card endpoints are always exempt so orchestrators
    can discover the agent without credentials.
    """
    from code_agent.config import get_settings

    api_key = get_settings().API_KEY
    if not api_key:
        logger.info("Auth: API_KEY not set — server is open (no auth)")
        return

    _EXEMPT_PATHS = {"/.well-known/agent.json", "/health", "/docs", "/redoc", "/openapi.json"}

    @app.middleware("http")
    async def api_key_auth(request: Request, call_next):
        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)
        provided = request.headers.get("X-API-Key", "")
        if provided != api_key:
            logger.warning("Auth: rejected request from %s — bad or missing X-API-Key", request.client)
            return JSONResponse(
                {"error": "Unauthorized — provide a valid X-API-Key header"},
                status_code=401,
            )
        return await call_next(request)

    logger.info("Auth: API key auth enabled")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_message(params: dict) -> tuple[str, str | None]:
    """Extract user text and optional session_id from JSON-RPC params.

    Supports both A2A spec format and a convenience 'text' shorthand.
    """
    session_id: str | None = params.get("sessionId") or params.get("session_id")

    message = params.get("message")
    if message and isinstance(message, dict):
        parts = message.get("parts", [])
        texts = [
            p.get("text", "")
            for p in parts
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        if texts:
            return "\n".join(texts), session_id

    if "text" in params:
        return str(params["text"]), session_id

    raise ValueError("params must contain message.parts[].text or a 'text' field")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Start the FastAPI A2A server."""
    load_dotenv()
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "8000"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()
