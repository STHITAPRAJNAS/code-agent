"""FastAPI A2A server — JSON-RPC 2.0 dispatcher with SSE streaming support."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator

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
    Task,
)
from code_agent.a2a.runner import AgentRunner

load_dotenv()

# ── Application factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    from code_agent.agent import root_agent  # lazy import to avoid circular deps

    runner = AgentRunner(root_agent)

    app = FastAPI(
        title="Code Agent",
        description="A Staff Software Engineer AI agent (A2A compatible)",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ─────────────────────────────────────────────────────────────────

    @app.get("/.well-known/agent.json", tags=["A2A"])
    async def agent_card(request: Request):
        """Return the A2A agent card for discovery."""
        base_url = str(request.base_url).rstrip("/")
        card = build_agent_card(base_url=base_url)
        return JSONResponse(card.model_dump(mode="json"))

    @app.get("/health", tags=["Ops"])
    async def health():
        """Health check endpoint."""
        return {"status": "ok", "version": "0.1.0", "agent": "code_agent"}

    @app.post("/", tags=["A2A"])
    async def rpc_handler(request: Request):
        """Main JSON-RPC 2.0 dispatcher for A2A methods."""
        # Parse request body
        try:
            body = await request.json()
        except Exception:
            resp = JsonRpcResponse(error=RpcError.PARSE_ERROR)
            return JSONResponse(resp.model_dump(mode="json"), status_code=400)

        # Validate JSON-RPC structure
        try:
            rpc_req = JsonRpcRequest.model_validate(body)
        except Exception:
            resp = JsonRpcResponse(error=RpcError.INVALID_REQUEST)
            return JSONResponse(resp.model_dump(mode="json"), status_code=400)

        method = rpc_req.method
        params = rpc_req.params or {}

        # ── tasks/send ──────────────────────────────────────────────────────────
        if method == "tasks/send":
            try:
                user_text, session_id = _extract_message(params)
            except ValueError as e:
                resp = JsonRpcResponse(
                    id=rpc_req.id,
                    error=JsonRpcError(code=-32602, message=str(e)),
                )
                return JSONResponse(resp.model_dump(mode="json"), status_code=400)

            task = await runner.invoke(user_text, session_id)
            resp = JsonRpcResponse(id=rpc_req.id, result={"task": task.model_dump(mode="json")})
            return JSONResponse(resp.model_dump(mode="json"))

        # ── tasks/sendSubscribe (SSE streaming) ─────────────────────────────────
        if method == "tasks/sendSubscribe":
            try:
                user_text, session_id = _extract_message(params)
            except ValueError as e:
                resp = JsonRpcResponse(
                    id=rpc_req.id,
                    error=JsonRpcError(code=-32602, message=str(e)),
                )
                return JSONResponse(resp.model_dump(mode="json"), status_code=400)

            async def sse_generator() -> AsyncGenerator[dict, None]:
                async for task_snapshot in runner.stream(user_text, session_id):
                    payload = JsonRpcResponse(
                        id=rpc_req.id,
                        result={"task": task_snapshot.model_dump(mode="json")},
                    )
                    yield {"data": payload.model_dump_json()}

            return EventSourceResponse(sse_generator())

        # ── tasks/get ───────────────────────────────────────────────────────────
        if method == "tasks/get":
            task_id = params.get("id") or params.get("taskId", "")
            if not task_id:
                resp = JsonRpcResponse(
                    id=rpc_req.id,
                    error=JsonRpcError(code=-32602, message="Missing task id"),
                )
                return JSONResponse(resp.model_dump(mode="json"), status_code=400)

            task = runner.get_task(task_id)
            if task is None:
                resp = JsonRpcResponse(id=rpc_req.id, error=RpcError.TASK_NOT_FOUND)
                return JSONResponse(resp.model_dump(mode="json"), status_code=404)

            resp = JsonRpcResponse(id=rpc_req.id, result={"task": task.model_dump(mode="json")})
            return JSONResponse(resp.model_dump(mode="json"))

        # ── tasks/cancel ────────────────────────────────────────────────────────
        if method == "tasks/cancel":
            task_id = params.get("id") or params.get("taskId", "")
            if not task_id:
                resp = JsonRpcResponse(
                    id=rpc_req.id,
                    error=JsonRpcError(code=-32602, message="Missing task id"),
                )
                return JSONResponse(resp.model_dump(mode="json"), status_code=400)

            found = runner.cancel_task(task_id)
            if not found:
                resp = JsonRpcResponse(id=rpc_req.id, error=RpcError.TASK_NOT_FOUND)
                return JSONResponse(resp.model_dump(mode="json"), status_code=404)

            task = runner.get_task(task_id)
            resp = JsonRpcResponse(id=rpc_req.id, result={"task": task.model_dump(mode="json")})
            return JSONResponse(resp.model_dump(mode="json"))

        # ── Unknown method ──────────────────────────────────────────────────────
        resp = JsonRpcResponse(id=rpc_req.id, error=RpcError.METHOD_NOT_FOUND)
        return JSONResponse(resp.model_dump(mode="json"), status_code=404)

    return app


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_message(params: dict) -> tuple[str, str | None]:
    """Extract user text and optional session_id from tasks/send params.

    Supports both A2A spec format and a convenience 'text' shorthand.
    """
    session_id: str | None = params.get("sessionId") or params.get("session_id")

    # Spec format: params.message.parts[].text
    message = params.get("message")
    if message and isinstance(message, dict):
        parts = message.get("parts", [])
        texts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") == "text"]
        if texts:
            return "\n".join(texts), session_id

    # Convenience shorthand
    if "text" in params:
        return str(params["text"]), session_id

    raise ValueError("params must contain message.parts[].text or a 'text' field")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Start the FastAPI A2A server."""
    load_dotenv()
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "8000"))
    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
