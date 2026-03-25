"""ADK callbacks — observe, guard, and enrich every stage of agent execution.

Six callbacks covering the full lifecycle:
  before_agent / after_agent  — invocation timing and context injection
  before_model / after_model  — system-instruction variable substitution,
                                guardrails, and token-usage logging
  before_tool  / after_tool   — dangerous-command blocking and audit trail

ADK 1.x callback signatures
────────────────────────────
  before_agent_callback(ctx: CallbackContext) -> Optional[types.Content]
  after_agent_callback(ctx: CallbackContext)  -> Optional[types.Content]
  before_model_callback(ctx: CallbackContext, llm_request: LlmRequest)
                                              -> Optional[LlmResponse]
  after_model_callback(ctx: CallbackContext,  llm_response: LlmResponse)
                                              -> Optional[LlmResponse]
  before_tool_callback(tool: BaseTool, args: dict, tool_ctx: ToolContext)
                                              -> Optional[dict]
  after_tool_callback(tool: BaseTool, args: dict, tool_ctx: ToolContext,
                      response: dict)         -> Optional[dict]

Returning None from any callback lets normal execution continue.
Returning a typed value short-circuits (skips) that stage.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    # Imported for type hints only — avoids hard failures if ADK internals
    # move between minor versions.
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.adk.tools.base_tool import BaseTool
    from google.adk.tools.tool_context import ToolContext

from code_agent.guardrails import (
    injection_guard_before_model,       # Layer 2a
    secret_redaction_after_model,       # Layer 2b
    tool_payload_guard_before_tool,     # Layer 2c
    secret_redaction_after_tool,        # Layer 2d
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Patterns blocked unconditionally in run_command / run_script
# ---------------------------------------------------------------------------
_BLOCKED_SHELL_PATTERNS: list[str] = [
    "rm -rf /",
    "rm -rf ~",
    "rm -rf $HOME",
    ":(){ :|:& };:",   # fork bomb
    "dd if=/dev/zero",
    "dd if=/dev/urandom of=/dev/",
    "mkfs",
    "> /dev/sda",
    "chmod -R 777 /",
    "chown -R root /",
]

# State key used to track cumulative LLM calls per invocation
_TOKEN_CALL_KEY = "llm_call_count"
_MAX_LLM_CALLS = 40  # hard guard against infinite loops


# ---------------------------------------------------------------------------
# 1. Before-agent callback
# ---------------------------------------------------------------------------

async def before_agent_callback(callback_context: "CallbackContext") -> None:
    """Record invocation start time and seed instruction template variables."""
    callback_context.state["invocation_start_ms"] = int(time.time() * 1000)
    callback_context.state[_TOKEN_CALL_KEY] = 0  # reset call counter
    # Seed keys used in the instruction template so ADK's inject_session_state
    # doesn't raise KeyError when they are absent from state.
    callback_context.state.setdefault("custom_instructions", "")
    callback_context.state.setdefault("active_repo", "")
    callback_context.state.setdefault("active_workspace", "")
    logger.info(
        "[agent:%s] invocation=%s started",
        callback_context.agent_name,
        callback_context.invocation_id,
    )
    return None


# ---------------------------------------------------------------------------
# 2. After-agent callback
# ---------------------------------------------------------------------------

async def after_agent_callback(callback_context: "CallbackContext") -> None:
    """Log elapsed time for the invocation."""
    start = callback_context.state.get("invocation_start_ms", 0)
    elapsed = int(time.time() * 1000) - start
    calls = callback_context.state.get(_TOKEN_CALL_KEY, 0)
    logger.info(
        "[agent:%s] invocation=%s completed in %dms (%d LLM call(s))",
        callback_context.agent_name,
        callback_context.invocation_id,
        elapsed,
        calls,
    )
    return None


# ---------------------------------------------------------------------------
# 3. Before-model callback
# ---------------------------------------------------------------------------

async def before_model_callback(
    callback_context: "CallbackContext",
    llm_request: "LlmRequest",
) -> "Optional[LlmResponse]":
    """Three responsibilities before each LLM call:

    1. Inject session-state variables into the system instruction, filling
       the {active_repo}, {active_workspace}, {custom_instructions}
       placeholders defined in agent.py.
    2. Guard against prompt-injection attempts in the latest user turn.
    3. Enforce a per-invocation LLM-call limit to prevent infinite loops.
    """
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types as genai_types

    # ── 3a. Inject context into system instruction ────────────────────────────
    _inject_context_into_system_instruction(callback_context, llm_request)

    # ── 3b. Prompt-injection guard (Layer 2a) ─────────────────────────────────
    injection_response = injection_guard_before_model(callback_context, llm_request)
    if injection_response is not None:
        return injection_response

    # ── 3c. LLM call-count guard ──────────────────────────────────────────────
    call_count = callback_context.state.get(_TOKEN_CALL_KEY, 0) + 1
    callback_context.state[_TOKEN_CALL_KEY] = call_count
    if call_count > _MAX_LLM_CALLS:
        logger.error(
            "[model-guard] LLM call limit (%d) exceeded — aborting invocation",
            _MAX_LLM_CALLS,
        )
        abort = LlmResponse(
            content=genai_types.Content(
                role="model",
                parts=[genai_types.Part(
                    text=(
                        "I've reached the maximum number of reasoning steps for "
                        "this task. Please break the request into smaller pieces."
                    )
                )],
            )
        )
        return abort

    logger.debug(
        "[model:%s] LLM call #%d for invocation=%s",
        callback_context.agent_name,
        call_count,
        callback_context.invocation_id,
    )
    return None


# ---------------------------------------------------------------------------
# 4. After-model callback
# ---------------------------------------------------------------------------

async def after_model_callback(
    callback_context: "CallbackContext",
    llm_response: "LlmResponse",
) -> "Optional[LlmResponse]":
    """Log token usage, then apply Layer 2b secret redaction."""
    try:
        usage = getattr(llm_response, "usage_metadata", None)
        if usage:
            logger.info(
                "[model:%s] tokens — prompt=%s, candidates=%s, total=%s",
                callback_context.agent_name,
                getattr(usage, "prompt_token_count", "?"),
                getattr(usage, "candidates_token_count", "?"),
                getattr(usage, "total_token_count", "?"),
            )
    except Exception:
        pass  # never let logging crash the agent
    # Layer 2b: redact any secrets that leaked into the model response.
    return secret_redaction_after_model(callback_context, llm_response)


# ---------------------------------------------------------------------------
# 5. Before-tool callback
# ---------------------------------------------------------------------------

async def before_tool_callback(
    tool: "BaseTool",
    args: dict[str, Any],
    tool_context: "ToolContext",
) -> Optional[dict[str, Any]]:
    """Block dangerous shell commands before the tool executes.

    For run_command and run_script, we check the command string against a
    known-bad pattern list.  Any match causes an immediate error response
    (the tool is never invoked).
    """
    tool_name: str = getattr(tool, "name", str(tool))

    # Layer 2c: block any tool call with dangerous payload in its string args.
    payload_block = tool_payload_guard_before_tool(tool, args, tool_context)
    if payload_block is not None:
        return payload_block

    if tool_name in ("run_command", "run_script"):
        cmd: str = args.get("command", "") or args.get("script", "")
        for pattern in _BLOCKED_SHELL_PATTERNS:
            if pattern in cmd:
                logger.warning(
                    "[tool-guard] BLOCKED dangerous command in '%s': %r",
                    tool_name,
                    cmd[:120],
                )
                return {
                    "error": (
                        f"Blocked: potentially destructive command pattern "
                        f"'{pattern}' detected. If this was intentional, "
                        f"please confirm explicitly."
                    )
                }

    logger.debug("[tool:%s] args=%s", tool_name, _truncate_args(args))
    return None


# ---------------------------------------------------------------------------
# 6. After-tool callback
# ---------------------------------------------------------------------------

async def after_tool_callback(
    tool: "BaseTool",
    args: dict[str, Any],
    tool_context: "ToolContext",
    tool_response: dict[str, Any],
) -> "Optional[dict[str, Any]]":
    """Audit trail + Layer 2d: redact secrets from tool output before the LLM sees it."""
    tool_name: str = getattr(tool, "name", str(tool))
    audit: list[dict] = tool_context.state.setdefault("tool_audit", [])
    audit.append(
        {
            "tool": tool_name,
            "args": _truncate_args(args),
            "ts_ms": int(time.time() * 1000),
        }
    )
    logger.debug("[tool:%s] completed — response len=%d", tool_name, len(str(tool_response)))
    # Layer 2d: redact any credentials that appeared in tool output.
    return secret_redaction_after_tool(tool, args, tool_context, tool_response)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate_args(args: dict[str, Any], max_len: int = 120) -> dict[str, str]:
    """Return a copy of *args* with each value truncated to *max_len* chars."""
    return {k: str(v)[:max_len] for k, v in args.items()}


def _inject_context_into_system_instruction(
    callback_context: "CallbackContext",
    llm_request: "LlmRequest",
) -> None:
    """Fill {active_repo}, {active_workspace}, {custom_instructions} placeholders.

    The root agent's instruction string contains these template variables.
    ADK sets them verbatim as the system instruction; we replace them here
    using values stored in session state so every LLM call has current context.

    Callers can set state keys before or during a conversation:
        session.state["active_repo"]          = "/path/to/repo  (or VCS URL)"
        session.state["active_workspace"]     = "workspace-slug"
        session.state["custom_instructions"]  = "Focus on the auth module."
    """
    active_repo = callback_context.state.get("active_repo") or "Not set — ask the user for a repository path or URL"
    active_workspace = callback_context.state.get("active_workspace") or "Not set"
    custom_instructions = callback_context.state.get("custom_instructions") or ""

    replacements = {
        "{active_repo}": f"Active repository: {active_repo}",
        "{active_workspace}": f"Active workspace: {active_workspace}",
        "{custom_instructions}": custom_instructions,
    }

    try:
        config = getattr(llm_request, "config", None)
        if config is None:
            return
        si = getattr(config, "system_instruction", None)
        if si is None:
            return

        # si is types.Content with .parts list
        parts = getattr(si, "parts", None)
        if parts:
            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    new_text = text
                    for placeholder, value in replacements.items():
                        new_text = new_text.replace(placeholder, value)
                    if new_text != text:
                        part.text = new_text
            return

        # Fallback: si is a plain string (older ADK builds)
        if isinstance(si, str):
            new_si = si
            for placeholder, value in replacements.items():
                new_si = new_si.replace(placeholder, value)
            if new_si != si:
                config.system_instruction = new_si
    except Exception as exc:
        # Never let context injection crash the agent
        logger.debug("Context injection skipped: %s", exc)
