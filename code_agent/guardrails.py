"""Guardrail functions for the code-agent.

Five layers of protection applied at different points in the ADK execution pipeline:

  Layer 2a — before_model_callback  : injection_guard_before_model
  Layer 2b — after_model_callback   : secret_redaction_after_model
  Layer 2c — before_tool_callback   : search_payload_guard_before_tool

Layers 1, 3, 4, and 5 are configured in agent.py / callbacks.py directly.

All three functions in this module are synchronous pure-logic helpers.
They have no project-internal imports and no I/O, which makes them easy
to unit-test without a live ADK runtime.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.adk.tools.base_tool import BaseTool
    from google.adk.tools.tool_context import ToolContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layer 2a — prompt-injection heuristics
# ---------------------------------------------------------------------------

_INJECTION_PHRASES: list[str] = [
    "ignore previous instructions",
    "ignore all previous",
    "disregard your instructions",
    "you are now",
    "new persona",
    "pretend you are",
    "act as an unrestricted",
    "jailbreak",
    "DAN mode",
]

# ---------------------------------------------------------------------------
# Layer 2b — secret / credential patterns to redact from model output
# ---------------------------------------------------------------------------

_SECRET_PATTERNS: list[re.Pattern] = [
    re.compile(r"api[_\-]?key\s*=\s*\S+", re.IGNORECASE),
    re.compile(r"sk-[A-Za-z0-9\-_]{20,}"),        # OpenAI-style keys
    re.compile(r"sk-ant-[A-Za-z0-9\-_]{20,}"),    # Anthropic keys
    re.compile(r"ghp_[A-Za-z0-9]{36}"),            # GitHub PATs
    re.compile(r"token\s*=\s*\S+", re.IGNORECASE),
    re.compile(r"password\s*=\s*\S+", re.IGNORECASE),
    re.compile(r"secret\s*=\s*\S+", re.IGNORECASE),
]

_REDACTION_PLACEHOLDER = "[REDACTED]"

# ---------------------------------------------------------------------------
# Layer 2c — dangerous search query patterns
# ---------------------------------------------------------------------------

_BLOCKED_SEARCH_PATTERNS: list[str] = [
    "drop table",
    "drop database",
    "exec(",
    "eval(",
    "__import__",
    "os.system",
    "subprocess",
    "; rm ",
    "&& rm ",
    "| rm ",
    "wget http",
    "curl http",
    "base64 -d",
]


# ---------------------------------------------------------------------------
# Layer 2a: injection guard — before_model_callback
# ---------------------------------------------------------------------------

def injection_guard_before_model(
    callback_context: "CallbackContext",
    llm_request: "LlmRequest",
) -> "Optional[LlmResponse]":
    """Scan user messages for prompt-injection patterns.

    Returns a canned LlmResponse to short-circuit the model call when an
    injection phrase is detected.  Returns None to allow normal execution.
    """
    # Imports are deferred to avoid hard failures if ADK internals move.
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types as genai_types

    try:
        for content in llm_request.contents or []:
            if getattr(content, "role", "") != "user":
                continue
            for part in getattr(content, "parts", []) or []:
                text: str = getattr(part, "text", "") or ""
                lower = text.lower()
                for phrase in _INJECTION_PHRASES:
                    if phrase.lower() in lower:
                        logger.warning(
                            "[guardrail-2a:%s] prompt injection detected: %r — blocking LLM call",
                            getattr(callback_context, "agent_name", "?"),
                            phrase,
                        )
                        return LlmResponse(
                            content=genai_types.Content(
                                role="model",
                                parts=[genai_types.Part(
                                    text=(
                                        "I'm unable to process that request as it appears "
                                        "to attempt to override my instructions."
                                    )
                                )],
                            )
                        )
    except Exception as exc:
        logger.debug("[guardrail-2a] injection check skipped: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Layer 2b: secret redaction — after_model_callback
# ---------------------------------------------------------------------------

def secret_redaction_after_model(
    callback_context: "CallbackContext",
    llm_response: "LlmResponse",
) -> "Optional[LlmResponse]":
    """Redact credential-like patterns from model output.

    Constructs and returns a new LlmResponse with redacted text if any
    secret pattern is found.  Returns None when the response is clean.
    """
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types as genai_types

    try:
        content = getattr(llm_response, "content", None)
        if content is None:
            return None

        parts = getattr(content, "parts", None) or []
        new_parts: list = []
        any_redacted = False

        for part in parts:
            text: str = getattr(part, "text", None)
            if text is None:
                # Non-text part (function_call, etc.) — pass through unchanged.
                new_parts.append(part)
                continue

            redacted = text
            for pattern in _SECRET_PATTERNS:
                redacted = pattern.sub(_REDACTION_PLACEHOLDER, redacted)

            if redacted != text:
                any_redacted = True
                new_parts.append(genai_types.Part(text=redacted))
            else:
                new_parts.append(part)

        if not any_redacted:
            return None

        logger.warning(
            "[guardrail-2b:%s] secrets redacted from model output",
            getattr(callback_context, "agent_name", "?"),
        )
        return LlmResponse(
            content=genai_types.Content(
                role=getattr(content, "role", "model"),
                parts=new_parts,
            )
        )
    except Exception as exc:
        logger.debug("[guardrail-2b] redaction skipped: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Layer 2c: search payload guard — before_tool_callback
# ---------------------------------------------------------------------------

def search_payload_guard_before_tool(
    tool: "BaseTool",
    args: dict[str, Any],
    tool_context: "ToolContext",
) -> Optional[dict[str, Any]]:
    """Block google_search calls whose query contains dangerous payloads.

    Only inspects the 'google_search' tool (exact name match).
    Returns an error dict to cancel the tool call, or None to allow it.
    """
    tool_name: str = getattr(tool, "name", "")
    if tool_name != "google_search":
        return None

    query: str = (args.get("query") or "").lower()
    for pattern in _BLOCKED_SEARCH_PATTERNS:
        if pattern in query:
            logger.warning(
                "[guardrail-2c:%s] blocked dangerous search payload — pattern=%r query=%r",
                getattr(tool_context, "agent_name", "?"),
                pattern,
                query[:120],
            )
            return {
                "error": (
                    f"Blocked: search query contains a dangerous pattern '{pattern}'. "
                    "If this was intentional, please rephrase the query."
                )
            }
    return None
