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
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import structlog

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
_slog: structlog.BoundLogger = structlog.get_logger(__name__)

# Per-session ContextManager registry keyed by session_id
# (ContextManager is lightweight — one per session, lives for session lifetime)
_context_managers: dict[str, Any] = {}

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
    """Record invocation start time, seed template vars, and initialise session services.

    On session start this callback:
      1. Seeds instruction template placeholders.
      2. Loads AGENT.md + MEMORY.md pointer index into session state.
      3. Fetches codebase snapshot (git branch, commits, index stats).
      4. Resets the ContextManager circuit breaker for this session.
    """
    callback_context.state["invocation_start_ms"] = int(time.time() * 1000)
    callback_context.state[_TOKEN_CALL_KEY] = 0  # reset LLM call counter

    # Seed keys used in the instruction template
    callback_context.state.setdefault("custom_instructions", "")
    callback_context.state.setdefault("active_repo", "")
    callback_context.state.setdefault("active_workspace", "")

    # Derive repo_workspace from active_repo if not explicitly provided.
    # This ensures the shared repo tier is automatically available without
    # requiring callers to compute the path themselves.
    if not callback_context.state.get("repo_workspace"):
        active_repo: str = callback_context.state.get("active_repo", "")
        if active_repo:
            from code_agent.tools.memory_tools import _repo_slug
            base = os.environ.get("REPO_WORKSPACE_BASE", "/data/repos")
            callback_context.state["repo_workspace"] = f"{base}/{_repo_slug(active_repo)}"

    session_id: str = getattr(callback_context, "session_id", "") or ""

    # ── Reset / create ContextManager circuit breaker ─────────────────────────
    from code_agent.services.context_manager import ContextManager
    ctx_mgr = _context_managers.get(session_id)
    if ctx_mgr is None:
        ctx_mgr = ContextManager(session_id=session_id)
        _context_managers[session_id] = ctx_mgr
    else:
        ctx_mgr.reset_circuit_breaker()

    # ── Load AGENT.md + MEMORY.md into session state (on first invocation) ────
    if not callback_context.state.get("_memory_loaded"):
        try:
            from code_agent.tools.memory_tools import read_agent_config

            class _FakeCtx:
                state = callback_context.state

            mem_data = await read_agent_config(_FakeCtx())  # type: ignore[arg-type]
            callback_context.state["agent_config"] = mem_data.get("agent_config", "")
            callback_context.state["memory_index"] = mem_data.get("memory_index", [])
            # L0: project identity for the static prompt section — shared across all
            # users on the same repo, kept before SYSTEM_PROMPT_DYNAMIC_BOUNDARY.
            callback_context.state["l0_identity"] = mem_data.get("l0_identity", "")
            callback_context.state["_memory_loaded"] = True

            memory_index = mem_data.get("memory_index", [])
            workspace_path = Path(
                callback_context.state.get("active_workspace")
                or os.environ.get("WORKSPACE_ROOT")
                or os.environ.get("WORKSPACE_DIR", "/tmp/code_agent_workspaces")
            )

            # ── Gap 2: detect stale topics before first LLM call ─────────────
            from code_agent.services.memory_preloader import (
                detect_stale_topics,
                warm_recent_topics,
            )
            stale = await detect_stale_topics(memory_index, workspace_path)
            if stale:
                callback_context.state["stale_topics"] = stale
                _slog.info(
                    "session.stale_topics_detected",
                    topics=stale,
                    session_id=session_id,
                )

            # ── Gap 4: warm the N most recent topics into memory ──────────────
            warm_cache = await warm_recent_topics(memory_index, workspace_path)
            callback_context.state["warm_topics_cache"] = warm_cache

            _slog.info(
                "session.memory_loaded",
                pointer_count=len(memory_index),
                stale_count=len(stale),
                warm_count=len(warm_cache),
                session_id=session_id,
            )
        except Exception as exc:
            _slog.warning("session.memory_load_failed", error=str(exc), session_id=session_id)

    # ── Fetch lightweight codebase snapshot ───────────────────────────────────
    try:
        from code_agent.tools.search_tools import get_current_codebase_state

        class _FakeCtx2:  # type: ignore[no-redef]
            state = callback_context.state

        snapshot = await get_current_codebase_state(_FakeCtx2())  # type: ignore[arg-type]
        callback_context.state["codebase_snapshot"] = snapshot
        _slog.debug(
            "session.codebase_snapshot",
            branch=snapshot.get("git_branch", ""),
            chunks=snapshot.get("total_chunks", 0),
            session_id=session_id,
        )
    except Exception as exc:
        _slog.debug("session.codebase_snapshot_failed", error=str(exc))

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
    """Log elapsed time and trigger AutoDream as a background task on session end."""
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

    # ── Increment session counter + launch AutoDream background consolidation ──
    workspace = (
        callback_context.state.get("active_workspace")
        or os.environ.get("WORKSPACE_ROOT")
        or os.environ.get("WORKSPACE_DIR", "/tmp/code_agent_workspaces")
    )
    try:
        from code_agent.services.auto_dream import increment_session_and_dream
        session_count = await increment_session_and_dream(Path(workspace))
        _slog.info(
            "session.end",
            session_count=session_count,
            elapsed_ms=elapsed,
            workspace=str(workspace),
        )
    except Exception as exc:
        _slog.warning("session.end.auto_dream_error", error=str(exc))

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

    # ── 3a. Check context budget and compress if needed ───────────────────────
    session_id = getattr(callback_context, "session_id", "") or ""
    ctx_mgr = _context_managers.get(session_id)
    if ctx_mgr is not None:
        try:
            contents = getattr(llm_request, "contents", None) or []
            messages = [
                {"role": getattr(c, "role", ""), "content": str(getattr(c, "parts", "") or "")}
                for c in contents
            ]
            active_files: list[str] = callback_context.state.get("active_files", [])
            current_plan: str | None = callback_context.state.get("current_plan")
            # Estimate max_tokens from model config (default 128k for Gemini)
            max_tokens: int = callback_context.state.get("max_context_tokens", 128_000)
            compressed_msgs, was_compressed = await ctx_mgr.check_and_compress(
                messages=messages,
                token_count=0,  # let ContextManager estimate
                max_tokens=max_tokens,
                active_files=active_files,
                current_plan=current_plan,
            )
            if was_compressed:
                _slog.info(
                    "callback.context_compressed",
                    session_id=session_id,
                    invocation_id=callback_context.invocation_id,
                )
        except Exception as exc:
            _slog.warning("callback.context_check_error", error=str(exc))

    # ── 3b. Gap 1: proactive L2 injection on first user message ─────────────
    # Front-run the agent's own recall_topic decision by keyword-matching the
    # user's message against the L1 pointer index.  Only fires once per session
    # (subsequent turns let the agent decide what else to load).
    if not callback_context.state.get("_l2_injected"):
        user_text = _get_latest_user_text(llm_request)
        if user_text:
            try:
                from code_agent.services.memory_preloader import (
                    match_topics_to_query,
                    preload_matched_topics,
                    format_proactive_context,
                )
                memory_index = callback_context.state.get("memory_index", [])
                warm_cache = callback_context.state.get("warm_topics_cache", {})
                matched = match_topics_to_query(user_text, memory_index)[:2]
                if matched:
                    class _FakeCtxL2:  # type: ignore[no-redef]
                        state = callback_context.state

                    contents = await preload_matched_topics(matched, warm_cache, _FakeCtxL2())  # type: ignore[arg-type]
                    stale_topics: set[str] = set(callback_context.state.get("stale_topics", []))
                    if contents:
                        callback_context.state["proactive_context"] = format_proactive_context(
                            contents, stale_topics
                        )
                        _slog.info(
                            "callback.l2_proactive_injected",
                            topics=list(contents.keys()),
                            session_id=session_id,
                        )
                callback_context.state["_l2_injected"] = True
            except Exception as exc:
                _slog.debug("callback.l2_proactive_error", error=str(exc))

    # ── 3c. Inject context into system instruction ────────────────────────────
    _inject_context_into_system_instruction(callback_context, llm_request)

    # ── 3d. Prompt-injection guard (Layer 2a) ─────────────────────────────────
    injection_response = injection_guard_before_model(callback_context, llm_request)
    if injection_response is not None:
        return injection_response

    # ── 3d. LLM call-count guard ──────────────────────────────────────────────
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
    """Audit trail, Gap-3 summarisation, and Layer 2d secret redaction.

    1. Writes full result to the audit trail unconditionally.
    2. Truncates verbose tool results (Gap 3) before they enter conversation history.
    3. Applies secret redaction (Layer 2d) to whatever is returned to the LLM.
    """
    tool_name: str = getattr(tool, "name", str(tool))

    # Always log the full result to the audit trail
    audit: list[dict] = tool_context.state.setdefault("tool_audit", [])
    audit.append(
        {
            "tool": tool_name,
            "args": _truncate_args(args),
            "ts_ms": int(time.time() * 1000),
        }
    )

    # Gap 3: truncate verbose tool results before they enter conversation history
    summarised = _summarise_tool_result(tool_name, tool_response)
    if summarised is not tool_response:
        logger.debug(
            "[tool:%s] result summarised — original=%d summarised=%d chars",
            tool_name,
            len(str(tool_response)),
            len(str(summarised)),
        )
        return summarised

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
    proactive = callback_context.state.get("proactive_context", "")

    # L0/L1 memory layer placeholders
    # {l0_project_summary} → static section (before boundary, maximises cache hits)
    # {l1_memory_index}    → dynamic section (loaded every session, ~120 tokens)
    from code_agent.services.prompt_builder import (
        extract_l0_for_static_section,
        format_memory_index_for_prompt,
    )
    raw_l0 = callback_context.state.get("l0_identity") or ""
    l0_project_summary = extract_l0_for_static_section(raw_l0) if raw_l0 else ""
    memory_index_list: list[dict] = callback_context.state.get("memory_index") or []
    l1_memory_index = format_memory_index_for_prompt(memory_index_list)

    replacements = {
        "{active_repo}": f"Active repository: {active_repo}",
        "{active_workspace}": f"Active workspace: {active_workspace}",
        "{custom_instructions}": custom_instructions,
        "{l0_project_summary}": l0_project_summary,
        "{l1_memory_index}": l1_memory_index,
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
                    new_text = _append_proactive_context(new_text, proactive)
                    if new_text != text:
                        part.text = new_text
            return

        # Fallback: si is a plain string (older ADK builds)
        if isinstance(si, str):
            new_si = si
            for placeholder, value in replacements.items():
                new_si = new_si.replace(placeholder, value)
            new_si = _append_proactive_context(new_si, proactive)
            if new_si != si:
                config.system_instruction = new_si
    except Exception as exc:
        # Never let context injection crash the agent
        logger.debug("Context injection skipped: %s", exc)


def _append_proactive_context(system_instruction: str, proactive: str) -> str:
    """Append pre-loaded L2 topic content after the dynamic boundary."""
    if not proactive:
        return system_instruction
    from code_agent.services.prompt_builder import SYSTEM_PROMPT_DYNAMIC_BOUNDARY
    marker = "## Pre-loaded Memory (L2)"
    if marker in system_instruction:
        return system_instruction  # already injected this call
    if SYSTEM_PROMPT_DYNAMIC_BOUNDARY in system_instruction:
        return system_instruction + f"\n\n{marker}\n{proactive}"
    return system_instruction + f"\n\n{marker}\n{proactive}"


def _get_latest_user_text(llm_request: "LlmRequest") -> str:
    """Extract the text of the most recent user turn from the LLM request."""
    try:
        contents = getattr(llm_request, "contents", None) or []
        for content in reversed(contents):
            if getattr(content, "role", "") != "user":
                continue
            for part in getattr(content, "parts", []) or []:
                text = getattr(part, "text", "") or ""
                if text.strip():
                    return text
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Gap 3 — tool result summarisation
# ---------------------------------------------------------------------------

# Tools whose output regularly exceeds useful density
_VERBOSE_TOOLS: frozenset[str] = frozenset({
    "git_log", "git_diff", "git_show", "git_blame",
    "search_in_files", "grep_code",
    "semantic_search", "lexical_search", "hybrid_search",
    "get_repo_file_tree", "find_files", "find_symbol_references",
})

_TOOL_RESULT_MAX_CHARS = 2_000  # ~500 tokens; full result preserved in audit log


def _summarise_tool_result(
    tool_name: str,
    response: dict[str, Any],
) -> dict[str, Any]:
    """Return a token-capped version of *response* for verbose tools.

    The original response is always written to the audit trail before this
    function is called — this copy is only what enters the conversation history.
    Non-verbose tools and small results are returned unchanged (same object).
    """
    if tool_name not in _VERBOSE_TOOLS:
        return response
    if not isinstance(response, dict):
        return response

    full_str = str(response)
    if len(full_str) <= _TOOL_RESULT_MAX_CHARS:
        return response

    # Walk keys and truncate values until we hit the budget
    summarised: dict[str, Any] = {}
    chars_used = 0
    for key, value in response.items():
        v_str = str(value)
        remaining = _TOOL_RESULT_MAX_CHARS - chars_used
        if remaining <= 0:
            summarised["_truncated"] = True
            break
        if len(v_str) > remaining:
            summarised[key] = v_str[:remaining] + f"… [{len(v_str) - remaining} chars omitted]"
            summarised["_truncated"] = True
            break
        summarised[key] = value
        chars_used += len(v_str)

    return summarised
