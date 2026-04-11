"""Prompt Cache Stability — build system prompts that maximise Gemini cache hits.

The system prompt is always structured in this exact order:

    [STATIC SECTION — identical byte-for-byte across all sessions and pods]
    {agent_identity_and_capabilities}
    {tool_definitions_summary}
    {coding_conventions}
    {a2a_protocol_instructions}
    ──── STABLE BOUNDARY ────
    [DYNAMIC SECTION — session/task specific, placed AFTER the boundary]
    {workspace_info}
    {agent_config_summary}      # from AGENT.md
    {memory_index}              # from MEMORY.md (pointer index only)
    {current_task_context}
    {active_files_summary}

Rules:
  - Static section must be identical byte-for-byte across all sessions and pods
  - Never inject timestamps, session IDs, or request IDs into the static section
  - Mode changes (plan mode, auto mode) go in dynamic section only
  - DANGEROUS_uncached_section() logs a warning when dynamic content appears
    before the boundary

The SYSTEM_PROMPT_DYNAMIC_BOUNDARY constant is the literal marker string.
Use it consistently everywhere a boundary is needed.
"""

from __future__ import annotations

import structlog

from pydantic import BaseModel

logger: structlog.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_DYNAMIC_BOUNDARY = "──── STABLE BOUNDARY ────"

# ---------------------------------------------------------------------------
# Config models
# ---------------------------------------------------------------------------


class StaticPromptConfig(BaseModel):
    """Content that must be identical across all sessions and pods.

    Populate once at startup from static configuration — never from
    per-request or per-session data.
    """

    agent_identity: str = ""
    tool_definitions_summary: str = ""
    coding_conventions: str = ""
    a2a_protocol_instructions: str = ""


class DynamicPromptContext(BaseModel):
    """Per-session context appended after the stable boundary."""

    workspace_info: str = ""
    agent_config_summary: str = ""   # summarised AGENT.md content
    memory_index: str = ""           # MEMORY.md pointer lines only (no topic files)
    current_task_context: str = ""
    active_files_summary: str = ""
    mode_hint: str = ""              # e.g. "plan_mode" or "auto_mode"


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def build_system_prompt(
    static_config: StaticPromptConfig,
    dynamic_context: DynamicPromptContext,
) -> str:
    """Construct the full system prompt with stable / dynamic sections.

    The static section is assembled first and is byte-for-byte identical
    across all invocations with the same StaticPromptConfig.  The
    SYSTEM_PROMPT_DYNAMIC_BOUNDARY marker separates it from the dynamic
    section, giving Gemini's prompt cache the maximum prefix to cache.

    Args:
        static_config:   Agent identity, tool summaries, conventions, A2A rules.
                         Must not contain any session-specific data.
        dynamic_context: Workspace state, memory index, task context.
                         Always placed after the stable boundary.

    Returns:
        The fully assembled system prompt string.
    """
    static_parts: list[str] = []

    if static_config.agent_identity:
        static_parts.append(static_config.agent_identity.rstrip())

    if static_config.tool_definitions_summary:
        static_parts.append(static_config.tool_definitions_summary.rstrip())

    if static_config.coding_conventions:
        static_parts.append(static_config.coding_conventions.rstrip())

    if static_config.a2a_protocol_instructions:
        static_parts.append(static_config.a2a_protocol_instructions.rstrip())

    static_section = "\n\n".join(static_parts) if static_parts else ""

    dynamic_parts: list[str] = []

    if dynamic_context.workspace_info:
        dynamic_parts.append(f"## Workspace\n{dynamic_context.workspace_info.rstrip()}")

    if dynamic_context.agent_config_summary:
        dynamic_parts.append(
            f"## Project Configuration (AGENT.md)\n{dynamic_context.agent_config_summary.rstrip()}"
        )

    if dynamic_context.memory_index:
        dynamic_parts.append(
            f"## Memory Index\n{dynamic_context.memory_index.rstrip()}"
        )

    if dynamic_context.current_task_context:
        dynamic_parts.append(
            f"## Current Task\n{dynamic_context.current_task_context.rstrip()}"
        )

    if dynamic_context.active_files_summary:
        dynamic_parts.append(
            f"## Active Files\n{dynamic_context.active_files_summary.rstrip()}"
        )

    if dynamic_context.mode_hint:
        dynamic_parts.append(f"## Mode\n{dynamic_context.mode_hint.rstrip()}")

    dynamic_section = "\n\n".join(dynamic_parts) if dynamic_parts else ""

    sections = [static_section, SYSTEM_PROMPT_DYNAMIC_BOUNDARY, dynamic_section]
    return "\n\n".join(s for s in sections if s.strip())


def DANGEROUS_uncached_section(content: str) -> str:
    """Wrap *content* and emit a warning — dynamic content must NOT appear before the boundary.

    In production this should never be called.  When it is called, it means
    session-specific data (timestamps, request IDs, etc.) is being injected
    into the static section, which kills prompt cache hits.

    Usage:
        # Bad — puts dynamic data before the boundary
        system_prompt = DANGEROUS_uncached_section(f"Request ID: {req_id}") + static_prompt

        # Correct — dynamic data goes after the boundary in build_system_prompt()
        ctx = DynamicPromptContext(current_task_context=f"Request ID: {req_id}")
        system_prompt = build_system_prompt(static, ctx)

    Returns the content unchanged (this is a log-only wrapper).
    """
    logger.warning(
        "prompt_builder.dangerous_uncached_section",
        content_preview=content[:200],
        note=(
            "Dynamic content is being placed before SYSTEM_PROMPT_DYNAMIC_BOUNDARY. "
            "This will invalidate the Gemini prompt cache for every call."
        ),
    )
    return content


def summarise_agent_config(agent_config_text: str, max_chars: int = 1500) -> str:
    """Truncate AGENT.md to a summary suitable for the dynamic section.

    Only the dynamic section carries AGENT.md content, but it should still
    be concise to avoid wasting tokens.  This helper keeps the first
    *max_chars* characters and appends a truncation note if needed.
    """
    if not agent_config_text:
        return ""
    if len(agent_config_text) <= max_chars:
        return agent_config_text
    return agent_config_text[:max_chars] + "\n\n[...truncated for context efficiency]"


def format_memory_index_for_prompt(memory_index: list[dict[str, str]]) -> str:
    """Format the MEMORY.md pointer list for injection into the dynamic section.

    Only the pointer lines are included — never the topic file contents.
    Topic files are loaded on demand via recall_topic().
    """
    if not memory_index:
        return ""
    lines = ["Available memory topics (use recall_topic to load details):"]
    for entry in memory_index:
        topic = entry.get("topic", "")
        summary = entry.get("summary", "")
        lines.append(f"  [{topic}] — {summary}")
    return "\n".join(lines)
