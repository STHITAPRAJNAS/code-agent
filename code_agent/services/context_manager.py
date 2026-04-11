"""Context Budget Circuit Breaker — prevents silent token burn in long A2A sessions.

Three-level compression hierarchy:
  1. MicroCompact  — strip old tool-call outputs from messages > 10 turns ago.
                     No API call. Always attempted first.
  2. AutoCompact   — call Gemini to generate a structured conversation summary.
                     Falls back to FullCompact on repeated failures.
  3. FullCompact   — compress entire conversation + re-inject active files +
                     current plan. Nuclear option; resets context budget.

Circuit breaker:
  - Opens after MAX_AUTOCOMPACT_FAILURES consecutive AutoCompact failures.
  - When open, skip AutoCompact and go straight to FullCompact.
  - Resets to CLOSED at the start of each new A2A session.

Wire this into before_model_callback by calling check_and_compress() before
every LLM request.

All logging uses structlog with context dicts.
"""

from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Any

import structlog

from pydantic import BaseModel, Field

logger: structlog.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants — mirror Claude Code's values
# ---------------------------------------------------------------------------

COMPRESSION_THRESHOLD = 0.80        # compress when 80% of budget used
COMPRESSION_RESERVE_TOKENS = 13_000
MAX_SUMMARY_TOKENS = 20_000
MAX_AUTOCOMPACT_FAILURES = 3        # circuit breaker trip threshold
POST_COMPRESS_BUDGET = 50_000       # token budget after full compression
MAX_FILE_REINJECT_TOKENS = 5_000    # per-file cap after full compression

# Rough heuristic: 1 token ≈ 4 characters (conservative for code-heavy context)
_CHARS_PER_TOKEN = 4
# Turns older than this have their tool outputs stripped in MicroCompact
_MICRO_COMPACT_TURN_AGE = 10

# ---------------------------------------------------------------------------
# Pydantic v2 models
# ---------------------------------------------------------------------------


class CompressionLevel(str, Enum):
    NONE = "none"
    MICRO = "micro"
    AUTO = "auto"
    FULL = "full"


class CircuitState(str, Enum):
    CLOSED = "closed"   # normal — AutoCompact is allowed
    OPEN = "open"       # tripped — skip AutoCompact, go straight to FullCompact


class CompressionEvent(BaseModel):
    """Logged on every compression attempt."""

    timestamp: float = Field(default_factory=time.time)
    level: CompressionLevel
    tokens_before: int
    tokens_after: int
    circuit_breaker_state: CircuitState
    session_id: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# Token counting helpers
# ---------------------------------------------------------------------------


def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate token count from messages using character heuristic."""
    total_chars = sum(len(str(m)) for m in messages)
    return total_chars // _CHARS_PER_TOKEN


def _estimate_str_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# Compression implementations
# ---------------------------------------------------------------------------


def _micro_compact(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip tool-call output from messages older than _MICRO_COMPACT_TURN_AGE.

    The most recent N turns are left intact so the model retains immediate
    context.  Older turns have their tool-result content replaced with a
    placeholder to free tokens cheaply without any API call.
    """
    if len(messages) <= _MICRO_COMPACT_TURN_AGE:
        return messages

    result = []
    cutoff = len(messages) - _MICRO_COMPACT_TURN_AGE
    for i, msg in enumerate(messages):
        if i >= cutoff:
            result.append(msg)
            continue

        role = msg.get("role", "")
        # Strip tool result content — keep the message skeleton for turn coherence
        if role == "tool":
            stripped = dict(msg)
            stripped["content"] = "[tool output stripped — older than 10 turns]"
            result.append(stripped)
        else:
            result.append(msg)

    return result


async def _auto_compact(
    messages: list[dict[str, Any]],
    max_tokens: int,
    session_id: str = "",
) -> list[dict[str, Any]] | None:
    """Call Gemini to produce a structured conversation summary.

    Returns the compressed message list, or None on failure.
    Reserves COMPRESSION_RESERVE_TOKENS so the summary prompt itself
    does not exceed the context window.
    """
    from code_agent.config import get_settings
    cfg = get_settings()

    available = max_tokens - COMPRESSION_RESERVE_TOKENS
    if available <= 0:
        logger.warning(
            "context.auto_compact.insufficient_reserve",
            max_tokens=max_tokens,
            reserve=COMPRESSION_RESERVE_TOKENS,
        )
        return None

    # Build a compact transcript for the summarisation prompt
    transcript_parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "?")
        content = str(msg.get("content", ""))[:2000]  # cap per message
        transcript_parts.append(f"[{role}]: {content}")
    transcript = "\n".join(transcript_parts)

    summary_prompt = (
        "You are summarising a software engineering conversation for context compression.\n\n"
        "Produce a concise structured summary that preserves:\n"
        "  - The original user request and goal\n"
        "  - Decisions made and the rationale\n"
        "  - Files created or modified (with paths)\n"
        "  - Current task state and next steps\n"
        "  - Any open questions or blockers\n\n"
        "Do NOT include tool call outputs verbatim. Keep the summary under "
        f"{MAX_SUMMARY_TOKENS // _CHARS_PER_TOKEN} words.\n\n"
        f"CONVERSATION:\n{transcript}"
    )

    try:
        import litellm  # type: ignore[import-untyped]

        response = await litellm.acompletion(
            model=cfg.LLM_MODEL,
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=MAX_SUMMARY_TOKENS,
        )
        summary_text = response.choices[0].message.content or ""
    except Exception as exc:
        logger.warning(
            "context.auto_compact.llm_error",
            error=str(exc),
            session_id=session_id,
        )
        return None

    compressed = [
        {
            "role": "system",
            "content": (
                "[CONTEXT COMPRESSED — conversation summarised to save tokens]\n\n"
                + summary_text
            ),
        }
    ]
    return compressed


def _full_compact(
    messages: list[dict[str, Any]],
    active_files: list[str],
    current_plan: str | None,
    file_contents: dict[str, str],
) -> list[dict[str, Any]]:
    """Nuclear compression — reduce to a minimal restart context.

    Injects:
      - A summary of what was being worked on (from the last user message)
      - Current plan (if any)
      - Active file contents (capped at MAX_FILE_REINJECT_TOKENS each)
    """
    # Extract last meaningful user message as task context
    last_user = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user = str(msg.get("content", ""))[:500]
            break

    parts = [
        "[FULL CONTEXT RESET — conversation history discarded to recover context budget]",
        "",
        "## Task Context",
        f"Last user request: {last_user}" if last_user else "No previous user message.",
        "",
    ]

    if current_plan:
        parts += ["## Current Plan", current_plan[:2000], ""]

    if active_files:
        parts.append("## Active Files (re-injected)")
        for fpath in active_files:
            content = file_contents.get(fpath, "")
            token_cap = MAX_FILE_REINJECT_TOKENS * _CHARS_PER_TOKEN
            if content:
                snippet = content[:token_cap]
                parts += [f"\n### {fpath}", "```", snippet, "```"]

    return [{"role": "system", "content": "\n".join(parts)}]


# ---------------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------------


class ContextManager:
    """Manages context budget and compression for a single A2A session.

    Create one instance per session and reset it at session start:
        ctx_mgr = ContextManager(session_id=session_id)
    """

    def __init__(self, session_id: str = "") -> None:
        self.session_id = session_id
        self._circuit_state = CircuitState.CLOSED
        self._auto_compact_failures = 0
        self._compression_history: list[CompressionEvent] = []

    def reset_circuit_breaker(self) -> None:
        """Call at the start of each new A2A session."""
        self._circuit_state = CircuitState.CLOSED
        self._auto_compact_failures = 0
        logger.info(
            "context.circuit_breaker.reset",
            session_id=self.session_id,
        )

    @property
    def circuit_state(self) -> CircuitState:
        return self._circuit_state

    async def check_and_compress(
        self,
        messages: list[dict[str, Any]],
        token_count: int,
        max_tokens: int,
        active_files: list[str] | None = None,
        current_plan: str | None = None,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Check token budget and compress if needed.

        Args:
            messages:     Current conversation message list.
            token_count:  Estimated or measured token count for *messages*.
                          Pass 0 to have this method estimate it.
            max_tokens:   Model context window limit.
            active_files: File paths to re-inject after FullCompact.
            current_plan: Current plan summary to preserve through compression.

        Returns:
            (messages, was_compressed) — if was_compressed is True, the
            returned messages list has been modified.
        """
        active_files = active_files or []
        effective_count = token_count if token_count > 0 else _estimate_tokens(messages)
        utilisation = effective_count / max(max_tokens, 1)

        if utilisation < COMPRESSION_THRESHOLD:
            return messages, False

        log = logger.bind(
            session_id=self.session_id,
            tokens_before=effective_count,
            max_tokens=max_tokens,
            utilisation=round(utilisation, 3),
            circuit_state=self._circuit_state.value,
        )
        log.info("context.compression.triggered")

        # ── Level 1: MicroCompact ─────────────────────────────────────────────
        chars_before = sum(len(str(m)) for m in messages)
        result = _micro_compact(messages)
        chars_after = sum(len(str(m)) for m in result)

        # Scale the caller-supplied token count by the character reduction ratio so
        # that the threshold check is consistent with the token_count semantics the
        # caller expects.  Re-estimating from scratch diverges when token_count was
        # provided (e.g. measured by the model) vs. derived from raw char length.
        if chars_before > 0:
            after_micro = int(effective_count * chars_after / chars_before)
        else:
            after_micro = effective_count

        micro_utilisation = after_micro / max(max_tokens, 1)

        if micro_utilisation < COMPRESSION_THRESHOLD:
            self._log_event(CompressionLevel.MICRO, effective_count, after_micro)
            log.info(
                "context.micro_compact.sufficient",
                tokens_after=after_micro,
            )
            return result, True

        # ── Level 2: AutoCompact (if circuit is closed) ───────────────────────
        # Note: _auto_compact itself guards against insufficient reserve (available<=0);
        # we always attempt it when CLOSED so that failures are counted correctly.

        if self._circuit_state == CircuitState.CLOSED:
            compressed = await _auto_compact(result, max_tokens, self.session_id)
            if compressed is not None:
                self._auto_compact_failures = 0
                after_auto = _estimate_tokens(compressed)
                self._log_event(CompressionLevel.AUTO, effective_count, after_auto)
                log.info(
                    "context.auto_compact.success",
                    tokens_after=after_auto,
                )
                return compressed, True
            else:
                self._auto_compact_failures += 1
                log.warning(
                    "context.auto_compact.failure",
                    failures=self._auto_compact_failures,
                    threshold=MAX_AUTOCOMPACT_FAILURES,
                )
                if self._auto_compact_failures >= MAX_AUTOCOMPACT_FAILURES:
                    self._circuit_state = CircuitState.OPEN
                    log.warning(
                        "context.circuit_breaker.opened",
                        failures=self._auto_compact_failures,
                    )

        # ── Level 3: FullCompact ──────────────────────────────────────────────
        file_contents = await self._load_file_contents(active_files)
        full_result = _full_compact(result, active_files, current_plan, file_contents)
        after_full = _estimate_tokens(full_result)
        self._log_event(
            CompressionLevel.FULL,
            effective_count,
            after_full,
            notes="circuit_open" if self._circuit_state == CircuitState.OPEN else "",
        )
        log.info(
            "context.full_compact.applied",
            tokens_after=after_full,
            circuit_breaker_open=(self._circuit_state == CircuitState.OPEN),
        )
        return full_result, True

    async def _load_file_contents(self, paths: list[str]) -> dict[str, str]:
        """Read active files for re-injection after FullCompact."""
        contents: dict[str, str] = {}
        for path in paths:
            try:
                import aiofiles
                async with aiofiles.open(path, "r", encoding="utf-8", errors="replace") as fh:
                    contents[path] = await fh.read()
            except Exception as exc:
                logger.debug("context.file_reinject.error", path=path, error=str(exc))
        return contents

    def _log_event(
        self,
        level: CompressionLevel,
        tokens_before: int,
        tokens_after: int,
        notes: str = "",
    ) -> None:
        event = CompressionEvent(
            level=level,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            circuit_breaker_state=self._circuit_state,
            session_id=self.session_id,
            notes=notes,
        )
        self._compression_history.append(event)
        logger.info(
            "context.compression.event",
            level=level.value,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            reduction_pct=round(100 * (1 - tokens_after / max(tokens_before, 1)), 1),
            circuit_breaker_open=(self._circuit_state == CircuitState.OPEN),
            session_id=self.session_id,
        )

    @property
    def compression_history(self) -> list[CompressionEvent]:
        return list(self._compression_history)
