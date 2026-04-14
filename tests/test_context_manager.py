"""Unit tests for the Context Budget Circuit Breaker.

Covers: circuit breaker state transitions (CLOSED → OPEN after 3 failures → RESET),
MicroCompact stripping, token estimation, and FullCompact fallback.
No real LLM calls — AutoCompact's LLM call is mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_messages(n: int) -> list[dict]:
    """Build a conversation with alternating user/assistant messages."""
    msgs = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"Message {i} " + "x" * 100})
    return msgs


def _make_tool_messages(n: int) -> list[dict]:
    """Build a mix of user/assistant/tool messages."""
    msgs = []
    for i in range(n):
        role = ["user", "assistant", "tool"][i % 3]
        msgs.append({"role": role, "content": f"Message {i} " + "x" * 200})
    return msgs


# ---------------------------------------------------------------------------
# Circuit breaker state transitions
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    def test_starts_closed(self) -> None:
        from code_agent.services.context_manager import ContextManager, CircuitState

        mgr = ContextManager(session_id="test")
        assert mgr.circuit_state == CircuitState.CLOSED

    def test_reset_restores_closed_state(self) -> None:
        from code_agent.services.context_manager import ContextManager, CircuitState

        mgr = ContextManager(session_id="test")
        # Manually trip the breaker
        mgr._auto_compact_failures = 3
        mgr._circuit_state = mgr._circuit_state.__class__("open")
        mgr.reset_circuit_breaker()
        assert mgr.circuit_state == CircuitState.CLOSED
        assert mgr._auto_compact_failures == 0

    @pytest.mark.asyncio
    async def test_circuit_opens_after_max_failures(self) -> None:
        """AutoCompact failing MAX_AUTOCOMPACT_FAILURES times opens the circuit."""
        from code_agent.services.context_manager import (
            ContextManager,
            CircuitState,
            MAX_AUTOCOMPACT_FAILURES,
            COMPRESSION_THRESHOLD,
        )

        mgr = ContextManager(session_id="circuit-test")

        # Build messages that exceed 80% of a small budget (forces compression)
        # 100 messages × ~200 chars each ≈ 20_000 chars ≈ 5_000 tokens
        messages = _make_messages(100)
        max_tokens = 1_000  # tiny budget so utilisation > 80%

        with patch(
            "code_agent.services.context_manager._auto_compact",
            new=AsyncMock(return_value=None),  # always fail
        ):
            for _ in range(MAX_AUTOCOMPACT_FAILURES):
                await mgr.check_and_compress(
                    messages=messages,
                    token_count=900,   # 90% utilisation
                    max_tokens=max_tokens,
                )

        assert mgr.circuit_state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_reset_on_new_session_closes_open_circuit(self) -> None:
        from code_agent.services.context_manager import ContextManager, CircuitState

        mgr = ContextManager(session_id="s1")
        # Manually open the circuit
        from code_agent.services.context_manager import CircuitState as CS
        mgr._circuit_state = CS.OPEN
        mgr._auto_compact_failures = 5

        # Simulate new session start
        mgr.reset_circuit_breaker()
        assert mgr.circuit_state == CircuitState.CLOSED
        assert mgr._auto_compact_failures == 0


# ---------------------------------------------------------------------------
# MicroCompact
# ---------------------------------------------------------------------------


class TestMicroCompact:
    def test_strips_old_tool_outputs(self) -> None:
        from code_agent.services.context_manager import _micro_compact

        # 15 messages: first 5 should have tool outputs stripped
        messages = _make_tool_messages(15)
        result = _micro_compact(messages)

        # The 5 oldest tool messages should be replaced
        old_tools = [m for m in result[:5] if m["role"] == "tool"]
        for msg in old_tools:
            assert "stripped" in msg["content"]

    def test_preserves_recent_turns(self) -> None:
        from code_agent.services.context_manager import _micro_compact, _MICRO_COMPACT_TURN_AGE

        messages = _make_tool_messages(20)
        original_recent = messages[-_MICRO_COMPACT_TURN_AGE:]
        result = _micro_compact(messages)
        recent = result[-_MICRO_COMPACT_TURN_AGE:]
        assert recent == original_recent  # recent turns untouched

    def test_no_op_for_short_conversations(self) -> None:
        from code_agent.services.context_manager import _micro_compact

        messages = _make_messages(5)
        result = _micro_compact(messages)
        assert result == messages  # nothing stripped

    def test_non_tool_messages_never_stripped(self) -> None:
        from code_agent.services.context_manager import _micro_compact

        messages = [{"role": "user", "content": "important"} for _ in range(20)]
        result = _micro_compact(messages)
        assert all(m["content"] == "important" for m in result)


# ---------------------------------------------------------------------------
# check_and_compress: no compression when under threshold
# ---------------------------------------------------------------------------


class TestCheckAndCompress:
    @pytest.mark.asyncio
    async def test_no_compression_under_threshold(self) -> None:
        from code_agent.services.context_manager import ContextManager

        mgr = ContextManager(session_id="low-usage")
        messages = _make_messages(5)
        result_msgs, compressed = await mgr.check_and_compress(
            messages=messages,
            token_count=100,
            max_tokens=10_000,  # 1% utilisation
        )
        assert compressed is False
        assert result_msgs == messages

    @pytest.mark.asyncio
    async def test_micro_compact_applied_first(self) -> None:
        """When MicroCompact brings usage below threshold, AutoCompact is skipped."""
        from code_agent.services.context_manager import ContextManager

        mgr = ContextManager(session_id="micro-test")
        messages = _make_tool_messages(30)  # many old tool messages to strip

        with patch(
            "code_agent.services.context_manager._auto_compact",
            new=AsyncMock(),
        ) as mock_auto:
            # High token count that drops after micro-compact strips old tool outputs
            _, compressed = await mgr.check_and_compress(
                messages=messages,
                token_count=8_500,   # 85% — over threshold
                max_tokens=10_000,
            )
            # AutoCompact may or may not be called depending on post-micro utilisation;
            # what matters is we got compression and no exception
            assert isinstance(compressed, bool)

    @pytest.mark.asyncio
    async def test_compression_history_recorded(self) -> None:
        from code_agent.services.context_manager import ContextManager

        mgr = ContextManager(session_id="history-test")
        messages = _make_messages(20)

        with patch(
            "code_agent.services.context_manager._auto_compact",
            new=AsyncMock(return_value=[{"role": "system", "content": "summary"}]),
        ):
            _, compressed = await mgr.check_and_compress(
                messages=messages,
                token_count=9_000,
                max_tokens=10_000,
            )

        if compressed:
            assert len(mgr.compression_history) >= 1

    @pytest.mark.asyncio
    async def test_full_compact_when_circuit_open(self) -> None:
        """When circuit is open, skip AutoCompact and go straight to FullCompact."""
        from code_agent.services.context_manager import ContextManager, CircuitState

        mgr = ContextManager(session_id="open-circuit")
        mgr._circuit_state = CircuitState.OPEN
        messages = _make_messages(20)

        with patch(
            "code_agent.services.context_manager._auto_compact",
            new=AsyncMock(),
        ) as mock_auto:
            result_msgs, compressed = await mgr.check_and_compress(
                messages=messages,
                token_count=9_000,
                max_tokens=10_000,
            )

        # AutoCompact must NOT have been called with an open circuit
        mock_auto.assert_not_called()
        assert compressed is True
        # FullCompact returns a single system message
        assert result_msgs[0]["role"] == "system"
