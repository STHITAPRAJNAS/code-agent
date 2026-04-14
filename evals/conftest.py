"""Shared pytest fixtures for guardrail evaluation tests.

All fixtures use unittest.mock.MagicMock so the full ADK runtime is NOT
required — tests run without any API keys or live services.

The guardrail functions use getattr() with defaults throughout, so MagicMock
objects satisfy the interface without importing real ADK types.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_callback_context():
    """Minimal CallbackContext-like object."""
    ctx = MagicMock()
    ctx.agent_name = "code_agent"
    ctx.state = {}
    ctx.invocation_id = "test-invocation-001"
    return ctx


@pytest.fixture
def make_llm_request():
    """Factory: build an LlmRequest-like mock with a single user message."""

    def _factory(user_text: str):
        part = MagicMock()
        part.text = user_text
        content = MagicMock()
        content.role = "user"
        content.parts = [part]
        req = MagicMock()
        req.contents = [content]
        req.config = None
        return req

    return _factory


@pytest.fixture
def make_llm_response():
    """Factory: build an LlmResponse-like mock with model text."""

    def _factory(text: str):
        part = MagicMock()
        part.text = text
        content = MagicMock()
        content.role = "model"
        content.parts = [part]
        resp = MagicMock()
        resp.content = content
        resp.usage_metadata = None
        return resp

    return _factory


@pytest.fixture
def make_tool():
    """Factory: build a BaseTool-like mock with a given name."""

    def _factory(name: str):
        tool = MagicMock()
        tool.name = name
        return tool

    return _factory


@pytest.fixture
def make_tool_context():
    """Build a ToolContext-like mock."""
    ctx = MagicMock()
    ctx.agent_name = "code_agent"
    ctx.state = {}
    return ctx
