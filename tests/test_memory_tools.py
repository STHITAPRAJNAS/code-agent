"""Unit tests for the 3-layer file-based memory system.

Covers: read/write/forget/list operations, atomic writes, pointer index
parsing, and multi-write idempotency.  No network calls — pure file I/O
using a temp directory.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import pytest_asyncio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Isolated workspace root in a temp directory."""
    return tmp_path / "workspace"


@pytest.fixture(autouse=True)
def patch_workspace_env(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point WORKSPACE_ROOT at the temp workspace for every test."""
    monkeypatch.setenv("WORKSPACE_ROOT", str(workspace))


# ---------------------------------------------------------------------------
# Helper: build a minimal fake ToolContext
# ---------------------------------------------------------------------------


class FakeToolContext:
    def __init__(self, workspace: Path) -> None:
        self.state: dict = {"active_workspace": str(workspace)}


# ---------------------------------------------------------------------------
# MemoryIndex parsing
# ---------------------------------------------------------------------------


class TestMemoryIndexParsing:
    def test_roundtrip_single_pointer(self) -> None:
        from code_agent.tools.memory_tools import MemoryIndex, MemoryPointer

        idx = MemoryIndex(
            pointers=[
                MemoryPointer(
                    topic="auth",
                    file_path=".agent/memory/auth.md",
                    summary="OAuth2 flow details",
                )
            ]
        )
        rendered = idx.render()
        parsed = MemoryIndex.parse(rendered)
        assert len(parsed.pointers) == 1
        assert parsed.pointers[0].topic == "auth"
        assert parsed.pointers[0].file_path == ".agent/memory/auth.md"
        assert parsed.pointers[0].summary == "OAuth2 flow details"

    def test_parse_ignores_headers_and_comments(self) -> None:
        from code_agent.tools.memory_tools import MemoryIndex

        text = "# Agent Memory Index\n<!-- comment -->\n\n[topic] → .agent/memory/t.md | summary\n"
        idx = MemoryIndex.parse(text)
        assert len(idx.pointers) == 1
        assert idx.pointers[0].topic == "topic"

    def test_parse_empty_returns_empty_index(self) -> None:
        from code_agent.tools.memory_tools import MemoryIndex

        assert MemoryIndex.parse("").pointers == []
        assert MemoryIndex.parse("# just headers\n").pointers == []

    def test_parse_multiple_pointers(self) -> None:
        from code_agent.tools.memory_tools import MemoryIndex

        text = (
            "[auth] → .agent/memory/auth.md | OAuth2\n"
            "[db] → .agent/memory/db.md | DB schema\n"
        )
        idx = MemoryIndex.parse(text)
        assert len(idx.pointers) == 2
        topics = {p.topic for p in idx.pointers}
        assert topics == {"auth", "db"}


# ---------------------------------------------------------------------------
# remember / recall_topic / forget / list_memory_topics
# ---------------------------------------------------------------------------


class TestMemoryReadWrite:
    @pytest.mark.asyncio
    async def test_remember_creates_topic_file(self, workspace: Path) -> None:
        from code_agent.tools.memory_tools import remember

        ctx = FakeToolContext(workspace)
        result = await remember("auth", "# Auth\nOAuth2 via Google.", tool_context=ctx)

        assert result["ok"] is True
        assert result["topic"] == "auth"
        topic_file = Path(result["file_path"])
        assert topic_file.exists()
        assert "OAuth2" in topic_file.read_text()

    @pytest.mark.asyncio
    async def test_remember_updates_memory_index(self, workspace: Path) -> None:
        from code_agent.tools.memory_tools import remember, list_memory_topics

        ctx = FakeToolContext(workspace)
        await remember("db", "DB schema notes.", tool_context=ctx)

        topics = await list_memory_topics(tool_context=ctx)
        assert topics["count"] == 1
        assert topics["topics"][0]["topic"] == "db"

    @pytest.mark.asyncio
    async def test_remember_is_idempotent(self, workspace: Path) -> None:
        """Writing the same topic twice replaces the old entry — no duplicates."""
        from code_agent.tools.memory_tools import remember, list_memory_topics

        ctx = FakeToolContext(workspace)
        await remember("auth", "first version", tool_context=ctx)
        await remember("auth", "updated version", tool_context=ctx)

        topics = await list_memory_topics(tool_context=ctx)
        assert topics["count"] == 1  # still only one pointer

        from code_agent.tools.memory_tools import recall_topic
        recalled = await recall_topic("auth", tool_context=ctx)
        assert recalled["found"] is True
        assert "updated version" in recalled["content"]

    @pytest.mark.asyncio
    async def test_recall_returns_content(self, workspace: Path) -> None:
        from code_agent.tools.memory_tools import remember, recall_topic

        ctx = FakeToolContext(workspace)
        await remember("api-contracts", "GET /api/v1/users → list of users", tool_context=ctx)

        result = await recall_topic("api-contracts", tool_context=ctx)
        assert result["found"] is True
        assert "GET /api/v1/users" in result["content"]

    @pytest.mark.asyncio
    async def test_recall_nonexistent_topic_returns_not_found(self, workspace: Path) -> None:
        from code_agent.tools.memory_tools import recall_topic

        ctx = FakeToolContext(workspace)
        result = await recall_topic("nonexistent", tool_context=ctx)
        assert result["found"] is False
        assert result["content"] == ""

    @pytest.mark.asyncio
    async def test_forget_removes_file_and_pointer(self, workspace: Path) -> None:
        from code_agent.tools.memory_tools import remember, forget, list_memory_topics

        ctx = FakeToolContext(workspace)
        r = await remember("temp", "temporary note", tool_context=ctx)
        topic_file = Path(r["file_path"])
        assert topic_file.exists()

        result = await forget("temp", tool_context=ctx)
        assert result["ok"] is True
        assert result["file_deleted"] is True
        assert not topic_file.exists()

        topics = await list_memory_topics(tool_context=ctx)
        assert topics["count"] == 0

    @pytest.mark.asyncio
    async def test_forget_nonexistent_topic_is_idempotent(self, workspace: Path) -> None:
        from code_agent.tools.memory_tools import forget

        ctx = FakeToolContext(workspace)
        result = await forget("does-not-exist", tool_context=ctx)
        assert result["ok"] is True
        assert result["file_deleted"] is False

    @pytest.mark.asyncio
    async def test_multiple_topics_tracked_independently(self, workspace: Path) -> None:
        from code_agent.tools.memory_tools import remember, list_memory_topics, recall_topic

        ctx = FakeToolContext(workspace)
        await remember("topic-a", "content A", tool_context=ctx)
        await remember("topic-b", "content B", tool_context=ctx)
        await remember("topic-c", "content C", tool_context=ctx)

        listing = await list_memory_topics(tool_context=ctx)
        assert listing["count"] == 3

        for name, expected in [("topic-a", "content A"), ("topic-b", "content B")]:
            r = await recall_topic(name, tool_context=ctx)
            assert r["found"] is True
            assert expected in r["content"]

    @pytest.mark.asyncio
    async def test_read_agent_config_returns_empty_when_no_files(self, workspace: Path) -> None:
        from code_agent.tools.memory_tools import read_agent_config

        ctx = FakeToolContext(workspace)
        result = await read_agent_config(tool_context=ctx)
        assert result["agent_config"] == ""
        assert result["memory_index"] == []

    @pytest.mark.asyncio
    async def test_read_agent_config_loads_agent_md(self, workspace: Path) -> None:
        from code_agent.tools.memory_tools import read_agent_config

        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "AGENT.md").write_text("# My Project\nPython 3.12, FastAPI.")

        ctx = FakeToolContext(workspace)
        result = await read_agent_config(tool_context=ctx)
        assert "My Project" in result["agent_config"]

    @pytest.mark.asyncio
    async def test_summary_auto_derived_from_content(self, workspace: Path) -> None:
        """When summary arg is empty, first line of content is used."""
        from code_agent.tools.memory_tools import remember, list_memory_topics

        ctx = FakeToolContext(workspace)
        await remember("auto-summary", "# Auto Heading\nDetails here.", tool_context=ctx)

        topics = await list_memory_topics(tool_context=ctx)
        assert topics["topics"][0]["summary"] == "Auto Heading"
