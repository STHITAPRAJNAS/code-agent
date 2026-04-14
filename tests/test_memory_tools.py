"""Unit tests for the 3-layer file-based memory system.

Covers: read/write/forget/list operations, atomic writes, pointer index
parsing, multi-write idempotency, two-tier workspace routing, scope parameter,
L0 extraction, and merged read_agent_config.  No network calls — pure file I/O
using temp directories.
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
# Helper: build minimal fake ToolContexts
# ---------------------------------------------------------------------------


class FakeToolContext:
    def __init__(self, workspace: Path) -> None:
        self.state: dict = {"active_workspace": str(workspace)}


class FakeTwoTierContext:
    """Fake context that exposes both user and repo workspace paths."""

    def __init__(self, user_workspace: Path, repo_workspace: Path) -> None:
        self.state: dict = {
            "active_workspace": str(user_workspace),
            "repo_workspace": str(repo_workspace),
        }


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


# ---------------------------------------------------------------------------
# Two-tier workspace routing
# ---------------------------------------------------------------------------


class TestTwoTierWorkspace:
    @pytest.mark.asyncio
    async def test_repo_scope_writes_to_repo_workspace(self, tmp_path: Path) -> None:
        from code_agent.tools.memory_tools import remember

        user_ws = tmp_path / "user"
        repo_ws = tmp_path / "repo"
        ctx = FakeTwoTierContext(user_ws, repo_ws)

        result = await remember("arch", "# Architecture\nMicroservices.", scope="repo", tool_context=ctx)

        assert result["scope"] == "repo"
        # File must be in the repo workspace, NOT the user workspace
        assert repo_ws.name in result["file_path"]
        assert not (user_ws / ".agent" / "memory" / "arch.md").exists()
        assert (repo_ws / ".agent" / "memory" / "arch.md").exists()

    @pytest.mark.asyncio
    async def test_user_scope_writes_to_user_workspace(self, tmp_path: Path) -> None:
        from code_agent.tools.memory_tools import remember

        user_ws = tmp_path / "user"
        repo_ws = tmp_path / "repo"
        ctx = FakeTwoTierContext(user_ws, repo_ws)

        result = await remember("task", "# Current task\nFix auth bug.", scope="user", tool_context=ctx)

        assert result["scope"] == "user"
        assert user_ws.name in result["file_path"]
        assert not (repo_ws / ".agent" / "memory" / "task.md").exists()

    @pytest.mark.asyncio
    async def test_recall_user_shadows_repo_on_same_topic(self, tmp_path: Path) -> None:
        from code_agent.tools.memory_tools import remember, recall_topic

        user_ws = tmp_path / "user"
        repo_ws = tmp_path / "repo"
        ctx = FakeTwoTierContext(user_ws, repo_ws)

        await remember("auth", "repo version of auth", scope="repo", tool_context=ctx)
        await remember("auth", "user override of auth", scope="user", tool_context=ctx)

        result = await recall_topic("auth", tool_context=ctx)
        assert result["found"] is True
        assert result["scope"] == "user"
        assert "user override" in result["content"]

    @pytest.mark.asyncio
    async def test_recall_falls_back_to_repo_when_not_in_user(self, tmp_path: Path) -> None:
        from code_agent.tools.memory_tools import remember, recall_topic

        user_ws = tmp_path / "user"
        repo_ws = tmp_path / "repo"
        ctx = FakeTwoTierContext(user_ws, repo_ws)

        await remember("db-schema", "# DB Schema\nPostgres tables.", scope="repo", tool_context=ctx)

        result = await recall_topic("db-schema", tool_context=ctx)
        assert result["found"] is True
        assert result["scope"] == "repo"
        assert "Postgres" in result["content"]

    @pytest.mark.asyncio
    async def test_list_topics_merges_both_tiers(self, tmp_path: Path) -> None:
        from code_agent.tools.memory_tools import remember, list_memory_topics

        user_ws = tmp_path / "user"
        repo_ws = tmp_path / "repo"
        ctx = FakeTwoTierContext(user_ws, repo_ws)

        await remember("arch", "Microservices.", scope="repo", tool_context=ctx)
        await remember("api-contracts", "REST API.", scope="repo", tool_context=ctx)
        await remember("current-task", "Fix login bug.", scope="user", tool_context=ctx)

        listing = await list_memory_topics(tool_context=ctx)
        assert listing["count"] == 3
        topics_by_name = {t["topic"]: t for t in listing["topics"]}
        assert topics_by_name["arch"]["scope"] == "repo"
        assert topics_by_name["current-task"]["scope"] == "user"

    @pytest.mark.asyncio
    async def test_list_topics_user_shadows_repo_scope(self, tmp_path: Path) -> None:
        """A topic written to both tiers shows as 'user' in the listing."""
        from code_agent.tools.memory_tools import remember, list_memory_topics

        user_ws = tmp_path / "user"
        repo_ws = tmp_path / "repo"
        ctx = FakeTwoTierContext(user_ws, repo_ws)

        await remember("auth", "repo auth", scope="repo", tool_context=ctx)
        await remember("auth", "user auth override", scope="user", tool_context=ctx)

        listing = await list_memory_topics(tool_context=ctx)
        assert listing["count"] == 1
        assert listing["topics"][0]["scope"] == "user"

    @pytest.mark.asyncio
    async def test_read_agent_config_merges_both_tiers(self, tmp_path: Path) -> None:
        from code_agent.tools.memory_tools import read_agent_config

        user_ws = tmp_path / "user"
        repo_ws = tmp_path / "repo"
        user_ws.mkdir(parents=True)
        repo_ws.mkdir(parents=True)
        (repo_ws / "AGENT.md").write_text("# MyRepo\nPython 3.12, FastAPI.")
        (user_ws / "AGENT.md").write_text("## User Preferences\nPrefer verbose logging.")

        ctx = FakeTwoTierContext(user_ws, repo_ws)
        result = await read_agent_config(tool_context=ctx)

        # Both configs appear in the merged output
        assert "MyRepo" in result["agent_config"]
        assert "User Preferences" in result["agent_config"]
        # Repo content must come first
        assert result["agent_config"].index("MyRepo") < result["agent_config"].index("User Preferences")

    @pytest.mark.asyncio
    async def test_l0_identity_extracted_from_repo_config(self, tmp_path: Path) -> None:
        from code_agent.tools.memory_tools import read_agent_config, L0_CHARS

        user_ws = tmp_path / "user"
        repo_ws = tmp_path / "repo"
        repo_ws.mkdir(parents=True)
        long_config = "# MyRepo\nPython 3.12, FastAPI.\n" + "extra detail " * 50
        (repo_ws / "AGENT.md").write_text(long_config)

        ctx = FakeTwoTierContext(user_ws, repo_ws)
        result = await read_agent_config(tool_context=ctx)

        assert "MyRepo" in result["l0_identity"]
        assert len(result["l0_identity"]) <= L0_CHARS

    @pytest.mark.asyncio
    async def test_forget_repo_scope_removes_only_repo_entry(self, tmp_path: Path) -> None:
        from code_agent.tools.memory_tools import remember, forget, list_memory_topics

        user_ws = tmp_path / "user"
        repo_ws = tmp_path / "repo"
        ctx = FakeTwoTierContext(user_ws, repo_ws)

        await remember("auth", "repo auth knowledge", scope="repo", tool_context=ctx)
        await remember("auth", "user auth notes", scope="user", tool_context=ctx)

        await forget("auth", scope="repo", tool_context=ctx)

        listing = await list_memory_topics(tool_context=ctx)
        # User entry survives; repo entry is gone
        assert listing["count"] == 1
        assert listing["topics"][0]["scope"] == "user"


# ---------------------------------------------------------------------------
# _repo_slug
# ---------------------------------------------------------------------------


class TestRepoSlug:
    def test_github_https_url(self) -> None:
        from code_agent.tools.memory_tools import _repo_slug
        assert _repo_slug("https://github.com/acme/my-service.git") == "acme--my-service"

    def test_github_ssh_url(self) -> None:
        from code_agent.tools.memory_tools import _repo_slug
        assert _repo_slug("git@github.com:acme/my-service.git") == "acme--my-service"

    def test_local_path(self) -> None:
        from code_agent.tools.memory_tools import _repo_slug
        slug = _repo_slug("/home/user/projects/my-service")
        assert "my" in slug or "service" in slug  # basename extracted

    def test_unknown_format_returns_hash(self) -> None:
        from code_agent.tools.memory_tools import _repo_slug
        slug = _repo_slug("???unknown???")
        assert slug.startswith("repo-")
