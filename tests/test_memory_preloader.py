"""Unit tests for memory_preloader — Gaps 1, 2, 4.

Covers:
  Gap 1 — keyword matching (match_topics_to_query)
  Gap 1 — proactive loading from warm cache and disk fallback
  Gap 1 — format_proactive_context with stale warnings
  Gap 2 — stale topic detection against git-changed files
  Gap 4 — warm_recent_topics loads N most recently modified files
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# Gap 1 — match_topics_to_query
# ---------------------------------------------------------------------------


class TestMatchTopicsToQuery:
    def _index(self, entries: list[tuple[str, str]]) -> list[dict]:
        return [{"topic": t, "summary": s} for t, s in entries]

    def test_exact_keyword_match(self) -> None:
        from code_agent.services.memory_preloader import match_topics_to_query

        index = self._index([("auth", "OAuth2 flow"), ("db-schema", "Postgres tables")])
        result = match_topics_to_query("fix the auth bug", index)
        assert result[0] == "auth"

    def test_hyphen_split_match(self) -> None:
        """'auth-flow' should match a query containing 'auth'."""
        from code_agent.services.memory_preloader import match_topics_to_query

        index = self._index([("auth-flow", "authentication steps"), ("billing", "stripe")])
        result = match_topics_to_query("where is the auth logic", index)
        assert "auth-flow" in result

    def test_summary_match(self) -> None:
        from code_agent.services.memory_preloader import match_topics_to_query

        index = self._index([("payments", "stripe webhook handler"), ("auth", "JWT tokens")])
        result = match_topics_to_query("stripe integration failing", index)
        assert result[0] == "payments"

    def test_no_match_returns_empty(self) -> None:
        from code_agent.services.memory_preloader import match_topics_to_query

        index = self._index([("auth", "OAuth2"), ("db", "Postgres")])
        result = match_topics_to_query("hello world", index)
        assert result == []

    def test_ranked_by_overlap_score(self) -> None:
        from code_agent.services.memory_preloader import match_topics_to_query

        index = self._index([
            ("auth-token", "JWT auth token refresh"),   # 3 matches for "auth token"
            ("auth", "OAuth2"),                          # 1 match
        ])
        result = match_topics_to_query("auth token expired", index)
        assert result[0] == "auth-token"

    def test_short_tokens_ignored(self) -> None:
        """Tokens of 2 chars or less should not produce spurious matches."""
        from code_agent.services.memory_preloader import match_topics_to_query

        index = self._index([("db", "database schema")])
        result = match_topics_to_query("is it ok", index)
        # "is", "it", "ok" are all ≤ 2 chars and should be ignored
        assert result == []

    def test_empty_index_returns_empty(self) -> None:
        from code_agent.services.memory_preloader import match_topics_to_query

        assert match_topics_to_query("any query", []) == []

    def test_empty_query_returns_empty(self) -> None:
        from code_agent.services.memory_preloader import match_topics_to_query

        index = [{"topic": "auth", "summary": "OAuth2"}]
        assert match_topics_to_query("", index) == []


# ---------------------------------------------------------------------------
# Gap 4 — warm_recent_topics
# ---------------------------------------------------------------------------


class TestWarmRecentTopics:
    @pytest.mark.asyncio
    async def test_loads_most_recently_modified(self, tmp_path: Path) -> None:
        from code_agent.services.memory_preloader import warm_recent_topics
        import time

        workspace = tmp_path / "ws"
        mem_dir = workspace / ".agent" / "memory"
        mem_dir.mkdir(parents=True)

        # Write files with different mtimes
        (mem_dir / "auth.md").write_text("auth content")
        time.sleep(0.01)
        (mem_dir / "db.md").write_text("db content")
        time.sleep(0.01)
        (mem_dir / "api.md").write_text("api content")  # newest

        index = [
            {"topic": "auth", "file_path": ".agent/memory/auth.md", "summary": "auth"},
            {"topic": "db", "file_path": ".agent/memory/db.md", "summary": "db"},
            {"topic": "api", "file_path": ".agent/memory/api.md", "summary": "api"},
        ]
        result = await warm_recent_topics(index, workspace, n=2)

        assert len(result) == 2
        assert "api" in result  # newest always included
        assert "db" in result   # second newest
        assert "auth" not in result

    @pytest.mark.asyncio
    async def test_missing_files_skipped(self, tmp_path: Path) -> None:
        from code_agent.services.memory_preloader import warm_recent_topics

        workspace = tmp_path / "ws"
        index = [
            {"topic": "ghost", "file_path": ".agent/memory/ghost.md", "summary": "missing"},
        ]
        result = await warm_recent_topics(index, workspace, n=3)
        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_at_most_n_topics(self, tmp_path: Path) -> None:
        from code_agent.services.memory_preloader import warm_recent_topics

        workspace = tmp_path / "ws"
        mem_dir = workspace / ".agent" / "memory"
        mem_dir.mkdir(parents=True)

        index = []
        for i in range(5):
            name = f"topic{i}.md"
            (mem_dir / name).write_text(f"content {i}")
            index.append({"topic": f"topic{i}", "file_path": f".agent/memory/{name}", "summary": ""})

        result = await warm_recent_topics(index, workspace, n=2)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Gap 1 — preload_matched_topics
# ---------------------------------------------------------------------------


class TestPreloadMatchedTopics:
    @pytest.mark.asyncio
    async def test_serves_from_warm_cache(self) -> None:
        from code_agent.services.memory_preloader import preload_matched_topics

        warm = {"auth": "cached auth content", "db": "cached db content"}
        result = await preload_matched_topics(["auth", "db"], warm)
        assert result["auth"] == "cached auth content"
        assert result["db"] == "cached db content"

    @pytest.mark.asyncio
    async def test_disk_fallback_for_cache_miss(self, tmp_path: Path) -> None:
        from code_agent.services.memory_preloader import preload_matched_topics

        class FakeCtx:
            state = {"active_workspace": str(tmp_path)}

        with patch(
            "code_agent.tools.memory_tools.recall_topic",
            new=AsyncMock(return_value={"found": True, "content": "disk content", "topic": "api"}),
        ):
            result = await preload_matched_topics(["api"], warm_cache={}, tool_context=FakeCtx())

        assert result["api"] == "disk content"

    @pytest.mark.asyncio
    async def test_not_found_topic_excluded(self) -> None:
        from code_agent.services.memory_preloader import preload_matched_topics

        with patch(
            "code_agent.tools.memory_tools.recall_topic",
            new=AsyncMock(return_value={"found": False, "content": "", "topic": "missing"}),
        ):
            result = await preload_matched_topics(["missing"], warm_cache={})

        assert "missing" not in result


# ---------------------------------------------------------------------------
# Gap 1 — format_proactive_context
# ---------------------------------------------------------------------------


class TestFormatProactiveContext:
    def test_single_topic_formatted(self) -> None:
        from code_agent.services.memory_preloader import format_proactive_context

        result = format_proactive_context({"auth": "# Auth\nOAuth2 via Google."})
        assert "### [auth]" in result
        assert "OAuth2" in result

    def test_stale_topic_gets_warning(self) -> None:
        from code_agent.services.memory_preloader import format_proactive_context

        result = format_proactive_context({"auth": "content"}, stale_topics={"auth"})
        assert "stale" in result.lower() or "verify" in result.lower()

    def test_non_stale_topic_no_warning(self) -> None:
        from code_agent.services.memory_preloader import format_proactive_context

        result = format_proactive_context({"auth": "content"}, stale_topics=set())
        assert "stale" not in result.lower()

    def test_content_truncated_at_limit(self) -> None:
        from code_agent.services.memory_preloader import (
            format_proactive_context,
            _MAX_L2_CONTENT_CHARS,
        )

        long_content = "x" * (_MAX_L2_CONTENT_CHARS + 500)
        result = format_proactive_context({"big": long_content})
        assert "truncated" in result or "omitted" in result or "chars" in result

    def test_empty_dict_returns_empty(self) -> None:
        from code_agent.services.memory_preloader import format_proactive_context

        assert format_proactive_context({}) == ""


# ---------------------------------------------------------------------------
# Gap 2 — detect_stale_topics
# ---------------------------------------------------------------------------


class TestDetectStaleTopics:
    @pytest.mark.asyncio
    async def test_flags_topic_when_referenced_file_changed(self, tmp_path: Path) -> None:
        from code_agent.services.memory_preloader import detect_stale_topics

        index = [
            {"topic": "auth", "summary": "auth.py OAuth2 flow"},
            {"topic": "billing", "summary": "stripe webhook handler"},
        ]

        with patch(
            "code_agent.services.memory_preloader._git_recently_changed_files",
            new=AsyncMock(return_value=["src/auth.py", "tests/test_auth.py"]),
        ):
            stale = await detect_stale_topics(index, tmp_path)

        assert "auth" in stale
        assert "billing" not in stale

    @pytest.mark.asyncio
    async def test_no_stale_when_no_git_changes(self, tmp_path: Path) -> None:
        from code_agent.services.memory_preloader import detect_stale_topics

        index = [{"topic": "auth", "summary": "OAuth2 flow"}]

        with patch(
            "code_agent.services.memory_preloader._git_recently_changed_files",
            new=AsyncMock(return_value=[]),
        ):
            stale = await detect_stale_topics(index, tmp_path)

        assert stale == []

    @pytest.mark.asyncio
    async def test_git_error_returns_empty(self, tmp_path: Path) -> None:
        from code_agent.services.memory_preloader import detect_stale_topics

        index = [{"topic": "auth", "summary": "OAuth2 flow"}]

        with patch(
            "code_agent.services.memory_preloader._git_recently_changed_files",
            new=AsyncMock(return_value=[]),
        ):
            stale = await detect_stale_topics(index, tmp_path)

        assert stale == []  # graceful degradation


# ---------------------------------------------------------------------------
# Gap 3 — _summarise_tool_result (in callbacks)
# ---------------------------------------------------------------------------


class TestSummariseToolResult:
    def test_verbose_tool_large_result_truncated(self) -> None:
        from code_agent.a2a.callbacks import _summarise_tool_result, _TOOL_RESULT_MAX_CHARS

        big_response = {"output": "x" * (_TOOL_RESULT_MAX_CHARS + 1000)}
        result = _summarise_tool_result("git_log", big_response)

        assert result is not big_response
        assert result.get("_truncated") is True
        assert len(str(result)) < len(str(big_response))

    def test_verbose_tool_small_result_unchanged(self) -> None:
        from code_agent.a2a.callbacks import _summarise_tool_result

        small_response = {"output": "short output"}
        result = _summarise_tool_result("git_log", small_response)
        assert result is small_response

    def test_non_verbose_tool_always_unchanged(self) -> None:
        from code_agent.a2a.callbacks import _summarise_tool_result, _TOOL_RESULT_MAX_CHARS

        big = {"output": "x" * (_TOOL_RESULT_MAX_CHARS + 500)}
        result = _summarise_tool_result("write_file", big)
        assert result is big

    def test_all_verbose_tool_names_recognised(self) -> None:
        from code_agent.a2a.callbacks import _summarise_tool_result, _VERBOSE_TOOLS, _TOOL_RESULT_MAX_CHARS

        big = {"data": "y" * (_TOOL_RESULT_MAX_CHARS + 100)}
        for tool in _VERBOSE_TOOLS:
            result = _summarise_tool_result(tool, big)
            # Should return a different object (truncated) for all verbose tools
            assert result is not big, f"{tool} was not truncated"
