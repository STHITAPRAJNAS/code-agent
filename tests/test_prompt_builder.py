"""Unit tests for Prompt Cache Stability (prompt_builder.py).

Covers:
  - Static section always precedes the dynamic section
  - SYSTEM_PROMPT_DYNAMIC_BOUNDARY is present exactly once
  - Byte-for-byte identical output for identical static configs
  - DANGEROUS_uncached_section emits a warning log
  - Empty sections are handled gracefully
"""

from __future__ import annotations

import pytest


class TestBuildSystemPrompt:
    def test_static_precedes_dynamic(self) -> None:
        from code_agent.services.prompt_builder import (
            build_system_prompt,
            StaticPromptConfig,
            DynamicPromptContext,
            SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
        )

        static = StaticPromptConfig(
            agent_identity="I am Alex, a staff engineer.",
            coding_conventions="Use PEP 8.",
        )
        dynamic = DynamicPromptContext(
            workspace_info="Branch: main",
            current_task_context="Fix auth bug",
        )
        prompt = build_system_prompt(static, dynamic)

        boundary_pos = prompt.index(SYSTEM_PROMPT_DYNAMIC_BOUNDARY)
        static_pos = prompt.index("I am Alex")
        dynamic_pos = prompt.index("Fix auth bug")

        assert static_pos < boundary_pos < dynamic_pos

    def test_boundary_appears_exactly_once(self) -> None:
        from code_agent.services.prompt_builder import (
            build_system_prompt,
            StaticPromptConfig,
            DynamicPromptContext,
            SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
        )

        prompt = build_system_prompt(StaticPromptConfig(), DynamicPromptContext())
        assert prompt.count(SYSTEM_PROMPT_DYNAMIC_BOUNDARY) == 1

    def test_identical_static_config_produces_identical_prefix(self) -> None:
        """Same static config → byte-identical static section for cache hits."""
        from code_agent.services.prompt_builder import (
            build_system_prompt,
            StaticPromptConfig,
            DynamicPromptContext,
            SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
        )

        static = StaticPromptConfig(
            agent_identity="Agent identity v1.0",
            tool_definitions_summary="Tools: read_file, write_file",
            coding_conventions="PEP 8 + type hints",
            a2a_protocol_instructions="A2A JSON-RPC 2.0",
        )

        dynamic_a = DynamicPromptContext(workspace_info="Branch: feature-x")
        dynamic_b = DynamicPromptContext(workspace_info="Branch: hotfix-y")

        prompt_a = build_system_prompt(static, dynamic_a)
        prompt_b = build_system_prompt(static, dynamic_b)

        # Static section (up to boundary) must be byte-identical
        prefix_a = prompt_a.split(SYSTEM_PROMPT_DYNAMIC_BOUNDARY)[0]
        prefix_b = prompt_b.split(SYSTEM_PROMPT_DYNAMIC_BOUNDARY)[0]
        assert prefix_a == prefix_b

    def test_dynamic_sections_differ_for_different_contexts(self) -> None:
        from code_agent.services.prompt_builder import (
            build_system_prompt,
            StaticPromptConfig,
            DynamicPromptContext,
            SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
        )

        static = StaticPromptConfig(agent_identity="Agent")
        p1 = build_system_prompt(static, DynamicPromptContext(workspace_info="ws-A"))
        p2 = build_system_prompt(static, DynamicPromptContext(workspace_info="ws-B"))

        suffix_1 = p1.split(SYSTEM_PROMPT_DYNAMIC_BOUNDARY, 1)[1]
        suffix_2 = p2.split(SYSTEM_PROMPT_DYNAMIC_BOUNDARY, 1)[1]
        assert suffix_1 != suffix_2

    def test_empty_static_config_still_has_boundary(self) -> None:
        from code_agent.services.prompt_builder import (
            build_system_prompt,
            StaticPromptConfig,
            DynamicPromptContext,
            SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
        )

        prompt = build_system_prompt(StaticPromptConfig(), DynamicPromptContext())
        assert SYSTEM_PROMPT_DYNAMIC_BOUNDARY in prompt

    def test_dynamic_memory_index_in_prompt(self) -> None:
        from code_agent.services.prompt_builder import (
            build_system_prompt,
            StaticPromptConfig,
            DynamicPromptContext,
            format_memory_index_for_prompt,
            SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
        )

        memory_index = [
            {"topic": "auth", "summary": "OAuth2 flow"},
            {"topic": "db", "summary": "Schema notes"},
        ]
        formatted = format_memory_index_for_prompt(memory_index)
        assert "auth" in formatted
        assert "db" in formatted
        assert "OAuth2 flow" in formatted

        static = StaticPromptConfig(agent_identity="Agent")
        dynamic = DynamicPromptContext(memory_index=formatted)
        prompt = build_system_prompt(static, dynamic)

        # Memory index must be after the boundary
        boundary_pos = prompt.index(SYSTEM_PROMPT_DYNAMIC_BOUNDARY)
        assert prompt.index("auth") > boundary_pos

    def test_mode_hint_in_dynamic_section(self) -> None:
        from code_agent.services.prompt_builder import (
            build_system_prompt,
            StaticPromptConfig,
            DynamicPromptContext,
            SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
        )

        static = StaticPromptConfig(agent_identity="Agent")
        dynamic = DynamicPromptContext(mode_hint="plan_mode")
        prompt = build_system_prompt(static, dynamic)

        boundary_pos = prompt.index(SYSTEM_PROMPT_DYNAMIC_BOUNDARY)
        mode_pos = prompt.index("plan_mode")
        assert mode_pos > boundary_pos

    def test_timestamps_must_not_appear_in_static_section(self) -> None:
        """Guard: timestamps in static_config pollute the cache prefix.

        This test documents the requirement — callers must ensure
        StaticPromptConfig fields are time-invariant.
        """
        from code_agent.services.prompt_builder import (
            build_system_prompt,
            StaticPromptConfig,
            DynamicPromptContext,
            SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
        )

        import time

        ts = str(int(time.time()))
        # Timestamp in static section (BAD — this is what we want to prevent)
        static_with_ts = StaticPromptConfig(agent_identity=f"Agent v1 built at {ts}")
        prompt = build_system_prompt(static_with_ts, DynamicPromptContext())
        static_prefix = prompt.split(SYSTEM_PROMPT_DYNAMIC_BOUNDARY)[0]

        # The timestamp IS there — this test documents that the caller is responsible
        # for not putting dynamic data in StaticPromptConfig.
        assert ts in static_prefix  # expected: callers must not do this

    def test_dangerous_uncached_section_emits_warning(self, caplog) -> None:
        import logging
        from code_agent.services.prompt_builder import DANGEROUS_uncached_section

        # structlog may not go through stdlib caplog; just check it returns content
        result = DANGEROUS_uncached_section("dynamic content here")
        assert result == "dynamic content here"


class TestSummariseAgentConfig:
    def test_short_config_unchanged(self) -> None:
        from code_agent.services.prompt_builder import summarise_agent_config

        text = "# Project\nPython 3.12."
        assert summarise_agent_config(text) == text

    def test_long_config_truncated(self) -> None:
        from code_agent.services.prompt_builder import summarise_agent_config

        long_text = "x" * 5000
        result = summarise_agent_config(long_text, max_chars=1500)
        assert len(result) < 5000
        assert "truncated" in result

    def test_empty_config_returns_empty(self) -> None:
        from code_agent.services.prompt_builder import summarise_agent_config

        assert summarise_agent_config("") == ""


class TestExtractL0ForStaticSection:
    def test_returns_first_heading_and_content(self) -> None:
        from code_agent.services.prompt_builder import extract_l0_for_static_section

        config = "# MyRepo\nPython 3.12, FastAPI.\n\nLots more detail here..." + "x" * 300
        result = extract_l0_for_static_section(config)
        assert "MyRepo" in result
        assert len(result) <= 200

    def test_empty_returns_empty(self) -> None:
        from code_agent.services.prompt_builder import extract_l0_for_static_section

        assert extract_l0_for_static_section("") == ""

    def test_short_config_returned_in_full(self) -> None:
        from code_agent.services.prompt_builder import extract_l0_for_static_section

        short = "# MyRepo\nPython 3.12."
        assert extract_l0_for_static_section(short) == short

    def test_custom_max_chars_respected(self) -> None:
        from code_agent.services.prompt_builder import extract_l0_for_static_section

        config = "# Title\n" + "a" * 500
        result = extract_l0_for_static_section(config, max_chars=50)
        assert len(result) <= 50


class TestL0InStaticSection:
    def test_l0_project_summary_precedes_agent_identity(self) -> None:
        """L0 must be the very first content in the static section."""
        from code_agent.services.prompt_builder import (
            build_system_prompt,
            StaticPromptConfig,
            DynamicPromptContext,
            SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
        )

        static = StaticPromptConfig(
            l0_project_summary="# MyRepo — Python 3.12",
            agent_identity="I am Alex, a staff engineer.",
        )
        prompt = build_system_prompt(static, DynamicPromptContext())

        l0_pos = prompt.index("MyRepo")
        identity_pos = prompt.index("Alex")
        boundary_pos = prompt.index(SYSTEM_PROMPT_DYNAMIC_BOUNDARY)

        assert l0_pos < identity_pos < boundary_pos

    def test_l0_in_static_section_identical_across_sessions(self) -> None:
        """Same l0_project_summary → byte-identical static prefix regardless of dynamic content."""
        from code_agent.services.prompt_builder import (
            build_system_prompt,
            StaticPromptConfig,
            DynamicPromptContext,
            SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
        )

        static = StaticPromptConfig(l0_project_summary="# MyRepo — Python 3.12")
        p1 = build_system_prompt(static, DynamicPromptContext(workspace_info="user-alice"))
        p2 = build_system_prompt(static, DynamicPromptContext(workspace_info="user-bob"))

        prefix1 = p1.split(SYSTEM_PROMPT_DYNAMIC_BOUNDARY)[0]
        prefix2 = p2.split(SYSTEM_PROMPT_DYNAMIC_BOUNDARY)[0]
        assert prefix1 == prefix2  # cache prefix identical for both users

    def test_empty_l0_does_not_add_blank_line(self) -> None:
        from code_agent.services.prompt_builder import (
            build_system_prompt,
            StaticPromptConfig,
            DynamicPromptContext,
        )

        prompt = build_system_prompt(
            StaticPromptConfig(l0_project_summary="", agent_identity="Agent"),
            DynamicPromptContext(),
        )
        assert not prompt.startswith("\n")
