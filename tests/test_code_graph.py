"""Unit tests for code_agent.services.code_graph.

Covers:
  - _parse_import_targets: Python and JS/TS import extraction
  - _resolve_to_file: module → workspace file mapping
  - build_code_graph: full build from a tmp workspace
  - _try_load_cached / _save_graph: cache round-trip
  - query_subgraph: BFS neighbourhood matching
  - format_graph_context: output shape and token budget
  - load_or_rebuild_graph: rebuilds on SHA mismatch, serves cache on match
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# _parse_import_targets
# ---------------------------------------------------------------------------

class TestParseImportTargets:
    def _call(self, raw: str, lang: str) -> list[str]:
        from code_agent.services.code_graph import _parse_import_targets
        return _parse_import_targets(raw, lang)

    def test_python_from_import(self) -> None:
        raw = "from code_agent.tools.memory_tools import remember"
        result = self._call(raw, "python")
        assert "code_agent.tools.memory_tools" in result

    def test_python_import_statement(self) -> None:
        raw = "import os\nimport pathlib"
        result = self._call(raw, "python")
        assert "os" in result
        assert "pathlib" in result

    def test_python_both_forms(self) -> None:
        raw = "import json\nfrom pathlib import Path"
        result = self._call(raw, "python")
        assert "json" in result
        assert "pathlib" in result

    def test_js_relative_import(self) -> None:
        raw = "import { foo } from './utils'\nimport bar from '../lib/helper'"
        result = self._call(raw, "javascript")
        assert "./utils" in result
        assert "../lib/helper" in result

    def test_js_external_import_excluded(self) -> None:
        raw = "import React from 'react'\nimport { useState } from 'react'"
        result = self._call(raw, "javascript")
        assert result == []  # external packages have no relative prefix

    def test_ts_relative_import(self) -> None:
        raw = "import { ApiClient } from './api/client'"
        result = self._call(raw, "typescript")
        assert "./api/client" in result

    def test_unsupported_language_empty(self) -> None:
        result = self._call("uses something.rb", "ruby")
        assert result == []

    def test_empty_raw_returns_empty(self) -> None:
        assert self._call("", "python") == []


# ---------------------------------------------------------------------------
# _resolve_to_file
# ---------------------------------------------------------------------------

class TestResolveToFile:
    def test_python_simple_module(self, tmp_path: Path) -> None:
        from code_agent.services.code_graph import _resolve_to_file

        (tmp_path / "myapp" / "utils.py").parent.mkdir(parents=True)
        (tmp_path / "myapp" / "utils.py").write_text("def helper(): pass")

        result = _resolve_to_file("myapp.utils", "python", "myapp/main.py", tmp_path)
        assert result == "myapp/utils.py"

    def test_python_package_init(self, tmp_path: Path) -> None:
        from code_agent.services.code_graph import _resolve_to_file

        pkg = tmp_path / "myapp" / "services"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")

        result = _resolve_to_file("myapp.services", "python", "myapp/main.py", tmp_path)
        assert result == "myapp/services/__init__.py"

    def test_python_missing_module_returns_none(self, tmp_path: Path) -> None:
        from code_agent.services.code_graph import _resolve_to_file

        result = _resolve_to_file("nonexistent.module", "python", "main.py", tmp_path)
        assert result is None

    def test_python_relative_import_returns_none(self, tmp_path: Path) -> None:
        from code_agent.services.code_graph import _resolve_to_file

        result = _resolve_to_file(".utils", "python", "myapp/main.py", tmp_path)
        assert result is None

    def test_js_relative_ts_file(self, tmp_path: Path) -> None:
        from code_agent.services.code_graph import _resolve_to_file

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "utils.ts").write_text("export function foo() {}")

        result = _resolve_to_file("./utils", "typescript", "src/main.ts", tmp_path)
        assert result == "src/utils.ts"

    def test_unknown_language_returns_none(self, tmp_path: Path) -> None:
        from code_agent.services.code_graph import _resolve_to_file

        result = _resolve_to_file("something", "go", "main.go", tmp_path)
        assert result is None


# ---------------------------------------------------------------------------
# build_code_graph
# ---------------------------------------------------------------------------

class TestBuildCodeGraph:
    def _make_workspace(self, tmp_path: Path) -> Path:
        """Create a minimal Python workspace."""
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "app.py").write_text(
            "import os\nfrom utils import helper\n\ndef main():\n    pass\n"
            "class App:\n    pass\n"
        )
        (ws / "utils.py").write_text("def helper():\n    return 42\n")
        return ws

    def test_nodes_contain_source_files(self, tmp_path: Path) -> None:
        from code_agent.services.code_graph import build_code_graph

        ws = self._make_workspace(tmp_path)
        graph = build_code_graph(ws, head_sha="abc123")

        assert "app.py" in graph.nodes
        assert "utils.py" in graph.nodes

    def test_symbols_extracted(self, tmp_path: Path) -> None:
        from code_agent.services.code_graph import build_code_graph

        ws = self._make_workspace(tmp_path)
        graph = build_code_graph(ws, head_sha="abc123")

        app_node = graph.nodes.get("app.py")
        assert app_node is not None
        assert "main" in app_node.symbols or "App" in app_node.symbols

    def test_import_edge_created(self, tmp_path: Path) -> None:
        from code_agent.services.code_graph import build_code_graph

        ws = self._make_workspace(tmp_path)
        graph = build_code_graph(ws, head_sha="abc123")

        # app.py imports utils → should produce an edge to utils.py
        edges_from_app = [(s, t, k) for s, t, k in graph.edges if s == "app.py"]
        targets = [t for _, t, _ in edges_from_app]
        assert "utils.py" in targets

    def test_head_sha_stored(self, tmp_path: Path) -> None:
        from code_agent.services.code_graph import build_code_graph

        ws = self._make_workspace(tmp_path)
        graph = build_code_graph(ws, head_sha="deadbeef")
        assert graph.head_sha == "deadbeef"

    def test_skip_dirs_excluded(self, tmp_path: Path) -> None:
        from code_agent.services.code_graph import build_code_graph

        ws = self._make_workspace(tmp_path)
        cache_dir = ws / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "app.cpython-311.pyc").write_bytes(b"\x00\x01")

        graph = build_code_graph(ws)
        assert not any("__pycache__" in fp for fp in graph.nodes)

    def test_empty_workspace_produces_empty_graph(self, tmp_path: Path) -> None:
        from code_agent.services.code_graph import build_code_graph

        ws = tmp_path / "empty"
        ws.mkdir()
        graph = build_code_graph(ws)
        assert graph.file_count == 0
        assert graph.nodes == {}


# ---------------------------------------------------------------------------
# Cache round-trip (_save_graph / _try_load_cached)
# ---------------------------------------------------------------------------

class TestCacheRoundTrip:
    def test_save_and_load_with_matching_sha(self, tmp_path: Path) -> None:
        from code_agent.services.code_graph import (
            FileNode, CodeGraph, _save_graph, _try_load_cached,
        )

        nodes = {"main.py": FileNode("main.py", "python", ["run"], "")}
        graph = CodeGraph(nodes=nodes, edges=[], head_sha="abc", built_at=time.time(), file_count=1)

        _save_graph(graph, tmp_path)
        loaded = _try_load_cached(tmp_path, "abc")

        assert loaded is not None
        assert "main.py" in loaded.nodes
        assert loaded.nodes["main.py"].symbols == ["run"]

    def test_sha_mismatch_returns_none(self, tmp_path: Path) -> None:
        from code_agent.services.code_graph import (
            FileNode, CodeGraph, _save_graph, _try_load_cached,
        )

        nodes = {"a.py": FileNode("a.py", "python", ["foo"], "")}
        graph = CodeGraph(nodes=nodes, edges=[], head_sha="old", built_at=time.time(), file_count=1)
        _save_graph(graph, tmp_path)

        result = _try_load_cached(tmp_path, "new_sha")
        assert result is None

    def test_missing_cache_returns_none(self, tmp_path: Path) -> None:
        from code_agent.services.code_graph import _try_load_cached

        result = _try_load_cached(tmp_path / "no_such_ws", "any")
        assert result is None

    def test_edges_round_trip(self, tmp_path: Path) -> None:
        from code_agent.services.code_graph import (
            FileNode, CodeGraph, _save_graph, _try_load_cached,
        )

        nodes = {
            "a.py": FileNode("a.py", "python", ["A"], ""),
            "b.py": FileNode("b.py", "python", ["B"], ""),
        }
        edges = [["a.py", "b.py", "import"]]
        graph = CodeGraph(nodes=nodes, edges=edges, head_sha="x", built_at=time.time(), file_count=2)
        _save_graph(graph, tmp_path)

        loaded = _try_load_cached(tmp_path, "x")
        assert loaded is not None
        assert loaded.edges == [["a.py", "b.py", "import"]]


# ---------------------------------------------------------------------------
# query_subgraph
# ---------------------------------------------------------------------------

class TestQuerySubgraph:
    def _make_graph(self) -> "CodeGraph":
        from code_agent.services.code_graph import FileNode, CodeGraph

        nodes = {
            "auth/login.py":  FileNode("auth/login.py",  "python", ["login", "logout"], ""),
            "auth/token.py":  FileNode("auth/token.py",  "python", ["verify_token"],    ""),
            "api/handler.py": FileNode("api/handler.py", "python", ["handle_request"],  ""),
            "utils/crypto.py":FileNode("utils/crypto.py","python", ["hash_password"],   ""),
        }
        edges = [
            ["api/handler.py",  "auth/login.py",  "import"],
            ["auth/login.py",   "auth/token.py",  "import"],
            ["auth/login.py",   "utils/crypto.py","import"],
        ]
        return CodeGraph(nodes=nodes, edges=edges, head_sha="x", built_at=0.0, file_count=4)

    def test_matches_file_path(self) -> None:
        from code_agent.services.code_graph import query_subgraph
        graph = self._make_graph()
        result = query_subgraph("auth/login", graph)
        assert "auth/login.py" in result

    def test_matches_symbol_name(self) -> None:
        from code_agent.services.code_graph import query_subgraph
        graph = self._make_graph()
        result = query_subgraph("verify_token", graph)
        assert "auth/token.py" in result

    def test_bfs_depth_1_includes_direct_neighbours(self) -> None:
        from code_agent.services.code_graph import query_subgraph
        graph = self._make_graph()
        # Seed: api/handler.py; depth=1 should include auth/login.py
        result = query_subgraph("handler", graph, depth=1)
        assert "api/handler.py" in result
        assert "auth/login.py" in result

    def test_bfs_depth_2_includes_transitive(self) -> None:
        from code_agent.services.code_graph import query_subgraph
        graph = self._make_graph()
        # api/handler → auth/login → auth/token (2 hops)
        result = query_subgraph("handler", graph, depth=2)
        assert "auth/token.py" in result
        assert "utils/crypto.py" in result

    def test_no_match_returns_empty(self) -> None:
        from code_agent.services.code_graph import query_subgraph
        graph = self._make_graph()
        result = query_subgraph("nonexistent_xyz", graph)
        assert result == {}

    def test_empty_query_returns_empty(self) -> None:
        from code_agent.services.code_graph import query_subgraph
        graph = self._make_graph()
        assert query_subgraph("", graph) == {}

    def test_case_insensitive_match(self) -> None:
        from code_agent.services.code_graph import query_subgraph
        graph = self._make_graph()
        result = query_subgraph("LOGIN", graph)
        assert "auth/login.py" in result


# ---------------------------------------------------------------------------
# format_graph_context
# ---------------------------------------------------------------------------

class TestFormatGraphContext:
    def _make_graph(self, file_count: int = 3) -> "CodeGraph":
        from code_agent.services.code_graph import FileNode, CodeGraph

        nodes = {
            f"module{i}.py": FileNode(f"module{i}.py", "python", [f"func{i}", f"Class{i}"], "")
            for i in range(file_count)
        }
        return CodeGraph(nodes=nodes, edges=[], head_sha="x", built_at=0.0, file_count=file_count)

    def test_contains_header_stats(self) -> None:
        from code_agent.services.code_graph import format_graph_context
        graph = self._make_graph()
        result = format_graph_context(graph)
        assert "files" in result
        assert "symbols" in result

    def test_respects_char_limit(self) -> None:
        from code_agent.services.code_graph import format_graph_context, _GRAPH_CONTEXT_MAX_CHARS
        # Build a very large graph
        from code_agent.services.code_graph import FileNode, CodeGraph
        nodes = {
            f"dir{i}/module{j}.py": FileNode(f"dir{i}/module{j}.py", "python",
                                              [f"func{k}" for k in range(20)], "")
            for i in range(10) for j in range(10)
        }
        graph = CodeGraph(nodes=nodes, edges=[], head_sha="x", built_at=0.0, file_count=100)
        result = format_graph_context(graph)
        assert len(result) <= _GRAPH_CONTEXT_MAX_CHARS + 50  # small buffer for last entry

    def test_empty_graph_returns_empty(self) -> None:
        from code_agent.services.code_graph import format_graph_context, CodeGraph
        graph = CodeGraph(nodes={}, edges=[], head_sha="x", built_at=0.0, file_count=0)
        assert format_graph_context(graph) == ""

    def test_focus_term_prioritised(self) -> None:
        from code_agent.services.code_graph import format_graph_context, FileNode, CodeGraph

        nodes = {
            "auth/login.py": FileNode("auth/login.py", "python", ["login"], ""),
            "billing/invoice.py": FileNode("billing/invoice.py", "python", ["invoice"] * 20, ""),
        }
        graph = CodeGraph(nodes=nodes, edges=[], head_sha="x", built_at=0.0, file_count=2)
        result = format_graph_context(graph, focus="auth")
        # auth should appear before billing despite fewer symbols
        auth_pos = result.find("auth")
        billing_pos = result.find("billing")
        assert auth_pos < billing_pos or billing_pos == -1

    def test_symbols_truncated_with_overflow_indicator(self) -> None:
        from code_agent.services.code_graph import format_graph_context, FileNode, CodeGraph

        nodes = {
            "big.py": FileNode("big.py", "python", [f"func{i}" for i in range(20)], ""),
        }
        graph = CodeGraph(nodes=nodes, edges=[], head_sha="x", built_at=0.0, file_count=1)
        result = format_graph_context(graph)
        assert "+" in result  # overflow indicator present


# ---------------------------------------------------------------------------
# load_or_rebuild_graph (async, integration)
# ---------------------------------------------------------------------------

class TestLoadOrRebuildGraph:
    @pytest.mark.asyncio
    async def test_builds_fresh_graph_when_no_cache(self, tmp_path: Path) -> None:
        from code_agent.services.code_graph import load_or_rebuild_graph

        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "main.py").write_text("def run(): pass\n")

        with patch(
            "code_agent.services.code_graph._get_head_sha",
            new=AsyncMock(return_value="abc123"),
        ):
            graph = await load_or_rebuild_graph(ws)

        assert graph is not None
        assert graph.file_count >= 1

    @pytest.mark.asyncio
    async def test_serves_cache_on_sha_match(self, tmp_path: Path) -> None:
        from code_agent.services.code_graph import (
            load_or_rebuild_graph, FileNode, CodeGraph, _save_graph,
        )

        ws = tmp_path / "ws"
        ws.mkdir()

        # Pre-populate cache
        nodes = {"cached.py": FileNode("cached.py", "python", ["cached_fn"], "")}
        pre = CodeGraph(nodes=nodes, edges=[], head_sha="sha1", built_at=time.time(), file_count=1)
        _save_graph(pre, ws)

        with patch(
            "code_agent.services.code_graph._get_head_sha",
            new=AsyncMock(return_value="sha1"),
        ):
            graph = await load_or_rebuild_graph(ws)

        assert graph is not None
        assert "cached.py" in graph.nodes

    @pytest.mark.asyncio
    async def test_rebuilds_on_sha_mismatch(self, tmp_path: Path) -> None:
        from code_agent.services.code_graph import (
            load_or_rebuild_graph, FileNode, CodeGraph, _save_graph,
        )

        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "new.py").write_text("def new_fn(): pass\n")

        # Stale cache with old SHA
        nodes = {"old.py": FileNode("old.py", "python", ["old_fn"], "")}
        old = CodeGraph(nodes=nodes, edges=[], head_sha="old_sha", built_at=0.0, file_count=1)
        _save_graph(old, ws)

        with patch(
            "code_agent.services.code_graph._get_head_sha",
            new=AsyncMock(return_value="new_sha"),
        ):
            graph = await load_or_rebuild_graph(ws)

        assert graph is not None
        assert "new.py" in graph.nodes
        assert "old.py" not in graph.nodes

    @pytest.mark.asyncio
    async def test_returns_none_on_build_failure(self, tmp_path: Path) -> None:
        from code_agent.services.code_graph import load_or_rebuild_graph

        ws = tmp_path / "ws"
        ws.mkdir()

        with patch(
            "code_agent.services.code_graph._get_head_sha",
            new=AsyncMock(return_value="sha1"),
        ), patch(
            "code_agent.services.code_graph.build_code_graph",
            side_effect=RuntimeError("boom"),
        ):
            graph = await load_or_rebuild_graph(ws)

        assert graph is None
