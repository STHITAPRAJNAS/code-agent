"""Code knowledge graph — lightweight structural map of the repository.

Builds a two-layer graph:
  • Nodes  — source files with their exported symbols (functions / classes)
  • Edges  — import/dependency relationships between files

The graph is stored in .agent/graph/ inside the workspace and rebuilt only
when the git HEAD SHA changes.  A compact text summary (~150 tokens) is
injected into the agent's system instruction via before_agent_callback so
the agent can navigate the codebase without loading individual files.

Token cost profile
  Without graph: codebase orientation ≈ 4 000 tok (file reads)
  With graph:    same orientation     ≈   300 tok (graph summary + targeted queries)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Storage layout
# ---------------------------------------------------------------------------
_GRAPH_DIR = ".agent/graph"
_META_FILE = "meta.json"
_INDEX_FILE = "index.json"
_EDGES_FILE = "edges.json"

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
_MAX_FILES = 2_000           # skip build on huge repos
_MAX_SYMBOLS_PER_FILE = 40   # cap stored symbols per file to keep index compact
_GRAPH_CONTEXT_MAX_CHARS = 600  # ~150 tokens for system-instruction injection
_BUILD_TIMEOUT_SECS = 15     # async executor timeout for graph build

_SKIP_DIRS: frozenset[str] = frozenset({
    ".git", ".agent", "node_modules", "__pycache__", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build", ".next",
    "vendor", "third_party", ".eggs", "htmlcov", "site-packages",
})

_SKIP_EXTENSIONS: frozenset[str] = frozenset({
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".exe", ".whl",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg",
    ".pdf", ".zip", ".tar", ".gz", ".bz2", ".lock", ".bin",
})

# ---------------------------------------------------------------------------
# Import-line regexes — one pass over the raw import block per language
# ---------------------------------------------------------------------------
_PY_FROM_RE = re.compile(r"^from\s+([\w.]+)\s+import", re.MULTILINE)
_PY_IMPORT_RE = re.compile(r"^import\s+([\w.]+)", re.MULTILINE)
_PY_DEF_RE = re.compile(r"^(?:async\s+)?def\s+(\w+)\s*\(", re.MULTILINE)
_PY_CLASS_RE = re.compile(r"^class\s+(\w+)[\s:(]", re.MULTILINE)
# Matches: from './foo', from "../bar", require('./baz')
_JS_RELATIVE_RE = re.compile(r"""(?:from\s+|require\s*\(\s*)['"](\.{1,2}/[^'"]+)['"]""", re.MULTILINE)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FileNode:
    file_path: str      # relative to workspace root
    language: str
    symbols: list[str]  # function / class names
    imports_raw: str    # raw import lines collected by tree-sitter


@dataclass
class CodeGraph:
    nodes: dict[str, FileNode]    # file_path → FileNode
    edges: list[list[str]]        # [[source_file, target_file, kind], ...]
    head_sha: str
    built_at: float
    file_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {k: asdict(v) for k, v in self.nodes.items()},
            "edges": self.edges,
            "head_sha": self.head_sha,
            "built_at": self.built_at,
            "file_count": self.file_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeGraph":
        nodes = {k: FileNode(**v) for k, v in data["nodes"].items()}
        return cls(
            nodes=nodes,
            edges=data["edges"],
            head_sha=data["head_sha"],
            built_at=data["built_at"],
            file_count=data["file_count"],
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def load_or_rebuild_graph(workspace: Path) -> CodeGraph | None:
    """Return the cached graph if still current; rebuild and cache if stale.

    Returns ``None`` if the workspace has no source files or if the build
    fails — callers should treat ``None`` as "graph unavailable" and skip
    graph-context injection rather than aborting.
    """
    current_sha = await _get_head_sha(workspace)
    cached = _try_load_cached(workspace, current_sha)
    if cached is not None:
        logger.debug("code_graph: loaded from cache (sha=%s, files=%d)", current_sha, cached.file_count)
        return cached

    try:
        loop = asyncio.get_event_loop()
        graph = await asyncio.wait_for(
            loop.run_in_executor(None, build_code_graph, workspace, current_sha),
            timeout=_BUILD_TIMEOUT_SECS,
        )
    except asyncio.TimeoutError:
        logger.warning("code_graph: build timed out after %ds", _BUILD_TIMEOUT_SECS)
        return None
    except Exception as exc:
        logger.warning("code_graph: build failed: %s", exc)
        return None

    try:
        _save_graph(graph, workspace)
    except Exception as exc:
        logger.debug("code_graph: save failed (non-fatal): %s", exc)

    logger.info("code_graph: built (sha=%s, files=%d, symbols=%d, edges=%d)",
                current_sha, graph.file_count,
                sum(len(n.symbols) for n in graph.nodes.values()),
                len(graph.edges))
    return graph


def build_code_graph(workspace: Path, head_sha: str = "") -> CodeGraph:
    """Walk *workspace* synchronously and build a code graph.

    Uses :class:`~code_agent.indexer.chunker.CodeChunker` for symbol
    extraction (tree-sitter where available, regex fallback otherwise).
    Import edges are resolved for Python; raw import strings are stored
    for all other languages.
    """
    from code_agent.indexer.chunker import CodeChunker

    chunker = CodeChunker()
    nodes: dict[str, FileNode] = {}
    edges: list[list[str]] = []
    file_count = 0

    for fpath in _walk_source_files(workspace):
        if file_count >= _MAX_FILES:
            break
        rel = str(fpath.relative_to(workspace))
        try:
            content = fpath.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        chunks = chunker.chunk_file(rel, content)
        if not chunks:
            continue

        language = chunks[0].language
        symbols = [c.symbol_name for c in chunks if c.symbol_name][:_MAX_SYMBOLS_PER_FILE]
        # When tree-sitter falls back to line chunks, symbol names are empty.
        # Recover top-level definitions with a lightweight regex pass.
        if not symbols:
            symbols = _extract_symbols_regex(content, language)[:_MAX_SYMBOLS_PER_FILE]

        # Prefer chunker-collected imports; fall back to direct regex scan of the
        # raw file content when tree-sitter couldn't parse it (line-window fallback
        # produces chunks with empty imports_context).
        imports_raw = chunks[0].imports_context or _extract_imports_from_content(content, language)

        nodes[rel] = FileNode(
            file_path=rel,
            language=language,
            symbols=symbols,
            imports_raw=imports_raw,
        )

        for target in _parse_import_targets(imports_raw, language):
            resolved = _resolve_to_file(target, language, rel, workspace)
            if resolved:
                edges.append([rel, resolved, "import"])

        file_count += 1

    return CodeGraph(
        nodes=nodes,
        edges=edges,
        head_sha=head_sha,
        built_at=time.time(),
        file_count=file_count,
    )


def query_subgraph(query: str, graph: CodeGraph, depth: int = 2) -> dict[str, FileNode]:
    """Return nodes within *depth* import hops of any file/symbol matching *query*.

    Matching is case-insensitive substring on file path and symbol names.
    Returns an empty dict when the graph is empty or nothing matches.
    """
    if not graph.nodes or not query:
        return {}

    q = query.lower()
    seeds: set[str] = set()
    for fp, node in graph.nodes.items():
        if q in fp.lower():
            seeds.add(fp)
            continue
        for sym in node.symbols:
            if q in sym.lower():
                seeds.add(fp)
                break

    if not seeds:
        return {}

    # Build a bidirectional adjacency index for BFS
    adj: dict[str, set[str]] = {}
    for src, tgt, _kind in graph.edges:
        adj.setdefault(src, set()).add(tgt)
        adj.setdefault(tgt, set()).add(src)

    visited: set[str] = set(seeds)
    frontier = set(seeds)
    for _ in range(depth):
        next_frontier: set[str] = set()
        for node_path in frontier:
            for neighbour in adj.get(node_path, set()):
                if neighbour not in visited:
                    visited.add(neighbour)
                    next_frontier.add(neighbour)
        frontier = next_frontier

    return {fp: graph.nodes[fp] for fp in visited if fp in graph.nodes}


def format_graph_context(graph: CodeGraph, focus: str = "") -> str:
    """Return a compact codebase summary for system-instruction injection.

    Stays within :data:`_GRAPH_CONTEXT_MAX_CHARS` (~150 tokens).  When
    *focus* is provided the output emphasises files relevant to that term.
    """
    if not graph.nodes:
        return ""

    total_symbols = sum(len(n.symbols) for n in graph.nodes.values())
    header = (
        f"Codebase graph: {graph.file_count} files · "
        f"{total_symbols} symbols · "
        f"{len(graph.edges)} import edges"
    )

    lines: list[str] = [header]
    chars_used = len(header)

    # Prioritise focus-matching nodes, then sort by symbol count descending
    def _priority(item: tuple[str, FileNode]) -> tuple[int, int]:
        fp, node = item
        is_focus = int(bool(focus and focus.lower() in fp.lower()))
        return (is_focus, len(node.symbols))

    sorted_nodes = sorted(graph.nodes.items(), key=_priority, reverse=True)

    # Group by top-level directory for readability
    by_dir: dict[str, list[tuple[str, FileNode]]] = {}
    for fp, node in sorted_nodes:
        top = fp.split("/")[0] if "/" in fp else "."
        by_dir.setdefault(top, []).append((fp, node))

    def _dir_key(item: tuple[str, list]) -> tuple[int, int]:
        dname, fnodes = item
        has_focus = int(bool(focus and any(focus.lower() in fp.lower() for fp, _ in fnodes)))
        return (-has_focus, -max(len(n.symbols) for _, n in fnodes))

    for dir_name, file_nodes in sorted(by_dir.items(), key=_dir_key):
        if chars_used >= _GRAPH_CONTEXT_MAX_CHARS:
            break
        # At most 3 richest files per directory
        for fp, node in file_nodes[:3]:
            if chars_used >= _GRAPH_CONTEXT_MAX_CHARS:
                break
            sym_list = node.symbols[:5]
            sym_str = ", ".join(sym_list)
            if len(node.symbols) > 5:
                sym_str += f" +{len(node.symbols) - 5}"
            entry = f"  {fp}: [{sym_str}]"
            if chars_used + len(entry) + 1 > _GRAPH_CONTEXT_MAX_CHARS:
                remaining = _GRAPH_CONTEXT_MAX_CHARS - chars_used - 20
                if remaining > 10:
                    lines.append(f"  {fp[:remaining]}…")
                break
            lines.append(entry)
            chars_used += len(entry) + 1  # +1 for newline

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers — file walking
# ---------------------------------------------------------------------------

def _walk_source_files(workspace: Path) -> list[Path]:
    """Yield source files from *workspace*, skipping binary/generated paths."""
    result: list[Path] = []
    for root, dirs, files in os.walk(workspace):
        dirs[:] = [
            d for d in dirs
            if d not in _SKIP_DIRS and not d.startswith(".")
        ]
        for fname in files:
            fpath = Path(root) / fname
            if fpath.suffix.lower() in _SKIP_EXTENSIONS:
                continue
            result.append(fpath)
    return result


# ---------------------------------------------------------------------------
# Internal helpers — import parsing & resolution
# ---------------------------------------------------------------------------

def _extract_symbols_regex(content: str, language: str) -> list[str]:
    """Regex-based symbol extraction — fallback when tree-sitter is unavailable."""
    if language == "python":
        syms: list[str] = []
        for m in _PY_CLASS_RE.finditer(content):
            syms.append(m.group(1))
        for m in _PY_DEF_RE.finditer(content):
            syms.append(m.group(1))
        return syms
    return []


def _extract_imports_from_content(content: str, language: str) -> str:
    """Extract import lines from raw file content by language-specific regex."""
    if language == "python":
        lines = [
            line for line in content.splitlines()
            if line.startswith("import ") or line.startswith("from ")
        ]
        return "\n".join(lines)
    if language in ("javascript", "typescript"):
        # Return the whole content so _JS_RELATIVE_RE can scan it
        return content
    return ""


def _parse_import_targets(imports_raw: str, language: str) -> list[str]:
    """Extract referenced module names from raw import text."""
    if not imports_raw:
        return []

    if language == "python":
        targets: list[str] = []
        for m in _PY_FROM_RE.finditer(imports_raw):
            targets.append(m.group(1))
        for m in _PY_IMPORT_RE.finditer(imports_raw):
            targets.append(m.group(1))
        return targets

    if language in ("javascript", "typescript"):
        return [m.group(1) for m in _JS_RELATIVE_RE.finditer(imports_raw)]

    return []


def _resolve_to_file(
    module: str,
    language: str,
    source_file: str,
    workspace: Path,
) -> str | None:
    """Map a module reference to a relative workspace file path, or None."""
    if language == "python":
        if module.startswith("."):
            return None  # relative imports — skip for now
        candidate = Path(module.replace(".", "/"))
        for ext in (".py",):
            if (workspace / (str(candidate) + ext)).exists():
                return str(candidate) + ext
        if (workspace / candidate / "__init__.py").exists():
            return str(candidate / "__init__.py")

    elif language in ("javascript", "typescript"):
        source_dir = Path(source_file).parent
        raw = module.lstrip("./")
        for ext in (".ts", ".tsx", ".js", ".jsx"):
            candidate_rel = source_dir / (raw + ext)
            if (workspace / candidate_rel).exists():
                return str(candidate_rel)

    return None


# ---------------------------------------------------------------------------
# Internal helpers — cache load / save / git SHA
# ---------------------------------------------------------------------------

def _try_load_cached(workspace: Path, head_sha: str) -> CodeGraph | None:
    """Return a cached graph if it matches *head_sha*, else None."""
    meta_path = workspace / _GRAPH_DIR / _META_FILE
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        if not head_sha or meta.get("head_sha") != head_sha:
            return None
        index_path = workspace / _GRAPH_DIR / _INDEX_FILE
        edges_path = workspace / _GRAPH_DIR / _EDGES_FILE
        if not (index_path.exists() and edges_path.exists()):
            return None
        nodes_raw = json.loads(index_path.read_text())
        edges = json.loads(edges_path.read_text())
        nodes = {k: FileNode(**v) for k, v in nodes_raw.items()}
        return CodeGraph(
            nodes=nodes,
            edges=edges,
            head_sha=head_sha,
            built_at=meta["built_at"],
            file_count=meta["file_count"],
        )
    except Exception as exc:
        logger.debug("code_graph: cache load failed: %s", exc)
        return None


def _save_graph(graph: CodeGraph, workspace: Path) -> None:
    """Write graph to .agent/graph/."""
    graph_dir = workspace / _GRAPH_DIR
    graph_dir.mkdir(parents=True, exist_ok=True)

    (graph_dir / _INDEX_FILE).write_text(
        json.dumps({fp: asdict(node) for fp, node in graph.nodes.items()}, indent=2)
    )
    (graph_dir / _EDGES_FILE).write_text(json.dumps(graph.edges))
    (graph_dir / _META_FILE).write_text(
        json.dumps({
            "head_sha": graph.head_sha,
            "built_at": graph.built_at,
            "file_count": graph.file_count,
        }, indent=2)
    )


async def _get_head_sha(workspace: Path) -> str:
    """Return the current git HEAD short SHA, empty string on error."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "-C", str(workspace), "rev-parse", "--short", "HEAD",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        return stdout.decode().strip()
    except Exception:
        return ""
