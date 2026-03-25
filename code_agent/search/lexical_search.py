"""Lexical (literal/regex) search using ripgrep or Python fallback.

Provides fast pattern search across a local directory tree, returning
structured match objects with surrounding context lines.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_RG_TIMEOUT = 30  # seconds


@dataclass
class LexicalMatch:
    """A single line match returned by :class:`LexicalSearcher`."""

    file_path: str
    line_number: int
    line_content: str
    context_before: list[str] = field(default_factory=list)
    context_after: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Language → ripgrep type name mapping
# ---------------------------------------------------------------------------
_LANG_TO_RG_TYPE: dict[str, str] = {
    "python": "py",
    "javascript": "js",
    "typescript": "ts",
    "java": "java",
    "go": "go",
    "rust": "rust",
    "ruby": "ruby",
    "c": "c",
    "cpp": "cpp",
    "csharp": "cs",
    "kotlin": "kotlin",
    "swift": "swift",
    "php": "php",
    "scala": "scala",
    "shell": "sh",
}


class LexicalSearcher:
    """Fast text search across a local directory.

    Tries ``rg`` (ripgrep) first for performance; falls back to a pure-Python
    implementation if ripgrep is not installed.

    Example::

        searcher = LexicalSearcher()
        matches = searcher.search("authenticate", "/repos/myapp", file_type="py")
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        pattern: str,
        path: str,
        file_type: str | None = None,
        case_sensitive: bool = True,
        max_results: int = 50,
        context_lines: int = 2,
    ) -> list[LexicalMatch]:
        """Search for *pattern* under *path*.

        Args:
            pattern:        Regular expression or literal pattern.
            path:           Root directory (or file) to search.
            file_type:      Optional language hint for file filtering (e.g. ``"py"``).
                            Accepts both ripgrep type names (``"py"``) and language
                            names (``"python"``).
            case_sensitive: Use case-sensitive matching (default ``True``).
            max_results:    Maximum number of matches to return.
            context_lines:  Lines of context before/after each match.

        Returns:
            List of :class:`LexicalMatch` objects, truncated to *max_results*.
        """
        if not Path(path).exists():
            logger.warning("LexicalSearcher: path does not exist: %s", path)
            return []

        # Normalise file_type: "python" → "py" etc.
        rg_type = _normalise_file_type(file_type)

        if _rg_available():
            try:
                return self._search_rg(pattern, path, rg_type, case_sensitive, max_results, context_lines)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ripgrep search failed, falling back to Python grep: %s", exc)

        return self._search_python(pattern, path, rg_type, case_sensitive, max_results, context_lines)

    def find_symbol_references(
        self,
        symbol: str,
        path: str,
        language: str | None = None,
    ) -> list[LexicalMatch]:
        """Find all references to *symbol* as a whole word.

        Searches for ``\\bsymbol\\b`` to avoid partial matches.

        Args:
            symbol:   Identifier to find (function name, class name, etc.).
            path:     Root directory to search.
            language: Optional language filter.

        Returns:
            List of :class:`LexicalMatch` objects.
        """
        # Word-boundary pattern to avoid partial matches
        pattern = rf"\b{re.escape(symbol)}\b"
        file_type = _lang_to_rg_type(language) if language else None
        return self.search(
            pattern=pattern,
            path=path,
            file_type=file_type,
            case_sensitive=True,
            max_results=100,
            context_lines=2,
        )

    # ------------------------------------------------------------------
    # ripgrep implementation
    # ------------------------------------------------------------------

    def _search_rg(
        self,
        pattern: str,
        path: str,
        rg_type: str | None,
        case_sensitive: bool,
        max_results: int,
        context_lines: int,
    ) -> list[LexicalMatch]:
        """Run ripgrep and parse its JSON output."""
        cmd = ["rg", "--json", "-e", pattern]

        if not case_sensitive:
            cmd.append("-i")

        if rg_type:
            cmd.extend(["-t", rg_type])

        if context_lines > 0:
            cmd.extend(["-C", str(context_lines)])

        cmd.append(path)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_RG_TIMEOUT,
        )

        # rg exits 1 when no matches found — that's fine
        if result.returncode not in (0, 1):
            raise RuntimeError(f"rg exited {result.returncode}: {result.stderr.strip()}")

        return _parse_rg_json(result.stdout, max_results, context_lines)

    # ------------------------------------------------------------------
    # Python fallback implementation
    # ------------------------------------------------------------------

    def _search_python(
        self,
        pattern: str,
        path: str,
        rg_type: str | None,
        case_sensitive: bool,
        max_results: int,
        context_lines: int,
    ) -> list[LexicalMatch]:
        """Pure-Python grep fallback using ``re``."""
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            compiled = re.compile(pattern, flags)
        except re.error as exc:
            logger.error("Invalid pattern %r: %s", pattern, exc)
            return []

        root = Path(path)
        ext_filter = _rg_type_to_extensions(rg_type)

        matches: list[LexicalMatch] = []

        for filepath in _iter_files(root):
            if ext_filter and filepath.suffix.lstrip(".") not in ext_filter:
                continue
            if len(matches) >= max_results:
                break
            try:
                file_lines = filepath.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue

            for i, line in enumerate(file_lines):
                if len(matches) >= max_results:
                    break
                if compiled.search(line):
                    before = file_lines[max(0, i - context_lines): i]
                    after = file_lines[i + 1: i + 1 + context_lines]
                    matches.append(
                        LexicalMatch(
                            file_path=str(filepath),
                            line_number=i + 1,
                            line_content=line,
                            context_before=before,
                            context_after=after,
                        )
                    )

        return matches


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rg_available() -> bool:
    """Return True if ``rg`` is on PATH."""
    try:
        subprocess.run(
            ["rg", "--version"],
            capture_output=True,
            timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _parse_rg_json(output: str, max_results: int, context_lines: int) -> list[LexicalMatch]:
    """Parse ripgrep ``--json`` output into :class:`LexicalMatch` objects.

    ripgrep emits one JSON object per line; types include ``"match"``,
    ``"context"``, ``"begin"``, ``"end"``, and ``"summary"``.
    """
    matches: list[LexicalMatch] = []
    current_match: LexicalMatch | None = None
    pending_context_before: list[str] = []

    for raw_line in output.splitlines():
        if not raw_line.strip():
            continue
        try:
            obj = json.loads(raw_line)
        except json.JSONDecodeError:
            continue

        msg_type = obj.get("type")
        data = obj.get("data", {})

        if msg_type == "begin":
            pending_context_before = []
            current_match = None

        elif msg_type == "context":
            text = data.get("lines", {}).get("text", "").rstrip("\n")
            if current_match is None:
                # context line BEFORE the match
                pending_context_before.append(text)
                if len(pending_context_before) > context_lines:
                    pending_context_before.pop(0)
            else:
                # context line AFTER the match
                current_match.context_after.append(text)

        elif msg_type == "match":
            if len(matches) >= max_results:
                break

            line_text = data.get("lines", {}).get("text", "").rstrip("\n")
            line_num = data.get("line_number", 0)
            file_path = data.get("path", {}).get("text", "")

            current_match = LexicalMatch(
                file_path=file_path,
                line_number=line_num,
                line_content=line_text,
                context_before=list(pending_context_before),
                context_after=[],
            )
            matches.append(current_match)
            pending_context_before = []

        elif msg_type == "end":
            current_match = None
            pending_context_before = []

    return matches


def _normalise_file_type(file_type: str | None) -> str | None:
    """Convert language name to ripgrep type if needed."""
    if file_type is None:
        return None
    # Already a short rg type like "py", "js"
    if len(file_type) <= 4 and file_type.isalpha():
        return file_type
    return _lang_to_rg_type(file_type)


def _lang_to_rg_type(language: str | None) -> str | None:
    if not language:
        return None
    return _LANG_TO_RG_TYPE.get(language.lower())


def _rg_type_to_extensions(rg_type: str | None) -> set[str]:
    """Return file extensions associated with a ripgrep type for Python fallback."""
    _TYPE_EXTS: dict[str, set[str]] = {
        "py": {"py"},
        "js": {"js", "mjs", "cjs"},
        "ts": {"ts", "tsx"},
        "java": {"java"},
        "go": {"go"},
        "rust": {"rs"},
        "ruby": {"rb"},
        "c": {"c", "h"},
        "cpp": {"cpp", "cc", "cxx", "hpp"},
        "cs": {"cs"},
        "kotlin": {"kt", "kts"},
        "swift": {"swift"},
        "php": {"php"},
        "scala": {"scala"},
        "sh": {"sh", "bash"},
    }
    if rg_type is None:
        return set()
    return _TYPE_EXTS.get(rg_type, set())


def _iter_files(root: Path):
    """Yield all non-hidden files under *root* recursively."""
    skip_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv", "build", "dist"}
    for p in root.rglob("*"):
        if p.is_file() and not any(part.startswith(".") for part in p.parts):
            if not any(skip in p.parts for skip in skip_dirs):
                yield p
