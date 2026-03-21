"""
context_builder.py — Utilities for building concise context strings from repos.

:class:`ContextBuilder` produces compact text representations of repository
content suitable for inclusion in an LLM prompt window.  It prioritises
high-signal files (READMEs, manifests, entry-points) and trims content to
respect context-window limits.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Files (by name) that are considered high-signal and always included when present
_HIGH_SIGNAL_FILES: list[str] = [
    "README.md",
    "README.rst",
    "README.txt",
    "README",
    # Python
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    # Node.js / JS
    "package.json",
    # Java / Kotlin
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    # Go
    "go.mod",
    # Rust
    "Cargo.toml",
    # Ruby
    "Gemfile",
    # .NET
    "*.csproj",
    "*.sln",
    # Docker / CI
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    ".github/workflows",
    # General
    "Makefile",
    "justfile",
]

# Common entry-point filenames searched near the repo root
_ENTRYPOINT_NAMES: list[str] = [
    "main.py",
    "app.py",
    "server.py",
    "index.js",
    "index.ts",
    "app.js",
    "app.ts",
    "main.go",
    "main.rs",
    "main.java",
    "Program.cs",
    "index.rb",
    "application.rb",
]

# Max bytes to read from a single file when building context
_MAX_FILE_BYTES = 4_000

# Custom instructions filenames, searched in order
_INSTRUCTION_FILES: list[str] = [
    ".agent-instructions.md",
    ".github/copilot-instructions.md",
]


class ContextBuilder:
    """
    Builds condensed context strings from local repository clones.

    No external I/O beyond the local filesystem is performed.  All methods
    are synchronous and safe to call from any thread.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_repo_context(self, local_path: str, max_files: int = 20) -> str:
        """
        Build a compact context string summarising a repository.

        The output contains:

        1. A directory tree (top 3 levels, pruned to keep it readable).
        2. The content of up to *max_files* high-signal files (READMEs,
           manifests, entry-points).

        Args:
            local_path: Absolute path to a local Git clone.
            max_files:  Maximum number of files whose content is included.

        Returns:
            A multi-section Markdown-ish string ready for inclusion in a
            prompt.
        """
        root = Path(local_path)
        if not root.exists():
            return f"[workspace not found: {local_path}]"

        sections: list[str] = []

        # ---- 1. Directory tree ----
        tree = self._build_tree(root, max_depth=3)
        sections.append("## Directory Tree\n```\n" + tree + "\n```")

        # ---- 2. High-signal file contents ----
        candidate_paths = self._collect_candidates(root)
        included = 0
        file_sections: list[str] = []
        for rel_path in candidate_paths:
            if included >= max_files:
                break
            abs_path = root / rel_path
            if not abs_path.is_file():
                continue
            content = self._safe_read(abs_path, max_bytes=_MAX_FILE_BYTES)
            if not content:
                continue
            file_sections.append(
                f"### {rel_path}\n```\n{content}\n```"
            )
            included += 1

        if file_sections:
            sections.append("## Key Files\n" + "\n\n".join(file_sections))

        return "\n\n".join(sections)

    def build_diff_context(self, diff: str, max_chars: int = 8000) -> str:
        """
        Clean and truncate a unified diff to fit within *max_chars*.

        Processing steps:

        1. Strip ANSI escape sequences.
        2. Remove binary diff noise (``Binary files … differ`` lines).
        3. Truncate to *max_chars*, appending a notice when truncation occurs.

        Args:
            diff:      Raw unified diff string.
            max_chars: Maximum number of characters in the returned string.

        Returns:
            A cleaned, possibly truncated diff string.
        """
        if not diff:
            return ""

        # Strip ANSI colour codes
        clean = re.sub(r"\x1b\[[0-9;]*m", "", diff)

        # Remove binary diff lines — they add no signal for code review
        clean = re.sub(
            r"^Binary files .+ differ\s*$", "", clean, flags=re.MULTILINE
        )

        # Collapse excessive blank lines
        clean = re.sub(r"\n{3,}", "\n\n", clean)

        clean = clean.strip()

        if len(clean) <= max_chars:
            return clean

        truncated = clean[:max_chars]
        # Try to cut at a natural boundary (end of a hunk header or line)
        last_newline = truncated.rfind("\n")
        if last_newline > max_chars // 2:
            truncated = truncated[:last_newline]

        notice = (
            f"\n\n[... diff truncated at {max_chars} characters — "
            f"{len(clean) - len(truncated)} characters omitted ...]"
        )
        return truncated + notice

    def load_custom_instructions(self, local_path: str) -> str | None:
        """
        Read custom agent instructions from the repository, if present.

        Checks the following paths in order, returning the first one found:

        * ``.agent-instructions.md``
        * ``.github/copilot-instructions.md``

        Args:
            local_path: Absolute path to a local Git clone.

        Returns:
            The instruction file content as a string, or ``None`` when no
            instruction file is present.
        """
        root = Path(local_path)
        for rel in _INSTRUCTION_FILES:
            candidate = root / rel
            if candidate.is_file():
                content = self._safe_read(candidate, max_bytes=16_000)
                if content:
                    logger.debug(
                        "Loaded custom instructions from %s", candidate
                    )
                    return content
        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_tree(self, root: Path, max_depth: int = 3) -> str:
        """Return an indented directory tree string for *root*."""
        lines: list[str] = [root.name + "/"]
        self._tree_recurse(root, prefix="", depth=0, max_depth=max_depth, lines=lines)
        return "\n".join(lines)

    def _tree_recurse(
        self,
        directory: Path,
        prefix: str,
        depth: int,
        max_depth: int,
        lines: list[str],
    ) -> None:
        if depth >= max_depth:
            return
        try:
            entries = sorted(
                directory.iterdir(),
                key=lambda p: (p.is_file(), p.name.lower()),
            )
        except PermissionError:
            return

        # Filter out noisy hidden/generated directories
        entries = [
            e for e in entries
            if e.name not in {
                ".git", "__pycache__", "node_modules", ".venv", "venv",
                ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
                ".eggs", "*.egg-info",
            }
            and not e.name.endswith(".egg-info")
        ]

        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            display = entry.name + ("/" if entry.is_dir() else "")
            lines.append(prefix + connector + display)
            if entry.is_dir():
                extension = "    " if is_last else "│   "
                self._tree_recurse(
                    entry,
                    prefix=prefix + extension,
                    depth=depth + 1,
                    max_depth=max_depth,
                    lines=lines,
                )

    def _collect_candidates(self, root: Path) -> list[str]:
        """
        Return relative paths of high-signal files, in priority order.

        1. Exact-name matches from :data:`_HIGH_SIGNAL_FILES`.
        2. Common entry-point files found in the root or one level deep.
        3. Any remaining top-level files not yet included.
        """
        seen: set[str] = set()
        result: list[str] = []

        def _add(rel: str) -> None:
            if rel not in seen:
                seen.add(rel)
                result.append(rel)

        # Pass 1: named high-signal files
        for name in _HIGH_SIGNAL_FILES:
            if "*" in name:
                # Glob pattern — only match at root level
                for match in root.glob(name):
                    _add(match.relative_to(root).as_posix())
            else:
                candidate = root / name
                if candidate.exists():
                    _add(name)

        # Pass 2: entry-point files (root or one sub-level)
        for ep in _ENTRYPOINT_NAMES:
            for candidate in [root / ep] + list(root.glob(f"*/{ep}")):
                if candidate.is_file():
                    _add(candidate.relative_to(root).as_posix())

        # Pass 3: remaining top-level files (sorted)
        try:
            for entry in sorted(root.iterdir(), key=lambda p: p.name.lower()):
                if entry.is_file() and not entry.name.startswith("."):
                    _add(entry.relative_to(root).as_posix())
        except PermissionError:
            pass

        return result

    @staticmethod
    def _safe_read(path: Path, max_bytes: int = _MAX_FILE_BYTES) -> str:
        """
        Read a text file safely, respecting *max_bytes* and encoding errors.

        Binary files are silently skipped (return empty string).
        """
        try:
            size = path.stat().st_size
            if size == 0:
                return ""
            with path.open("rb") as fh:
                raw = fh.read(max_bytes)
            # Heuristic: if >30 % of the first 512 bytes are non-text, skip it
            sample = raw[:512]
            non_text = sum(
                1 for b in sample if b < 9 or (13 < b < 32 and b != 27)
            )
            if len(sample) > 0 and non_text / len(sample) > 0.30:
                return ""  # Likely binary
            text = raw.decode("utf-8", errors="replace")
            if size > max_bytes:
                text += f"\n... [{size - max_bytes} bytes truncated]"
            return text
        except (OSError, PermissionError) as exc:
            logger.debug("Could not read %s: %s", path, exc)
            return ""
