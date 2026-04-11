"""AutoDream — background memory consolidation service.

Runs as a forked asyncio task at the end of each successful A2A session.
It never blocks the main agent loop.

Trigger conditions (both must be true):
  - >= 24 hours since last AutoDream run
  - >= 5 sessions since last AutoDream run

Operations performed (in order):
  1. Replace relative time references with absolute dates.
  2. Resolve contradictions — keep the newer / more specific fact.
  3. Delete entries that point to files that no longer exist (stale pointers).
  4. Keep MEMORY.md under 200 lines — archive overflow to .agent/memory/archive/.

Meta state is persisted to .agent/meta.json (also updated by the session
lifecycle on every successful session completion).

All I/O is async (aiofiles).  MEMORY.md writes use the same fcntl advisory
lock as memory_tools.py so multi-pod EKS deployments sharing EFS are safe.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os
import structlog

from pydantic import BaseModel

logger: structlog.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_HOURS_BETWEEN_RUNS = 24
_MIN_SESSIONS_BETWEEN_RUNS = 5
_MAX_MEMORY_LINES = 200
_POINTER_RE = re.compile(
    r"^\[(?P<topic>[^\]]+)\]\s*→\s*(?P<path>[^|]+)\s*\|\s*(?P<summary>.+)$"
)

# Relative-time patterns to replace with absolute dates
_RELATIVE_TIME_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\byesterday\b", re.IGNORECASE), ""),
    (re.compile(r"\blast week\b", re.IGNORECASE), ""),
    (re.compile(r"\blast month\b", re.IGNORECASE), ""),
    (re.compile(r"\brecently\b", re.IGNORECASE), ""),
    (re.compile(r"\bjust now\b", re.IGNORECASE), ""),
]

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class AgentMeta(BaseModel):
    """Persisted to .agent/meta.json."""

    session_count: int = 0
    last_dream_ts: float = 0.0   # Unix timestamp of last AutoDream run
    last_dream_sessions: int = 0  # session_count when last dream ran


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _agent_dir(workspace: Path) -> Path:
    return workspace / ".agent"


def _memory_dir(workspace: Path) -> Path:
    return _agent_dir(workspace) / "memory"


def _archive_dir(workspace: Path) -> Path:
    return _agent_dir(workspace) / "memory" / "archive"


def _memory_index_path(workspace: Path) -> Path:
    return _agent_dir(workspace) / "MEMORY.md"


def _meta_path(workspace: Path) -> Path:
    return _agent_dir(workspace) / "meta.json"


async def _read_text(path: Path) -> str:
    try:
        async with aiofiles.open(path, "r", encoding="utf-8") as fh:
            return await fh.read()
    except FileNotFoundError:
        return ""


async def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
    try:
        async with aiofiles.open(fd, "w", encoding="utf-8") as fh:
            await fh.write(content)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


async def _read_meta(workspace: Path) -> AgentMeta:
    text = await _read_text(_meta_path(workspace))
    if not text.strip():
        return AgentMeta()
    try:
        return AgentMeta.model_validate_json(text)
    except Exception:
        return AgentMeta()


async def _write_meta(workspace: Path, meta: AgentMeta) -> None:
    await _atomic_write(_meta_path(workspace), meta.model_dump_json(indent=2))


async def _locked_write_memory_index(workspace: Path, lines: list[str]) -> None:
    """Write MEMORY.md lines under an fcntl advisory lock."""
    lock_path = _agent_dir(workspace) / ".memory.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            content = "\n".join(lines) + "\n"
            await _atomic_write(_memory_index_path(workspace), content)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
    finally:
        os.close(lock_fd)


# ---------------------------------------------------------------------------
# Dream operations
# ---------------------------------------------------------------------------


def _resolve_relative_times(content: str, now: datetime) -> str:
    """Replace relative time references with absolute ISO dates."""
    date_str = now.strftime("%Y-%m-%d")
    result = content
    for pattern, _ in _RELATIVE_TIME_PATTERNS:
        result = pattern.sub(date_str, result)
    return result


def _resolve_contradictions(lines: list[str]) -> list[str]:
    """When the same topic appears twice, keep only the last (newest) entry."""
    seen_topics: dict[str, int] = {}  # topic → last index
    result = list(lines)
    for i, line in enumerate(lines):
        m = _POINTER_RE.match(line.strip())
        if m:
            topic = m.group("topic")
            if topic in seen_topics:
                # Mark the earlier duplicate for removal
                result[seen_topics[topic]] = ""
            seen_topics[topic] = i
    return [ln for ln in result if ln != ""]


async def _prune_stale_pointers(workspace: Path, lines: list[str]) -> list[str]:
    """Remove pointers to topic files that no longer exist on disk."""
    kept: list[str] = []
    for line in lines:
        m = _POINTER_RE.match(line.strip())
        if m:
            topic_path = workspace / m.group("path").strip()
            exists = await aiofiles.os.path.exists(str(topic_path))
            if not exists:
                logger.info(
                    "auto_dream.prune_stale",
                    topic=m.group("topic"),
                    path=str(topic_path),
                )
                continue
        kept.append(line)
    return kept


async def _archive_overflow(workspace: Path, lines: list[str]) -> list[str]:
    """If MEMORY.md > 200 lines, move the oldest entries to .agent/memory/archive/."""
    if len(lines) <= _MAX_MEMORY_LINES:
        return lines

    await aiofiles.os.makedirs(str(_archive_dir(workspace)), exist_ok=True)
    overflow_count = len(lines) - _MAX_MEMORY_LINES
    overflow = lines[:overflow_count]
    kept = lines[overflow_count:]

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = _archive_dir(workspace) / f"overflow_{ts}.md"
    header = [
        f"# Archived Memory Overflow — {ts}",
        "# These entries exceeded the 200-line MEMORY.md limit.",
        "",
    ]
    await _atomic_write(archive_path, "\n".join(header + overflow) + "\n")
    logger.info(
        "auto_dream.archive_overflow",
        archived=overflow_count,
        kept=len(kept),
        archive_path=str(archive_path),
    )
    return kept


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class AutoDream:
    """Background memory consolidation.

    Usage (at session end)::

        asyncio.create_task(AutoDream.maybe_run(workspace, session_count))
    """

    @staticmethod
    async def maybe_run(workspace: Path | str, session_count: int) -> None:
        """Trigger a dream cycle if conditions are met.

        This is the public entry point.  Designed to be launched as an
        ``asyncio.create_task()`` so it never blocks the caller.

        Args:
            workspace:     Path to the active workspace root.
            session_count: Current total session count (after incrementing).
        """
        workspace = Path(workspace)
        meta = await _read_meta(workspace)

        now_ts = datetime.now(tz=timezone.utc).timestamp()
        hours_since = (now_ts - meta.last_dream_ts) / 3600
        sessions_since = session_count - meta.last_dream_sessions

        if hours_since < _MIN_HOURS_BETWEEN_RUNS or sessions_since < _MIN_SESSIONS_BETWEEN_RUNS:
            logger.debug(
                "auto_dream.skip",
                hours_since=round(hours_since, 1),
                sessions_since=sessions_since,
                workspace=str(workspace),
            )
            return

        logger.info(
            "auto_dream.start",
            hours_since=round(hours_since, 1),
            sessions_since=sessions_since,
            workspace=str(workspace),
        )

        try:
            await AutoDream._run(workspace)
            meta.last_dream_ts = now_ts
            meta.last_dream_sessions = session_count
            await _write_meta(workspace, meta)
            logger.info("auto_dream.complete", workspace=str(workspace))
        except Exception as exc:
            logger.error("auto_dream.error", error=str(exc), workspace=str(workspace))

    @staticmethod
    async def _run(workspace: Path) -> None:
        """Execute the four consolidation steps."""
        index_path = _memory_index_path(workspace)
        raw = await _read_text(index_path)
        if not raw.strip():
            return

        now = datetime.now(tz=timezone.utc)
        lines = raw.splitlines()

        # Separate header comments from pointer lines
        headers: list[str] = []
        pointers: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("<!--") or not stripped:
                headers.append(line)
            else:
                pointers.append(line)

        # Step 1 — replace relative time refs in topic files
        await AutoDream._resolve_topic_times(workspace, now)

        # Step 2 — resolve duplicate/contradicting pointer entries
        pointers = _resolve_contradictions(pointers)

        # Step 3 — prune stale pointers
        pointers = await _prune_stale_pointers(workspace, pointers)

        # Step 4 — archive overflow
        pointers = await _archive_overflow(workspace, pointers)

        # Write consolidated index
        final_lines = headers + [""] + pointers
        await _locked_write_memory_index(workspace, final_lines)

    @staticmethod
    async def _resolve_topic_times(workspace: Path, now: datetime) -> None:
        """Walk all topic files and replace relative time references in-place."""
        mem_dir = _memory_dir(workspace)
        if not await aiofiles.os.path.isdir(str(mem_dir)):
            return

        for entry in os.scandir(str(mem_dir)):
            if not entry.is_file() or not entry.name.endswith(".md"):
                continue
            path = Path(entry.path)
            content = await _read_text(path)
            updated = _resolve_relative_times(content, now)
            if updated != content:
                await _atomic_write(path, updated)
                logger.debug("auto_dream.resolved_times", path=str(path))


async def increment_session_and_dream(workspace: Path | str) -> int:
    """Increment session_count in meta.json and launch AutoDream as a background task.

    Returns the new session count.  Call this at the end of each successful
    A2A session from the after_agent_callback.
    """
    workspace = Path(workspace)
    await aiofiles.os.makedirs(str(_agent_dir(workspace)), exist_ok=True)

    meta = await _read_meta(workspace)
    meta.session_count += 1
    await _write_meta(workspace, meta)

    asyncio.create_task(AutoDream.maybe_run(workspace, meta.session_count))
    return meta.session_count
