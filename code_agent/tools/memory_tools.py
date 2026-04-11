"""Memory tools — 3-layer file-based persistent memory for the agent.

File layout under the mounted workspace root::

    {workspace}/
        AGENT.md                          # static project config, loaded every session
        .agent/
            MEMORY.md                     # pointer index only (~150 chars per line)
            meta.json                     # session counter + last AutoDream timestamp
            memory/
                {topic}.md                # topic files, loaded on demand
                archive/                  # overflow from MEMORY.md > 200 lines

Rules encoded here and enforced in tool docstrings:
  - MEMORY.md holds ONLY pointers: "[topic] → .agent/memory/{topic}.md | one-line summary"
  - Write topic file FIRST, then update MEMORY.md pointer (atomic: temp + os.replace)
  - If a fact can be re-derived from live codebase search, do NOT store it
  - Treat stored facts as hints — always verify before acting
  - If memory contradicts the live codebase, update memory (not vice versa)

All writes use write-to-temp + os.replace() for atomicity.
MEMORY.md writes are protected by an fcntl advisory lock so multiple EKS
pods sharing EFS do not clobber each other.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiofiles
import aiofiles.os
import structlog

from pydantic import BaseModel

if TYPE_CHECKING:
    from google.adk.tools import ToolContext

logger: structlog.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Pydantic v2 models
# ---------------------------------------------------------------------------


class MemoryPointer(BaseModel):
    """A single line in MEMORY.md."""

    topic: str
    file_path: str  # relative path, e.g. ".agent/memory/auth.md"
    summary: str    # one-line summary, ≤ 120 chars


class MemoryIndex(BaseModel):
    """Parsed contents of MEMORY.md."""

    pointers: list[MemoryPointer] = []

    def render(self) -> str:
        """Render to the MEMORY.md wire format."""
        lines = ["# Agent Memory Index\n", "<!-- Auto-managed. Do not edit manually. -->\n", ""]
        for p in self.pointers:
            lines.append(f"[{p.topic}] → {p.file_path} | {p.summary}")
        return "\n".join(lines) + "\n"

    @classmethod
    def parse(cls, text: str) -> "MemoryIndex":
        """Parse MEMORY.md text into a MemoryIndex."""
        pointers: list[MemoryPointer] = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("<!--"):
                continue
            # Format: [topic] → file_path | summary
            if line.startswith("[") and "→" in line and "|" in line:
                try:
                    topic_part, rest = line.split("→", 1)
                    topic = topic_part.strip().lstrip("[").rstrip("]").strip()
                    file_path, summary = rest.split("|", 1)
                    pointers.append(
                        MemoryPointer(
                            topic=topic,
                            file_path=file_path.strip(),
                            summary=summary.strip(),
                        )
                    )
                except ValueError:
                    continue
        return cls(pointers=pointers)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _workspace_root(tool_context: "ToolContext | None") -> Path:
    """Resolve the workspace root from session state or WORKSPACE_ROOT env var."""
    root: str | None = None
    if tool_context is not None:
        root = tool_context.state.get("active_workspace") or None
    root = root or os.environ.get("WORKSPACE_ROOT") or os.environ.get("WORKSPACE_DIR", "/tmp/code_agent_workspaces")
    return Path(root)


def _agent_dir(workspace: Path) -> Path:
    return workspace / ".agent"


def _memory_dir(workspace: Path) -> Path:
    return _agent_dir(workspace) / "memory"


def _archive_dir(workspace: Path) -> Path:
    return _agent_dir(workspace) / "memory" / "archive"


def _memory_index_path(workspace: Path) -> Path:
    return _agent_dir(workspace) / "MEMORY.md"


def _agent_config_path(workspace: Path) -> Path:
    return workspace / "AGENT.md"


def _meta_path(workspace: Path) -> Path:
    return _agent_dir(workspace) / "meta.json"


async def _ensure_dirs(workspace: Path) -> None:
    """Create .agent/ and .agent/memory/ if they do not exist."""
    for d in (_agent_dir(workspace), _memory_dir(workspace), _archive_dir(workspace)):
        await aiofiles.os.makedirs(str(d), exist_ok=True)


async def _read_text(path: Path) -> str:
    """Read a text file; return empty string if it does not exist."""
    try:
        async with aiofiles.open(path, "r", encoding="utf-8") as fh:
            return await fh.read()
    except FileNotFoundError:
        return ""


async def _atomic_write(path: Path, content: str) -> None:
    """Write *content* to *path* atomically using a sibling temp file."""
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


async def _locked_write_memory_index(workspace: Path, index: MemoryIndex) -> None:
    """Write MEMORY.md under an fcntl advisory lock (EFS-safe for multi-pod EKS)."""
    lock_path = _agent_dir(workspace) / ".memory.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    # Open (or create) the lock file in a sync context — fcntl needs a real fd
    lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            await _atomic_write(_memory_index_path(workspace), index.render())
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
    finally:
        os.close(lock_fd)


async def _read_memory_index(workspace: Path) -> MemoryIndex:
    text = await _read_text(_memory_index_path(workspace))
    return MemoryIndex.parse(text)


# ---------------------------------------------------------------------------
# Public ADK tools
# ---------------------------------------------------------------------------


async def read_agent_config(
    tool_context: "ToolContext | None" = None,
) -> dict[str, Any]:
    """Load AGENT.md and the MEMORY.md pointer index.

    Should be called at the start of every session to seed the agent's
    knowledge of the project's static configuration and what topics have
    been stored in memory.  Loading is cheap — topic files are NOT read
    here; use recall_topic() to load a specific topic on demand.

    Returns a dict with keys:
        agent_config  — full text of AGENT.md, or "" if not present
        memory_index  — list of {"topic", "file_path", "summary"} dicts
        workspace     — resolved workspace root path (string)
    """
    workspace = _workspace_root(tool_context)
    await _ensure_dirs(workspace)

    agent_config = await _read_text(_agent_config_path(workspace))
    index = await _read_memory_index(workspace)

    log = logger.bind(workspace=str(workspace), pointer_count=len(index.pointers))
    log.info("memory.read_agent_config")

    return {
        "agent_config": agent_config,
        "memory_index": [p.model_dump() for p in index.pointers],
        "workspace": str(workspace),
    }


async def recall_topic(
    topic: str,
    tool_context: "ToolContext | None" = None,
) -> dict[str, Any]:
    """Fetch a specific memory topic file by name.

    Lazy-loads only the requested topic — do NOT pre-load all topics.
    If the topic does not exist in MEMORY.md, returns an empty result;
    this is not an error.

    Treat the returned content as a *hint* — always verify facts against
    the live codebase before acting.

    Args:
        topic: Topic name exactly as it appears in the MEMORY.md pointer
               (e.g. "auth", "db-schema", "api-contracts").
        tool_context: ADK ToolContext for workspace resolution.

    Returns a dict with keys:
        found    — bool, whether the topic file exists
        topic    — the requested topic name
        content  — file content, or "" if not found
        file_path — resolved path string
    """
    workspace = _workspace_root(tool_context)
    index = await _read_memory_index(workspace)

    pointer = next((p for p in index.pointers if p.topic == topic), None)
    if pointer is None:
        logger.info("memory.recall_topic.not_found", topic=topic, workspace=str(workspace))
        return {"found": False, "topic": topic, "content": "", "file_path": ""}

    full_path = workspace / pointer.file_path
    content = await _read_text(full_path)

    logger.info(
        "memory.recall_topic.loaded",
        topic=topic,
        file_path=str(full_path),
        chars=len(content),
    )
    return {
        "found": True,
        "topic": topic,
        "content": content,
        "file_path": str(full_path),
    }


async def remember(
    topic: str,
    content: str,
    summary: str = "",
    tool_context: "ToolContext | None" = None,
) -> dict[str, Any]:
    """Store content under a named topic; update the MEMORY.md pointer index.

    Write discipline:
      1. Topic file is written first (atomic temp + os.replace).
      2. MEMORY.md pointer index is updated second under an fcntl lock.
    This ordering ensures MEMORY.md never points to a file that does not exist.

    Do NOT store facts that can be re-derived from live codebase search.
    Prefer short, structured notes.

    Args:
        topic:   Short identifier, lowercase with hyphens (e.g. "auth-flow").
        content: Full markdown content to store in the topic file.
        summary: One-line summary (≤ 120 chars) for the MEMORY.md pointer.
                 If empty, the first non-blank line of content is used.
        tool_context: ADK ToolContext for workspace resolution.

    Returns a dict with keys:
        ok        — bool
        topic     — topic name
        file_path — path to the written topic file (string)
    """
    workspace = _workspace_root(tool_context)
    await _ensure_dirs(workspace)

    # Derive a safe filename from the topic name
    safe_name = "".join(c if (c.isalnum() or c in "-_") else "_" for c in topic.lower())
    rel_path = f".agent/memory/{safe_name}.md"
    full_path = workspace / rel_path

    # 1. Write topic file first
    await _atomic_write(full_path, content)

    # 2. Build one-line summary for the pointer
    effective_summary = (summary.strip() or next(
        (line.strip().lstrip("#").strip() for line in content.splitlines() if line.strip()),
        topic,
    ))[:120]

    # 3. Update MEMORY.md pointer under lock
    index = await _read_memory_index(workspace)
    # Replace existing pointer for this topic (or append)
    index.pointers = [p for p in index.pointers if p.topic != topic]
    index.pointers.append(
        MemoryPointer(topic=topic, file_path=rel_path, summary=effective_summary)
    )
    await _locked_write_memory_index(workspace, index)

    logger.info(
        "memory.remember.wrote",
        topic=topic,
        file_path=str(full_path),
        summary=effective_summary,
    )
    return {"ok": True, "topic": topic, "file_path": str(full_path)}


async def forget(
    topic: str,
    tool_context: "ToolContext | None" = None,
) -> dict[str, Any]:
    """Delete a topic file and remove its pointer from MEMORY.md.

    Safe to call even if the topic does not exist (idempotent).

    Args:
        topic: Topic name to delete.
        tool_context: ADK ToolContext for workspace resolution.

    Returns a dict with keys:
        ok           — bool
        topic        — topic name
        file_deleted — bool, whether a file was actually removed
    """
    workspace = _workspace_root(tool_context)
    index = await _read_memory_index(workspace)

    pointer = next((p for p in index.pointers if p.topic == topic), None)
    file_deleted = False

    if pointer is not None:
        full_path = workspace / pointer.file_path
        try:
            await aiofiles.os.remove(str(full_path))
            file_deleted = True
        except FileNotFoundError:
            pass

        index.pointers = [p for p in index.pointers if p.topic != topic]
        await _locked_write_memory_index(workspace, index)

    logger.info(
        "memory.forget",
        topic=topic,
        file_deleted=file_deleted,
        workspace=str(workspace),
    )
    return {"ok": True, "topic": topic, "file_deleted": file_deleted}


async def list_memory_topics(
    tool_context: "ToolContext | None" = None,
) -> dict[str, Any]:
    """Return all current pointers from MEMORY.md without loading topic files.

    Cheap — reads only the index file.  Use this to decide which topics
    to recall_topic() for a given task.

    Returns a dict with keys:
        topics — list of {"topic", "summary"} dicts (file_path excluded)
        count  — int
    """
    workspace = _workspace_root(tool_context)
    index = await _read_memory_index(workspace)

    topics = [{"topic": p.topic, "summary": p.summary} for p in index.pointers]
    logger.debug("memory.list_topics", count=len(topics), workspace=str(workspace))
    return {"topics": topics, "count": len(topics)}
