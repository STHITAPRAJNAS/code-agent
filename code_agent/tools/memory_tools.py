"""Memory tools — 3-layer file-based persistent memory for the agent.

Two-tier workspace model
------------------------
  repo tier  — shared across all users working on the same repository.
               Stores code understanding: architecture, API contracts, schema
               notes, discovered patterns.  Built once, reused by every user.

  user tier  — private per-user.
               Stores personal task context, WIP notes, user preferences.
               Never visible to other users.

File layout under each workspace root::

    {workspace}/
        AGENT.md                          # static project/user config
        .agent/
            MEMORY.md                     # pointer index only (~150 chars per line)
            meta.json                     # session counter + last AutoDream timestamp
            memory/
                {topic}.md                # topic files, loaded on demand
                archive/                  # overflow from MEMORY.md > 200 lines

MemPalace-inspired memory layers
---------------------------------
  L0 (~50 tok)  — first L0_CHARS of AGENT.md (project name, tech stack).
                  Injected into the STATIC prompt section (before boundary).
                  Byte-identical across all sessions → maximises cache hits.

  L1 (~120 tok) — full MEMORY.md pointer index (topic + one-line summaries).
                  Injected into the DYNAMIC section every session.
                  Shows what is stored; topic files are NOT loaded here.

  L2 (on-demand) — individual topic files via recall_topic().
  L3 (on-demand) — live codebase search via hybrid_search / verify_symbol_exists.

Write discipline:
  - MEMORY.md holds ONLY pointers: "[topic] → .agent/memory/{topic}.md | summary"
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
import hashlib
import json
import logging
import os
import re
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
# Layer constants (MemPalace-inspired)
# ---------------------------------------------------------------------------

L0_CHARS: int = 200        # ~50 tokens — project identity for static prompt section
L1_MAX_POINTERS: int = 50  # max pointer lines injected every session (L1 layer)

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
# Workspace resolution — two-tier model
# ---------------------------------------------------------------------------


def _repo_slug(active_repo: str) -> str:
    """Derive a filesystem-safe slug from a repo URL or local path.

    Examples:
        "https://github.com/acme/my-service.git"  → "acme--my-service"
        "/home/user/projects/my-service"           → "my-service"
        "git@github.com:acme/my-service.git"       → "acme--my-service"
    """
    # GitHub/GitLab style URLs and SSH remotes: extract owner/repo
    m = re.search(r"[:/]([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?$", active_repo)
    if m:
        slug = m.group(1).lower().replace("/", "--")
        return re.sub(r"[^a-z0-9-]", "_", slug)
    # Local path: use directory basename
    basename = Path(active_repo).name
    if basename and re.match(r"^[a-zA-Z0-9]", basename):
        return re.sub(r"[^a-z0-9-]", "_", basename.lower())
    # Unknown format: stable short hash
    return "repo-" + hashlib.sha1(active_repo.encode()).hexdigest()[:10]


def _user_workspace(tool_context: "ToolContext | None") -> Path:
    """Resolve the per-user workspace root.

    Resolution order:
      1. tool_context.state["active_workspace"]
      2. WORKSPACE_ROOT env var
      3. WORKSPACE_DIR env var
      4. /tmp/code_agent_workspaces/{session_id} (or base dir if no session)
    """
    root: str | None = None
    if tool_context is not None:
        root = tool_context.state.get("active_workspace") or None
    root = root or os.environ.get("WORKSPACE_ROOT") or os.environ.get("WORKSPACE_DIR")
    if not root:
        sid = getattr(tool_context, "session_id", None) if tool_context else None
        base = "/tmp/code_agent_workspaces"
        root = f"{base}/{sid}" if sid else base
    return Path(root)


def _repo_workspace(tool_context: "ToolContext | None") -> Path:
    """Resolve the shared repo workspace root.

    Resolution order:
      1. tool_context.state["repo_workspace"]  (explicitly set by session creator)
      2. Derived from tool_context.state["active_repo"] + REPO_WORKSPACE_BASE env
      3. REPO_WORKSPACE_ROOT env var
      4. Falls back to user workspace (single-user / backward-compat setups)
    """
    root: str | None = None
    if tool_context is not None:
        root = tool_context.state.get("repo_workspace") or None
        if not root:
            active_repo: str = tool_context.state.get("active_repo") or ""
            if active_repo:
                base = os.environ.get("REPO_WORKSPACE_BASE", "/data/repos")
                root = f"{base}/{_repo_slug(active_repo)}"
    root = root or os.environ.get("REPO_WORKSPACE_ROOT") or None
    return Path(root) if root else _user_workspace(tool_context)


# Keep alias so existing internal helpers and tests that call _workspace_root still work.
_workspace_root = _user_workspace


def _extract_l0(agent_config_text: str) -> str:
    """Return the L0 identity slice from AGENT.md (first L0_CHARS, ~50 tokens).

    This is the only slice injected into the STATIC prompt section — it must
    contain only time-invariant facts (project name, tech stack) so the
    Gemini prompt cache prefix stays byte-identical across all sessions.
    """
    return agent_config_text[:L0_CHARS].rstrip() if agent_config_text else ""


# ---------------------------------------------------------------------------
# Internal path helpers
# ---------------------------------------------------------------------------


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


def _merged_index(
    repo_index: MemoryIndex,
    user_index: MemoryIndex,
) -> list[MemoryPointer]:
    """Merge repo and user indexes; user pointers shadow repo on the same topic.

    Capped at L1_MAX_POINTERS so the injected L1 block stays within ~120 tokens.
    """
    merged: dict[str, MemoryPointer] = {p.topic: p for p in repo_index.pointers}
    merged.update({p.topic: p for p in user_index.pointers})
    return list(merged.values())[:L1_MAX_POINTERS]


# ---------------------------------------------------------------------------
# Public ADK tools
# ---------------------------------------------------------------------------


async def read_agent_config(
    tool_context: "ToolContext | None" = None,
) -> dict[str, Any]:
    """Load AGENT.md and the MEMORY.md pointer index from both workspace tiers.

    Merges repo-level (shared) and user-level (private) memory into a single
    view.  User pointers shadow repo pointers for the same topic.

    Layer mapping:
      l0_identity   — first ~50 tokens of repo AGENT.md.  Inject into the
                      STATIC prompt section (before SYSTEM_PROMPT_DYNAMIC_BOUNDARY)
                      so Gemini can cache it across all sessions and users.
      memory_index  — merged L1 pointer index.  Inject into the DYNAMIC section
                      every session (~120 tokens of topic summaries).
      agent_config  — full merged AGENT.md text.  Summarise before injecting.

    Returns a dict with keys:
        l0_identity   — L0 project identity slice (string, ~50 tokens)
        agent_config  — merged AGENT.md text (repo first, user appended)
        memory_index  — merged pointer list as dicts (topic, file_path, summary)
        workspace     — resolved user workspace path (string)
        repo_workspace — resolved repo workspace path (string)
    """
    user_ws = _user_workspace(tool_context)
    repo_ws = _repo_workspace(tool_context)

    await _ensure_dirs(user_ws)

    # Load both tiers in parallel
    user_config, repo_config = await asyncio_gather(
        _read_text(_agent_config_path(user_ws)),
        _read_text(_agent_config_path(repo_ws)) if repo_ws != user_ws else _noop_str(),
    )

    # L0: extracted from the repo config (shared, time-invariant project facts).
    # Fall back to user config if no repo config exists yet.
    primary_config = repo_config or user_config
    l0_identity = _extract_l0(primary_config)

    # Merged full config: repo context first, user-specific appended at the end.
    merged_config_parts = [p for p in (repo_config, user_config) if p]
    merged_config = "\n\n".join(merged_config_parts)

    # Merge memory indexes (L1)
    user_index = await _read_memory_index(user_ws)
    repo_index = (
        await _read_memory_index(repo_ws)
        if repo_ws != user_ws
        else MemoryIndex()
    )
    merged_pointers = _merged_index(repo_index, user_index)

    log = logger.bind(
        user_workspace=str(user_ws),
        repo_workspace=str(repo_ws),
        pointer_count=len(merged_pointers),
    )
    log.info("memory.read_agent_config")

    return {
        "l0_identity": l0_identity,
        "agent_config": merged_config,
        "memory_index": [p.model_dump() for p in merged_pointers],
        "workspace": str(user_ws),
        "repo_workspace": str(repo_ws),
    }


async def recall_topic(
    topic: str,
    tool_context: "ToolContext | None" = None,
) -> dict[str, Any]:
    """Fetch a specific memory topic file by name (L2 on-demand load).

    Searches user workspace first; falls back to repo workspace if not found.
    User topics shadow repo topics for the same name.

    Treat the returned content as a *hint* — always verify facts against
    the live codebase before acting.

    Args:
        topic: Topic name exactly as it appears in MEMORY.md
               (e.g. "auth", "db-schema", "api-contracts").
        tool_context: ADK ToolContext for workspace resolution.

    Returns a dict with keys:
        found     — bool
        topic     — requested topic name
        content   — file content, or "" if not found
        file_path — resolved absolute path string
        scope     — "user" or "repo" indicating which tier served the result
    """
    user_ws = _user_workspace(tool_context)
    repo_ws = _repo_workspace(tool_context)

    # Check user tier first (user shadows repo)
    for workspace, scope in [(user_ws, "user"), (repo_ws, "repo")]:
        if scope == "repo" and workspace == user_ws:
            continue  # same path, already checked
        index = await _read_memory_index(workspace)
        pointer = next((p for p in index.pointers if p.topic == topic), None)
        if pointer is not None:
            full_path = workspace / pointer.file_path
            content = await _read_text(full_path)
            logger.info(
                "memory.recall_topic.loaded",
                topic=topic,
                scope=scope,
                file_path=str(full_path),
                chars=len(content),
            )
            return {
                "found": True,
                "topic": topic,
                "content": content,
                "file_path": str(full_path),
                "scope": scope,
            }

    logger.info("memory.recall_topic.not_found", topic=topic)
    return {"found": False, "topic": topic, "content": "", "file_path": "", "scope": ""}


async def remember(
    topic: str,
    content: str,
    summary: str = "",
    scope: str = "user",
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
        scope:   "user" (default) — write to per-user workspace.
                 "repo"           — write to shared repo workspace (code knowledge).
        tool_context: ADK ToolContext for workspace resolution.

    Returns a dict with keys:
        ok        — bool
        topic     — topic name
        scope     — tier that was written to ("user" or "repo")
        file_path — path to the written topic file (string)
    """
    workspace = _repo_workspace(tool_context) if scope == "repo" else _user_workspace(tool_context)
    await _ensure_dirs(workspace)

    safe_name = "".join(c if (c.isalnum() or c in "-_") else "_" for c in topic.lower())
    rel_path = f".agent/memory/{safe_name}.md"
    full_path = workspace / rel_path

    # 1. Write topic file first
    await _atomic_write(full_path, content)

    # 2. Build one-line summary
    effective_summary = (summary.strip() or next(
        (line.strip().lstrip("#").strip() for line in content.splitlines() if line.strip()),
        topic,
    ))[:120]

    # 3. Update MEMORY.md pointer under lock
    index = await _read_memory_index(workspace)
    index.pointers = [p for p in index.pointers if p.topic != topic]
    index.pointers.append(
        MemoryPointer(topic=topic, file_path=rel_path, summary=effective_summary)
    )
    await _locked_write_memory_index(workspace, index)

    logger.info(
        "memory.remember.wrote",
        topic=topic,
        scope=scope,
        file_path=str(full_path),
        summary=effective_summary,
    )
    return {"ok": True, "topic": topic, "scope": scope, "file_path": str(full_path)}


async def forget(
    topic: str,
    scope: str = "user",
    tool_context: "ToolContext | None" = None,
) -> dict[str, Any]:
    """Delete a topic file and remove its pointer from MEMORY.md.

    Safe to call even if the topic does not exist (idempotent).

    Args:
        topic: Topic name to delete.
        scope: "user" (default) or "repo" — which tier to remove from.
        tool_context: ADK ToolContext for workspace resolution.

    Returns a dict with keys:
        ok           — bool
        topic        — topic name
        scope        — tier that was targeted
        file_deleted — bool, whether a file was actually removed
    """
    workspace = _repo_workspace(tool_context) if scope == "repo" else _user_workspace(tool_context)
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
        scope=scope,
        file_deleted=file_deleted,
    )
    return {"ok": True, "topic": topic, "scope": scope, "file_deleted": file_deleted}


async def list_memory_topics(
    tool_context: "ToolContext | None" = None,
) -> dict[str, Any]:
    """Return all current pointers from both workspace tiers without loading topic files.

    Cheap — reads only the two index files.  User topics shadow repo topics for
    the same name.  Use this to decide which topics to recall_topic() for a task.

    Returns a dict with keys:
        topics — list of {"topic", "summary", "scope"} dicts
        count  — int
    """
    user_ws = _user_workspace(tool_context)
    repo_ws = _repo_workspace(tool_context)

    user_index = await _read_memory_index(user_ws)
    repo_index = (
        await _read_memory_index(repo_ws)
        if repo_ws != user_ws
        else MemoryIndex()
    )

    user_topics = {p.topic for p in user_index.pointers}
    merged = _merged_index(repo_index, user_index)

    topics = [
        {
            "topic": p.topic,
            "summary": p.summary,
            "scope": "user" if p.topic in user_topics else "repo",
        }
        for p in merged
    ]
    logger.debug("memory.list_topics", count=len(topics))
    return {"topics": topics, "count": len(topics)}


# ---------------------------------------------------------------------------
# asyncio helpers (avoid importing asyncio at module level for test compat)
# ---------------------------------------------------------------------------


async def asyncio_gather(*coros):  # type: ignore[no-untyped-def]
    import asyncio
    return await asyncio.gather(*coros)


async def _noop_str() -> str:
    return ""
