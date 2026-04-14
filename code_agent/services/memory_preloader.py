"""Memory pre-loader — eliminates reactive recall roundtrips.

Problem
-------
Without this module the agent needs 1-2 extra LLM calls per task:
  call 1 — agent decides which memory topics are relevant
  call 2 — agent uses the loaded content to actually answer

This module front-runs that decision by:
  Gap 1  keyword-matching the user's first message against the L1 pointer
          index and injecting the top-matching topic files before the LLM
          ever sees the request.
  Gap 4  warming the N most recently modified topics at session start so
          topic file reads come from an in-memory cache, not disk.

Gap 2 (stale detection) is also here: it compares recent git changes against
the in-memory topic summaries to flag entries that may reference outdated code.

None of these operations make LLM calls — all work is deterministic so the
overhead is a handful of file reads + a subprocess call at session start.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from google.adk.tools import ToolContext

logger: structlog.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WARM_TOPIC_COUNT = 3        # Gap 4: how many recent topics to load at session start
_MAX_L2_CONTENT_CHARS = 800  # cap per topic file when injecting into system instruction
_MAX_GIT_LOOKBACK = 5        # commits to scan for stale detection

# ---------------------------------------------------------------------------
# Gap 1 — keyword matching
# ---------------------------------------------------------------------------


def match_topics_to_query(
    query: str,
    memory_index: list[dict[str, Any]],
) -> list[str]:
    """Return memory topic names ordered by keyword overlap with *query*.

    Scoring:
      +1 for each query token that appears in the topic name or summary
      +1 extra for each hyphen/underscore segment of the topic name that matches
         (so "auth-flow" gives both "auth" and "flow" as matchable tokens)

    Returns topic names ranked highest-first; unmatched topics are excluded.
    """
    query_tokens = set(_tokenise(query))
    if not query_tokens:
        return []

    scored: list[tuple[int, str]] = []
    for entry in memory_index:
        topic: str = entry.get("topic", "")
        summary: str = entry.get("summary", "")

        topic_tokens = _tokenise(topic + " " + summary)
        # Also treat each segment of a hyphenated/underscored topic as a token
        topic_parts = set(re.split(r"[-_]", topic.lower()))

        all_tokens = set(topic_tokens) | topic_parts
        score = len(query_tokens & all_tokens)
        if score > 0:
            scored.append((score, topic))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored]


def _tokenise(text: str) -> list[str]:
    """Lower-case, strip punctuation, split on whitespace."""
    return [w for w in re.sub(r"[^a-z0-9]", " ", text.lower()).split() if len(w) > 2]


# ---------------------------------------------------------------------------
# Gap 4 — warm cache (load N most recently modified topics at session start)
# ---------------------------------------------------------------------------


async def warm_recent_topics(
    memory_index: list[dict[str, Any]],
    workspace: Path,
    n: int = _WARM_TOPIC_COUNT,
) -> dict[str, str]:
    """Load the N most recently modified topic files into an in-memory cache.

    Called once in before_agent_callback.  Subsequent topic accesses in
    before_model_callback hit the cache instead of disk, eliminating one
    async file-read per injected topic.

    Returns a dict mapping topic name → file content (empty string on error).
    """
    candidates: list[tuple[float, str, Path]] = []
    for entry in memory_index:
        rel = entry.get("file_path", "")
        topic = entry.get("topic", "")
        if not rel or not topic:
            continue
        # Check both user and repo workspace paths
        for candidate_path in _candidate_paths(workspace, rel):
            try:
                mtime = candidate_path.stat().st_mtime
                candidates.append((mtime, topic, candidate_path))
                break  # use first existing path
            except OSError:
                continue

    candidates.sort(key=lambda x: x[0], reverse=True)

    warm: dict[str, str] = {}
    for _, topic, path in candidates[:n]:
        try:
            import aiofiles
            async with aiofiles.open(path, "r", encoding="utf-8", errors="replace") as fh:
                warm[topic] = await fh.read()
        except Exception as exc:
            logger.debug("memory_preloader.warm.read_error", topic=topic, error=str(exc))

    logger.info(
        "memory_preloader.warmed",
        warmed_count=len(warm),
        topics=list(warm.keys()),
    )
    return warm


def _candidate_paths(workspace: Path, rel_path: str) -> list[Path]:
    """Given a relative file path, return candidate absolute paths to try."""
    return [workspace / rel_path]


# ---------------------------------------------------------------------------
# Gap 1 — proactive topic loading (uses warm cache, falls back to disk)
# ---------------------------------------------------------------------------


async def preload_matched_topics(
    topics: list[str],
    warm_cache: dict[str, str],
    tool_context: "ToolContext | None" = None,
) -> dict[str, str]:
    """Load content for *topics*, using the warm cache when available.

    Disk fallback uses recall_topic() so workspace resolution is consistent.
    Returns a dict of topic → content (empty topics excluded).
    """
    result: dict[str, str] = {}
    disk_needed = [t for t in topics if t not in warm_cache]

    # Serve from warm cache first
    for topic in topics:
        if topic in warm_cache and warm_cache[topic]:
            result[topic] = warm_cache[topic]

    # Disk fallback for cache misses
    if disk_needed:
        from code_agent.tools.memory_tools import recall_topic
        tasks = [recall_topic(t, tool_context) for t in disk_needed]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        for topic, resp in zip(disk_needed, responses):
            if isinstance(resp, dict) and resp.get("found") and resp.get("content"):
                result[topic] = resp["content"]

    logger.debug(
        "memory_preloader.preloaded",
        requested=topics,
        loaded=list(result.keys()),
        cache_hits=[t for t in topics if t in warm_cache],
    )
    return result


def format_proactive_context(
    topic_contents: dict[str, str],
    stale_topics: set[str] | None = None,
) -> str:
    """Format pre-loaded topic content for injection into the system instruction.

    Caps each topic at _MAX_L2_CONTENT_CHARS to stay within token budget.
    Stale topics get a warning prefix so the agent knows to re-verify.
    """
    if not topic_contents:
        return ""

    stale = stale_topics or set()
    parts: list[str] = []
    for topic, content in topic_contents.items():
        header = f"### [{topic}]"
        if topic in stale:
            header += " ⚠ may be stale — verify against live codebase"
        snippet = content[:_MAX_L2_CONTENT_CHARS]
        if len(content) > _MAX_L2_CONTENT_CHARS:
            snippet += f"\n… [{len(content) - _MAX_L2_CONTENT_CHARS} chars truncated]"
        parts.append(f"{header}\n{snippet}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Gap 2 — stale topic detection
# ---------------------------------------------------------------------------


async def detect_stale_topics(
    memory_index: list[dict[str, Any]],
    workspace: Path,
) -> list[str]:
    """Return topic names that may reference code changed in recent commits.

    Runs ``git log --name-only`` against the workspace to get the list of
    files touched in the last _MAX_GIT_LOOKBACK commits, then checks each
    topic's summary for references to those filenames.

    Returns an empty list if the workspace is not a git repo or git is
    unavailable — stale detection is best-effort, never blocking.
    """
    changed_files = await _git_recently_changed_files(workspace)
    if not changed_files:
        return []

    # Build a set of lower-case basenames for fuzzy matching
    changed_basenames = {Path(f).name.lower() for f in changed_files if f}
    changed_stems = {Path(f).stem.lower() for f in changed_files if f}

    stale: list[str] = []
    for entry in memory_index:
        topic = entry.get("topic", "")
        summary = entry.get("summary", "").lower()
        topic_lower = topic.lower()

        # Match if any changed file basename or stem appears in the topic name/summary
        for token in (changed_basenames | changed_stems):
            if len(token) > 2 and (token in summary or token in topic_lower):
                stale.append(topic)
                break

    if stale:
        logger.info(
            "memory_preloader.stale_detected",
            stale_topics=stale,
            changed_files=len(changed_files),
        )
    return stale


async def _git_recently_changed_files(workspace: Path) -> list[str]:
    """Return filenames changed in the last _MAX_GIT_LOOKBACK commits."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "-C", str(workspace),
            "log",
            f"--max-count={_MAX_GIT_LOOKBACK}",
            "--name-only",
            "--pretty=format:",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            return []
        return [line for line in stdout.decode(errors="replace").splitlines() if line.strip()]
    except Exception as exc:
        logger.debug("memory_preloader.git_error", error=str(exc))
        return []
