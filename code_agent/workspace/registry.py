"""
registry.py — Persistent registry of named agent workspaces.

The registry stores metadata about named workspaces (collections of
repositories the agent should operate on) in a JSON file on disk.
It provides simple CRUD operations with no external dependencies beyond
the standard library.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from code_agent.config import get_settings

logger = logging.getLogger(__name__)

_REGISTRY_FILENAME = ".registry.json"


@dataclass
class WorkspaceEntry:
    """
    Metadata record for a named workspace.

    Attributes:
        name:             Unique human-readable identifier for the workspace
                          (e.g. ``"backend-services"``).
        repos:            List of HTTPS clone URLs (credentials may be
                          embedded) for repositories that belong to this
                          workspace.
        vcs_type:         VCS provider type: ``"github"``,
                          ``"bitbucket-cloud"``, or ``"bitbucket-server"``.
        workspace_or_org: Organisation slug, Bitbucket workspace, or project
                          key associated with the repos.
        instructions:     Free-text custom instructions for the agent (e.g.
                          coding conventions, architecture notes).
        index_collection: Name of the ChromaDB collection that indexes this
                          workspace's code.
        created_at:       ISO-8601 timestamp of when the entry was created.
    """

    name: str
    repos: list[str] = field(default_factory=list)
    vcs_type: str = ""
    workspace_or_org: str = ""
    instructions: str = ""
    index_collection: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        """Serialise the entry to a JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "WorkspaceEntry":
        """Deserialise from a dictionary (tolerates missing optional keys)."""
        return cls(
            name=data["name"],
            repos=data.get("repos", []),
            vcs_type=data.get("vcs_type", ""),
            workspace_or_org=data.get("workspace_or_org", ""),
            instructions=data.get("instructions", ""),
            index_collection=data.get("index_collection", ""),
            created_at=data.get(
                "created_at", datetime.now(timezone.utc).isoformat()
            ),
        )


class WorkspaceRegistry:
    """
    Persistent registry of named workspaces backed by a JSON file.

    The registry file is located at
    ``{WORKSPACE_DIR}/.registry.json``.  All operations read and write the
    file atomically (write-to-temp then rename) to avoid corruption.

    Args:
        registry_path: Override the default registry file path.  Mainly
                       useful for testing.
    """

    def __init__(self, registry_path: str | None = None) -> None:
        if registry_path:
            self._path = Path(registry_path)
        else:
            settings = get_settings()
            base = Path(settings.WORKSPACE_DIR)
            base.mkdir(parents=True, exist_ok=True)
            self._path = base / _REGISTRY_FILENAME

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, entry: WorkspaceEntry) -> None:
        """
        Persist a workspace entry, creating or overwriting any existing
        entry with the same *name*.

        Args:
            entry: The :class:`WorkspaceEntry` to persist.
        """
        data = self._load_raw()
        data[entry.name] = entry.to_dict()
        self._save_raw(data)
        logger.debug("Saved workspace entry: %s", entry.name)

    def get(self, name: str) -> WorkspaceEntry | None:
        """
        Retrieve a single workspace entry by name.

        Args:
            name: Workspace name.

        Returns:
            The :class:`WorkspaceEntry`, or ``None`` if not found.
        """
        data = self._load_raw()
        raw = data.get(name)
        if raw is None:
            return None
        try:
            return WorkspaceEntry.from_dict(raw)
        except (KeyError, TypeError) as exc:
            logger.error("Failed to deserialise workspace entry %r: %s", name, exc)
            return None

    def list_all(self) -> list[WorkspaceEntry]:
        """
        Return all workspace entries, sorted by name.

        Returns:
            A list of :class:`WorkspaceEntry` objects.
        """
        data = self._load_raw()
        entries: list[WorkspaceEntry] = []
        for name, raw in sorted(data.items()):
            try:
                entries.append(WorkspaceEntry.from_dict(raw))
            except (KeyError, TypeError) as exc:
                logger.warning(
                    "Skipping malformed registry entry %r: %s", name, exc
                )
        return entries

    def delete(self, name: str) -> bool:
        """
        Remove a workspace entry from the registry.

        Args:
            name: Workspace name to delete.

        Returns:
            ``True`` if the entry existed and was deleted, ``False`` otherwise.
        """
        data = self._load_raw()
        if name not in data:
            logger.warning("delete() called for unknown workspace %r", name)
            return False
        del data[name]
        self._save_raw(data)
        logger.debug("Deleted workspace entry: %s", name)
        return True

    def update_instructions(self, name: str, instructions: str) -> bool:
        """
        Update the custom instructions for an existing workspace entry.

        Args:
            name:         Workspace name.
            instructions: New instructions text (replaces the existing value).

        Returns:
            ``True`` on success, ``False`` when the entry was not found.
        """
        data = self._load_raw()
        if name not in data:
            logger.warning(
                "update_instructions() called for unknown workspace %r", name
            )
            return False
        data[name]["instructions"] = instructions
        self._save_raw(data)
        logger.debug("Updated instructions for workspace: %s", name)
        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_raw(self) -> dict[str, dict]:
        """Read the registry JSON file, returning an empty dict if absent."""
        if not self._path.exists():
            return {}
        try:
            with self._path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error(
                "Failed to read registry at %s: %s — starting with empty registry",
                self._path,
                exc,
            )
            return {}

    def _save_raw(self, data: dict[str, dict]) -> None:
        """
        Write *data* to the registry file atomically.

        Writes to a temporary file first and then renames it so that
        concurrent readers never see a partial write.
        """
        tmp_path = self._path.with_suffix(".json.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
                fh.write("\n")
            tmp_path.replace(self._path)
        except OSError as exc:
            logger.error("Failed to write registry at %s: %s", self._path, exc)
            tmp_path.unlink(missing_ok=True)
            raise
