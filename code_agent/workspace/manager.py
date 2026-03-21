"""
manager.py — Local workspace management for cloned repositories.

Provides :class:`WorkspaceManager`, which handles cloning, pulling, and
cleaning up local Git working trees used by the agent during analysis tasks.
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

import git
from git import Repo, InvalidGitRepositoryError, NoSuchPathError

from code_agent.config import get_settings

logger = logging.getLogger(__name__)


class WorkspaceManager:
    """
    Manages a directory of local Git clones used as ephemeral workspaces.

    All workspace directories live under *base_dir*.  The class is safe to
    use from a single thread; for concurrent cloning callers should use a
    :class:`concurrent.futures.ThreadPoolExecutor` and call :meth:`clone` or
    :meth:`get_or_clone` from worker threads.

    Args:
        base_dir: Root directory under which all workspaces are created.
                  Defaults to ``WORKSPACE_DIR`` from application settings.
    """

    def __init__(self, base_dir: str | None = None) -> None:
        settings = get_settings()
        self.base_dir = Path(base_dir or settings.WORKSPACE_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("WorkspaceManager initialised at %s", self.base_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clone(
        self,
        clone_url: str,
        shallow: bool = True,
        branch: str | None = None,
    ) -> str:
        """
        Clone a repository into a new unique sub-directory.

        Each call creates a fresh directory named by a UUID, so multiple
        simultaneous clones of the same repository do not interfere.

        Args:
            clone_url: HTTPS URL (may contain embedded credentials).
            shallow:   When ``True`` (default), performs a shallow clone
                       (``--depth 1``).
            branch:    Specific branch to check out.  When ``None`` the remote
                       default branch is used.

        Returns:
            Absolute path to the cloned directory as a string.
        """
        dest = self.base_dir / str(uuid.uuid4())
        dest.mkdir(parents=True, exist_ok=True)
        logger.info("Cloning %s -> %s (shallow=%s)", _safe_url(clone_url), dest, shallow)

        kwargs: dict = {}
        if shallow:
            settings = get_settings()
            kwargs["depth"] = settings.SHALLOW_CLONE_DEPTH
        if branch:
            kwargs["branch"] = branch

        try:
            Repo.clone_from(clone_url, str(dest), **kwargs)
        except git.GitCommandError as exc:
            # Clean up the empty directory on failure
            shutil.rmtree(dest, ignore_errors=True)
            raise RuntimeError(
                f"Failed to clone {_safe_url(clone_url)}: {exc}"
            ) from exc

        logger.info("Clone complete: %s", dest)
        return str(dest)

    def get_or_clone(
        self,
        clone_url: str,
        repo_id: str,
        branch: str | None = None,
    ) -> str:
        """
        Return the path to a local clone, creating one if it does not exist.

        When the directory already contains a valid Git repository it is
        updated via ``git fetch`` + ``git pull``.  This makes the method
        idempotent and suitable for repeated calls.

        Args:
            clone_url: HTTPS URL (may contain embedded credentials).
            repo_id:   Stable identifier used as the directory name under
                       *base_dir* (e.g. ``"my-org-my-repo"``).
            branch:    Branch to check out / pull.  Uses remote default when
                       ``None``.

        Returns:
            Absolute path to the local clone as a string.
        """
        dest = self.base_dir / _sanitise_id(repo_id)

        if dest.exists():
            try:
                repo = Repo(str(dest))
                logger.info(
                    "Workspace %s already exists — pulling latest changes", dest
                )
                self._fetch_and_pull(repo, branch)
                return str(dest)
            except (InvalidGitRepositoryError, NoSuchPathError):
                logger.warning(
                    "Directory %s exists but is not a valid git repo — re-cloning",
                    dest,
                )
                shutil.rmtree(dest, ignore_errors=True)

        dest.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Cloning %s -> %s", _safe_url(clone_url), dest
        )
        kwargs: dict = {}
        if branch:
            kwargs["branch"] = branch
        try:
            Repo.clone_from(clone_url, str(dest), **kwargs)
        except git.GitCommandError as exc:
            shutil.rmtree(dest, ignore_errors=True)
            raise RuntimeError(
                f"Failed to clone {_safe_url(clone_url)}: {exc}"
            ) from exc

        return str(dest)

    def pull(self, local_path: str) -> None:
        """
        Run ``git pull`` on an existing local clone.

        Args:
            local_path: Absolute path to the local clone.

        Raises:
            RuntimeError: When the pull fails.
        """
        repo = Repo(local_path)
        self._fetch_and_pull(repo, branch=None)

    def get_changed_files(
        self, local_path: str, from_sha: str, to_sha: str = "HEAD"
    ) -> list[str]:
        """
        Return the list of files that changed between two commits.

        Equivalent to ``git diff --name-only {from_sha} {to_sha}``.

        Args:
            local_path: Absolute path to the local clone.
            from_sha:   Starting commit SHA.
            to_sha:     Ending commit SHA.  Defaults to ``"HEAD"``.

        Returns:
            A list of repository-relative file paths.
        """
        repo = Repo(local_path)
        try:
            diff_output: str = repo.git.diff(
                "--name-only", from_sha, to_sha
            )
        except git.GitCommandError as exc:
            raise RuntimeError(
                f"git diff --name-only {from_sha}..{to_sha} failed: {exc}"
            ) from exc
        files = [line for line in diff_output.splitlines() if line.strip()]
        return files

    def get_current_sha(self, local_path: str) -> str:
        """
        Return the SHA of HEAD for a local clone.

        Args:
            local_path: Absolute path to the local clone.

        Returns:
            The full 40-character commit SHA string.
        """
        repo = Repo(local_path)
        return repo.head.commit.hexsha

    def cleanup(self, local_path: str) -> None:
        """
        Remove a local workspace directory entirely.

        Args:
            local_path: Absolute path to the workspace directory to delete.
        """
        path = Path(local_path)
        if path.exists():
            shutil.rmtree(path)
            logger.info("Removed workspace: %s", local_path)
        else:
            logger.warning("cleanup() called on non-existent path: %s", local_path)

    def list_workspaces(self) -> list[dict]:
        """
        Return a list of all workspace directories under *base_dir*.

        Each entry is a dict with keys:

        * ``path`` — absolute path to the workspace
        * ``name`` — directory name
        * ``is_git_repo`` — whether it contains a valid ``.git`` directory
        * ``head_sha`` — current HEAD SHA (empty string if not a valid repo)
        * ``size_bytes`` — total size on disk in bytes

        Returns:
            A list of workspace metadata dicts.
        """
        results: list[dict] = []
        if not self.base_dir.exists():
            return results

        for entry in sorted(self.base_dir.iterdir()):
            if not entry.is_dir():
                continue
            # Skip the registry file directory marker
            if entry.name.startswith("."):
                continue

            size = _dir_size(entry)
            is_git = (entry / ".git").exists()
            head_sha = ""
            if is_git:
                try:
                    repo = Repo(str(entry))
                    head_sha = repo.head.commit.hexsha
                except Exception:
                    pass

            results.append(
                {
                    "path": str(entry),
                    "name": entry.name,
                    "is_git_repo": is_git,
                    "head_sha": head_sha,
                    "size_bytes": size,
                }
            )
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_and_pull(self, repo: Repo, branch: str | None) -> None:
        """Fetch from origin and pull the current or specified branch."""
        try:
            origin = repo.remotes.origin
            origin.fetch()
            if branch:
                # Check out the branch if not already on it
                try:
                    repo.git.checkout(branch)
                except git.GitCommandError:
                    # Branch may be a remote tracking branch
                    repo.git.checkout("-b", branch, f"origin/{branch}")
            origin.pull()
        except git.GitCommandError as exc:
            raise RuntimeError(f"git pull failed: {exc}") from exc


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _safe_url(url: str) -> str:
    """Return *url* with any embedded password replaced by ``***``."""
    try:
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(url)
        if parsed.password:
            netloc = f"{parsed.username}:***@{parsed.hostname}"
            if parsed.port:
                netloc += f":{parsed.port}"
            return urlunparse(parsed._replace(netloc=netloc))
    except Exception:
        pass
    return url


def _sanitise_id(repo_id: str) -> str:
    """Replace characters that are problematic in directory names."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in repo_id)


def _dir_size(path: Path) -> int:
    """Return the total size in bytes of all files under *path*."""
    total = 0
    try:
        for dirpath, _, filenames in os.walk(path):
            for fname in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, fname))
                except OSError:
                    pass
    except OSError:
        pass
    return total
