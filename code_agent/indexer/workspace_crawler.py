"""Workspace crawler — clone and index all repositories in a VCS workspace.

Iterates over repositories provided by a :class:`~code_agent.vcs.base.VCSProvider`,
clones each one (shallow by default), indexes it with
:class:`~code_agent.indexer.repository_indexer.RepositoryIndexer`, and
optionally cleans up the local clone.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from code_agent.indexer.repository_indexer import IndexingResult, RepositoryIndexer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimal structural types so this module does not require the VCS package
# ---------------------------------------------------------------------------

@runtime_checkable
class RepoInfo(Protocol):
    """Minimal interface expected from VCS provider repository objects."""

    @property
    def full_name(self) -> str:
        """Unique name, e.g. ``"acme/myapp"``."""
        ...

    @property
    def clone_url(self) -> str:
        """HTTPS or SSH clone URL."""
        ...

    @property
    def default_branch(self) -> str:
        """Default branch name (``"main"``, ``"master"``, etc.)."""
        ...


@runtime_checkable
class VCSProvider(Protocol):
    """Minimal interface expected from VCS provider objects."""

    def list_repos(self) -> list[Any]:
        """Return a list of repository objects (must satisfy :class:`RepoInfo`)."""
        ...


@runtime_checkable
class WorkspaceManager(Protocol):
    """Minimal interface for a workspace manager that can provide clone paths."""

    def get_clone_path(self, repo_full_name: str) -> str:
        """Return the local directory path to clone *repo_full_name* into."""
        ...


class WorkspaceCrawler:
    """Clones and indexes every repository in a VCS workspace.

    Uses a :class:`~concurrent.futures.ThreadPoolExecutor` to clone and
    index repositories in parallel.

    Example::

        crawler = WorkspaceCrawler(
            provider=github_provider,
            indexer=repo_indexer,
            workspace_manager=manager,
        )
        results = crawler.crawl_and_index(
            workspace_id="acme",
            max_repos=50,
            languages=["python", "go"],
        )
    """

    def __init__(
        self,
        provider: VCSProvider,
        indexer: RepositoryIndexer,
        workspace_manager: WorkspaceManager | None = None,
        max_workers: int = 4,
        shallow_clone_depth: int = 1,
        cleanup_after_index: bool = True,
    ) -> None:
        """Initialise the crawler.

        Args:
            provider:            VCS provider that lists and describes repos.
            indexer:             :class:`RepositoryIndexer` instance.
            workspace_manager:   Optional manager supplying clone paths.
                                 If ``None``, temporary directories are used.
            max_workers:         Thread-pool parallelism for clone+index tasks.
            shallow_clone_depth: ``--depth`` value for ``git clone``.
            cleanup_after_index: Remove the local clone when indexing finishes.
        """
        self._provider = provider
        self._indexer = indexer
        self._workspace_manager = workspace_manager
        self._max_workers = max_workers
        self._shallow_clone_depth = shallow_clone_depth
        self._cleanup = cleanup_after_index

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def crawl_and_index(
        self,
        workspace_id: str,
        max_repos: int | None = None,
        languages: list[str] | None = None,
        exclude_repos: list[str] | None = None,
        mode: str = "full",
        exclude_patterns: list[str] | None = None,
    ) -> list[IndexingResult]:
        """Discover, clone, and index repositories from the configured provider.

        Args:
            workspace_id:     Friendly name / namespace for the workspace.
            max_repos:        Cap on total repos processed (``None`` = no limit).
            languages:        If set, only files in these languages are indexed.
            exclude_repos:    Repository full names to skip (e.g. ``["org/huge-repo"]``).
            mode:             ``"full"`` or ``"incremental"`` passed to indexer.
            exclude_patterns: Glob file patterns passed to indexer.

        Returns:
            List of :class:`IndexingResult` objects, one per processed repo.
        """
        exclude_set = set(exclude_repos or [])

        try:
            all_repos = self._provider.list_repos()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to list repos from provider: %s", exc)
            return []

        repos_to_process = [
            r for r in all_repos
            if not _repo_name(r) or _repo_name(r) not in exclude_set
        ]

        if max_repos is not None:
            repos_to_process = repos_to_process[:max_repos]

        logger.info(
            "WorkspaceCrawler: processing %d repos for workspace %s",
            len(repos_to_process), workspace_id,
        )

        results: list[IndexingResult] = []

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_repo = {
                executor.submit(
                    self._process_repo,
                    repo,
                    workspace_id,
                    languages,
                    exclude_patterns,
                    mode,
                ): repo
                for repo in repos_to_process
            }

            for future in as_completed(future_to_repo):
                repo = future_to_repo[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Unhandled error processing repo %s: %s",
                        _repo_name(repo), exc,
                    )

        total_chunks = sum(r.total_chunks for r in results)
        logger.info(
            "WorkspaceCrawler: finished workspace %s — %d repos, %d total chunks",
            workspace_id, len(results), total_chunks,
        )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_repo(
        self,
        repo: Any,
        workspace_id: str,
        languages: list[str] | None,
        exclude_patterns: list[str] | None,
        mode: str,
    ) -> IndexingResult | None:
        """Clone (or locate) a single repo and index it.

        Args:
            repo:             Repository object from the VCS provider.
            workspace_id:     Workspace name for metadata.
            languages:        Language filter passed to indexer.
            exclude_patterns: File glob exclusions passed to indexer.
            mode:             Indexing mode.

        Returns:
            :class:`IndexingResult` on success, ``None`` if skipped.
        """
        repo_name = _repo_name(repo)
        clone_url = _repo_clone_url(repo)
        default_branch = _repo_default_branch(repo)

        if not clone_url:
            logger.warning("Skipping repo %s — no clone URL", repo_name)
            return None

        repo_id = f"{workspace_id}/{repo_name}" if repo_name else workspace_id
        metadata = {
            "workspace": workspace_id,
            "repo": repo_name,
            "repo_id": repo_id,
            "default_branch": default_branch,
            "clone_url": clone_url,
        }

        # Determine local clone path
        tmp_dir: str | None = None
        if self._workspace_manager is not None and repo_name:
            try:
                local_path = self._workspace_manager.get_clone_path(repo_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "workspace_manager.get_clone_path failed for %s: %s — using tmpdir",
                    repo_name, exc,
                )
                tmp_dir = tempfile.mkdtemp(prefix="code_agent_")
                local_path = tmp_dir
        else:
            tmp_dir = tempfile.mkdtemp(prefix="code_agent_")
            local_path = tmp_dir

        try:
            self._clone_repo(clone_url, local_path, default_branch)
        except Exception as exc:  # noqa: BLE001
            logger.error("Clone failed for %s: %s", repo_name, exc)
            _maybe_cleanup(local_path if tmp_dir else None)
            err_result = IndexingResult(repo_id=repo_id)
            err_result.errors.append(f"Clone failed: {exc}")
            return err_result

        try:
            result = self._indexer.index_repository(
                local_path=local_path,
                repo_id=repo_id,
                metadata=metadata,
                mode=mode,
                languages=languages,
                exclude_patterns=exclude_patterns,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Indexing failed for %s: %s", repo_name, exc)
            result = IndexingResult(repo_id=repo_id)
            result.errors.append(f"Indexing failed: {exc}")
        finally:
            if self._cleanup:
                cleanup_path = tmp_dir or (local_path if tmp_dir is None else None)
                if cleanup_path:
                    _maybe_cleanup(cleanup_path)

        return result

    def _clone_repo(
        self, clone_url: str, local_path: str, branch: str | None
    ) -> None:
        """Shallow-clone *clone_url* into *local_path*.

        If *local_path* already contains a ``.git`` directory the clone is
        skipped (assumed already present from a previous run).

        Args:
            clone_url:  Repository clone URL.
            local_path: Destination directory.
            branch:     Branch to clone (``None`` clones the default branch).

        Raises:
            RuntimeError: If ``git clone`` exits with a non-zero status.
        """
        git_dir = Path(local_path) / ".git"
        if git_dir.exists():
            logger.debug("Skipping clone — %s already exists", local_path)
            return

        cmd = [
            "git", "clone",
            "--depth", str(self._shallow_clone_depth),
            "--single-branch",
        ]
        if branch:
            cmd.extend(["--branch", branch])
        cmd.extend([clone_url, local_path])

        logger.info("Cloning %s → %s (depth=%d)", clone_url, local_path, self._shallow_clone_depth)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"git clone exited {result.returncode}: {result.stderr.strip()}"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repo_name(repo: Any) -> str:
    """Extract the full name from a repo object."""
    for attr in ("full_name", "name", "slug"):
        if hasattr(repo, attr):
            v = getattr(repo, attr)
            if callable(v):
                v = v()
            if v:
                return str(v)
    return str(repo)


def _repo_clone_url(repo: Any) -> str:
    """Extract the clone URL from a repo object."""
    for attr in ("clone_url", "html_url", "http_url", "ssh_url"):
        if hasattr(repo, attr):
            v = getattr(repo, attr)
            if callable(v):
                v = v()
            if v:
                return str(v)
    if isinstance(repo, dict):
        return str(repo.get("clone_url") or repo.get("http_url") or "")
    return ""


def _repo_default_branch(repo: Any) -> str:
    """Extract the default branch from a repo object."""
    for attr in ("default_branch", "mainbranch", "master_branch"):
        if hasattr(repo, attr):
            v = getattr(repo, attr)
            if callable(v):
                v = v()
            if v:
                return str(v)
    return "main"


def _maybe_cleanup(path: str | None) -> None:
    """Remove *path* if it exists, ignoring errors."""
    if not path:
        return
    try:
        shutil.rmtree(path, ignore_errors=True)
        logger.debug("Cleaned up %s", path)
    except Exception:  # noqa: BLE001
        pass
