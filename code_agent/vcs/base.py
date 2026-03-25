"""
base.py — Abstract VCS provider interface and shared data-transfer objects.

Every concrete VCS backend (GitHub, Bitbucket Cloud, Bitbucket Server …)
must subclass :class:`VCSProvider` and implement all abstract methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RepoInfo:
    """Lightweight descriptor for a repository."""

    slug: str
    """Short, URL-safe repository identifier (e.g. ``my-repo``)."""

    name: str
    """Human-readable repository name."""

    clone_url_https: str
    """HTTPS clone URL with credentials already embedded, ready for ``git clone``."""

    default_branch: str
    """Name of the default branch (e.g. ``main``, ``master``)."""

    description: str = ""
    """Optional repository description."""

    vcs_type: str = ""
    """Provider tag: ``"github"``, ``"bitbucket-cloud"``, or ``"bitbucket-server"``."""

    workspace: str = ""
    """Organisation / workspace / project that owns the repository."""


@dataclass
class PRInfo:
    """Lightweight descriptor for a pull / merge request."""

    id: int | str
    """Provider-assigned pull-request identifier."""

    title: str
    """Pull-request title."""

    description: str
    """Body / description text of the pull request."""

    source_branch: str
    """Branch that contains the proposed changes (head)."""

    target_branch: str
    """Branch that the changes will be merged into (base)."""

    state: str
    """Current state, e.g. ``"open"``, ``"merged"``, ``"declined"``, ``"closed"``."""

    author: str
    """Username or display name of the author."""

    diff: str = ""
    """Unified diff of the pull request (populated on demand)."""


@dataclass
class FileTreeEntry:
    """A single node in a repository's file tree."""

    path: str
    """Repository-relative path, e.g. ``src/main.py``."""

    type: str
    """Node type: ``"file"`` or ``"dir"``."""

    size: int = 0
    """File size in bytes (0 for directories or when not reported by the API)."""


class VCSProvider(ABC):
    """
    Abstract interface that every VCS backend must implement.

    Concrete subclasses are expected to hold their own authentication
    credentials and configuration, set up during ``__init__``.
    """

    @abstractmethod
    def list_repos(self) -> list[RepoInfo]:
        """
        Return all repositories visible to the authenticated user / org.

        Returns:
            A list of :class:`RepoInfo` objects, one per repository.
        """
        ...

    @abstractmethod
    def get_repo_info(self, repo_slug: str) -> RepoInfo:
        """
        Return metadata for a single repository.

        Args:
            repo_slug: Short identifier of the repository.

        Returns:
            A populated :class:`RepoInfo` instance.
        """
        ...

    @abstractmethod
    def get_file_content(self, repo_slug: str, path: str, ref: str = "HEAD") -> str:
        """
        Retrieve the text content of a single file.

        Args:
            repo_slug: Short identifier of the repository.
            path:      Repository-relative path to the file.
            ref:       Git ref (branch name, tag, or commit SHA).  Defaults to
                       ``"HEAD"``.

        Returns:
            The decoded file content as a UTF-8 string.

        Raises:
            FileNotFoundError: When the path does not exist at *ref*.
        """
        ...

    @abstractmethod
    def get_file_tree(
        self, repo_slug: str, path: str = "", ref: str = "HEAD"
    ) -> list[FileTreeEntry]:
        """
        List the contents of a directory (or the entire repository).

        Args:
            repo_slug: Short identifier of the repository.
            path:      Sub-directory to list.  Empty string means the root.
            ref:       Git ref (branch name, tag, or commit SHA).

        Returns:
            A list of :class:`FileTreeEntry` objects.
        """
        ...

    @abstractmethod
    def get_pr(self, repo_slug: str, pr_id: int | str) -> PRInfo:
        """
        Fetch a single pull request by ID.

        Args:
            repo_slug: Short identifier of the repository.
            pr_id:     Provider-assigned pull-request identifier.

        Returns:
            A populated :class:`PRInfo` instance (without diff).
        """
        ...

    @abstractmethod
    def list_prs(self, repo_slug: str, state: str = "open") -> list[PRInfo]:
        """
        List pull requests for a repository, filtered by state.

        Args:
            repo_slug: Short identifier of the repository.
            state:     Filter string.  Common values: ``"open"``, ``"closed"``,
                       ``"merged"``, ``"all"``.

        Returns:
            A list of :class:`PRInfo` objects.
        """
        ...

    @abstractmethod
    def get_pr_diff(self, repo_slug: str, pr_id: int | str) -> str:
        """
        Retrieve the unified diff for a pull request.

        Args:
            repo_slug: Short identifier of the repository.
            pr_id:     Provider-assigned pull-request identifier.

        Returns:
            The raw unified diff as a string.
        """
        ...

    @abstractmethod
    def post_review_comment(
        self, repo_slug: str, pr_id: int | str, body: str
    ) -> None:
        """
        Post a top-level review comment on a pull request.

        Args:
            repo_slug: Short identifier of the repository.
            pr_id:     Provider-assigned pull-request identifier.
            body:      Markdown-formatted comment body.
        """
        ...

    @abstractmethod
    def create_pr(
        self,
        repo_slug: str,
        title: str,
        body: str,
        head: str,
        base: str,
    ) -> PRInfo:
        """
        Open a new pull request.

        Args:
            repo_slug: Short identifier of the repository.
            title:     Pull-request title.
            body:      Pull-request description (Markdown).
            head:      Source branch (contains the proposed changes).
            base:      Target branch (will receive the merge).

        Returns:
            A :class:`PRInfo` describing the newly created pull request.
        """
        ...
