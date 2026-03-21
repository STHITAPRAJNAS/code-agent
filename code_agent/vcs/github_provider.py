"""
github_provider.py — VCS provider implementation backed by the GitHub REST API.

Authentication is handled via a Personal Access Token (PAT) or a GitHub App
installation token.  Both classic PATs and fine-grained PATs are supported.
"""

from __future__ import annotations

import base64
import logging
from typing import Optional
from urllib.parse import urlparse

import requests
from github import Github, GithubException
from github.GithubException import UnknownObjectException
from github.Repository import Repository

from code_agent.vcs.base import (
    FileTreeEntry,
    PRInfo,
    RepoInfo,
    VCSProvider,
)

logger = logging.getLogger(__name__)


class GitHubProvider(VCSProvider):
    """
    VCS provider that talks to GitHub via PyGithub and the GitHub REST API.

    Args:
        token: GitHub Personal Access Token (classic or fine-grained).
        org:   Optional organisation name.  When set, :meth:`list_repos`
               enumerates repositories belonging to that organisation rather
               than to the authenticated user.
    """

    def __init__(self, token: str, org: str | None = None) -> None:
        self._token = token
        self._org = org
        self._gh = Github(token)
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_repo(self, repo_slug: str) -> Repository:
        """
        Resolve *repo_slug* to a PyGithub :class:`Repository` object.

        The slug may be either ``owner/repo`` (full name) or just ``repo``
        (resolved against the configured org or authenticated user).
        """
        if "/" in repo_slug:
            return self._gh.get_repo(repo_slug)
        if self._org:
            return self._gh.get_repo(f"{self._org}/{repo_slug}")
        user = self._gh.get_user()
        return user.get_repo(repo_slug)

    def _build_clone_url(self, repo: Repository) -> str:
        """Return an HTTPS clone URL with the token embedded."""
        return f"https://{self._token}@github.com/{repo.full_name}.git"

    def _repo_to_info(self, repo: Repository) -> RepoInfo:
        """Convert a PyGithub Repository into a :class:`RepoInfo`."""
        default_branch = repo.default_branch or "main"
        # Determine owner / workspace
        workspace = ""
        try:
            workspace = repo.organization.login if repo.organization else repo.owner.login
        except Exception:
            workspace = repo.owner.login if repo.owner else ""

        return RepoInfo(
            slug=repo.name,
            name=repo.full_name,
            clone_url_https=self._build_clone_url(repo),
            default_branch=default_branch,
            description=repo.description or "",
            vcs_type="github",
            workspace=workspace,
        )

    # ------------------------------------------------------------------
    # VCSProvider interface
    # ------------------------------------------------------------------

    def list_repos(self) -> list[RepoInfo]:
        """
        Return all repositories for the configured organisation or the
        authenticated user.
        """
        repos: list[RepoInfo] = []
        try:
            if self._org:
                org = self._gh.get_organization(self._org)
                for repo in org.get_repos():
                    repos.append(self._repo_to_info(repo))
            else:
                user = self._gh.get_user()
                for repo in user.get_repos():
                    repos.append(self._repo_to_info(repo))
        except GithubException as exc:
            logger.error("GitHub list_repos failed: %s", exc)
            raise
        return repos

    def get_repo_info(self, repo_slug: str) -> RepoInfo:
        """Return metadata for the named repository."""
        repo = self._get_repo(repo_slug)
        return self._repo_to_info(repo)

    def get_file_content(self, repo_slug: str, path: str, ref: str = "HEAD") -> str:
        """
        Retrieve the decoded text content of a file at *ref*.

        Args:
            repo_slug: Repository slug or ``owner/repo`` full name.
            path:      Repository-relative path to the file.
            ref:       Branch name, tag, or commit SHA.  Defaults to
                       ``"HEAD"`` which GitHub resolves to the default branch.

        Returns:
            UTF-8 decoded file content.

        Raises:
            FileNotFoundError: When the path does not exist.
        """
        repo = self._get_repo(repo_slug)
        try:
            # get_contents accepts a ref kwarg; "HEAD" works for default branch
            contents = repo.get_contents(path, ref=ref)
        except UnknownObjectException:
            raise FileNotFoundError(
                f"Path {path!r} not found in {repo_slug!r} at ref {ref!r}"
            )
        if isinstance(contents, list):
            raise IsADirectoryError(
                f"Path {path!r} is a directory in {repo_slug!r}"
            )
        if contents.encoding == "base64" and contents.content:
            return base64.b64decode(contents.content).decode("utf-8", errors="replace")
        # Fallback: use download_url
        if contents.download_url:
            resp = self._session.get(contents.download_url, timeout=30)
            resp.raise_for_status()
            return resp.text
        return ""

    def get_file_tree(
        self, repo_slug: str, path: str = "", ref: str = "HEAD"
    ) -> list[FileTreeEntry]:
        """
        Return a recursive listing of the repository (or a sub-directory).

        For large repositories this uses the Git Trees API with
        ``recursive=True`` for efficiency, falling back to a shallow listing
        when *path* is non-empty.
        """
        repo = self._get_repo(repo_slug)
        entries: list[FileTreeEntry] = []

        if not path:
            # Use the efficient recursive trees endpoint for the full tree
            try:
                # Resolve "HEAD" to an actual SHA the Trees API understands
                actual_ref = ref if ref != "HEAD" else repo.default_branch
                tree = repo.get_git_tree(actual_ref, recursive=True)
                for item in tree.tree:
                    if item.type == "blob":
                        entries.append(
                            FileTreeEntry(path=item.path, type="file", size=item.size or 0)
                        )
                    elif item.type == "tree":
                        entries.append(FileTreeEntry(path=item.path, type="dir", size=0))
            except GithubException as exc:
                logger.warning(
                    "get_git_tree failed for %s, falling back to get_contents: %s",
                    repo_slug,
                    exc,
                )
                self._recursive_contents(repo, path, ref, entries)
        else:
            self._recursive_contents(repo, path, ref, entries)

        return entries

    def _recursive_contents(
        self,
        repo: Repository,
        path: str,
        ref: str,
        out: list[FileTreeEntry],
    ) -> None:
        """Recursively walk ``get_contents`` results into *out*."""
        try:
            items = repo.get_contents(path, ref=ref)
        except UnknownObjectException:
            return
        if not isinstance(items, list):
            items = [items]
        for item in items:
            if item.type == "dir":
                out.append(FileTreeEntry(path=item.path, type="dir", size=0))
                self._recursive_contents(repo, item.path, ref, out)
            else:
                out.append(
                    FileTreeEntry(path=item.path, type="file", size=item.size or 0)
                )

    def get_pr(self, repo_slug: str, pr_id: int | str) -> PRInfo:
        """Fetch a single pull request by its numeric ID."""
        repo = self._get_repo(repo_slug)
        pr = repo.get_pull(int(pr_id))
        return PRInfo(
            id=pr.number,
            title=pr.title,
            description=pr.body or "",
            source_branch=pr.head.ref,
            target_branch=pr.base.ref,
            state=pr.state,
            author=pr.user.login if pr.user else "",
        )

    def list_prs(self, repo_slug: str, state: str = "open") -> list[PRInfo]:
        """
        List pull requests filtered by *state*.

        Args:
            repo_slug: Repository slug or ``owner/repo`` full name.
            state:     ``"open"``, ``"closed"``, or ``"all"``.

        Returns:
            A list of :class:`PRInfo` objects (without diff content).
        """
        # GitHub API accepts "open", "closed", "all" — map "merged" → "closed"
        api_state = "closed" if state == "merged" else state
        repo = self._get_repo(repo_slug)
        result: list[PRInfo] = []
        for pr in repo.get_pulls(state=api_state, sort="updated", direction="desc"):
            result.append(
                PRInfo(
                    id=pr.number,
                    title=pr.title,
                    description=pr.body or "",
                    source_branch=pr.head.ref,
                    target_branch=pr.base.ref,
                    state=pr.state,
                    author=pr.user.login if pr.user else "",
                )
            )
        return result

    def get_pr_diff(self, repo_slug: str, pr_id: int | str) -> str:
        """
        Retrieve the unified diff for a pull request via the GitHub REST API.

        Uses the ``application/vnd.github.v3.diff`` Accept header to get the
        raw diff format.
        """
        repo = self._get_repo(repo_slug)
        url = f"https://api.github.com/repos/{repo.full_name}/pulls/{pr_id}"
        resp = self._session.get(
            url,
            headers={"Accept": "application/vnd.github.v3.diff"},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.text

    def post_review_comment(
        self, repo_slug: str, pr_id: int | str, body: str
    ) -> None:
        """Post a top-level issue comment on the pull request."""
        repo = self._get_repo(repo_slug)
        pr = repo.get_pull(int(pr_id))
        pr.create_issue_comment(body)
        logger.info("Posted review comment on PR #%s in %s", pr_id, repo_slug)

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
            repo_slug: Repository slug or ``owner/repo`` full name.
            title:     Pull-request title.
            body:      Pull-request description (Markdown).
            head:      Source branch name.
            base:      Target branch name.

        Returns:
            A :class:`PRInfo` for the newly created pull request.
        """
        repo = self._get_repo(repo_slug)
        pr = repo.create_pull(title=title, body=body, head=head, base=base)
        logger.info("Created PR #%s in %s: %s", pr.number, repo_slug, title)
        return PRInfo(
            id=pr.number,
            title=pr.title,
            description=pr.body or "",
            source_branch=pr.head.ref,
            target_branch=pr.base.ref,
            state=pr.state,
            author=pr.user.login if pr.user else "",
        )
