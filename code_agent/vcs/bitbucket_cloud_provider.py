"""
bitbucket_cloud_provider.py — VCS provider implementation for Bitbucket Cloud.

Uses the Bitbucket Cloud REST API v2.0 (api.bitbucket.org/2.0) with
username / app-password authentication.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator
from urllib.parse import urljoin

import requests
from requests.auth import HTTPBasicAuth

from code_agent.vcs.base import (
    FileTreeEntry,
    PRInfo,
    RepoInfo,
    VCSProvider,
)

logger = logging.getLogger(__name__)

_BB_API = "https://api.bitbucket.org/2.0"


class BitbucketCloudProvider(VCSProvider):
    """
    VCS provider that talks to Bitbucket Cloud via REST API v2.0.

    Args:
        username:     Bitbucket account username (used for auth and clone URLs).
        app_password: Bitbucket App Password (not your account password).
        workspace:    Bitbucket workspace slug (previously called "team").
    """

    def __init__(self, username: str, app_password: str, workspace: str) -> None:
        self._username = username
        self._app_password = app_password
        self._workspace = workspace
        self._auth = HTTPBasicAuth(username, app_password)
        self._session = requests.Session()
        self._session.auth = self._auth
        self._session.headers.update({"Accept": "application/json"})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, url: str, **kwargs: Any) -> dict[str, Any]:
        """Perform a GET request and return parsed JSON."""
        resp = self._session.get(url, timeout=30, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def _paginate(self, url: str, params: dict | None = None) -> Iterator[dict]:
        """
        Yield individual items from a paginated Bitbucket Cloud response.

        Follows the ``next`` link until exhausted.
        """
        next_url: str | None = url
        while next_url:
            resp = self._session.get(next_url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("values", []):
                yield item
            next_url = data.get("next")
            # params only apply to the first request; subsequent URLs include
            # all query parameters already.
            params = None

    def _clone_url(self, slug: str) -> str:
        """Return an HTTPS clone URL with credentials embedded."""
        return (
            f"https://{self._username}:{self._app_password}"
            f"@bitbucket.org/{self._workspace}/{slug}.git"
        )

    def _pr_state_to_bb(self, state: str) -> str:
        """Map a generic state string to a Bitbucket Cloud API state value."""
        mapping = {
            "open": "OPEN",
            "merged": "MERGED",
            "closed": "DECLINED",
            "declined": "DECLINED",
            "all": "ALL",
        }
        return mapping.get(state.lower(), "OPEN")

    def _parse_pr(self, data: dict) -> PRInfo:
        """Convert a Bitbucket Cloud pull-request payload to :class:`PRInfo`."""
        author_data = data.get("author") or {}
        author = author_data.get("display_name") or author_data.get("nickname", "")
        state_raw = data.get("state", "OPEN")
        state_map = {
            "OPEN": "open",
            "MERGED": "merged",
            "DECLINED": "declined",
            "SUPERSEDED": "declined",
        }
        state = state_map.get(state_raw, state_raw.lower())
        return PRInfo(
            id=data["id"],
            title=data.get("title", ""),
            description=data.get("description", ""),
            source_branch=data.get("source", {}).get("branch", {}).get("name", ""),
            target_branch=data.get("destination", {}).get("branch", {}).get("name", ""),
            state=state,
            author=author,
        )

    # ------------------------------------------------------------------
    # VCSProvider interface
    # ------------------------------------------------------------------

    def list_repos(self) -> list[RepoInfo]:
        """List all repositories in the workspace."""
        url = f"{_BB_API}/repositories/{self._workspace}"
        repos: list[RepoInfo] = []
        for item in self._paginate(url, params={"pagelen": 100}):
            slug = item.get("slug", "")
            name = item.get("name", slug)
            description = item.get("description", "")
            # Locate the HTTPS clone link
            clone_links = item.get("links", {}).get("clone", [])
            clone_https = ""
            for link in clone_links:
                if link.get("name") == "https":
                    clone_https = link.get("href", "")
                    break
            default_branch = (
                (item.get("mainbranch") or {}).get("name") or "main"
            )
            repos.append(
                RepoInfo(
                    slug=slug,
                    name=name,
                    clone_url_https=self._clone_url(slug),
                    default_branch=default_branch,
                    description=description,
                    vcs_type="bitbucket-cloud",
                    workspace=self._workspace,
                )
            )
        return repos

    def get_repo_info(self, repo_slug: str) -> RepoInfo:
        """Return metadata for a single repository."""
        url = f"{_BB_API}/repositories/{self._workspace}/{repo_slug}"
        data = self._get(url)
        default_branch = (
            (data.get("mainbranch") or {}).get("name") or "main"
        )
        return RepoInfo(
            slug=repo_slug,
            name=data.get("name", repo_slug),
            clone_url_https=self._clone_url(repo_slug),
            default_branch=default_branch,
            description=data.get("description", ""),
            vcs_type="bitbucket-cloud",
            workspace=self._workspace,
        )

    def get_file_content(self, repo_slug: str, path: str, ref: str = "HEAD") -> str:
        """
        Retrieve the raw text content of a file at *ref*.

        Bitbucket Cloud returns the raw file via
        ``/2.0/repositories/{ws}/{slug}/src/{ref}/{path}``.
        """
        # Bitbucket Cloud doesn't understand "HEAD" — use the default branch
        if ref == "HEAD":
            ref = self.get_repo_info(repo_slug).default_branch
        url = f"{_BB_API}/repositories/{self._workspace}/{repo_slug}/src/{ref}/{path}"
        resp = self._session.get(url, timeout=30)
        if resp.status_code == 404:
            raise FileNotFoundError(
                f"Path {path!r} not found in {repo_slug!r} at ref {ref!r}"
            )
        resp.raise_for_status()
        return resp.text

    def get_file_tree(
        self, repo_slug: str, path: str = "", ref: str = "HEAD"
    ) -> list[FileTreeEntry]:
        """
        List the directory at *path* (recursively follows sub-directories).

        Uses the ``?format=meta`` query parameter to get size information.
        """
        if ref == "HEAD":
            ref = self.get_repo_info(repo_slug).default_branch
        url_path = path.rstrip("/")
        base_url = (
            f"{_BB_API}/repositories/{self._workspace}/{repo_slug}"
            f"/src/{ref}/{url_path}"
        ).rstrip("/") + "/"
        entries: list[FileTreeEntry] = []
        self._walk_tree(base_url, entries)
        return entries

    def _walk_tree(self, url: str, out: list[FileTreeEntry]) -> None:
        """Recursively walk a directory URL into *out*."""
        for item in self._paginate(url, params={"format": "meta", "pagelen": 100}):
            item_type = item.get("type", "")
            path = item.get("path", "")
            if item_type == "commit_file":
                size = item.get("size", 0) or 0
                out.append(FileTreeEntry(path=path, type="file", size=size))
            elif item_type == "commit_directory":
                out.append(FileTreeEntry(path=path, type="dir", size=0))
                # Recurse into sub-directory
                links = item.get("links", {})
                self_href = links.get("self", {}).get("href", "")
                if self_href:
                    self._walk_tree(self_href.rstrip("/") + "/", out)

    def get_pr(self, repo_slug: str, pr_id: int | str) -> PRInfo:
        """Fetch a single pull request by ID."""
        url = (
            f"{_BB_API}/repositories/{self._workspace}/{repo_slug}"
            f"/pullrequests/{pr_id}"
        )
        data = self._get(url)
        return self._parse_pr(data)

    def list_prs(self, repo_slug: str, state: str = "open") -> list[PRInfo]:
        """
        List pull requests filtered by *state*.

        Args:
            repo_slug: Repository slug.
            state:     ``"open"``, ``"merged"``, ``"declined"``, or ``"all"``.
        """
        url = (
            f"{_BB_API}/repositories/{self._workspace}/{repo_slug}/pullrequests"
        )
        bb_state = self._pr_state_to_bb(state)
        params: dict[str, Any] = {"pagelen": 50}
        if bb_state != "ALL":
            params["state"] = bb_state

        result: list[PRInfo] = []
        for item in self._paginate(url, params=params):
            result.append(self._parse_pr(item))
        return result

    def get_pr_diff(self, repo_slug: str, pr_id: int | str) -> str:
        """
        Retrieve the unified diff for a pull request.

        Bitbucket Cloud returns a raw diff at
        ``/2.0/repositories/{ws}/{slug}/pullrequests/{id}/diff``.
        """
        url = (
            f"{_BB_API}/repositories/{self._workspace}/{repo_slug}"
            f"/pullrequests/{pr_id}/diff"
        )
        resp = self._session.get(url, timeout=60)
        resp.raise_for_status()
        return resp.text

    def post_review_comment(
        self, repo_slug: str, pr_id: int | str, body: str
    ) -> None:
        """Post a top-level comment on a pull request."""
        url = (
            f"{_BB_API}/repositories/{self._workspace}/{repo_slug}"
            f"/pullrequests/{pr_id}/comments"
        )
        resp = self._session.post(
            url,
            json={"content": {"raw": body}},
            timeout=30,
        )
        resp.raise_for_status()
        logger.info(
            "Posted review comment on PR #%s in %s/%s",
            pr_id,
            self._workspace,
            repo_slug,
        )

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
            repo_slug: Repository slug.
            title:     Pull-request title.
            body:      Pull-request description.
            head:      Source branch name.
            base:      Target branch name.

        Returns:
            A :class:`PRInfo` for the newly created pull request.
        """
        url = (
            f"{_BB_API}/repositories/{self._workspace}/{repo_slug}/pullrequests"
        )
        payload = {
            "title": title,
            "description": body,
            "source": {"branch": {"name": head}},
            "destination": {"branch": {"name": base}},
            "close_source_branch": False,
        }
        resp = self._session.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        logger.info(
            "Created PR #%s in %s/%s: %s",
            data.get("id"),
            self._workspace,
            repo_slug,
            title,
        )
        return self._parse_pr(data)
