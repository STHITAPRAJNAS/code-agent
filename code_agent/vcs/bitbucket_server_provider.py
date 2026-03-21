"""
bitbucket_server_provider.py — VCS provider for Bitbucket Server / Data Center.

Uses the Bitbucket Server REST API 1.0 with Bearer-token authentication.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator
from urllib.parse import urlparse

import requests

from code_agent.vcs.base import (
    FileTreeEntry,
    PRInfo,
    RepoInfo,
    VCSProvider,
)

logger = logging.getLogger(__name__)


class BitbucketServerProvider(VCSProvider):
    """
    VCS provider that talks to Bitbucket Server / Data Center via REST API 1.0.

    Args:
        base_url:    Root URL of the Bitbucket Server instance, e.g.
                     ``https://bitbucket.example.com``.
        token:       HTTP access token (personal or project-scoped).
        project_key: The Bitbucket Server project key, e.g. ``"MYPROJ"``.
    """

    def __init__(self, base_url: str, token: str, project_key: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._project_key = project_key
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _api(self, path: str) -> str:
        """Build a full REST API 1.0 URL from *path*."""
        return f"{self._base_url}/rest/api/1.0{path}"

    def _get(self, url: str, **kwargs: Any) -> dict[str, Any]:
        """Perform a GET request and return parsed JSON, raising on error."""
        resp = self._session.get(url, timeout=30, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def _paginate(self, url: str, params: dict | None = None) -> Iterator[dict]:
        """
        Yield individual items from a Bitbucket Server paginated response.

        Bitbucket Server uses ``isLastPage`` / ``nextPageStart`` pagination.
        """
        params = dict(params or {})
        params.setdefault("limit", 100)
        start = 0
        while True:
            params["start"] = start
            resp = self._session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("values", []):
                yield item
            if data.get("isLastPage", True):
                break
            start = data.get("nextPageStart", start + 1)

    def _clone_url(self, slug: str) -> str:
        """
        Build an HTTPS clone URL with the token embedded.

        Format:  ``https://x-token-auth:{token}@{host}/scm/{project}/{slug}.git``
        """
        parsed = urlparse(self._base_url)
        host = parsed.netloc  # e.g. "bitbucket.example.com"
        project_lower = self._project_key.lower()
        return (
            f"https://x-token-auth:{self._token}"
            f"@{host}/scm/{project_lower}/{slug}.git"
        )

    def _parse_pr(self, data: dict) -> PRInfo:
        """Convert a Bitbucket Server pull-request payload to :class:`PRInfo`."""
        author = (
            (data.get("author") or {}).get("user", {}).get("displayName", "")
            or (data.get("author") or {}).get("user", {}).get("name", "")
        )
        state_raw = data.get("state", "OPEN")
        state_map = {
            "OPEN": "open",
            "MERGED": "merged",
            "DECLINED": "declined",
        }
        state = state_map.get(state_raw, state_raw.lower())
        from_ref = data.get("fromRef", {})
        to_ref = data.get("toRef", {})
        return PRInfo(
            id=data["id"],
            title=data.get("title", ""),
            description=data.get("description", ""),
            source_branch=from_ref.get("displayId", from_ref.get("id", "")),
            target_branch=to_ref.get("displayId", to_ref.get("id", "")),
            state=state,
            author=author,
        )

    # ------------------------------------------------------------------
    # VCSProvider interface
    # ------------------------------------------------------------------

    def list_repos(self) -> list[RepoInfo]:
        """List all repositories in the configured project."""
        url = self._api(f"/projects/{self._project_key}/repos")
        repos: list[RepoInfo] = []
        for item in self._paginate(url):
            slug = item.get("slug", "")
            name = item.get("name", slug)
            # Find HTTPS clone link from the links block
            clone_links = item.get("links", {}).get("clone", [])
            clone_https = ""
            for link in clone_links:
                if link.get("name") == "http":
                    clone_https = link.get("href", "")
                    break
            # Prefer the credential-embedded URL over the raw API link
            clone_url = self._clone_url(slug)
            repos.append(
                RepoInfo(
                    slug=slug,
                    name=name,
                    clone_url_https=clone_url,
                    default_branch="main",  # resolved lazily if needed
                    description=item.get("description", ""),
                    vcs_type="bitbucket-server",
                    workspace=self._project_key,
                )
            )
        return repos

    def get_repo_info(self, repo_slug: str) -> RepoInfo:
        """Return metadata for a single repository."""
        url = self._api(f"/projects/{self._project_key}/repos/{repo_slug}")
        data = self._get(url)
        # Attempt to determine the default branch
        default_branch = self._get_default_branch(repo_slug)
        return RepoInfo(
            slug=repo_slug,
            name=data.get("name", repo_slug),
            clone_url_https=self._clone_url(repo_slug),
            default_branch=default_branch,
            description=data.get("description", ""),
            vcs_type="bitbucket-server",
            workspace=self._project_key,
        )

    def _get_default_branch(self, repo_slug: str) -> str:
        """Retrieve the default branch for *repo_slug*, falling back to ``main``."""
        try:
            url = self._api(
                f"/projects/{self._project_key}/repos/{repo_slug}/branches/default"
            )
            data = self._get(url)
            return data.get("displayId", data.get("id", "main").replace("refs/heads/", ""))
        except Exception:
            return "main"

    def get_file_content(self, repo_slug: str, path: str, ref: str = "HEAD") -> str:
        """
        Retrieve the raw text content of a file.

        Bitbucket Server returns file lines as a paginated JSON response from
        ``/browse/{path}?at={ref}``.  We reassemble all pages into a single
        string.
        """
        url = self._api(
            f"/projects/{self._project_key}/repos/{repo_slug}/browse/{path}"
        )
        params: dict[str, Any] = {"at": ref, "limit": 500}
        lines: list[str] = []
        start = 0
        while True:
            params["start"] = start
            resp = self._session.get(url, params=params, timeout=30)
            if resp.status_code == 404:
                raise FileNotFoundError(
                    f"Path {path!r} not found in {repo_slug!r} at ref {ref!r}"
                )
            resp.raise_for_status()
            data = resp.json()
            # The "lines" field contains [{"text": "..."}, ...]
            for line_obj in data.get("lines", []):
                lines.append(line_obj.get("text", ""))
            if data.get("isLastPage", True):
                break
            start = data.get("nextPageStart", start + 500)
        return "\n".join(lines)

    def get_file_tree(
        self, repo_slug: str, path: str = "", ref: str = "HEAD"
    ) -> list[FileTreeEntry]:
        """
        List files and directories under *path* in the repository.

        Bitbucket Server's ``/files`` endpoint returns a flat list of all file
        paths under a given directory.  We use this for efficiency and infer
        ``"dir"`` entries from path prefixes.
        """
        url = self._api(
            f"/projects/{self._project_key}/repos/{repo_slug}/files/{path}".rstrip("/")
        )
        params: dict[str, Any] = {"at": ref}
        entries: list[FileTreeEntry] = []
        seen_dirs: set[str] = set()

        for file_path in self._paginate(url, params):
            # Each item is a file path string (relative to the requested dir)
            full_path = f"{path}/{file_path}".lstrip("/") if path else file_path
            entries.append(FileTreeEntry(path=full_path, type="file", size=0))
            # Synthesise directory entries for each intermediate component
            parts = full_path.split("/")
            for depth in range(1, len(parts)):
                dir_path = "/".join(parts[:depth])
                if dir_path not in seen_dirs:
                    seen_dirs.add(dir_path)
                    entries.append(FileTreeEntry(path=dir_path, type="dir", size=0))

        return entries

    def get_pr(self, repo_slug: str, pr_id: int | str) -> PRInfo:
        """Fetch a single pull request by ID."""
        url = self._api(
            f"/projects/{self._project_key}/repos/{repo_slug}/pull-requests/{pr_id}"
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
        state_map = {
            "open": "OPEN",
            "merged": "MERGED",
            "declined": "DECLINED",
            "all": "ALL",
        }
        bb_state = state_map.get(state.lower(), "OPEN")
        url = self._api(
            f"/projects/{self._project_key}/repos/{repo_slug}/pull-requests"
        )
        params: dict[str, Any] = {}
        if bb_state != "ALL":
            params["state"] = bb_state

        result: list[PRInfo] = []
        for item in self._paginate(url, params):
            result.append(self._parse_pr(item))
        return result

    def get_pr_diff(self, repo_slug: str, pr_id: int | str) -> str:
        """
        Retrieve the unified diff for a pull request.

        Bitbucket Server returns a structured diff JSON; we reconstruct a
        human-readable unified diff from the hunks.
        """
        url = self._api(
            f"/projects/{self._project_key}/repos/{repo_slug}"
            f"/pull-requests/{pr_id}/diff"
        )
        resp = self._session.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return self._render_diff(data)

    def _render_diff(self, data: dict) -> str:
        """
        Convert a Bitbucket Server diff JSON payload into a unified diff string.
        """
        lines: list[str] = []
        for diff in data.get("diffs", []):
            src = diff.get("source") or {}
            dst = diff.get("destination") or {}
            src_path = src.get("toString", "/dev/null")
            dst_path = dst.get("toString", "/dev/null")
            lines.append(f"--- a/{src_path}")
            lines.append(f"+++ b/{dst_path}")
            for hunk in diff.get("hunks", []):
                src_line = hunk.get("sourceLine", 0)
                src_span = hunk.get("sourceSpan", 0)
                dst_line = hunk.get("destinationLine", 0)
                dst_span = hunk.get("destinationSpan", 0)
                lines.append(
                    f"@@ -{src_line},{src_span} +{dst_line},{dst_span} @@"
                )
                for seg in hunk.get("segments", []):
                    seg_type = seg.get("type", "CONTEXT")
                    prefix = {"ADDED": "+", "REMOVED": "-", "CONTEXT": " "}.get(
                        seg_type, " "
                    )
                    for line_obj in seg.get("lines", []):
                        lines.append(f"{prefix}{line_obj.get('line', '')}")
        return "\n".join(lines)

    def post_review_comment(
        self, repo_slug: str, pr_id: int | str, body: str
    ) -> None:
        """Post a top-level comment on a pull request."""
        url = self._api(
            f"/projects/{self._project_key}/repos/{repo_slug}"
            f"/pull-requests/{pr_id}/comments"
        )
        resp = self._session.post(url, json={"text": body}, timeout=30)
        resp.raise_for_status()
        logger.info(
            "Posted review comment on PR #%s in %s/%s",
            pr_id,
            self._project_key,
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
        url = self._api(
            f"/projects/{self._project_key}/repos/{repo_slug}/pull-requests"
        )
        payload = {
            "title": title,
            "description": body,
            "state": "OPEN",
            "open": True,
            "closed": False,
            "fromRef": {
                "id": f"refs/heads/{head}",
                "repository": {
                    "slug": repo_slug,
                    "project": {"key": self._project_key},
                },
            },
            "toRef": {
                "id": f"refs/heads/{base}",
                "repository": {
                    "slug": repo_slug,
                    "project": {"key": self._project_key},
                },
            },
            "locked": False,
        }
        resp = self._session.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        logger.info(
            "Created PR #%s in %s/%s: %s",
            data.get("id"),
            self._project_key,
            repo_slug,
            title,
        )
        return self._parse_pr(data)
