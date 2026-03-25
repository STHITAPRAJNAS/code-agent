"""VCS tools — unified GitHub and Bitbucket operations.

ADK tool functions for listing repos, cloning, PR management, and remote file
access.  All functions return plain strings consumed directly by LLM agents.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_provider(vcs_type: str, workspace_or_org: str = ""):
    """Build a VCSProvider from application settings.

    Raises ValueError with a descriptive message if required credentials are
    missing or if vcs_type is unknown.
    """
    from code_agent.config import get_settings
    from code_agent.vcs.factory import create_vcs_provider

    cfg = get_settings()
    vt = vcs_type.lower().strip()

    if vt == "github":
        if not cfg.GITHUB_TOKEN:
            raise ValueError(
                "GITHUB_TOKEN is not configured.  Set it in .env or as an "
                "environment variable."
            )
        return create_vcs_provider(
            "github",
            token=cfg.GITHUB_TOKEN,
            org=workspace_or_org or None,
        )

    if vt == "bitbucket-cloud":
        if not cfg.BITBUCKET_USERNAME or not cfg.BITBUCKET_APP_PASSWORD:
            raise ValueError(
                "BITBUCKET_USERNAME and BITBUCKET_APP_PASSWORD must both be "
                "configured for Bitbucket Cloud."
            )
        workspace = workspace_or_org or cfg.BITBUCKET_WORKSPACE or ""
        return create_vcs_provider(
            "bitbucket-cloud",
            username=cfg.BITBUCKET_USERNAME,
            app_password=cfg.BITBUCKET_APP_PASSWORD,
            workspace=workspace,
        )

    if vt == "bitbucket-server":
        if not cfg.BITBUCKET_SERVER_URL or not cfg.BITBUCKET_SERVER_TOKEN:
            raise ValueError(
                "BITBUCKET_SERVER_URL and BITBUCKET_SERVER_TOKEN must both be "
                "configured for Bitbucket Server."
            )
        return create_vcs_provider(
            "bitbucket-server",
            base_url=cfg.BITBUCKET_SERVER_URL,
            token=cfg.BITBUCKET_SERVER_TOKEN,
            project_key=workspace_or_org or cfg.BITBUCKET_SERVER_PROJECT or "",
        )

    raise ValueError(
        f"Unknown vcs_type '{vcs_type}'.  Must be one of: "
        "github, bitbucket-cloud, bitbucket-server"
    )


# ---------------------------------------------------------------------------
# Repository discovery
# ---------------------------------------------------------------------------

def list_repositories(
    vcs_type: str,
    workspace_or_org: str = "",
) -> str:
    """List all repositories in a GitHub org or Bitbucket workspace.

    Returns repo names, descriptions, and default branches.
    Use to discover available repositories before cloning or analyzing.

    Args:
        vcs_type: VCS provider — one of 'github', 'bitbucket-cloud', or
            'bitbucket-server'.
        workspace_or_org: GitHub organisation name, Bitbucket Cloud workspace
            slug, or Bitbucket Server project key.  Falls back to the value
            configured in settings when empty.

    Returns:
        Formatted list of repositories with slug, default branch, and
        description, or an error message.
    """
    try:
        provider = _make_provider(vcs_type, workspace_or_org)
        repos = provider.list_repos()
    except Exception as exc:
        return f"Error listing repositories: {exc}"

    if not repos:
        return f"No repositories found for {vcs_type} / {workspace_or_org or '(default workspace)'}"

    lines = [
        f"Repositories in {vcs_type} / {workspace_or_org or '(default workspace)'}"
        f" ({len(repos)} total):\n"
    ]
    for r in repos:
        desc = f" — {r.description}" if r.description else ""
        lines.append(f"  {r.slug:<40} [{r.default_branch}]{desc}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cloning
# ---------------------------------------------------------------------------

def clone_repository(
    repo_url: str,
    repo_id: str = "",
    branch: str = "",
    shallow: bool = True,
) -> str:
    """Clone a git repository to a local workspace.

    Supports GitHub (https://) and Bitbucket (https://) repositories.
    shallow=True does a fast depth-1 clone.
    Returns the local path where the repo was cloned.

    Uses WorkspaceManager to store clones under the configured WORKSPACE_DIR.
    If the repo has been cloned before (by repo_id), the existing clone is
    updated via git pull instead of re-cloning.

    Args:
        repo_url: HTTPS clone URL of the repository.
        repo_id: Stable identifier used as the local directory name
            (e.g. 'my-org-my-repo').  Derived from the URL when empty.
        branch: Branch to check out.  Uses the remote default when empty.
        shallow: Perform a shallow depth-1 clone when True (default).

    Returns:
        Absolute path to the local clone, or an error message.
    """
    try:
        from code_agent.workspace.manager import WorkspaceManager

        manager = WorkspaceManager()
    except Exception as exc:
        return f"Error initialising workspace manager: {exc}"

    # Derive repo_id from URL if not provided
    if not repo_id:
        # e.g. https://github.com/org/repo.git -> org-repo
        slug = repo_url.rstrip("/").rstrip(".git").split("/")[-2:]
        repo_id = "-".join(slug)

    try:
        if shallow:
            # get_or_clone doesn't support shallow; use clone() for fresh clones
            # but get_or_clone for updates (idempotent)
            local_path = manager.clone(
                clone_url=repo_url,
                shallow=True,
                branch=branch if branch else None,
            )
        else:
            local_path = manager.get_or_clone(
                clone_url=repo_url,
                repo_id=repo_id,
                branch=branch if branch else None,
            )
    except Exception as exc:
        return f"Error cloning {repo_url}: {exc}"

    return f"Repository cloned to: {local_path}"


# ---------------------------------------------------------------------------
# Pull request operations
# ---------------------------------------------------------------------------

def get_pull_request(
    vcs_type: str,
    repo_slug: str,
    pr_id: str,
    workspace_or_org: str = "",
) -> str:
    """Get details of a pull request including title, description, diff, and changed files.

    Returns the full PR context needed for code review.

    Args:
        vcs_type: VCS provider — 'github', 'bitbucket-cloud', or
            'bitbucket-server'.
        repo_slug: Repository slug (short name, e.g. 'my-repo').
        pr_id: Pull request ID or number as a string.
        workspace_or_org: Org/workspace/project.  Falls back to settings.

    Returns:
        PR details and unified diff, or an error message.
    """
    try:
        provider = _make_provider(vcs_type, workspace_or_org)
    except Exception as exc:
        return f"Error: {exc}"

    try:
        pr = provider.get_pr(repo_slug, pr_id)
    except Exception as exc:
        return f"Error fetching PR #{pr_id} from {repo_slug}: {exc}"

    try:
        diff = provider.get_pr_diff(repo_slug, pr_id)
    except Exception as exc:
        diff = f"(diff unavailable: {exc})"

    lines = [
        f"Pull Request #{pr.id} — {repo_slug}",
        f"  Title:   {pr.title}",
        f"  Author:  {pr.author}",
        f"  State:   {pr.state}",
        f"  Source:  {pr.source_branch} → {pr.target_branch}",
        "",
        "Description:",
        pr.description or "(no description)",
        "",
        "Diff:",
        diff or "(no diff)",
    ]
    return "\n".join(lines)


def list_pull_requests(
    vcs_type: str,
    repo_slug: str,
    state: str = "open",
    workspace_or_org: str = "",
) -> str:
    """List pull requests for a repository.

    state: 'open', 'closed', 'merged', 'all'
    Returns PR titles, authors, branches, and IDs.

    Args:
        vcs_type: VCS provider — 'github', 'bitbucket-cloud', or
            'bitbucket-server'.
        repo_slug: Repository slug.
        state: Filter by state — 'open', 'closed', 'merged', or 'all'.
        workspace_or_org: Org/workspace/project.  Falls back to settings.

    Returns:
        Formatted list of pull requests, or an error message.
    """
    try:
        provider = _make_provider(vcs_type, workspace_or_org)
        prs = provider.list_prs(repo_slug, state=state)
    except Exception as exc:
        return f"Error listing PRs for {repo_slug}: {exc}"

    if not prs:
        return f"No {state} pull requests found for {repo_slug}"

    lines = [f"Pull requests for {repo_slug} (state={state}, {len(prs)} found):\n"]
    for pr in prs:
        lines.append(
            f"  #{pr.id:<6} [{pr.state:<8}] {pr.source_branch} → {pr.target_branch}"
            f"  by {pr.author}"
        )
        lines.append(f"           {pr.title}")
        lines.append("")
    return "\n".join(lines)


def post_pr_review(
    vcs_type: str,
    repo_slug: str,
    pr_id: str,
    review_body: str,
    workspace_or_org: str = "",
) -> str:
    """Post a code review comment on a pull request.

    Posts the review as a comment on the PR.
    Include line-specific feedback formatted as markdown.
    Returns confirmation of posted review.

    Args:
        vcs_type: VCS provider — 'github', 'bitbucket-cloud', or
            'bitbucket-server'.
        repo_slug: Repository slug.
        pr_id: Pull request ID or number.
        review_body: Markdown-formatted review comment body.
        workspace_or_org: Org/workspace/project.  Falls back to settings.

    Returns:
        Confirmation that the review was posted, or an error message.
    """
    if not review_body.strip():
        return "Error: review_body is empty — nothing to post"

    try:
        provider = _make_provider(vcs_type, workspace_or_org)
        provider.post_review_comment(repo_slug, pr_id, review_body)
    except Exception as exc:
        return f"Error posting review on PR #{pr_id}: {exc}"

    return f"Review posted on PR #{pr_id} in {repo_slug}"


def create_pull_request(
    vcs_type: str,
    repo_slug: str,
    title: str,
    body: str,
    head_branch: str,
    base_branch: str = "main",
    workspace_or_org: str = "",
) -> str:
    """Create a new pull request on GitHub or Bitbucket.

    head_branch: the branch with your changes
    base_branch: the branch to merge into (usually 'main' or 'master')
    Returns the PR URL and ID.

    Args:
        vcs_type: VCS provider — 'github', 'bitbucket-cloud', or
            'bitbucket-server'.
        repo_slug: Repository slug.
        title: Pull request title.
        body: Pull request description in Markdown format.
        head_branch: Branch containing your proposed changes (source).
        base_branch: Target branch to merge into (default 'main').
        workspace_or_org: Org/workspace/project.  Falls back to settings.

    Returns:
        PR ID and URL on success, or an error message.
    """
    if not title.strip():
        return "Error: PR title is required"
    if not head_branch.strip():
        return "Error: head_branch is required"

    try:
        provider = _make_provider(vcs_type, workspace_or_org)
        pr = provider.create_pr(
            repo_slug=repo_slug,
            title=title,
            body=body,
            head=head_branch,
            base=base_branch,
        )
    except Exception as exc:
        return f"Error creating PR in {repo_slug}: {exc}"

    lines = [
        f"Pull request created in {repo_slug}:",
        f"  ID:     #{pr.id}",
        f"  Title:  {pr.title}",
        f"  State:  {pr.state}",
        f"  Branch: {pr.source_branch} → {pr.target_branch}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Remote file access
# ---------------------------------------------------------------------------

def get_file_from_remote(
    vcs_type: str,
    repo_slug: str,
    file_path: str,
    ref: str = "HEAD",
    workspace_or_org: str = "",
) -> str:
    """Fetch a file's content directly from GitHub or Bitbucket without cloning.

    Use for quickly inspecting a single file from a remote repo.
    Returns the file content as text.

    Args:
        vcs_type: VCS provider — 'github', 'bitbucket-cloud', or
            'bitbucket-server'.
        repo_slug: Repository slug.
        file_path: Repository-relative path to the file (e.g. 'src/main.py').
        ref: Git ref — branch name, tag, or commit SHA (default 'HEAD').
        workspace_or_org: Org/workspace/project.  Falls back to settings.

    Returns:
        The file content as a string, or an error message if not found.
    """
    try:
        provider = _make_provider(vcs_type, workspace_or_org)
        content = provider.get_file_content(repo_slug, file_path, ref=ref)
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found in {repo_slug} at ref '{ref}'"
    except Exception as exc:
        return f"Error fetching {file_path} from {repo_slug}: {exc}"

    header = f"File: {repo_slug}/{file_path}  (ref: {ref})\n" + "-" * 60 + "\n"
    return header + content


def get_repo_file_tree(
    vcs_type: str,
    repo_slug: str,
    path: str = "",
    workspace_or_org: str = "",
) -> str:
    """Get the directory/file tree of a remote repository without cloning.

    Returns a tree view of files and directories.
    Use to understand repo structure before cloning.

    Args:
        vcs_type: VCS provider — 'github', 'bitbucket-cloud', or
            'bitbucket-server'.
        repo_slug: Repository slug.
        path: Sub-directory path to list (empty means the repo root).
        workspace_or_org: Org/workspace/project.  Falls back to settings.

    Returns:
        Tree listing of files and directories, or an error message.
    """
    try:
        provider = _make_provider(vcs_type, workspace_or_org)
        entries = provider.get_file_tree(repo_slug, path=path)
    except Exception as exc:
        return f"Error fetching file tree from {repo_slug}: {exc}"

    if not entries:
        display_path = path or "(root)"
        return f"No entries found at '{display_path}' in {repo_slug}"

    # Sort dirs first, then files; alphabetical within each group
    dirs = sorted([e for e in entries if e.type == "dir"], key=lambda e: e.path)
    files = sorted([e for e in entries if e.type != "dir"], key=lambda e: e.path)
    sorted_entries = dirs + files

    display_path = path or "(root)"
    lines = [f"File tree of {repo_slug}/{display_path} ({len(entries)} entries):\n"]
    for e in sorted_entries:
        icon = "DIR  " if e.type == "dir" else "FILE "
        size = f"{e.size:>10} B" if e.type != "dir" and e.size else " " * 12
        lines.append(f"  {icon} {size}  {e.path}")
    return "\n".join(lines)
