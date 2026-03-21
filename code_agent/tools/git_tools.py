"""Git tools — status, diff, log, commit, branch, push, and clone operations.

ADK tool functions that return plain strings consumed directly by LLM agents.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _git(args: list[str], repo_path: str, timeout: int = 60) -> tuple[bool, str]:
    """Run a git subcommand in *repo_path*.

    Returns (success, output_text).  stderr is appended to stdout on failure
    so the agent always sees the full picture.
    """
    p = Path(repo_path).expanduser().resolve()
    try:
        result = subprocess.run(
            ["git", "-C", str(p)] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return True, result.stdout
        # On failure, combine stdout + stderr for useful context
        combined = (result.stdout + result.stderr).strip()
        return False, combined
    except subprocess.TimeoutExpired:
        return False, "Error: git command timed out"
    except FileNotFoundError:
        return False, "Error: git not found in PATH"
    except Exception as exc:
        return False, f"Error: {exc}"


# ---------------------------------------------------------------------------
# Read-only operations
# ---------------------------------------------------------------------------

def git_status(repo_path: str = ".") -> str:
    """Show the working tree status of a git repository.

    Reports which files are modified, staged, untracked, or have merge
    conflicts.  Use before committing to understand what has changed.

    Args:
        repo_path: Path to the git repository root (default: current directory).

    Returns:
        Git status output as a string, or an error message.
    """
    ok, out = _git(["status", "--short", "--branch"], repo_path)
    if not ok:
        return f"Error running git status: {out}"
    return out or "(working tree clean)"


def git_diff(
    repo_path: str = ".",
    ref1: str = "",
    ref2: str = "",
    file_path: str = "",
) -> str:
    """Show changes between commits, working tree, or index.

    Examples:
      git_diff()                         — unstaged working tree changes
      git_diff(ref1='HEAD')              — staged + unstaged vs HEAD
      git_diff(ref1='main', ref2='HEAD') — diff between two refs
      git_diff(ref1='HEAD~1', file_path='src/main.py') — diff one file

    Args:
        repo_path: Path to the git repository root (default: current directory).
        ref1: First ref to diff.  When empty, shows unstaged changes.
        ref2: Second ref to diff.  When empty, diffs ref1 against working tree.
        file_path: Restrict diff to this specific file path (relative to repo).

    Returns:
        Unified diff output as a string, or an error message.
    """
    args = ["diff"]
    if ref1 and ref2:
        args.extend([ref1, ref2])
    elif ref1:
        args.append(ref1)
    if file_path:
        args.extend(["--", file_path])
    ok, out = _git(args, repo_path)
    if not ok:
        return f"Error running git diff: {out}"
    return out or "(no differences)"


def git_log(
    repo_path: str = ".",
    limit: int = 20,
    author: str = "",
) -> str:
    """Show the commit history for a repository.

    Returns a concise log with hashes, author, date, and subject line.
    Use to understand what has changed recently before reviewing a PR.

    Args:
        repo_path: Path to the git repository root (default: current directory).
        limit: Maximum number of commits to show (default 20).
        author: Filter commits by this author name or email (substring match).

    Returns:
        Formatted commit log as a string, or an error message.
    """
    args = [
        "log",
        f"--max-count={max(1, limit)}",
        "--pretty=format:%h  %an  %ad  %s",
        "--date=short",
    ]
    if author:
        args.append(f"--author={author}")
    ok, out = _git(args, repo_path)
    if not ok:
        return f"Error running git log: {out}"
    return out or "(no commits)"


def git_show(ref: str, repo_path: str = ".") -> str:
    """Show the contents and diff of a specific commit.

    Returns the commit metadata (author, date, message) and the full unified
    diff of all changes introduced by that commit.

    Args:
        ref: Commit hash, tag, or branch name to inspect.
        repo_path: Path to the git repository root (default: current directory).

    Returns:
        Commit details and diff as a string, or an error message.
    """
    ok, out = _git(["show", ref], repo_path)
    if not ok:
        return f"Error running git show {ref}: {out}"
    return out


def git_blame(file_path: str, repo_path: str = ".") -> str:
    """Show who last modified each line of a file and in which commit.

    Use to understand the history of a specific piece of code or to identify
    which commit introduced a bug.

    Args:
        file_path: Repository-relative path to the file.
        repo_path: Path to the git repository root (default: current directory).

    Returns:
        Blame output (commit, author, date, line) as a string, or an error.
    """
    ok, out = _git(["blame", "--date=short", file_path], repo_path)
    if not ok:
        return f"Error running git blame on {file_path}: {out}"
    return out


def git_branch(repo_path: str = ".") -> str:
    """List all local and remote branches with their tracking information.

    Shows which branch is currently checked out (marked with '*') and the
    upstream tracking branch for each local branch.

    Args:
        repo_path: Path to the git repository root (default: current directory).

    Returns:
        Branch listing as a string, or an error message.
    """
    ok, out = _git(["branch", "-a", "-vv"], repo_path)
    if not ok:
        return f"Error running git branch: {out}"
    return out or "(no branches)"


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------

def git_checkout(
    branch: str,
    repo_path: str = ".",
    create: bool = False,
) -> str:
    """Switch to an existing branch or create and switch to a new one.

    Args:
        branch: Name of the branch to check out or create.
        repo_path: Path to the git repository root (default: current directory).
        create: When True, create the branch if it does not already exist
            (equivalent to git checkout -b).

    Returns:
        Confirmation message, or an error message if checkout failed.
    """
    args = ["checkout"]
    if create:
        args.append("-b")
    args.append(branch)
    ok, out = _git(args, repo_path)
    if not ok:
        return f"Error checking out branch '{branch}': {out}"
    return out or f"Switched to branch '{branch}'"


def git_commit(
    message: str,
    files: list[str] | None = None,
    repo_path: str = ".",
) -> str:
    """Stage files and create a commit with the given message.

    Stages specified files (or all tracked changes if files is None) and
    creates a new commit.  Use Conventional Commits format for message
    (e.g. 'feat: add JWT authentication').

    Args:
        message: Commit message (use Conventional Commits: feat:, fix:, etc.).
        files: List of file paths to stage (relative to repo root).  When
            None or empty, stages all tracked modifications with 'git add -A'.
        repo_path: Path to the git repository root (default: current directory).

    Returns:
        The commit output showing the new commit hash, or an error message.
    """
    # Stage files
    stage_args = ["add"] + (files if files else ["-A"])
    ok, out = _git(stage_args, repo_path)
    if not ok:
        return f"Error staging files: {out}"

    ok, out = _git(["commit", "-m", message], repo_path)
    if not ok:
        return f"Error creating commit: {out}"
    return out


def git_clone(
    url: str,
    destination: str,
    shallow: bool = True,
) -> str:
    """Clone a remote git repository to a local path.

    Performs a shallow clone (depth 1) by default for speed.  Use
    shallow=False when you need the full commit history.

    Args:
        url: Remote repository URL (HTTPS or SSH).
        destination: Local destination path for the clone.
        shallow: When True (default), perform a shallow depth-1 clone.

    Returns:
        Confirmation of successful clone with the destination path, or an
        error message if the clone failed.
    """
    cmd = ["git", "clone"]
    if shallow:
        cmd.extend(["--depth", "1"])
    cmd.extend([url, destination])
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            return f"Cloned {url} -> {destination}"
        combined = (result.stdout + result.stderr).strip()
        return f"Error cloning {url}: {combined}"
    except subprocess.TimeoutExpired:
        return f"Error: git clone timed out after 300s for {url}"
    except FileNotFoundError:
        return "Error: git not found in PATH"
    except Exception as exc:
        return f"Error: {exc}"


def git_create_branch(
    branch_name: str,
    repo_path: str = ".",
    from_ref: str = "",
) -> str:
    """Create a new branch and switch to it.

    Creates the branch from from_ref (or HEAD if not specified) and
    immediately checks it out.

    Args:
        branch_name: Name for the new branch (e.g. 'feature/add-auth').
        repo_path: Path to the git repository root (default: current directory).
        from_ref: Git ref (branch, tag, or commit SHA) to branch from.
            Defaults to HEAD when empty.

    Returns:
        Confirmation message, or an error message if branch creation failed.
    """
    args = ["checkout", "-b", branch_name]
    if from_ref:
        args.append(from_ref)
    ok, out = _git(args, repo_path)
    if not ok:
        return f"Error creating branch '{branch_name}': {out}"
    return out or f"Created and switched to branch '{branch_name}'"


def git_push(
    branch: str = "",
    remote: str = "origin",
    repo_path: str = ".",
) -> str:
    """Push a local branch to a remote repository.

    Pushes the specified branch (or the current branch if empty) to the given
    remote.  Sets the upstream tracking reference automatically.

    Args:
        branch: Local branch name to push.  When empty, pushes the currently
            checked-out branch.
        remote: Name of the remote to push to (default 'origin').
        repo_path: Path to the git repository root (default: current directory).

    Returns:
        Push output confirming what was pushed, or an error message.
    """
    args = ["push", "--set-upstream", remote]
    if branch:
        args.append(branch)
    ok, out = _git(args, repo_path, timeout=120)
    if not ok:
        return f"Error pushing to {remote}: {out}"
    return out or f"Pushed to {remote}"
