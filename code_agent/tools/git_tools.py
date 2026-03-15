"""Git tools — status, diff, log, commit, branch, clone operations."""

import subprocess
from pathlib import Path


def _git(args: list[str], repo_path: str, input_text: str = "") -> dict:
    """Run a git subcommand in the given repo path."""
    p = Path(repo_path).expanduser().resolve()
    try:
        result = subprocess.run(
            ["git", "-C", str(p)] + args,
            capture_output=True,
            text=True,
            timeout=60,
            input=input_text or None,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
            "error": result.stderr.strip() if result.returncode != 0 else "",
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": "", "exit_code": -1, "error": "git command timed out"}
    except FileNotFoundError:
        return {"success": False, "stdout": "", "stderr": "", "exit_code": -1, "error": "git not found in PATH"}
    except Exception as exc:
        return {"success": False, "stdout": "", "stderr": "", "exit_code": -1, "error": str(exc)}


def git_status(repo_path: str) -> dict:
    """Show the working tree status of a git repository.

    Args:
        repo_path: Path to the git repository root.

    Returns:
        dict with stdout (status text), success, error.
    """
    return _git(["status", "--short", "--branch"], repo_path)


def git_diff(repo_path: str, ref: str = "", staged: bool = False) -> dict:
    """Show changes between commits, index, and working tree.

    Args:
        repo_path: Path to the git repository root.
        ref: Commit ref, branch, or range to diff against (e.g. "HEAD~1", "main..feature").
        staged: Show staged (index) changes if True; unstaged if False (default).

    Returns:
        dict with stdout (diff text), success, error.
    """
    args = ["diff"]
    if staged:
        args.append("--staged")
    if ref:
        args.append(ref)
    return _git(args, repo_path)


def git_log(repo_path: str, n: int = 20, branch: str = "") -> dict:
    """Show the commit history.

    Args:
        repo_path: Path to the git repository root.
        n: Number of commits to show (default 20).
        branch: Branch or ref to show log for (default: current HEAD).

    Returns:
        dict with stdout (log text), success, error.
    """
    args = ["log", f"--max-count={n}", "--oneline", "--decorate", "--graph"]
    if branch:
        args.append(branch)
    return _git(args, repo_path)


def git_show(repo_path: str, ref: str) -> dict:
    """Show the contents of a commit (diff + metadata).

    Args:
        repo_path: Path to the git repository root.
        ref: Commit hash, tag, or branch name to show.

    Returns:
        dict with stdout (commit details + diff), success, error.
    """
    return _git(["show", ref], repo_path)


def git_branch(repo_path: str) -> dict:
    """List all local and remote branches.

    Args:
        repo_path: Path to the git repository root.

    Returns:
        dict with stdout (branch list), success, error.
    """
    return _git(["branch", "-a", "-vv"], repo_path)


def git_create_branch(repo_path: str, name: str, from_ref: str = "HEAD") -> dict:
    """Create and switch to a new branch.

    Args:
        repo_path: Path to the git repository root.
        name: New branch name (e.g. "feature/add-auth").
        from_ref: Starting ref (default "HEAD").

    Returns:
        dict with stdout, success, error.
    """
    return _git(["checkout", "-b", name, from_ref], repo_path)


def git_commit(repo_path: str, message: str, files: list[str] | None = None) -> dict:
    """Stage files and create a commit.

    Args:
        repo_path: Path to the git repository root.
        message: Commit message (use Conventional Commits: feat:, fix:, etc.).
        files: List of file paths to stage. Stages all tracked changes if empty.

    Returns:
        dict with stdout, success, error.
    """
    # Stage files
    stage_args = ["add"] + (files if files else ["-A"])
    stage = _git(stage_args, repo_path)
    if not stage["success"]:
        return stage

    return _git(["commit", "-m", message], repo_path)


def git_clone(url: str, dest: str) -> dict:
    """Clone a remote git repository.

    Args:
        url: Remote repository URL (https or ssh).
        dest: Local destination path.

    Returns:
        dict with stdout, success, error.
    """
    try:
        result = subprocess.run(
            ["git", "clone", url, dest],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
            "error": result.stderr.strip() if result.returncode != 0 else "",
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": "", "exit_code": -1, "error": "git clone timed out (120s)"}
    except Exception as exc:
        return {"success": False, "stdout": "", "stderr": "", "exit_code": -1, "error": str(exc)}
