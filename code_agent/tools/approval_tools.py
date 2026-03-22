"""Human-in-the-loop approval gates using ADK's LongRunningFunctionTool.

These tools act as mandatory pause points before irreversible operations.
When the agent calls one of these tools:

  1. The tool function runs immediately and returns a "pending" dict
     describing what is about to happen.
  2. ADK wraps this in an event with `long_running_tool_ids` set, which
     signals the runner to pause and set the A2A task to INPUT_REQUIRED.
  3. The human reviews the pending action details in the A2A task metadata
     and sends a tasks/resume request with `{approved: true/false, comment: "..."}`.
  4. ADK resumes the agent with the FunctionResponse containing the human's
     decision, and the agent proceeds (or aborts) based on it.

Gates currently defined
───────────────────────
  request_pr_approval   — must be called before create_pull_request
  request_push_approval — must be called before git_push

Both gate tools are also used directly as `LongRunningFunctionTool` instances
(`request_pr_approval_tool`, `request_push_approval_tool`) that are registered
with the agents that can trigger the irreversible operations.
"""

from __future__ import annotations

from typing import Any

from google.adk.tools import LongRunningFunctionTool


# ---------------------------------------------------------------------------
# PR creation approval gate
# ---------------------------------------------------------------------------

def request_pr_approval(
    vcs_type: str,
    repo: str,
    title: str,
    body: str,
    head_branch: str,
    base_branch: str,
) -> dict[str, Any]:
    """Request human approval before creating a pull request.

    Call this tool BEFORE create_pull_request.  It pauses execution and
    surfaces the PR details to the human operator for review.

    The human will receive:
      - The PR title, body, and branch diff (head → base)
      - An approve/reject prompt

    When the human approves, the agent will receive:
      {"approved": true, "comment": "<optional comment>"}

    When the human rejects, the agent will receive:
      {"approved": false, "comment": "<reason>"}

    The agent MUST check the response and only call create_pull_request
    if approved is True.  If approved is False, the agent must inform the
    user and stop without creating the PR.

    Args:
        vcs_type:    VCS provider — "github", "bitbucket-cloud", or
                     "bitbucket-server".
        repo:        Repository slug or full name (e.g. "myorg/myrepo").
        title:       Proposed PR title.
        body:        Proposed PR description (markdown).
        head_branch: Source branch (the branch with your changes).
        base_branch: Target branch to merge into (e.g. "main").

    Returns:
        A dict with status "pending_approval" and the PR details that will
        be shown to the human for review.
    """
    return {
        "status": "pending_approval",
        "action": "create_pull_request",
        "vcs_type": vcs_type,
        "repo": repo,
        "title": title,
        "body": body,
        "head_branch": head_branch,
        "base_branch": base_branch,
        "message": (
            f"Human approval required: create PR '{title}' "
            f"({head_branch} → {base_branch}) in {repo}"
        ),
    }


request_pr_approval_tool = LongRunningFunctionTool(func=request_pr_approval)


# ---------------------------------------------------------------------------
# Git push approval gate
# ---------------------------------------------------------------------------

def request_push_approval(
    repo_path: str,
    branch: str,
    remote: str = "origin",
    commit_summary: str = "",
) -> dict[str, Any]:
    """Request human approval before executing git push.

    Call this tool BEFORE git_push.  It pauses execution and surfaces
    the push details (branch, remote, and recent commits) to the human
    operator for review.

    When the human approves, the agent will receive:
      {"approved": true, "comment": "<optional comment>"}

    When the human rejects, the agent will receive:
      {"approved": false, "comment": "<reason>"}

    The agent MUST check the response and only call git_push if
    approved is True.  If rejected, the agent must inform the user.

    Args:
        repo_path:      Path to the local git repository root.
        branch:         Branch to be pushed (e.g. "feat/add-auth").
        remote:         Git remote name (default "origin").
        commit_summary: Short summary of commits about to be pushed
                        (e.g. output of git log --oneline -5).

    Returns:
        A dict with status "pending_approval" and push details.
    """
    return {
        "status": "pending_approval",
        "action": "git_push",
        "repo_path": repo_path,
        "branch": branch,
        "remote": remote,
        "commit_summary": commit_summary,
        "message": (
            f"Human approval required: push branch '{branch}' "
            f"to {remote} from {repo_path}"
        ),
    }


request_push_approval_tool = LongRunningFunctionTool(func=request_push_approval)
