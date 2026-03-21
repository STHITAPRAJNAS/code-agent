"""VCS Agent — GitHub and Bitbucket API workflow specialist."""
import os
from google.adk.agents import LlmAgent
from code_agent.tools import (
    list_repositories, clone_repository, get_pull_request, list_pull_requests,
    post_pr_review, create_pull_request, get_file_from_remote, get_repo_file_tree,
    index_local_repository, semantic_search, hybrid_search,
    git_clone, git_status, git_diff, git_log,
)

_INSTRUCTION = """You are a Staff Software Engineer who works fluently with GitHub and Bitbucket APIs to manage
repositories, pull requests, and code reviews without leaving the terminal.

## Capabilities
- List and explore repositories in GitHub orgs and Bitbucket workspaces
- Clone repos for local analysis
- Fetch PR details, diffs, and reviews
- Post structured code reviews as PR comments
- Create PRs with well-crafted descriptions
- Fetch files from remote repos without cloning
- Index repos for semantic search

## Workflow for PR Creation
1. Understand the changes (git_diff, git_log)
2. Write a PR title following: `type: concise description`
3. Write a body with: Summary, Changes, Testing, Breaking Changes (if any)
4. Create the PR and return the URL

## Workflow for PR Review
1. get_pull_request to fetch diff + metadata
2. Read context files using get_file_from_remote
3. Use semantic_search if the repo is indexed
4. Write structured review (see pr_reviewer format)
5. post_pr_review with the full review
"""

vcs_agent = LlmAgent(
    model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
    name="vcs_agent",
    description="GitHub and Bitbucket API: list repos, fetch PRs, post reviews, create PRs, remote file access",
    instruction=_INSTRUCTION,
    tools=[
        list_repositories, clone_repository, get_pull_request, list_pull_requests,
        post_pr_review, create_pull_request, get_file_from_remote, get_repo_file_tree,
        index_local_repository, semantic_search, hybrid_search,
        git_clone, git_status, git_diff, git_log,
    ],
)
