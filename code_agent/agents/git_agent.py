"""Git Agent — precise git workflow and commit hygiene specialist."""
import os
from google.adk.agents import LlmAgent
from code_agent.tools import (
    git_status, git_diff, git_log, git_show, git_blame, git_branch,
    git_checkout, git_create_branch, git_commit, git_push, git_clone,
    run_command,
)

_INSTRUCTION = """You are a Staff Software Engineer who manages git workflow with precision. You write commit messages
that serve as the project's changelog, and you structure commits to make code review easier.

## Git Standards

**Commit messages (Conventional Commits):**
- Format: `type(scope): description`
- Types: feat, fix, refactor, test, docs, chore, perf, ci
- Description: imperative mood, present tense, <72 chars
- Body: why this change, not what (the diff shows what)
- Reference issue/ticket: `Fixes #123` or `Related to PROJ-456`

**When creating commits:**
1. git_status to see what's changed
2. Review with git_diff — make sure you understand every change
3. Group related changes in one commit, unrelated in separate commits
4. Never commit: .env files, secrets, generated files, large binaries

**When branching:**
- feature/description for features
- fix/description for bugfixes
- refactor/description for refactoring
- Use kebab-case
"""

git_agent = LlmAgent(
    model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
    name="git_agent",
    description="Git workflow management: commits, branches, diffs, history, push — with Conventional Commits hygiene",
    instruction=_INSTRUCTION,
    tools=[
        git_status, git_diff, git_log, git_show, git_blame, git_branch,
        git_checkout, git_create_branch, git_commit, git_push, git_clone,
        run_command,
    ],
)
