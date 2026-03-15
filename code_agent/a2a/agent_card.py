"""Agent card definition for the Code Agent."""

import os
from code_agent.a2a.models import AgentCard, AgentCapabilities, AgentSkill


def build_agent_card(base_url: str | None = None) -> AgentCard:
    """Build the A2A agent card with current server URL."""
    url = base_url or f"http://{os.getenv('APP_HOST', '0.0.0.0')}:{os.getenv('APP_PORT', '8000')}"

    return AgentCard(
        name="Code Agent",
        description=(
            "A Staff Software Engineer AI agent powered by Google Gemini. "
            "Builds projects from scratch, analyzes codebases, writes and modifies code, "
            "reviews PRs, debugs issues, and runs shell commands. "
            "Operates like a senior engineer: reads before writing, reasons before acting."
        ),
        version="0.1.0",
        url=url,
        capabilities=AgentCapabilities(
            streaming=True,
            push_notifications=False,
            state_transition_history=True,
        ),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[
            AgentSkill(
                id="build_project",
                name="Build Project",
                description="Scaffold a complete new project from scratch with directory structure, dependencies, config, and README.",
                tags=["scaffold", "create", "new project"],
                examples=[
                    "Create a FastAPI CRUD app with PostgreSQL in /tmp/myapp",
                    "Scaffold a React + TypeScript frontend in /tmp/frontend",
                    "Build a CLI tool in Go for parsing log files",
                ],
            ),
            AgentSkill(
                id="analyze_code",
                name="Analyze Codebase",
                description="Understand an existing project's architecture, tech stack, patterns, and code quality.",
                tags=["analyze", "understand", "overview", "architecture"],
                examples=[
                    "Analyze the project at /path/to/project and summarize its architecture",
                    "What tech stack does /home/user/myapp use?",
                    "Explain the data flow in this codebase",
                ],
            ),
            AgentSkill(
                id="modify_code",
                name="Modify Code",
                description="Add features, refactor existing code, fix bugs, or make targeted modifications to source files.",
                tags=["modify", "edit", "add feature", "refactor"],
                examples=[
                    "Add JWT authentication to the FastAPI app at /tmp/myapp",
                    "Refactor the database module to use connection pooling",
                    "Add input validation to all API endpoints",
                ],
            ),
            AgentSkill(
                id="review_pr",
                name="Review Pull Request",
                description=(
                    "Perform a thorough code review of a git diff or PR. "
                    "Checks for security issues, bugs, performance problems, and code quality."
                ),
                tags=["review", "PR", "code review", "diff", "security"],
                examples=[
                    "Review the diff between main and feature/add-auth branches",
                    "Review this PR diff: <paste diff here>",
                    "Do a security review of the authentication module",
                ],
            ),
            AgentSkill(
                id="debug",
                name="Debug Issue",
                description="Analyze errors, stack traces, and unexpected behavior to find the root cause and propose a minimal fix.",
                tags=["debug", "error", "bug", "fix", "stack trace"],
                examples=[
                    "This KeyError occurs when processing large batches: <stacktrace>",
                    "The API returns 500 for authenticated users only — investigate",
                    "Find why the background job silently fails every 3rd run",
                ],
            ),
            AgentSkill(
                id="git_ops",
                name="Git Operations",
                description="Inspect and manage git repositories: status, diffs, history, commits, branches.",
                tags=["git", "commit", "branch", "diff", "history"],
                examples=[
                    "Show me the last 10 commits in /path/to/repo",
                    "What changed between main and the feature branch?",
                    "Commit all staged changes with a meaningful message",
                ],
            ),
            AgentSkill(
                id="run_tests",
                name="Run Tests",
                description="Execute test suites, interpret results, and identify failing tests.",
                tags=["test", "pytest", "jest", "unit test", "CI"],
                examples=[
                    "Run the test suite for the project at /tmp/myapp",
                    "Run only the failing tests and show me the output",
                    "Check if all tests pass after my recent changes",
                ],
            ),
        ],
    )
