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
            "reviews PRs, debugs issues, runs security scans, writes documentation, "
            "indexes repos for semantic search, and manages Jira tickets. "
            "Works with GitHub and Bitbucket. "
            "Operates like a senior engineer: reads before writing, reasons before acting."
        ),
        version="0.2.0",
        url=url,
        capabilities=AgentCapabilities(
            streaming=True,
            push_notifications=False,
            state_transition_history=True,
        ),
        # Advertise async submit support so orchestrators know they can fire-
        # and-forget via tasks/submit and poll/subscribe separately.
        # (stored as extra metadata on the card)

        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[
            AgentSkill(
                id="analyze_codebase",
                name="Analyze Codebase",
                description=(
                    "Understand an existing project's architecture, tech stack, patterns, and code quality. "
                    "Supports semantic search after indexing."
                ),
                tags=["analyze", "understand", "overview", "architecture", "semantic search"],
                examples=[
                    "Analyze the project at /path/to/project and summarize its architecture",
                    "What tech stack does /home/user/myapp use?",
                    "Explain the data flow in this codebase",
                ],
            ),
            AgentSkill(
                id="review_pr",
                name="Review Pull Request",
                description=(
                    "Perform a thorough code review of a GitHub or Bitbucket PR. "
                    "Checks correctness, security (OWASP), performance, and maintainability. "
                    "Can post the review directly to the PR."
                ),
                tags=["review", "PR", "code review", "diff", "security", "GitHub", "Bitbucket"],
                examples=[
                    "Review PR #42 in github.com/myorg/myrepo",
                    "Review the diff between main and feature/add-auth branches",
                    "Do a security review of the authentication module",
                ],
            ),
            AgentSkill(
                id="implement_feature",
                name="Implement Feature",
                description=(
                    "Implement a feature end-to-end: read the Jira ticket, understand the codebase, "
                    "write the code, commit, create a PR, and update the ticket."
                ),
                tags=["implement", "feature", "code", "Jira", "PR"],
                examples=[
                    "Implement the feature described in PROJ-123",
                    "Add JWT authentication as described in ticket PROJ-456",
                ],
            ),
            AgentSkill(
                id="fix_bug",
                name="Fix Bug",
                description=(
                    "Diagnose and fix a bug using root cause analysis. "
                    "Traces stack traces, uses git blame, forms hypotheses, and applies minimal fixes."
                ),
                tags=["debug", "bug", "fix", "error", "stack trace", "root cause"],
                examples=[
                    "This KeyError occurs when processing large batches: <stacktrace>",
                    "The API returns 500 for authenticated users only — investigate",
                    "Find why the background job silently fails every 3rd run",
                ],
            ),
            AgentSkill(
                id="refactor_code",
                name="Refactor Code",
                description=(
                    "Refactor existing code to improve structure, readability, or performance "
                    "while preserving behavior."
                ),
                tags=["refactor", "clean up", "restructure", "improve"],
                examples=[
                    "Refactor the database module to use connection pooling",
                    "Extract the payment logic into a dedicated PaymentService class",
                    "Simplify the authentication middleware without changing its behavior",
                ],
            ),
            AgentSkill(
                id="design_architecture",
                name="Design Architecture",
                description=(
                    "Design system architecture for new projects or propose improvements to existing ones. "
                    "Produces directory trees, ADRs, data models, and API surfaces."
                ),
                tags=["architecture", "design", "ADR", "system design", "scaffold"],
                examples=[
                    "Design the architecture for a multi-tenant SaaS API",
                    "Analyze the current architecture of /path/to/project and suggest improvements",
                    "Create an ADR for switching from REST to GraphQL",
                ],
            ),
            AgentSkill(
                id="explain_code",
                name="Explain Code",
                description=(
                    "Explain what a piece of code does, how a system works, or trace a call chain "
                    "through a codebase with file:line references."
                ),
                tags=["explain", "understand", "trace", "how does", "what does"],
                examples=[
                    "Explain how authentication works in this codebase",
                    "Trace the request lifecycle from HTTP handler to database",
                    "What does the UserRepository.find_by_email method do?",
                ],
            ),
            AgentSkill(
                id="write_tests",
                name="Write Tests",
                description=(
                    "Write unit, integration, or end-to-end tests for existing code. "
                    "Reads the code first to understand behavior, then writes tests that cover edge cases."
                ),
                tags=["test", "unit test", "pytest", "jest", "coverage"],
                examples=[
                    "Write unit tests for the PaymentService class",
                    "Add integration tests for the /auth/login endpoint",
                    "Write a test that would have caught the bug in PR #42",
                ],
            ),
            AgentSkill(
                id="search_code",
                name="Search Code",
                description=(
                    "Search a codebase using semantic (meaning-based) or lexical (exact text) search. "
                    "Requires the repo to be indexed for semantic search."
                ),
                tags=["search", "find", "grep", "semantic search", "symbol"],
                examples=[
                    "Find all places where user authentication is checked",
                    "Search for usages of the deprecated send_email function",
                    "Where is the database connection pool initialized?",
                ],
            ),
            AgentSkill(
                id="scaffold_project",
                name="Scaffold Project",
                description=(
                    "Create a new project from scratch with idiomatic directory structure, "
                    "dependency manifest, configuration, entrypoint, and README."
                ),
                tags=["scaffold", "create", "new project", "boilerplate"],
                examples=[
                    "Create a FastAPI CRUD app with PostgreSQL in /tmp/myapp",
                    "Scaffold a React + TypeScript frontend in /tmp/frontend",
                    "Build a CLI tool in Go for parsing log files",
                ],
            ),
            AgentSkill(
                id="security_review",
                name="Security Review",
                description=(
                    "Perform a security review covering OWASP Top 10, secret exposure, "
                    "dependency CVEs, and license compliance. Findings are CVSS-ranked."
                ),
                tags=["security", "OWASP", "CVE", "secrets", "vulnerability", "audit"],
                examples=[
                    "Run a security review of the authentication module",
                    "Scan dependencies for known CVEs",
                    "Check the codebase for hardcoded secrets",
                ],
            ),
            AgentSkill(
                id="update_jira",
                name="Update Jira",
                description=(
                    "Read, create, update, or transition Jira issues. "
                    "Add comments, change status, and link PRs to tickets."
                ),
                tags=["Jira", "ticket", "issue", "project management"],
                examples=[
                    "Mark PROJ-123 as In Review and add a comment with the PR URL",
                    "Create a bug ticket in the PROJ project for the login failure",
                    "List all open tickets assigned to me in PROJ",
                ],
            ),
            AgentSkill(
                id="index_repository",
                name="Index Repository",
                description=(
                    "Index a local or remote repository for semantic search. "
                    "After indexing, semantic_search and hybrid_search become available for that repo."
                ),
                tags=["index", "semantic search", "vector", "embeddings"],
                examples=[
                    "Index the repository at /path/to/myrepo for semantic search",
                    "Clone and index github.com/myorg/myrepo",
                ],
            ),
        ],
    )
