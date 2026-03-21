"""ADK tool functions for the code agent.

This package exposes every tool function from all tool modules so they can be
imported and registered with an LlmAgent in a single import.

Usage::

    from code_agent.tools import (
        read_file, write_file, run_command,
        git_status, semantic_search, list_repositories,
        # ... etc.
    )
"""

# ---------------------------------------------------------------------------
# File tools
# ---------------------------------------------------------------------------
from code_agent.tools.file_tools import (
    read_file,
    write_file,
    list_directory,
    find_files,
    delete_file,
    create_directory,
    get_file_info,
    search_in_files,
)

# ---------------------------------------------------------------------------
# Shell tools
# ---------------------------------------------------------------------------
from code_agent.tools.shell_tools import (
    run_command,
    run_script,
)

# ---------------------------------------------------------------------------
# Git tools
# ---------------------------------------------------------------------------
from code_agent.tools.git_tools import (
    git_status,
    git_diff,
    git_log,
    git_show,
    git_blame,
    git_branch,
    git_checkout,
    git_commit,
    git_clone,
    git_create_branch,
    git_push,
)

# ---------------------------------------------------------------------------
# Code analysis tools
# ---------------------------------------------------------------------------
from code_agent.tools.code_tools import (
    extract_symbols,
    get_symbol,
    check_syntax,
    get_imports,
    count_lines,
    get_file_outline,
    grep_code,
    find_symbol,
)

# ---------------------------------------------------------------------------
# Search tools
# ---------------------------------------------------------------------------
from code_agent.tools.search_tools import (
    semantic_search,
    lexical_search,
    hybrid_search,
    find_symbol_references,
    index_local_repository,
)

# ---------------------------------------------------------------------------
# VCS tools
# ---------------------------------------------------------------------------
from code_agent.tools.vcs_tools import (
    list_repositories,
    clone_repository,
    get_pull_request,
    list_pull_requests,
    post_pr_review,
    create_pull_request,
    get_file_from_remote,
    get_repo_file_tree,
)

# ---------------------------------------------------------------------------
# Security tools
# ---------------------------------------------------------------------------
from code_agent.tools.security_tools import (
    run_security_scan,
    scan_dependencies,
    detect_secrets,
    check_license_compliance,
)

# ---------------------------------------------------------------------------
# Jira / Confluence tools
# ---------------------------------------------------------------------------
from code_agent.tools.jira_tools import (
    get_jira_issue,
    update_jira_issue,
    create_jira_issue,
    search_jira_issues,
    get_confluence_page,
    update_confluence_page,
)

__all__ = [
    # file tools
    "read_file",
    "write_file",
    "list_directory",
    "find_files",
    "delete_file",
    "create_directory",
    "get_file_info",
    "search_in_files",
    # shell tools
    "run_command",
    "run_script",
    # git tools
    "git_status",
    "git_diff",
    "git_log",
    "git_show",
    "git_blame",
    "git_branch",
    "git_checkout",
    "git_commit",
    "git_clone",
    "git_create_branch",
    "git_push",
    # code analysis tools
    "extract_symbols",
    "get_symbol",
    "check_syntax",
    "get_imports",
    "count_lines",
    "get_file_outline",
    "grep_code",
    "find_symbol",
    # search tools
    "semantic_search",
    "lexical_search",
    "hybrid_search",
    "find_symbol_references",
    "index_local_repository",
    # VCS tools
    "list_repositories",
    "clone_repository",
    "get_pull_request",
    "list_pull_requests",
    "post_pr_review",
    "create_pull_request",
    "get_file_from_remote",
    "get_repo_file_tree",
    # security tools
    "run_security_scan",
    "scan_dependencies",
    "detect_secrets",
    "check_license_compliance",
    # Jira / Confluence tools
    "get_jira_issue",
    "update_jira_issue",
    "create_jira_issue",
    "search_jira_issues",
    "get_confluence_page",
    "update_confluence_page",
]
