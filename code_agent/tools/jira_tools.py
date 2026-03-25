"""Jira and Confluence tools — issue management and documentation access.

ADK tool functions for reading and updating Atlassian Jira issues and
Confluence pages.  All functions return plain strings consumed by LLM agents.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _jira_client():
    """Build and return a JIRA client from application settings.

    Raises:
        RuntimeError: If required credentials are not configured.
    """
    from jira import JIRA  # type: ignore[import]
    from code_agent.config import get_settings

    cfg = get_settings()

    missing: list[str] = []
    if not cfg.JIRA_SERVER_URL:
        missing.append("JIRA_SERVER_URL")
    if not cfg.JIRA_USERNAME:
        missing.append("JIRA_USERNAME")
    if not cfg.JIRA_API_TOKEN:
        missing.append("JIRA_API_TOKEN")

    if missing:
        raise RuntimeError(
            "Jira credentials not configured.  Set the following environment "
            "variables (or add them to .env):\n  " + ", ".join(missing)
        )

    return JIRA(
        server=cfg.JIRA_SERVER_URL,
        basic_auth=(cfg.JIRA_USERNAME, cfg.JIRA_API_TOKEN),
    )


def _confluence_client():
    """Build and return an Atlassian Confluence client from application settings.

    Raises:
        RuntimeError: If required credentials are not configured.
    """
    from atlassian import Confluence  # type: ignore[import]
    from code_agent.config import get_settings

    cfg = get_settings()

    # Fall back to Jira credentials if Confluence-specific ones are absent
    server_url = cfg.CONFLUENCE_SERVER_URL or cfg.JIRA_SERVER_URL
    username = cfg.CONFLUENCE_USERNAME or cfg.JIRA_USERNAME
    api_token = cfg.CONFLUENCE_API_TOKEN or cfg.JIRA_API_TOKEN

    missing: list[str] = []
    if not server_url:
        missing.append("CONFLUENCE_SERVER_URL (or JIRA_SERVER_URL)")
    if not username:
        missing.append("CONFLUENCE_USERNAME (or JIRA_USERNAME)")
    if not api_token:
        missing.append("CONFLUENCE_API_TOKEN (or JIRA_API_TOKEN)")

    if missing:
        raise RuntimeError(
            "Confluence credentials not configured.  Set:\n  "
            + ", ".join(missing)
        )

    return Confluence(
        url=server_url,
        username=username,
        password=api_token,
        cloud=True,
    )


def _format_issue(issue) -> str:
    """Format a JIRA Issue object into a readable string."""
    fields = issue.fields
    summary = fields.summary or ""
    status = fields.status.name if fields.status else "Unknown"
    assignee = fields.assignee.displayName if fields.assignee else "Unassigned"
    reporter = fields.reporter.displayName if fields.reporter else "Unknown"
    priority = fields.priority.name if fields.priority else "None"
    issue_type = fields.issuetype.name if fields.issuetype else "Unknown"
    labels = ", ".join(fields.labels) if fields.labels else "None"
    description = fields.description or "(no description)"

    # Linked issues
    links: list[str] = []
    for link in (fields.issuelinks or []):
        if hasattr(link, "outwardIssue"):
            links.append(f"  {link.type.outward}: {link.outwardIssue.key} — {link.outwardIssue.fields.summary}")
        elif hasattr(link, "inwardIssue"):
            links.append(f"  {link.type.inward}: {link.inwardIssue.key} — {link.inwardIssue.fields.summary}")

    lines = [
        f"Issue: {issue.key}",
        f"  Type:        {issue_type}",
        f"  Summary:     {summary}",
        f"  Status:      {status}",
        f"  Priority:    {priority}",
        f"  Assignee:    {assignee}",
        f"  Reporter:    {reporter}",
        f"  Labels:      {labels}",
        "",
        "Description:",
        description,
    ]
    if links:
        lines.append("\nLinked Issues:")
        lines.extend(links)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Jira issue operations
# ---------------------------------------------------------------------------

def get_jira_issue(issue_key: str) -> str:
    """Fetch a Jira issue by key (e.g. 'PROJ-123').

    Returns: title, description, status, assignee, labels, acceptance criteria, linked issues.
    Use to understand requirements before implementing a feature.

    Args:
        issue_key: Jira issue key in the format PROJECT-NUMBER (e.g. 'PROJ-123').

    Returns:
        Formatted issue details including description, status, assignee, and
        linked issues, or an error message if the issue is not found or
        credentials are not configured.
    """
    if not issue_key.strip():
        return "Error: issue_key is required"

    try:
        jira = _jira_client()
    except RuntimeError as exc:
        return f"Error: {exc}"
    except ImportError:
        return (
            "Error: jira library not installed.  "
            "Install with: pip install jira"
        )

    try:
        issue = jira.issue(
            issue_key.strip().upper(),
            fields="summary,description,status,assignee,reporter,priority,"
                   "issuetype,labels,issuelinks",
        )
    except Exception as exc:
        err_str = str(exc)
        if "404" in err_str or "does not exist" in err_str.lower():
            return f"Issue '{issue_key}' not found"
        return f"Error fetching issue {issue_key}: {exc}"

    return _format_issue(issue)


def update_jira_issue(
    issue_key: str,
    status: str = "",
    comment: str = "",
    assignee: str = "",
) -> str:
    """Update a Jira issue status or add a comment.

    status: transition name like 'In Progress', 'In Review', 'Done'
    comment: text to add as a comment
    Returns confirmation of the update.

    At least one of status, comment, or assignee must be provided.

    Args:
        issue_key: Jira issue key (e.g. 'PROJ-123').
        status: Transition name to move the issue to (e.g. 'In Progress',
            'Done', 'In Review').  Empty means no status change.
        comment: Text comment to add to the issue.  Empty means no comment.
        assignee: Username or account ID to assign the issue to.  Empty means
            no assignment change.

    Returns:
        Confirmation of each action taken, or an error message.
    """
    if not issue_key.strip():
        return "Error: issue_key is required"
    if not any([status.strip(), comment.strip(), assignee.strip()]):
        return "Error: At least one of status, comment, or assignee must be provided"

    try:
        jira = _jira_client()
    except RuntimeError as exc:
        return f"Error: {exc}"
    except ImportError:
        return "Error: jira library not installed.  Install with: pip install jira"

    key = issue_key.strip().upper()
    results: list[str] = []

    # --- Status transition ---
    if status.strip():
        try:
            transitions = jira.transitions(key)
            matched = [
                t for t in transitions
                if t["name"].lower() == status.strip().lower()
            ]
            if not matched:
                available = [t["name"] for t in transitions]
                results.append(
                    f"Status transition '{status}' not found.  "
                    f"Available: {', '.join(available)}"
                )
            else:
                jira.transition_issue(key, matched[0]["id"])
                results.append(f"Status changed to '{matched[0]['name']}'")
        except Exception as exc:
            results.append(f"Error changing status: {exc}")

    # --- Add comment ---
    if comment.strip():
        try:
            jira.add_comment(key, comment.strip())
            results.append("Comment added")
        except Exception as exc:
            results.append(f"Error adding comment: {exc}")

    # --- Assign ---
    if assignee.strip():
        try:
            jira.assign_issue(key, assignee.strip())
            results.append(f"Assigned to '{assignee.strip()}'")
        except Exception as exc:
            results.append(f"Error assigning issue: {exc}")

    prefix = f"Updates for {key}:\n"
    return prefix + "\n".join(f"  - {r}" for r in results)


def create_jira_issue(
    project_key: str,
    summary: str,
    description: str,
    issue_type: str = "Task",
    labels: str = "",
) -> str:
    """Create a new Jira issue.

    issue_type: 'Bug', 'Story', 'Task', 'Epic'
    labels: comma-separated list of labels
    Returns the new issue key and URL.

    Args:
        project_key: Jira project key (e.g. 'PROJ', 'BACKEND').
        summary: Issue title / summary line.
        description: Full issue description (supports Jira wiki markup).
        issue_type: Issue type — 'Bug', 'Story', 'Task', or 'Epic'.
        labels: Comma-separated list of label strings to attach to the issue.

    Returns:
        The new issue key and browse URL, or an error message.
    """
    if not project_key.strip():
        return "Error: project_key is required"
    if not summary.strip():
        return "Error: summary is required"

    try:
        jira = _jira_client()
    except RuntimeError as exc:
        return f"Error: {exc}"
    except ImportError:
        return "Error: jira library not installed.  Install with: pip install jira"

    fields: dict = {
        "project": {"key": project_key.strip().upper()},
        "summary": summary.strip(),
        "description": description,
        "issuetype": {"name": issue_type or "Task"},
    }
    if labels.strip():
        fields["labels"] = [l.strip() for l in labels.split(",") if l.strip()]

    try:
        new_issue = jira.create_issue(fields=fields)
    except Exception as exc:
        return f"Error creating issue in {project_key}: {exc}"

    # Build browse URL
    try:
        from code_agent.config import get_settings
        cfg = get_settings()
        base = (cfg.JIRA_SERVER_URL or "").rstrip("/")
        url = f"{base}/browse/{new_issue.key}" if base else new_issue.key
    except Exception:
        url = new_issue.key

    return (
        f"Issue created: {new_issue.key}\n"
        f"  Summary: {summary.strip()}\n"
        f"  Type:    {issue_type}\n"
        f"  URL:     {url}"
    )


def search_jira_issues(jql: str, max_results: int = 20) -> str:
    """Search Jira issues using JQL (Jira Query Language).

    Examples:
      jql='project = PROJ AND status = "In Progress"'
      jql='assignee = currentUser() AND sprint in openSprints()'
    Returns matching issues with key, summary, status, assignee.

    Args:
        jql: Jira Query Language expression.  Must be a valid JQL string.
        max_results: Maximum number of issues to return (default 20, max 100).

    Returns:
        Formatted list of matching issues with key, summary, status, and
        assignee, or an error message.
    """
    if not jql.strip():
        return "Error: jql query is required"

    try:
        jira = _jira_client()
    except RuntimeError as exc:
        return f"Error: {exc}"
    except ImportError:
        return "Error: jira library not installed.  Install with: pip install jira"

    limit = min(max(1, max_results), 100)
    try:
        issues = jira.search_issues(
            jql.strip(),
            maxResults=limit,
            fields="summary,status,assignee,priority,issuetype",
        )
    except Exception as exc:
        return f"Error executing JQL '{jql}': {exc}"

    if not issues:
        return f"No issues found for JQL: {jql}"

    lines = [f"JQL: {jql}", f"Results: {len(issues)}\n"]
    for issue in issues:
        f = issue.fields
        status = f.status.name if f.status else "?"
        assignee = f.assignee.displayName if f.assignee else "Unassigned"
        itype = f.issuetype.name if f.issuetype else "?"
        lines.append(
            f"  {issue.key:<12} [{status:<15}] [{itype:<10}] {f.summary or ''}"
        )
        lines.append(f"                   Assignee: {assignee}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Confluence operations
# ---------------------------------------------------------------------------

def get_confluence_page(
    page_id: str = "",
    space_key: str = "",
    title: str = "",
) -> str:
    """Fetch a Confluence page by ID or by space + title.

    Returns the page content as plain text.
    Use to read architecture docs, runbooks, or specifications.

    Provide either page_id alone, or both space_key and title together.

    Args:
        page_id: Numeric Confluence page ID (e.g. '123456').
        space_key: Confluence space key (e.g. 'ARCH', 'TEAM').  Required when
            using title-based lookup.
        title: Exact page title.  Required when using title-based lookup.

    Returns:
        Page title and content as plain text, or an error message.
    """
    if not page_id and not (space_key and title):
        return (
            "Error: provide either page_id or both space_key and title"
        )

    try:
        confluence = _confluence_client()
    except RuntimeError as exc:
        return f"Error: {exc}"
    except ImportError:
        return (
            "Error: atlassian-python-api not installed.  "
            "Install with: pip install atlassian-python-api"
        )

    try:
        if page_id:
            page = confluence.get_page_by_id(
                page_id=str(page_id).strip(),
                expand="body.storage",
            )
        else:
            page = confluence.get_page_by_title(
                space=space_key.strip().upper(),
                title=title.strip(),
                expand="body.storage",
            )
    except Exception as exc:
        ident = page_id or f"{space_key}/{title}"
        return f"Error fetching Confluence page '{ident}': {exc}"

    if not page:
        ident = page_id or f"{space_key}/{title}"
        return f"Confluence page not found: {ident}"

    page_title = page.get("title", "Unknown")
    body_storage = page.get("body", {}).get("storage", {}).get("value", "")
    page_url = page.get("_links", {}).get("base", "") + page.get("_links", {}).get("webui", "")

    # Strip HTML tags for plain-text representation
    plain_text = _strip_html(body_storage)

    lines = [
        f"Confluence Page: {page_title}",
        f"URL: {page_url}",
        "-" * 60,
        plain_text,
    ]
    return "\n".join(lines)


def update_confluence_page(
    page_id: str,
    title: str,
    content: str,
    version_comment: str = "Updated by Code Agent",
) -> str:
    """Update a Confluence page with new content.

    content: the new page body in Confluence storage format or plain text.
    Returns confirmation with page URL.

    The page version is automatically incremented.  The content should be
    provided as Confluence Storage Format (XHTML-like) for rich formatting,
    or as plain text which will be wrapped in a paragraph tag.

    Args:
        page_id: Numeric Confluence page ID of the page to update.
        title: New page title (can be the same as the current title).
        content: New page body.  Provide as Confluence storage format
            (XHTML) or plain text.
        version_comment: Optional comment describing the change
            (default: 'Updated by Code Agent').

    Returns:
        Confirmation with the page URL, or an error message.
    """
    if not page_id.strip():
        return "Error: page_id is required"
    if not title.strip():
        return "Error: title is required"
    if not content.strip():
        return "Error: content is required"

    try:
        confluence = _confluence_client()
    except RuntimeError as exc:
        return f"Error: {exc}"
    except ImportError:
        return (
            "Error: atlassian-python-api not installed.  "
            "Install with: pip install atlassian-python-api"
        )

    # Get current version
    try:
        existing = confluence.get_page_by_id(page_id.strip(), expand="version")
        if not existing:
            return f"Error: Confluence page '{page_id}' not found"
        current_version = existing.get("version", {}).get("number", 1)
    except Exception as exc:
        return f"Error fetching current version for page {page_id}: {exc}"

    # Wrap plain text if not already in storage format
    body = content if content.strip().startswith("<") else f"<p>{content}</p>"

    try:
        result = confluence.update_page(
            page_id=page_id.strip(),
            title=title.strip(),
            body=body,
            version_comment=version_comment,
        )
    except Exception as exc:
        return f"Error updating Confluence page {page_id}: {exc}"

    page_url = (
        result.get("_links", {}).get("base", "")
        + result.get("_links", {}).get("webui", "")
    ) if result else ""

    return (
        f"Confluence page updated: {title}\n"
        f"  ID:      {page_id}\n"
        f"  Version: {current_version} → {current_version + 1}\n"
        f"  URL:     {page_url or '(URL unavailable)'}"
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _strip_html(html: str) -> str:
    """Very lightweight HTML tag stripper for Confluence storage format."""
    import re
    # Replace block tags with newlines
    html = re.sub(r"<(?:p|br|h[1-6]|li|tr|div)[^>]*>", "\n", html, flags=re.IGNORECASE)
    # Remove remaining tags
    html = re.sub(r"<[^>]+>", "", html)
    # Decode common entities
    html = (
        html.replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&nbsp;", " ")
            .replace("&quot;", '"')
            .replace("&#39;", "'")
    )
    # Collapse multiple blank lines
    html = re.sub(r"\n{3,}", "\n\n", html)
    return html.strip()
