"""Security analysis tools — SAST, dependency audit, secret detection, and license checks.

ADK tool functions that return plain strings consumed directly by LLM agents.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# SAST — static security analysis
# ---------------------------------------------------------------------------

def run_security_scan(
    path: str = ".",
    language: str = "python",
) -> str:
    """Run static security analysis (SAST) on code files.

    Python: uses bandit to detect security issues.
    All languages: uses semgrep if available.
    Returns categorized security findings with severity levels.

    Findings are grouped by severity (HIGH, MEDIUM, LOW) and include the
    file path, line number, issue description, and remediation advice.

    Args:
        path: Root directory or file to scan (default: current directory).
        language: Primary language of the codebase ('python', 'javascript',
            etc.).  Used to select the appropriate scanner.

    Returns:
        Security findings report as a formatted string, or an error message
        if no scanner is available.
    """
    root = Path(path).expanduser().resolve()
    if not root.exists():
        return f"Error: Path not found: {path}"

    lang = language.lower().strip()
    results: list[str] = []

    # ---- Bandit for Python ----
    if lang == "python":
        bandit_result = _run_bandit(root)
        if bandit_result is not None:
            return bandit_result
        results.append("bandit not installed (pip install bandit).")

    # ---- Semgrep for all languages ----
    semgrep_result = _run_semgrep(root, lang)
    if semgrep_result is not None:
        return semgrep_result
    results.append("semgrep not installed (pip install semgrep).")

    if results:
        note = "\n".join(f"  - {r}" for r in results)
        return (
            f"No security scanner available for {path}:\n{note}\n\n"
            "Install bandit (Python) or semgrep (all languages) and try again."
        )
    return "No findings — security scan completed with no issues detected."


def _run_bandit(root: Path) -> str | None:
    """Run bandit and return a formatted report, or None if bandit is absent."""
    try:
        result = subprocess.run(
            ["bandit", "-r", str(root), "-f", "json", "-q"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return "Error: bandit scan timed out after 120s"

    # bandit exits 1 when issues found — that's fine
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        raw = (result.stdout + result.stderr).strip()
        return f"bandit output (could not parse JSON):\n{raw}" if raw else None

    issues: list[dict] = data.get("results", [])
    errors: list[dict] = data.get("errors", [])
    metrics: dict = data.get("metrics", {})

    if not issues and not errors:
        return "bandit: No security issues found."

    # Group by severity
    by_severity: dict[str, list[dict]] = {"HIGH": [], "MEDIUM": [], "LOW": []}
    for issue in issues:
        sev = issue.get("issue_severity", "LOW").upper()
        by_severity.setdefault(sev, []).append(issue)

    lines = [f"bandit security scan: {root}"]
    total = len(issues)
    lines.append(
        f"  Total findings: {total}  "
        f"(HIGH={len(by_severity['HIGH'])}, "
        f"MEDIUM={len(by_severity['MEDIUM'])}, "
        f"LOW={len(by_severity['LOW'])})\n"
    )

    for sev in ("HIGH", "MEDIUM", "LOW"):
        sev_issues = by_severity.get(sev, [])
        if not sev_issues:
            continue
        lines.append(f"[{sev}] {len(sev_issues)} issue(s):")
        for issue in sev_issues:
            filename = issue.get("filename", "?")
            lineno = issue.get("line_number", "?")
            test_id = issue.get("test_id", "?")
            test_name = issue.get("test_name", "?")
            issue_text = issue.get("issue_text", "")
            more_info = issue.get("more_info", "")
            lines.append(f"  {filename}:{lineno}  [{test_id}] {test_name}")
            lines.append(f"    {issue_text}")
            if more_info:
                lines.append(f"    More info: {more_info}")
        lines.append("")

    if errors:
        lines.append(f"Scan errors ({len(errors)}):")
        for err in errors[:5]:
            lines.append(f"  - {err.get('filename', '?')}: {err.get('reason', '?')}")

    return "\n".join(lines)


def _run_semgrep(root: Path, lang: str) -> str | None:
    """Run semgrep and return a formatted report, or None if semgrep is absent."""
    _SEMGREP_CONFIGS: dict[str, str] = {
        "python": "p/python",
        "javascript": "p/javascript",
        "typescript": "p/typescript",
        "java": "p/java",
        "go": "p/go",
        "ruby": "p/ruby",
    }
    config = _SEMGREP_CONFIGS.get(lang, "p/default")

    try:
        result = subprocess.run(
            ["semgrep", "--config", config, "--json", "--quiet", str(root)],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return "Error: semgrep timed out after 180s"

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        raw = (result.stdout + result.stderr).strip()
        return f"semgrep output (could not parse JSON):\n{raw}" if raw else None

    findings: list[dict] = data.get("results", [])
    if not findings:
        return "semgrep: No security issues found."

    lines = [f"semgrep security scan ({config}): {root}",
             f"  Total findings: {len(findings)}\n"]

    for f in findings:
        check_id = f.get("check_id", "?")
        path_str = f.get("path", "?")
        start = f.get("start", {})
        lineno = start.get("line", "?")
        message = f.get("extra", {}).get("message", "")
        severity = f.get("extra", {}).get("severity", "").upper()
        lines.append(f"  [{severity}] {path_str}:{lineno}  {check_id}")
        lines.append(f"    {message}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dependency vulnerability scanning
# ---------------------------------------------------------------------------

def scan_dependencies(
    path: str = ".",
    package_manager: str = "auto",
) -> str:
    """Scan project dependencies for known CVEs and vulnerabilities.

    pip: uses pip-audit
    npm/yarn: uses npm audit
    auto: detects from requirements.txt, pyproject.toml, package.json
    Returns vulnerability report with CVE IDs and fix recommendations.

    Args:
        path: Root directory of the project (default: current directory).
        package_manager: One of 'auto', 'pip', 'npm', or 'yarn'.  When 'auto',
            the package manager is detected from manifest files present in path.

    Returns:
        Vulnerability report with CVE IDs and remediation advice, or an error
        message.
    """
    root = Path(path).expanduser().resolve()
    if not root.exists():
        return f"Error: Path not found: {path}"

    pm = package_manager.lower().strip()

    # Auto-detect
    if pm == "auto":
        if (root / "requirements.txt").exists() or (root / "pyproject.toml").exists():
            pm = "pip"
        elif (root / "yarn.lock").exists():
            pm = "yarn"
        elif (root / "package.json").exists():
            pm = "npm"
        else:
            return (
                "Could not detect package manager.  No requirements.txt, "
                "pyproject.toml, or package.json found in:\n  " + str(root)
            )

    if pm == "pip":
        return _run_pip_audit(root)
    if pm in ("npm", "yarn"):
        return _run_npm_audit(root, pm)

    return f"Error: Unsupported package manager '{package_manager}'.  Use: auto, pip, npm, yarn"


def _run_pip_audit(root: Path) -> str:
    """Run pip-audit and return a formatted report."""
    try:
        result = subprocess.run(
            ["pip-audit", "--format", "json", "--progress-spinner", "off"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(root),
        )
    except FileNotFoundError:
        return (
            "pip-audit not installed.  Install with: pip install pip-audit\n"
            "Then re-run the scan."
        )
    except subprocess.TimeoutExpired:
        return "Error: pip-audit timed out after 120s"

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        raw = (result.stdout + result.stderr).strip()
        return f"pip-audit output:\n{raw}"

    vulnerabilities = data.get("vulnerabilities", [])
    if not vulnerabilities:
        return "pip-audit: No vulnerabilities found."

    lines = [f"pip-audit: {len(vulnerabilities)} vulnerability/ies found\n"]
    for vuln in vulnerabilities:
        pkg = vuln.get("name", "?")
        version = vuln.get("version", "?")
        for v in vuln.get("vulns", []):
            vuln_id = v.get("id", "?")
            desc = v.get("description", "")
            fix = v.get("fix_versions", [])
            fix_str = f"  Fix: upgrade to {', '.join(fix)}" if fix else "  No fix available"
            lines.append(f"  {pkg}=={version}  [{vuln_id}]")
            if desc:
                lines.append(f"    {desc[:200]}")
            lines.append(fix_str)
            lines.append("")
    return "\n".join(lines)


def _run_npm_audit(root: Path, pm: str) -> str:
    """Run npm/yarn audit and return a formatted report."""
    cmd = ["yarn", "audit", "--json"] if pm == "yarn" else ["npm", "audit", "--json"]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(root),
        )
    except FileNotFoundError:
        return f"Error: '{pm}' not found in PATH.  Install Node.js and try again."
    except subprocess.TimeoutExpired:
        return f"Error: {pm} audit timed out after 120s"

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        raw = (result.stdout + result.stderr).strip()
        return f"{pm} audit output:\n{raw[:2000]}"

    # npm audit JSON structure
    metadata = data.get("metadata", {})
    vulns = data.get("vulnerabilities", {})
    total = metadata.get("vulnerabilities", {})
    if not vulns:
        return f"{pm} audit: No vulnerabilities found."

    total_count = sum(total.values()) if isinstance(total, dict) else len(vulns)
    lines = [f"{pm} audit: {total_count} vulnerability/ies found\n"]
    for pkg_name, info in list(vulns.items())[:50]:
        severity = info.get("severity", "?")
        via = info.get("via", [])
        advisories = [v for v in via if isinstance(v, dict)]
        lines.append(f"  {pkg_name}  [{severity.upper()}]")
        for adv in advisories[:2]:
            title = adv.get("title", "")
            url = adv.get("url", "")
            if title:
                lines.append(f"    {title}")
            if url:
                lines.append(f"    {url}")
        lines.append("")
    if len(vulns) > 50:
        lines.append(f"  ... and {len(vulns) - 50} more packages")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Secret detection
# ---------------------------------------------------------------------------

#: Patterns that may indicate hardcoded secrets.
#: Each entry is (label, pattern) where pattern matches the whole line.
_SECRET_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("API Key",        re.compile(r'(?i)(api[_\-]?key|apikey)\s*[=:]\s*["\']?([A-Za-z0-9\-_]{16,})["\']?')),
    ("Secret",         re.compile(r'(?i)(secret[_\-]?key|secret)\s*[=:]\s*["\']?([A-Za-z0-9\-_+/]{16,})["\']?')),
    ("Password",       re.compile(r'(?i)(password|passwd|pwd)\s*[=:]\s*["\']?([^\s"\']{8,})["\']?')),
    ("Token",          re.compile(r'(?i)(token|auth[_\-]?token|access[_\-]?token)\s*[=:]\s*["\']?([A-Za-z0-9\-_.]{16,})["\']?')),
    ("AWS Access Key", re.compile(r'(?i)(aws[_\-]?access[_\-]?key[_\-]?id)\s*[=:]\s*["\']?(AKIA[0-9A-Z]{16})["\']?')),
    ("AWS Secret",     re.compile(r'(?i)(aws[_\-]?secret[_\-]?access[_\-]?key)\s*[=:]\s*["\']?([A-Za-z0-9/+=]{40})["\']?')),
    ("Private Key",    re.compile(r'-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----')),
    ("GitHub Token",   re.compile(r'ghp_[A-Za-z0-9]{36}')),
    ("Generic Bearer", re.compile(r'(?i)bearer\s+([A-Za-z0-9\-_.+/]{20,})')),
    ("DB Password",    re.compile(r'(?i)(db[_\-]?pass(?:word)?|database[_\-]?password)\s*[=:]\s*["\']?([^\s"\']{6,})["\']?')),
]

_SKIP_DIRS_SECRETS = frozenset(
    [".git", "__pycache__", "node_modules", ".venv", "venv", "build", "dist"]
)
_SKIP_EXTENSIONS = frozenset([".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip",
                               ".tar", ".gz", ".woff", ".ttf", ".pyc", ".class"])

# Patterns that are clearly false positives (placeholder / example values)
_FP_PATTERNS = re.compile(
    r'(?i)(example|sample|placeholder|your[_\-]?key|your[_\-]?secret|changeme|'
    r'<[^>]+>|\$\{[^}]+\}|%[A-Z_]+%|xxx+|dummy)'
)


def detect_secrets(path: str = ".") -> str:
    """Scan for accidentally committed secrets, API keys, and credentials.

    Looks for patterns matching: API keys, tokens, passwords, private keys.
    Returns findings with file paths and line numbers (values redacted).

    Skips binary files, build artifacts, and common false-positive patterns
    (e.g. placeholder values like 'your_api_key_here').

    Args:
        path: Root directory to scan (default: current directory).

    Returns:
        Report of potential secrets with file locations (values redacted), or
        a clear message if no secrets are found.
    """
    root = Path(path).expanduser().resolve()
    if not root.exists():
        return f"Error: Path not found: {path}"

    findings: list[tuple[str, int, str, str]] = []  # (file, line, label, redacted)

    for filepath in root.rglob("*"):
        if not filepath.is_file():
            continue
        if filepath.suffix.lower() in _SKIP_EXTENSIONS:
            continue
        if any(part in _SKIP_DIRS_SECRETS for part in filepath.parts):
            continue
        # Skip .env* files content scanning — the agent should know they hold secrets
        # but we still want to flag actual inline secrets in source files

        try:
            for lineno, line in enumerate(
                filepath.read_text(encoding="utf-8", errors="replace").splitlines(),
                start=1,
            ):
                for label, pat in _SECRET_PATTERNS:
                    m = pat.search(line)
                    if not m:
                        continue
                    # Skip obvious false positives
                    if _FP_PATTERNS.search(line):
                        continue
                    # Redact the sensitive value
                    redacted = pat.sub(
                        lambda mo: mo.group(0).replace(
                            mo.group(mo.lastindex or 0), "[REDACTED]"
                        ) if mo.lastindex else "[REDACTED]",
                        line,
                    ).strip()
                    findings.append((str(filepath), lineno, label, redacted))
                    break  # only report first pattern match per line
        except OSError:
            continue

    if not findings:
        return f"No secrets detected in {root}"

    lines = [
        f"Potential secrets detected in {root} ({len(findings)} finding(s)):\n",
        "NOTE: Values have been redacted.  Review each finding manually.\n",
    ]
    for fpath, lineno, label, redacted in findings:
        lines.append(f"  {fpath}:{lineno}  [{label}]")
        lines.append(f"    {redacted}")
        lines.append("")
    lines.append(
        "Recommendation: remove secrets from source, use environment variables, "
        "and consider rotating any exposed credentials."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# License compliance
# ---------------------------------------------------------------------------

#: Licenses that may have significant usage restrictions
_RESTRICTIVE_LICENSES = frozenset([
    "GPL-2.0", "GPL-3.0", "AGPL-3.0", "LGPL-2.1", "LGPL-3.0",
    "EUPL-1.1", "EUPL-1.2", "CC-BY-SA-4.0", "CDDL-1.0",
])

#: Permissive licenses generally safe for commercial use
_PERMISSIVE_LICENSES = frozenset([
    "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC",
    "Unlicense", "0BSD", "CC0-1.0", "WTFPL",
])


def check_license_compliance(path: str = ".") -> str:
    """Check licenses of all dependencies for compliance.

    Returns a report of dependency licenses grouped by license type.
    Highlights potentially problematic licenses (GPL, AGPL, etc.).

    Supports Python (pip-licenses or requirements.txt + PyPI lookup) and
    Node.js (package.json + node_modules).

    Args:
        path: Root directory of the project (default: current directory).

    Returns:
        License report grouped by license type with a compliance summary, or
        an error message.
    """
    root = Path(path).expanduser().resolve()
    if not root.exists():
        return f"Error: Path not found: {path}"

    # ---- Try pip-licenses for Python ----
    if (root / "requirements.txt").exists() or (root / "pyproject.toml").exists():
        result = _check_pip_licenses(root)
        if result:
            return result

    # ---- Try license-checker / node_modules for Node.js ----
    if (root / "package.json").exists():
        result = _check_node_licenses(root)
        if result:
            return result

    # ---- Basic fallback: parse requirements.txt ----
    req_file = root / "requirements.txt"
    if req_file.exists():
        return _parse_requirements_txt(req_file)

    return (
        f"No package manifest found in {root}.\n"
        "Supported: requirements.txt, pyproject.toml, package.json\n"
        "Install pip-licenses (pip) or license-checker (npm) for full analysis."
    )


def _check_pip_licenses(root: Path) -> str | None:
    """Run pip-licenses and return a formatted report."""
    try:
        result = subprocess.run(
            ["pip-licenses", "--format", "json", "--with-license-file",
             "--no-license-path"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(root),
        )
    except FileNotFoundError:
        return None  # pip-licenses not installed
    except subprocess.TimeoutExpired:
        return "Error: pip-licenses timed out"

    try:
        packages: list[dict] = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None

    if not packages:
        return "pip-licenses: No packages found."

    return _format_license_report(
        "pip", root, {p["Name"]: p.get("License", "UNKNOWN") for p in packages}
    )


def _check_node_licenses(root: Path) -> str | None:
    """Run license-checker and return a formatted report."""
    try:
        result = subprocess.run(
            ["npx", "license-checker", "--json", "--production"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(root),
        )
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return "Error: license-checker timed out"

    try:
        data: dict = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None

    pkg_licenses = {
        pkg: info.get("licenses", "UNKNOWN")
        for pkg, info in data.items()
    }
    return _format_license_report("npm", root, pkg_licenses)


def _parse_requirements_txt(req_file: Path) -> str:
    """Basic report from requirements.txt without version resolution."""
    lines_out = [
        f"requirements.txt found at {req_file}.",
        "Install pip-licenses for full license analysis: pip install pip-licenses\n",
        "Dependencies listed (licenses not resolved without pip-licenses):",
    ]
    try:
        for line in req_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                lines_out.append(f"  {stripped}")
    except OSError as exc:
        return f"Error reading requirements.txt: {exc}"
    return "\n".join(lines_out)


def _format_license_report(
    ecosystem: str,
    root: Path,
    pkg_licenses: dict[str, str],
) -> str:
    """Format a license compliance report grouped by license type."""
    # Group packages by license
    by_license: dict[str, list[str]] = {}
    for pkg, lic in sorted(pkg_licenses.items()):
        # Some entries have multiple licenses like "MIT OR Apache-2.0"
        lic_clean = str(lic).strip()
        by_license.setdefault(lic_clean, []).append(pkg)

    restrictive = {
        lic: pkgs
        for lic, pkgs in by_license.items()
        if any(r in lic for r in _RESTRICTIVE_LICENSES)
    }
    unknown = {
        lic: pkgs
        for lic, pkgs in by_license.items()
        if "UNKNOWN" in lic.upper() or not lic.strip()
    }

    lines = [
        f"License compliance report ({ecosystem}): {root}",
        f"  Total packages: {len(pkg_licenses)}",
        f"  Unique licenses: {len(by_license)}",
        f"  Potentially restrictive: {sum(len(v) for v in restrictive.values())}",
        f"  Unknown licenses: {sum(len(v) for v in unknown.values())}",
        "",
    ]

    if restrictive:
        lines.append("POTENTIALLY RESTRICTIVE LICENSES (review before commercial use):")
        for lic, pkgs in sorted(restrictive.items()):
            lines.append(f"  [{lic}] ({len(pkgs)} packages):")
            for pkg in pkgs[:10]:
                lines.append(f"    - {pkg}")
            if len(pkgs) > 10:
                lines.append(f"    ... and {len(pkgs) - 10} more")
        lines.append("")

    if unknown:
        lines.append("UNKNOWN LICENSES (manual review required):")
        for lic, pkgs in sorted(unknown.items()):
            for pkg in pkgs[:10]:
                lines.append(f"  - {pkg}")
            if len(pkgs) > 10:
                lines.append(f"  ... and {len(pkgs) - 10} more")
        lines.append("")

    lines.append("ALL LICENSES:")
    for lic, pkgs in sorted(by_license.items()):
        flag = " [REVIEW]" if lic in restrictive else ""
        lines.append(f"  {lic} ({len(pkgs)} packages){flag}")

    return "\n".join(lines)
