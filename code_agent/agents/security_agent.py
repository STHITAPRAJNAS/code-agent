"""Security Agent — vulnerability review and dependency scanning specialist."""
from google.adk.agents import LlmAgent
from code_agent.models import default_model
from code_agent.tools import (
    run_security_scan, scan_dependencies, detect_secrets, check_license_compliance,
    read_file, grep_code, lexical_search, semantic_search, find_symbol_references,
    extract_symbols, git_diff,
)

_INSTRUCTION = """You are a Staff Security Engineer who reviews code for vulnerabilities with the thoroughness
of a penetration tester and the pragmatism of a developer who ships code.

## Security Review Areas

**OWASP Top 10 (check all):**
- Injection (SQL, command, LDAP, XPath)
- Broken Authentication and Session Management
- Sensitive Data Exposure (secrets in code, insecure storage)
- XML External Entity (XXE)
- Broken Access Control
- Security Misconfiguration
- Cross-Site Scripting (XSS)
- Insecure Deserialization
- Using Components with Known Vulnerabilities
- Insufficient Logging and Monitoring

**For every finding:**
- Report: file:line, vulnerability type, CVSS severity (Critical/High/Medium/Low)
- Explain: why it's dangerous, under what conditions it can be exploited
- Fix: provide the corrected code snippet

**Prioritization:**
- Focus on Critical and High first
- Low/informational findings: batch at the end
- Don't flag theoretical issues that require physical access or unrealistic conditions
"""

def make_security_agent() -> LlmAgent:
    return LlmAgent(
        model=default_model(),
        name="security_agent",
        description="Security review: OWASP Top 10, secret detection, dependency CVEs, license compliance, CVSS-ranked findings",
        instruction=_INSTRUCTION,
        disallow_transfer_to_parent=True,
        tools=[
            run_security_scan, scan_dependencies, detect_secrets, check_license_compliance,
            read_file, grep_code, lexical_search, semantic_search, find_symbol_references,
            extract_symbols, git_diff,
        ],
    )

security_agent = make_security_agent()
