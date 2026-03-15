"""Shell execution tools — run commands safely with timeout and cwd support."""

import os
import subprocess
from pathlib import Path

# Commands that are unconditionally blocked regardless of arguments
_BLOCKED_PATTERNS = [
    "rm -rf /",
    "rm -rf ~",
    "mkfs",
    "dd if=",
    ":(){:|:&};:",   # fork bomb
    "chmod -R 777 /",
    "chown -R",
    "> /dev/sda",
]


def _is_dangerous(command: str) -> tuple[bool, str]:
    """Return (True, reason) if the command matches a blocked pattern."""
    cmd_lower = command.lower().strip()
    for pattern in _BLOCKED_PATTERNS:
        if pattern in cmd_lower:
            return True, f"Blocked dangerous pattern: '{pattern}'"
    return False, ""


def run_command(
    command: str,
    cwd: str = "",
    timeout: int = 60,
    env_vars: dict | None = None,
) -> dict:
    """Execute a shell command and return its output.

    Args:
        command: Shell command string to execute (passed to bash -c).
        cwd: Working directory for the command. Defaults to current directory.
        timeout: Maximum execution time in seconds (default 60, max 300).
        env_vars: Additional environment variables to set for this command.

    Returns:
        dict with keys: stdout (str), stderr (str), exit_code (int), success (bool), error (str), command (str).
    """
    # Safety check
    blocked, reason = _is_dangerous(command)
    if blocked:
        return {
            "success": False,
            "stdout": "",
            "stderr": "",
            "exit_code": -1,
            "command": command,
            "error": f"Command refused: {reason}",
        }

    # Clamp timeout
    timeout = min(max(1, timeout), 300)

    # Resolve working directory
    work_dir: str | None = None
    if cwd:
        p = Path(cwd).expanduser().resolve()
        if p.exists() and p.is_dir():
            work_dir = str(p)
        else:
            return {
                "success": False,
                "stdout": "",
                "stderr": "",
                "exit_code": -1,
                "command": command,
                "error": f"Working directory not found: {cwd}",
            }

    # Build environment
    env = os.environ.copy()
    if env_vars:
        env.update({str(k): str(v) for k, v in env_vars.items()})

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=work_dir,
            env=env,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
            "command": command,
            "error": "",
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "",
            "exit_code": -1,
            "command": command,
            "error": f"Command timed out after {timeout}s",
        }
    except Exception as exc:
        return {
            "success": False,
            "stdout": "",
            "stderr": "",
            "exit_code": -1,
            "command": command,
            "error": str(exc),
        }
