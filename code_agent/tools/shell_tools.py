"""Shell execution tools — run commands and scripts with timeout and safety checks.

ADK tool functions that return plain strings consumed directly by LLM agents.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

# Commands that are unconditionally blocked regardless of arguments
_BLOCKED_PATTERNS: list[str] = [
    "rm -rf /",
    "rm -rf ~",
    "mkfs",
    "dd if=",
    ":(){:|:&};:",   # fork bomb
    "chmod -R 777 /",
    "> /dev/sda",
]

_LANG_EXECUTORS: dict[str, tuple[str, str]] = {
    "bash": ("bash", ".sh"),
    "python": ("python3", ".py"),
    "node": ("node", ".js"),
}


def _is_dangerous(command: str) -> tuple[bool, str]:
    """Return (True, reason) if the command matches a blocked pattern."""
    cmd_lower = command.lower().strip()
    for pattern in _BLOCKED_PATTERNS:
        if pattern in cmd_lower:
            return True, f"Blocked dangerous pattern: '{pattern}'"
    return False, ""


def run_command(
    command: str,
    cwd: str | None = None,
    timeout: int = 120,
) -> str:
    """Execute a shell command and return its output.

    Runs the command via bash -c in the given working directory.  A small set
    of obviously destructive patterns (rm -rf /, mkfs, fork bomb, etc.) are
    refused unconditionally.

    Args:
        command: Shell command string to execute (passed to bash -c).
        cwd: Working directory for the command.  When None, the current
            process working directory is used.
        timeout: Maximum execution time in seconds (default 120, clamped to
            1–600).

    Returns:
        Combined stdout and stderr output as a string, prefixed with the exit
        code.  Returns an error description string if the command is refused,
        the working directory is not found, or the process times out.
    """
    # Safety check
    blocked, reason = _is_dangerous(command)
    if blocked:
        return f"Error: Command refused — {reason}"

    # Clamp timeout
    timeout = min(max(1, timeout), 600)

    # Resolve working directory
    work_dir: str | None = None
    if cwd:
        p = Path(cwd).expanduser().resolve()
        if not p.exists() or not p.is_dir():
            return f"Error: Working directory not found: {cwd}"
        work_dir = str(p)

    env = os.environ.copy()

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
        parts = [f"Exit code: {result.returncode}"]
        if result.stdout:
            parts.append(f"STDOUT:\n{result.stdout.rstrip()}")
        if result.stderr:
            parts.append(f"STDERR:\n{result.stderr.rstrip()}")
        if not result.stdout and not result.stderr:
            parts.append("(no output)")
        return "\n".join(parts)
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout}s: {command}"
    except Exception as exc:
        return f"Error running command: {exc}"


def run_script(
    script: str,
    language: str = "bash",
    cwd: str | None = None,
) -> str:
    """Execute a multi-line script by writing it to a temp file and running it.

    Supports bash, python, and node.  The temporary script file is always
    cleaned up after execution, even on error.

    Args:
        script: The full script source code to execute.
        language: One of 'bash', 'python', or 'node' (default 'bash').
        cwd: Working directory for script execution.  When None, the current
            process working directory is used.

    Returns:
        Combined stdout and stderr output as a string, prefixed with the exit
        code.  Returns an error description if the language is unsupported,
        the interpreter is not found, the working directory is missing, or
        execution times out.
    """
    lang = language.lower().strip()
    if lang not in _LANG_EXECUTORS:
        supported = ", ".join(sorted(_LANG_EXECUTORS))
        return f"Error: Unsupported language '{language}'. Supported: {supported}"

    executor, suffix = _LANG_EXECUTORS[lang]

    # Resolve working directory
    work_dir: str | None = None
    if cwd:
        p = Path(cwd).expanduser().resolve()
        if not p.exists() or not p.is_dir():
            return f"Error: Working directory not found: {cwd}"
        work_dir = str(p)

    tmp_path: str | None = None
    try:
        # Write to a named temp file so the interpreter can read it
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(script)

        result = subprocess.run(
            [executor, tmp_path],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=work_dir,
            env=os.environ.copy(),
        )
        parts = [f"Exit code: {result.returncode}"]
        if result.stdout:
            parts.append(f"STDOUT:\n{result.stdout.rstrip()}")
        if result.stderr:
            parts.append(f"STDERR:\n{result.stderr.rstrip()}")
        if not result.stdout and not result.stderr:
            parts.append("(no output)")
        return "\n".join(parts)
    except FileNotFoundError:
        return f"Error: Interpreter '{executor}' not found in PATH"
    except subprocess.TimeoutExpired:
        return f"Error: Script timed out after 120s"
    except Exception as exc:
        return f"Error running {language} script: {exc}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
