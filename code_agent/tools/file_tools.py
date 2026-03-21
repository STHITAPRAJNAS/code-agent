"""File system tools — read, write, search, and manage files.

ADK tool functions that return plain strings consumed directly by LLM agents.
"""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Read / Write
# ---------------------------------------------------------------------------

def read_file(
    file_path: str,
    start_line: int = 1,
    end_line: int | None = None,
) -> str:
    """Read the contents of a file, optionally restricting to a line range.

    Use this to inspect source code, configuration files, or any text file.
    For large files, specify start_line and end_line to read only the relevant
    portion.

    Args:
        file_path: Absolute or relative path to the file to read.
        start_line: First line to return (1-indexed, default 1).
        end_line: Last line to return inclusive (default: read to end of file).

    Returns:
        The file contents as a string, annotated with the resolved path and
        line range.  Returns an error message string if the file is not found
        or cannot be read.
    """
    try:
        p = Path(file_path).expanduser().resolve()
        if not p.exists():
            return f"Error: File not found: {file_path}"
        if p.is_dir():
            return f"Error: Path is a directory, not a file: {file_path}"

        text = p.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        total = len(lines)

        # Normalise to 0-indexed slice bounds
        s = max(0, start_line - 1)
        e = end_line if end_line is not None else total

        selected = lines[s:e]
        content = "".join(selected)

        header = f"File: {p}\nLines {s + 1}–{min(e, total)} of {total}\n"
        header += "-" * 60 + "\n"
        return header + content
    except Exception as exc:
        return f"Error reading {file_path}: {exc}"


def write_file(
    file_path: str,
    content: str,
    create_dirs: bool = True,
) -> str:
    """Write content to a file, replacing it entirely if it already exists.

    Creates missing parent directories by default.  Use this to save code
    edits, generate new files, or write configuration.

    Args:
        file_path: Absolute or relative path to write to.
        content: Text content to write (UTF-8 encoded).
        create_dirs: When True (default), create missing parent directories
            automatically.

    Returns:
        A success message with the resolved path and bytes written, or an
        error message if the write failed.
    """
    try:
        p = Path(file_path).expanduser().resolve()
        if create_dirs:
            p.parent.mkdir(parents=True, exist_ok=True)
        encoded = content.encode("utf-8")
        p.write_bytes(encoded)
        return f"Written {len(encoded)} bytes to {p}"
    except Exception as exc:
        return f"Error writing {file_path}: {exc}"


# ---------------------------------------------------------------------------
# Directory listing
# ---------------------------------------------------------------------------

def list_directory(
    path: str,
    recursive: bool = False,
    pattern: str = "*",
) -> str:
    """List files and directories at a given path.

    Use to explore a repository's structure before reading individual files.
    Supports glob patterns to filter results (e.g. '*.py', '*.ts').

    Args:
        path: Directory path to list.
        recursive: When True, list all files in subdirectories as well.
        pattern: Glob pattern to filter entries (default '*' = everything).

    Returns:
        A formatted directory listing with type, size, and modification time,
        or an error message if the path does not exist.
    """
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"Error: Path not found: {path}"
        if not p.is_dir():
            return f"Error: Not a directory: {path}"

        if recursive:
            glob_pat = f"**/{pattern}" if pattern != "*" else "**/*"
            items = sorted(p.glob(glob_pat))
        else:
            items = sorted(p.glob(pattern))

        if not items:
            return f"Directory {p} is empty (pattern: {pattern})"

        lines = [f"Directory: {p}\n"]
        for item in items:
            try:
                stat = item.stat()
                kind = "DIR " if item.is_dir() else "FILE"
                size = f"{stat.st_size:>10} B" if item.is_file() else " " * 12
                mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                rel = item.relative_to(p)
                lines.append(f"  {kind}  {size}  {mtime}  {rel}")
            except OSError:
                continue

        lines.append(f"\n{len(items)} entries")
        return "\n".join(lines)
    except Exception as exc:
        return f"Error listing {path}: {exc}"


# ---------------------------------------------------------------------------
# Find / search
# ---------------------------------------------------------------------------

def find_files(
    pattern: str,
    path: str = ".",
) -> str:
    """Find files matching a glob pattern under a directory.

    Use to locate files by name across a repository, e.g. find all '*.py'
    files or locate a specific file by name.

    Args:
        pattern: Glob pattern to match against file names or paths
            (e.g. '*.py', 'config.json', '**/tests/**').
        path: Root directory to search from (default: current directory).

    Returns:
        Newline-separated list of matching absolute file paths, or an error
        message if the path does not exist.
    """
    try:
        root = Path(path).expanduser().resolve()
        if not root.exists():
            return f"Error: Path not found: {path}"

        # Support both filename-only patterns and path patterns
        if "**" in pattern or "/" in pattern:
            matches = sorted(root.glob(pattern))
        else:
            matches = sorted(root.rglob(pattern))

        files = [str(m) for m in matches if m.is_file()]
        if not files:
            return f"No files matching '{pattern}' found under {root}"

        return f"Found {len(files)} file(s) matching '{pattern}':\n" + "\n".join(files)
    except Exception as exc:
        return f"Error finding files: {exc}"


def search_in_files(
    pattern: str,
    path: str = ".",
    file_extension: str = "",
) -> str:
    """Search for a text pattern inside files using ripgrep (or Python fallback).

    Use this for fast text search across a codebase.  Supports regex patterns.
    For exact symbol/identifier lookup prefer grep_code in code_tools.

    Args:
        pattern: Regular expression or literal string to search for.
        path: Root directory to search (default: current directory).
        file_extension: Restrict search to files with this extension, without
            the leading dot (e.g. 'py', 'js', 'ts').  Empty means all files.

    Returns:
        Matching lines with file path and line number, or an error message.
    """
    try:
        root = Path(path).expanduser().resolve()
        if not root.exists():
            return f"Error: Path not found: {path}"

        # Try ripgrep first for speed
        cmd = ["rg", "--with-filename", "--line-number", "--no-heading", "-e", pattern]
        if file_extension:
            ext = file_extension.lstrip(".")
            cmd.extend(["-g", f"*.{ext}"])
        cmd.append(str(root))

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            if result.returncode in (0, 1):  # 1 = no matches
                output = result.stdout.strip()
                if not output:
                    return f"No matches for '{pattern}' in {root}"
                count = output.count("\n") + 1
                return f"Found {count} match(es) for '{pattern}':\n\n{output}"
        except FileNotFoundError:
            pass  # rg not installed, fall through to Python

        # Python fallback
        flags = re.MULTILINE
        try:
            compiled = re.compile(pattern, flags)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"

        matches: list[str] = []
        glob_pat = f"**/*.{file_extension.lstrip('.')}" if file_extension else "**/*"
        for filepath in sorted(root.glob(glob_pat)):
            if not filepath.is_file():
                continue
            try:
                for i, line in enumerate(
                    filepath.read_text(encoding="utf-8", errors="replace").splitlines(),
                    start=1,
                ):
                    if compiled.search(line):
                        matches.append(f"{filepath}:{i}: {line.rstrip()}")
            except OSError:
                continue

        if not matches:
            return f"No matches for '{pattern}' in {root}"
        return f"Found {len(matches)} match(es) for '{pattern}':\n\n" + "\n".join(matches)
    except Exception as exc:
        return f"Error searching files: {exc}"


# ---------------------------------------------------------------------------
# File management
# ---------------------------------------------------------------------------

def delete_file(file_path: str) -> str:
    """Delete a file or directory tree.

    Removes a single file or an entire directory recursively.  Use with
    caution — deletion is permanent.

    Args:
        file_path: Absolute or relative path to the file or directory to delete.

    Returns:
        A success message, or an error message if the path does not exist or
        deletion failed.
    """
    try:
        p = Path(file_path).expanduser().resolve()
        if not p.exists():
            return f"Error: Path not found: {file_path}"
        if p.is_dir():
            shutil.rmtree(p)
            return f"Deleted directory: {p}"
        p.unlink()
        return f"Deleted file: {p}"
    except Exception as exc:
        return f"Error deleting {file_path}: {exc}"


def create_directory(path: str) -> str:
    """Create a directory and all missing parent directories.

    Equivalent to 'mkdir -p'.  Safe to call even if the directory already
    exists.

    Args:
        path: Absolute or relative path of the directory to create.

    Returns:
        A success message with the resolved path, or an error message if
        creation failed.
    """
    try:
        p = Path(path).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return f"Directory created: {p}"
    except Exception as exc:
        return f"Error creating directory {path}: {exc}"


def get_file_info(file_path: str) -> str:
    """Get metadata about a file or directory: size, type, and timestamps.

    Use to check whether a file exists, determine its size, or read its
    modification time before deciding whether to re-read or re-index it.

    Args:
        file_path: Absolute or relative path to the file or directory.

    Returns:
        A human-readable summary of file metadata, or an error message if the
        path does not exist.
    """
    try:
        p = Path(file_path).expanduser().resolve()
        if not p.exists():
            return f"Error: Path not found: {file_path}"
        stat = p.stat()
        kind = "directory" if p.is_dir() else "file"
        size = stat.st_size
        mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()
        ctime = datetime.fromtimestamp(stat.st_ctime).isoformat()
        lines = [
            f"Path:     {p}",
            f"Type:     {kind}",
            f"Size:     {size} bytes",
            f"Modified: {mtime}",
            f"Created:  {ctime}",
        ]
        if p.is_file():
            lines.append(f"Extension: {p.suffix or '(none)'}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Error getting info for {file_path}: {exc}"
