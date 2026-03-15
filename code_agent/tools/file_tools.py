"""File system tools — read, write, search, and manage files."""

import fnmatch
import os
import shutil
from datetime import datetime
from pathlib import Path


def read_file(path: str, offset: int = 0, limit: int = 0) -> dict:
    """Read the contents of a file.

    Args:
        path: Absolute or relative path to the file.
        offset: Line number to start reading from (0-indexed, default 0).
        limit: Maximum number of lines to return (0 = all lines).

    Returns:
        dict with keys: content (str), total_lines (int), path (str), success (bool), error (str).
    """
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return {"success": False, "error": f"File not found: {path}", "content": "", "path": str(p)}
        if p.is_dir():
            return {"success": False, "error": f"Path is a directory: {path}", "content": "", "path": str(p)}

        text = p.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        total = len(lines)

        if offset or limit:
            sliced = lines[offset: (offset + limit) if limit else None]
            text = "".join(sliced)

        return {
            "success": True,
            "content": text,
            "total_lines": total,
            "path": str(p),
            "error": "",
        }
    except Exception as exc:
        return {"success": False, "error": str(exc), "content": "", "path": path}


def write_file(path: str, content: str, create_dirs: bool = True) -> dict:
    """Write content to a file, optionally creating parent directories.

    Args:
        path: Absolute or relative path to write to.
        content: Text content to write.
        create_dirs: Create missing parent directories if True (default True).

    Returns:
        dict with keys: success (bool), path (str), bytes_written (int), error (str).
    """
    try:
        p = Path(path).expanduser().resolve()
        if create_dirs:
            p.parent.mkdir(parents=True, exist_ok=True)
        encoded = content.encode("utf-8")
        p.write_bytes(encoded)
        return {"success": True, "path": str(p), "bytes_written": len(encoded), "error": ""}
    except Exception as exc:
        return {"success": False, "path": path, "bytes_written": 0, "error": str(exc)}


def list_directory(path: str, recursive: bool = False, pattern: str = "") -> dict:
    """List files and directories at a given path.

    Args:
        path: Directory path to list.
        recursive: List recursively if True.
        pattern: Optional glob pattern to filter (e.g. "*.py").

    Returns:
        dict with keys: entries (list of dicts), count (int), path (str), success (bool), error (str).
    """
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return {"success": False, "error": f"Path not found: {path}", "entries": [], "count": 0, "path": str(p)}
        if not p.is_dir():
            return {"success": False, "error": f"Not a directory: {path}", "entries": [], "count": 0, "path": str(p)}

        glob_pattern = f"**/{pattern}" if (recursive and pattern) else ("**/*" if recursive else (pattern or "*"))
        items = sorted(p.glob(glob_pattern)) if recursive else sorted(p.glob(pattern or "*"))

        entries = []
        for item in items:
            try:
                stat = item.stat()
                entries.append({
                    "name": item.name,
                    "path": str(item),
                    "type": "dir" if item.is_dir() else "file",
                    "size": stat.st_size if item.is_file() else 0,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })
            except OSError:
                continue

        return {"success": True, "entries": entries, "count": len(entries), "path": str(p), "error": ""}
    except Exception as exc:
        return {"success": False, "error": str(exc), "entries": [], "count": 0, "path": path}


def search_in_files(directory: str, pattern: str, file_glob: str = "**/*") -> dict:
    """Search for a text pattern across files in a directory.

    Args:
        directory: Root directory to search in.
        pattern: Text string to search for (case-sensitive substring match).
        file_glob: Glob pattern to filter files (default "**/*").

    Returns:
        dict with keys: matches (list of {file, line_number, line}), total_matches (int), success (bool), error (str).
    """
    try:
        root = Path(directory).expanduser().resolve()
        if not root.exists():
            return {"success": False, "error": f"Directory not found: {directory}", "matches": [], "total_matches": 0}

        matches = []
        for filepath in sorted(root.glob(file_glob)):
            if not filepath.is_file():
                continue
            try:
                text = filepath.read_text(encoding="utf-8", errors="replace")
                for i, line in enumerate(text.splitlines(), start=1):
                    if pattern in line:
                        matches.append({
                            "file": str(filepath),
                            "line_number": i,
                            "line": line.rstrip(),
                        })
            except OSError:
                continue

        return {"success": True, "matches": matches, "total_matches": len(matches), "error": ""}
    except Exception as exc:
        return {"success": False, "error": str(exc), "matches": [], "total_matches": 0}


def delete_file(path: str) -> dict:
    """Delete a file or empty directory.

    Args:
        path: Path to the file or directory to delete.

    Returns:
        dict with keys: success (bool), path (str), error (str).
    """
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return {"success": False, "path": str(p), "error": f"Path not found: {path}"}
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        return {"success": True, "path": str(p), "error": ""}
    except Exception as exc:
        return {"success": False, "path": path, "error": str(exc)}


def create_directory(path: str) -> dict:
    """Create a directory (and all missing parent directories).

    Args:
        path: Directory path to create.

    Returns:
        dict with keys: success (bool), path (str), error (str).
    """
    try:
        p = Path(path).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return {"success": True, "path": str(p), "error": ""}
    except Exception as exc:
        return {"success": False, "path": path, "error": str(exc)}


def get_file_info(path: str) -> dict:
    """Get metadata about a file or directory.

    Args:
        path: Path to inspect.

    Returns:
        dict with keys: path, type, size, modified, created, exists, success, error.
    """
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return {"success": False, "exists": False, "path": str(p), "error": f"Not found: {path}"}
        stat = p.stat()
        return {
            "success": True,
            "exists": True,
            "path": str(p),
            "type": "dir" if p.is_dir() else "file",
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "error": "",
        }
    except Exception as exc:
        return {"success": False, "exists": False, "path": path, "error": str(exc)}
