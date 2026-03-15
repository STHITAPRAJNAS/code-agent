"""Code analysis tools — grep, symbol search, syntax check, outline, LOC count."""

import ast
import re
import subprocess
from pathlib import Path


def grep_code(
    directory: str,
    pattern: str,
    file_glob: str = "**/*.py",
    case_sensitive: bool = True,
) -> dict:
    """Search for a regex pattern across source files.

    Args:
        directory: Root directory to search in.
        pattern: Regular expression pattern to search for.
        file_glob: Glob pattern to filter files (default "**/*.py").
        case_sensitive: Case-sensitive search if True (default).

    Returns:
        dict with matches (list of {file, line_number, line}), total_matches, success, error.
    """
    try:
        root = Path(directory).expanduser().resolve()
        if not root.exists():
            return {"success": False, "error": f"Directory not found: {directory}", "matches": [], "total_matches": 0}

        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            compiled = re.compile(pattern, flags)
        except re.error as e:
            return {"success": False, "error": f"Invalid regex: {e}", "matches": [], "total_matches": 0}

        matches = []
        for filepath in sorted(root.glob(file_glob)):
            if not filepath.is_file():
                continue
            try:
                text = filepath.read_text(encoding="utf-8", errors="replace")
                for i, line in enumerate(text.splitlines(), start=1):
                    if compiled.search(line):
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


def find_symbol(directory: str, symbol: str, language: str = "") -> dict:
    """Find where a class, function, or variable is defined.

    Args:
        directory: Root directory to search in.
        symbol: Symbol name to find (exact match of definition).
        language: Optional language hint ("python", "javascript", etc.) to narrow file types.

    Returns:
        dict with definitions (list of {file, line_number, line, kind}), success, error.
    """
    lang_globs: dict[str, list[str]] = {
        "python": ["**/*.py"],
        "javascript": ["**/*.js", "**/*.mjs"],
        "typescript": ["**/*.ts", "**/*.tsx"],
        "go": ["**/*.go"],
        "java": ["**/*.java"],
        "rust": ["**/*.rs"],
    }

    globs = lang_globs.get(language.lower(), ["**/*.py", "**/*.js", "**/*.ts", "**/*.go", "**/*.java", "**/*.rs"])

    # Patterns for common definition styles
    def_patterns = [
        (r"^\s*def\s+" + re.escape(symbol) + r"\s*\(", "function"),
        (r"^\s*class\s+" + re.escape(symbol) + r"[\s\(:]", "class"),
        (r"^\s*(async\s+def)\s+" + re.escape(symbol) + r"\s*\(", "async_function"),
        (r"^\s*(const|let|var|function)\s+" + re.escape(symbol) + r"[\s=\(]", "js_declaration"),
        (r"^\s*func\s+" + re.escape(symbol) + r"\s*\(", "go_function"),
        (r"^\s*type\s+" + re.escape(symbol) + r"\s+", "type"),
    ]
    compiled_patterns = [(re.compile(p), kind) for p, kind in def_patterns]

    root = Path(directory).expanduser().resolve()
    definitions = []

    for glob in globs:
        for filepath in sorted(root.glob(glob)):
            if not filepath.is_file():
                continue
            try:
                text = filepath.read_text(encoding="utf-8", errors="replace")
                for i, line in enumerate(text.splitlines(), start=1):
                    for cpat, kind in compiled_patterns:
                        if cpat.search(line):
                            definitions.append({
                                "file": str(filepath),
                                "line_number": i,
                                "line": line.strip(),
                                "kind": kind,
                            })
                            break
            except OSError:
                continue

    return {"success": True, "definitions": definitions, "count": len(definitions), "error": ""}


def syntax_check(file_path: str) -> dict:
    """Check a file for syntax errors.

    Supports Python (via ast.parse) and JavaScript/TypeScript (via node --check if available).

    Args:
        file_path: Path to the source file.

    Returns:
        dict with valid (bool), errors (list of str), language, success, error.
    """
    p = Path(file_path).expanduser().resolve()
    if not p.exists():
        return {"success": False, "valid": False, "errors": [f"File not found: {file_path}"], "language": "unknown"}

    suffix = p.suffix.lower()

    if suffix == ".py":
        try:
            source = p.read_text(encoding="utf-8", errors="replace")
            ast.parse(source, filename=str(p))
            return {"success": True, "valid": True, "errors": [], "language": "python", "error": ""}
        except SyntaxError as e:
            return {
                "success": True,
                "valid": False,
                "errors": [f"SyntaxError at line {e.lineno}: {e.msg}"],
                "language": "python",
                "error": "",
            }
        except Exception as exc:
            return {"success": False, "valid": False, "errors": [str(exc)], "language": "python", "error": str(exc)}

    if suffix in {".js", ".mjs", ".cjs", ".ts", ".tsx"}:
        try:
            result = subprocess.run(
                ["node", "--check", str(p)],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                return {"success": True, "valid": True, "errors": [], "language": "javascript", "error": ""}
            return {
                "success": True,
                "valid": False,
                "errors": [result.stderr.strip()],
                "language": "javascript",
                "error": "",
            }
        except FileNotFoundError:
            return {"success": False, "valid": False, "errors": ["node not found in PATH"], "language": "javascript", "error": "node not available"}
        except Exception as exc:
            return {"success": False, "valid": False, "errors": [str(exc)], "language": "javascript", "error": str(exc)}

    return {"success": True, "valid": True, "errors": [], "language": "unknown", "error": "No syntax checker for this file type"}


def get_file_outline(file_path: str) -> dict:
    """Extract the structural outline of a source file (classes, functions, imports).

    Currently supports Python via AST. Falls back to regex for other languages.

    Args:
        file_path: Path to the source file.

    Returns:
        dict with outline (list of {kind, name, line}), language, success, error.
    """
    p = Path(file_path).expanduser().resolve()
    if not p.exists():
        return {"success": False, "outline": [], "language": "unknown", "error": f"File not found: {file_path}"}

    suffix = p.suffix.lower()

    if suffix == ".py":
        try:
            source = p.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
            outline = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    outline.append({"kind": "function", "name": node.name, "line": node.lineno})
                elif isinstance(node, ast.ClassDef):
                    outline.append({"kind": "class", "name": node.name, "line": node.lineno})
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        outline.append({"kind": "import", "name": alias.name, "line": node.lineno})
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    outline.append({"kind": "import_from", "name": module, "line": node.lineno})
            outline.sort(key=lambda x: x["line"])
            return {"success": True, "outline": outline, "language": "python", "error": ""}
        except SyntaxError as e:
            return {"success": False, "outline": [], "language": "python", "error": f"SyntaxError: {e}"}

    # Regex fallback for other languages
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
        patterns = [
            (re.compile(r"^\s*(?:async\s+)?def\s+(\w+)\s*\("), "function"),
            (re.compile(r"^\s*class\s+(\w+)"), "class"),
            (re.compile(r"^\s*(?:const|let|var)\s+(\w+)\s*="), "variable"),
            (re.compile(r"^\s*function\s+(\w+)\s*\("), "function"),
            (re.compile(r"^\s*func\s+(\w+)\s*\("), "function"),
        ]
        outline = []
        for i, line in enumerate(text.splitlines(), start=1):
            for cpat, kind in patterns:
                m = cpat.match(line)
                if m:
                    outline.append({"kind": kind, "name": m.group(1), "line": i})
                    break
        return {"success": True, "outline": outline, "language": suffix.lstrip("."), "error": ""}
    except Exception as exc:
        return {"success": False, "outline": [], "language": "unknown", "error": str(exc)}


def count_lines(path: str) -> dict:
    """Count lines of code, blank lines, and comment lines in a file or directory.

    Args:
        path: File path or directory path (counts all .py files recursively if directory).

    Returns:
        dict with total (int), code (int), blank (int), comment (int), files (int), success, error.
    """
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return {"success": False, "error": f"Not found: {path}", "total": 0, "code": 0, "blank": 0, "comment": 0, "files": 0}

        files_to_count = [p] if p.is_file() else list(p.rglob("*.py"))
        total = blank = comment = code = 0

        for f in files_to_count:
            try:
                for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                    stripped = line.strip()
                    total += 1
                    if not stripped:
                        blank += 1
                    elif stripped.startswith(("#", "//", "/*", "*", "'")):
                        comment += 1
                    else:
                        code += 1
            except OSError:
                continue

        return {
            "success": True,
            "total": total,
            "code": code,
            "blank": blank,
            "comment": comment,
            "files": len(files_to_count),
            "error": "",
        }
    except Exception as exc:
        return {"success": False, "error": str(exc), "total": 0, "code": 0, "blank": 0, "comment": 0, "files": 0}
