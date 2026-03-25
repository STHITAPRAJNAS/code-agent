"""Code analysis tools — grep, symbol search, syntax check, outline, LOC count.

ADK tool functions that return plain strings consumed directly by LLM agents.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# Language helpers
# ---------------------------------------------------------------------------

_LANG_BY_EXT: dict[str, str] = {
    ".py": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go",
    ".java": "java",
    ".rs": "rust",
    ".rb": "ruby",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
    ".c": "c",
    ".cs": "csharp",
    ".kt": "kotlin",
    ".swift": "swift",
    ".php": "php",
    ".scala": "scala",
    ".sh": "bash", ".bash": "bash",
}

_EXT_BY_LANG: dict[str, list[str]] = {
    "python": ["**/*.py"],
    "javascript": ["**/*.js", "**/*.mjs", "**/*.cjs"],
    "typescript": ["**/*.ts", "**/*.tsx"],
    "go": ["**/*.go"],
    "java": ["**/*.java"],
    "rust": ["**/*.rs"],
    "ruby": ["**/*.rb"],
}

_SKIP_DIRS = frozenset(
    [".git", "__pycache__", "node_modules", ".venv", "venv", "build", "dist"]
)


def _detect_language(path: Path, hint: str = "") -> str:
    if hint:
        return hint.lower()
    return _LANG_BY_EXT.get(path.suffix.lower(), "unknown")


def _iter_source_files(root: Path, language: str = ""):
    """Yield source files under root, optionally filtered by language."""
    globs = _EXT_BY_LANG.get(language.lower(), ["**/*"]) if language else ["**/*"]
    seen: set[Path] = set()
    for g in globs:
        for p in sorted(root.glob(g)):
            if p in seen or not p.is_file():
                continue
            if any(part in _SKIP_DIRS for part in p.parts):
                continue
            seen.add(p)
            yield p


# ---------------------------------------------------------------------------
# Symbol extraction (tree-sitter or regex fallback)
# ---------------------------------------------------------------------------

def extract_symbols(file_path: str, language: str = "") -> str:
    """List all functions, classes, and methods in a source file with line numbers.

    Uses tree-sitter when available for accurate parsing; falls back to regex
    heuristics for all other languages.  Returns a structured list of symbol
    names and their definition line numbers.

    Args:
        file_path: Absolute or relative path to the source file.
        language: Optional language hint ('python', 'javascript', etc.).
            Auto-detected from file extension when empty.

    Returns:
        Formatted list of symbols (kind, name, line) or an error message.
    """
    try:
        p = Path(file_path).expanduser().resolve()
        if not p.exists():
            return f"Error: File not found: {file_path}"

        lang = _detect_language(p, language)

        # --- tree-sitter path ---
        symbols = _extract_with_treesitter(p, lang)
        if symbols is None:
            symbols = _extract_with_regex(p, lang)

        if not symbols:
            return f"No symbols found in {p}"

        lines = [f"Symbols in {p} ({lang}):"]
        for kind, name, lineno in sorted(symbols, key=lambda x: x[2]):
            lines.append(f"  {lineno:>5}  {kind:<15} {name}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Error extracting symbols from {file_path}: {exc}"


def _extract_with_treesitter(
    p: Path, lang: str
) -> list[tuple[str, str, int]] | None:
    """Return [(kind, name, line), ...] via tree-sitter, or None if unavailable."""
    _TS_LANG_MAP = {
        "python": "python",
        "javascript": "javascript",
        "typescript": "typescript",
        "go": "go",
        "rust": "rust",
        "java": "java",
    }
    ts_lang = _TS_LANG_MAP.get(lang)
    if not ts_lang:
        return None
    try:
        import tree_sitter_languages  # type: ignore[import]
        from tree_sitter import Language, Parser  # type: ignore[import]

        language_obj = tree_sitter_languages.get_language(ts_lang)
        parser = Parser()
        parser.set_language(language_obj)

        source = p.read_bytes()
        tree = parser.parse(source)

        _QUERIES: dict[str, list[tuple[str, str]]] = {
            "python": [
                ("(function_definition name: (identifier) @name)", "function"),
                ("(async_function_definition name: (identifier) @name)", "async_function"),
                ("(class_definition name: (identifier) @name)", "class"),
            ],
            "javascript": [
                ("(function_declaration name: (identifier) @name)", "function"),
                ("(class_declaration name: (identifier) @name)", "class"),
                ("(method_definition name: (property_identifier) @name)", "method"),
            ],
            "typescript": [
                ("(function_declaration name: (identifier) @name)", "function"),
                ("(class_declaration name: (identifier) @name)", "class"),
                ("(method_definition name: (property_identifier) @name)", "method"),
                ("(interface_declaration name: (type_identifier) @name)", "interface"),
            ],
            "go": [
                ("(function_declaration name: (identifier) @name)", "function"),
                ("(method_declaration name: (field_identifier) @name)", "method"),
                ("(type_spec name: (type_identifier) @name)", "type"),
            ],
        }

        results: list[tuple[str, str, int]] = []
        for query_str, kind in _QUERIES.get(ts_lang, []):
            try:
                query = language_obj.query(query_str)
                for node, _ in query.captures(tree.root_node):
                    name = source[node.start_byte: node.end_byte].decode("utf-8", errors="replace")
                    lineno = node.start_point[0] + 1
                    results.append((kind, name, lineno))
            except Exception:
                continue

        return results if results else None
    except ImportError:
        return None
    except Exception:
        return None


def _extract_with_regex(
    p: Path, lang: str
) -> list[tuple[str, str, int]]:
    """Regex-based symbol extraction fallback."""
    patterns: list[tuple[re.Pattern, str]] = [
        (re.compile(r"^\s*(?:async\s+)?def\s+(\w+)\s*\("), "function"),
        (re.compile(r"^\s*class\s+(\w+)[\s\(:]"), "class"),
        (re.compile(r"^\s*(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?function"), "function"),
        (re.compile(r"^\s*function\s+(\w+)\s*\("), "function"),
        (re.compile(r"^\s*(?:async\s+)?(\w+)\s*\([^)]*\)\s*\{"), "function"),
        (re.compile(r"^\s*func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\("), "function"),
        (re.compile(r"^\s*type\s+(\w+)\s+(?:struct|interface)\s*\{"), "type"),
        (re.compile(r"^\s*(?:public|private|protected|static).*\s+(\w+)\s*\("), "method"),
    ]
    results: list[tuple[str, str, int]] = []
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), start=1):
            for cpat, kind in patterns:
                m = cpat.search(line)
                if m:
                    results.append((kind, m.group(1), i))
                    break
    except OSError:
        pass
    return results


# ---------------------------------------------------------------------------
# Get symbol definition
# ---------------------------------------------------------------------------

def get_symbol(symbol_name: str, search_path: str = ".") -> str:
    """Find the definition of a function, class, or variable across all files.

    Searches for where a symbol is defined (not just referenced).  Useful for
    jumping to the implementation of a function you found in another file.

    Args:
        symbol_name: Name of the symbol to find (exact match).
        search_path: Directory to search recursively (default: current directory).

    Returns:
        Each definition location with file, line number, and surrounding context,
        or an error message.
    """
    try:
        root = Path(search_path).expanduser().resolve()
        if not root.exists():
            return f"Error: Path not found: {search_path}"

        def_patterns: list[tuple[re.Pattern, str]] = [
            (re.compile(r"^\s*(?:async\s+)?def\s+" + re.escape(symbol_name) + r"\s*\("), "function"),
            (re.compile(r"^\s*class\s+" + re.escape(symbol_name) + r"[\s\(:]"), "class"),
            (re.compile(r"^\s*(?:const|let|var|function)\s+" + re.escape(symbol_name) + r"[\s=\(]"), "declaration"),
            (re.compile(r"^\s*func\s+(?:\(\w+\s+\*?\w+\)\s+)?" + re.escape(symbol_name) + r"\s*\("), "function"),
            (re.compile(r"^\s*type\s+" + re.escape(symbol_name) + r"\s+"), "type"),
        ]

        results: list[str] = []
        for filepath in _iter_source_files(root):
            try:
                file_lines = filepath.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for i, line in enumerate(file_lines, start=1):
                for cpat, kind in def_patterns:
                    if cpat.search(line):
                        # Include a couple of lines of context
                        ctx_start = max(0, i - 1)
                        ctx_end = min(len(file_lines), i + 2)
                        context = "\n".join(
                            f"  {ctx_start + j + 1}: {file_lines[ctx_start + j]}"
                            for j in range(ctx_end - ctx_start)
                        )
                        results.append(
                            f"{filepath}:{i} [{kind}]\n{context}"
                        )
                        break

        if not results:
            return f"Symbol '{symbol_name}' not found in {root}"
        header = f"Found {len(results)} definition(s) of '{symbol_name}':\n"
        return header + "\n\n".join(results)
    except Exception as exc:
        return f"Error searching for symbol '{symbol_name}': {exc}"


# ---------------------------------------------------------------------------
# Syntax checking
# ---------------------------------------------------------------------------

def check_syntax(file_path: str) -> str:
    """Check a source file for syntax errors.

    Python: uses ast.parse for fast in-process checking.
    JavaScript/TypeScript: uses 'node --check' if node is available.
    Other languages: reported as 'not checked' (no suitable checker found).

    Args:
        file_path: Absolute or relative path to the source file.

    Returns:
        'Syntax OK' message, a description of any syntax errors found, or a
        message indicating the language is not supported.
    """
    try:
        p = Path(file_path).expanduser().resolve()
        if not p.exists():
            return f"Error: File not found: {file_path}"

        suffix = p.suffix.lower()

        if suffix == ".py":
            try:
                source = p.read_text(encoding="utf-8", errors="replace")
                ast.parse(source, filename=str(p))
                return f"Syntax OK: {p} (python)"
            except SyntaxError as e:
                return f"Syntax error in {p} (python) at line {e.lineno}: {e.msg}"
            except Exception as exc:
                return f"Error checking syntax: {exc}"

        if suffix in {".js", ".mjs", ".cjs"}:
            try:
                result = subprocess.run(
                    ["node", "--check", str(p)],
                    capture_output=True, text=True, timeout=15,
                )
                if result.returncode == 0:
                    return f"Syntax OK: {p} (javascript)"
                return f"Syntax error in {p} (javascript):\n{result.stderr.strip()}"
            except FileNotFoundError:
                return f"Syntax not checked: {p} — node not found in PATH"
            except subprocess.TimeoutExpired:
                return f"Syntax check timed out: {p}"

        if suffix in {".ts", ".tsx"}:
            # Try tsc --noEmit if available
            try:
                result = subprocess.run(
                    ["tsc", "--noEmit", "--allowJs", str(p)],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0:
                    return f"Syntax OK: {p} (typescript)"
                return f"Syntax error in {p} (typescript):\n{result.stderr.strip()}"
            except FileNotFoundError:
                pass
            # Fall back to node --check (works for .ts without type-checking)
            try:
                result = subprocess.run(
                    ["node", "--check", str(p)],
                    capture_output=True, text=True, timeout=15,
                )
                if result.returncode == 0:
                    return f"Syntax OK: {p} (typescript/node)"
                return f"Syntax error in {p} (typescript):\n{result.stderr.strip()}"
            except FileNotFoundError:
                return f"Syntax not checked: {p} — neither tsc nor node found in PATH"

        lang = _LANG_BY_EXT.get(suffix, suffix.lstrip(".") or "unknown")
        return f"Syntax not checked: {p} — no checker available for {lang}"
    except Exception as exc:
        return f"Error checking syntax of {file_path}: {exc}"


# ---------------------------------------------------------------------------
# Import extraction
# ---------------------------------------------------------------------------

def get_imports(file_path: str) -> str:
    """Extract all import statements from a source file.

    Supports Python (via AST for accuracy), and regex-based extraction for
    JavaScript/TypeScript and other languages.

    Args:
        file_path: Absolute or relative path to the source file.

    Returns:
        List of import statements with line numbers, or an error message.
    """
    try:
        p = Path(file_path).expanduser().resolve()
        if not p.exists():
            return f"Error: File not found: {file_path}"

        suffix = p.suffix.lower()
        source = p.read_text(encoding="utf-8", errors="replace")

        if suffix == ".py":
            try:
                tree = ast.parse(source)
                imports: list[tuple[int, str]] = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            as_part = f" as {alias.asname}" if alias.asname else ""
                            imports.append((node.lineno, f"import {alias.name}{as_part}"))
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        names = ", ".join(
                            (f"{a.name} as {a.asname}" if a.asname else a.name)
                            for a in node.names
                        )
                        dots = "." * (node.level or 0)
                        imports.append((node.lineno, f"from {dots}{module} import {names}"))
                imports.sort()
                if not imports:
                    return f"No imports found in {p}"
                lines = [f"Imports in {p} ({len(imports)} total):"]
                for lineno, stmt in imports:
                    lines.append(f"  {lineno:>5}: {stmt}")
                return "\n".join(lines)
            except SyntaxError:
                pass  # Fall through to regex

        # Regex-based for JS/TS and others
        import_patterns = [
            re.compile(r"^(?:import|from)\s+.+$", re.MULTILINE),
            re.compile(r"^(?:const|let|var)\s+\{[^}]+\}\s*=\s*require\(['\"][^'\"]+['\"]\)", re.MULTILINE),
            re.compile(r"^require\(['\"][^'\"]+['\"]\)", re.MULTILINE),
            re.compile(r'^#include\s*[<"][^>"]+[>"]', re.MULTILINE),
        ]

        file_lines = source.splitlines()
        found: list[tuple[int, str]] = []
        for lineno, line in enumerate(file_lines, start=1):
            stripped = line.strip()
            for pat in import_patterns:
                if pat.match(stripped):
                    found.append((lineno, stripped))
                    break

        if not found:
            return f"No imports found in {p}"
        out_lines = [f"Imports in {p} ({len(found)} total):"]
        for lineno, stmt in found:
            out_lines.append(f"  {lineno:>5}: {stmt}")
        return "\n".join(out_lines)
    except Exception as exc:
        return f"Error extracting imports from {file_path}: {exc}"


# ---------------------------------------------------------------------------
# Line counting
# ---------------------------------------------------------------------------

def count_lines(file_path: str) -> str:
    """Count total, code, comment, and blank lines in a file.

    Provides a quick LOC (lines of code) summary.  Supports Python (AST-
    assisted comment detection) and heuristic detection for other languages.

    Args:
        file_path: Absolute or relative path to a source file or directory.
            When a directory is given, counts all supported source files
            recursively.

    Returns:
        A summary table with total, code, comment, and blank line counts, or
        an error message.
    """
    try:
        p = Path(file_path).expanduser().resolve()
        if not p.exists():
            return f"Error: Not found: {file_path}"

        if p.is_dir():
            files = list(_iter_source_files(p))
            if not files:
                return f"No source files found in {p}"
            total = blank = comment = code = 0
            for f in files:
                t, b, c, k = _count_file_lines(f)
                total += t; blank += b; comment += c; code += k
            return (
                f"Line count for {p} ({len(files)} files):\n"
                f"  Total:    {total:>8}\n"
                f"  Code:     {code:>8}\n"
                f"  Comment:  {comment:>8}\n"
                f"  Blank:    {blank:>8}"
            )

        t, b, c, k = _count_file_lines(p)
        return (
            f"Line count for {p}:\n"
            f"  Total:    {t:>8}\n"
            f"  Code:     {k:>8}\n"
            f"  Comment:  {c:>8}\n"
            f"  Blank:    {b:>8}"
        )
    except Exception as exc:
        return f"Error counting lines in {file_path}: {exc}"


def _count_file_lines(p: Path) -> tuple[int, int, int, int]:
    """Return (total, blank, comment, code) for a single file."""
    total = blank = comment = code = 0
    try:
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            total += 1
            if not stripped:
                blank += 1
            elif stripped.startswith(("#", "//", "/*", "*", "'''", '"""', "--", ";")):
                comment += 1
            else:
                code += 1
    except OSError:
        pass
    return total, blank, comment, code


# ---------------------------------------------------------------------------
# File outline
# ---------------------------------------------------------------------------

def get_file_outline(file_path: str) -> str:
    """Generate a hierarchical structural outline of a source file.

    Shows classes, functions, and methods with their nesting relationships and
    line numbers.  Python uses AST for accurate hierarchy; other languages use
    regex-based heuristics.

    Args:
        file_path: Absolute or relative path to the source file.

    Returns:
        Indented outline of file structure with line numbers, or an error
        message.
    """
    try:
        p = Path(file_path).expanduser().resolve()
        if not p.exists():
            return f"Error: File not found: {file_path}"

        suffix = p.suffix.lower()

        if suffix == ".py":
            return _python_outline(p)

        # Regex fallback
        return _regex_outline(p)
    except Exception as exc:
        return f"Error generating outline for {file_path}: {exc}"


def _python_outline(p: Path) -> str:
    """AST-based hierarchical outline for Python files."""
    try:
        source = p.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"Cannot generate outline — syntax error at line {e.lineno}: {e.msg}"

    lines: list[str] = [f"Outline: {p} (python)"]

    def _visit(node: ast.AST, indent: int = 0) -> None:
        prefix = "  " * indent
        if isinstance(node, ast.ClassDef):
            lines.append(f"{prefix}{node.lineno:>5}: class {node.name}")
            for child in ast.iter_child_nodes(node):
                _visit(child, indent + 1)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            args = [a.arg for a in node.args.args]
            lines.append(f"{prefix}{node.lineno:>5}: {kind} {node.name}({', '.join(args)})")
            for child in ast.iter_child_nodes(node):
                _visit(child, indent + 1)

    for child in ast.iter_child_nodes(tree):
        _visit(child, 0)

    return "\n".join(lines) if len(lines) > 1 else f"No structural symbols found in {p}"


def _regex_outline(p: Path) -> str:
    """Regex-based outline for non-Python files."""
    patterns: list[tuple[re.Pattern, str]] = [
        (re.compile(r"^\s*class\s+(\w+)"), "class"),
        (re.compile(r"^\s*(?:async\s+)?def\s+(\w+)\s*\("), "def"),
        (re.compile(r"^\s*function\s+(\w+)\s*\("), "function"),
        (re.compile(r"^\s*(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\("), "arrow"),
        (re.compile(r"^\s*func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\("), "func"),
        (re.compile(r"^\s*(?:public|private|protected).*\s+(\w+)\s*\("), "method"),
        (re.compile(r"^\s*interface\s+(\w+)"), "interface"),
        (re.compile(r"^\s*type\s+(\w+)\s+(?:struct|interface)\s*\{"), "type"),
    ]
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
        lang = _LANG_BY_EXT.get(p.suffix.lower(), p.suffix.lstrip("."))
        lines: list[str] = [f"Outline: {p} ({lang})"]
        for i, line in enumerate(text.splitlines(), start=1):
            for cpat, kind in patterns:
                m = cpat.match(line)
                if m:
                    lines.append(f"  {i:>5}: {kind} {m.group(1)}")
                    break
        return "\n".join(lines) if len(lines) > 1 else f"No structural symbols found in {p}"
    except OSError as exc:
        return f"Error reading {p}: {exc}"


# ---------------------------------------------------------------------------
# Grep
# ---------------------------------------------------------------------------

def grep_code(
    pattern: str,
    path: str = ".",
    file_type: str = "",
) -> str:
    """Search for a regex pattern across source files using ripgrep.

    Fast and accurate code search.  Use for finding function calls, variable
    usages, imports, string literals, or any pattern across a codebase.

    Args:
        pattern: Regular expression or literal string to search for.
        path: Root directory or file to search (default: current directory).
        file_type: Restrict search to this file type using ripgrep type names
            (e.g. 'py', 'js', 'ts', 'go') or language names ('python',
            'javascript').  Empty means search all files.

    Returns:
        Matching lines with file paths and line numbers, or an error message.
    """
    try:
        root = Path(path).expanduser().resolve()
        if not root.exists():
            return f"Error: Path not found: {path}"

        # Try ripgrep first
        cmd = ["rg", "--with-filename", "--line-number", "--no-heading",
               "--color=never", "-e", pattern]
        if file_type:
            # Normalise: "python" -> "py"
            _LANG_TO_RG = {
                "python": "py", "javascript": "js", "typescript": "ts",
                "java": "java", "go": "go", "rust": "rust",
                "ruby": "ruby", "csharp": "cs", "cpp": "cpp",
            }
            rg_type = _LANG_TO_RG.get(file_type.lower(), file_type)
            cmd.extend(["-t", rg_type])
        cmd.append(str(root))

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode in (0, 1):
                output = result.stdout.strip()
                if not output:
                    return f"No matches for pattern '{pattern}'"
                count = len(output.splitlines())
                return f"Found {count} match(es) for '{pattern}':\n\n{output}"
            # rg error — fall through to Python
        except FileNotFoundError:
            pass

        # Python fallback
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            return f"Error: Invalid regex: {e}"

        _EXT_MAP: dict[str, set[str]] = {
            "py": {".py"}, "js": {".js", ".mjs"}, "ts": {".ts", ".tsx"},
            "go": {".go"}, "java": {".java"}, "rs": {".rs"},
        }
        allowed_exts = _EXT_MAP.get(file_type.lower(), set()) if file_type else set()

        matches: list[str] = []
        for filepath in _iter_source_files(root):
            if allowed_exts and filepath.suffix.lower() not in allowed_exts:
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
            return f"No matches for pattern '{pattern}'"
        return f"Found {len(matches)} match(es) for '{pattern}':\n\n" + "\n".join(matches)
    except Exception as exc:
        return f"Error searching code: {exc}"


# ---------------------------------------------------------------------------
# Symbol reference finder
# ---------------------------------------------------------------------------

def find_symbol(symbol: str, path: str = ".") -> str:
    """Find all usages and references to a symbol across the codebase.

    Searches for every place where a symbol (function, class, variable) is
    used or referenced — not just its definition.  Uses whole-word matching
    to avoid partial hits.

    Args:
        symbol: Symbol name to search for (function, class, or variable name).
        path: Root directory to search (default: current directory).

    Returns:
        All reference locations with file path, line number, and the matching
        line, or an error message.
    """
    try:
        root = Path(path).expanduser().resolve()
        if not root.exists():
            return f"Error: Path not found: {path}"

        # Word-boundary pattern to avoid partial matches
        pattern = rf"\b{re.escape(symbol)}\b"

        # Try ripgrep
        cmd = ["rg", "--with-filename", "--line-number", "--no-heading",
               "--color=never", "-e", pattern, str(root)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode in (0, 1):
                output = result.stdout.strip()
                if not output:
                    return f"No references to '{symbol}' found in {root}"
                count = len(output.splitlines())
                return f"Found {count} reference(s) to '{symbol}':\n\n{output}"
        except FileNotFoundError:
            pass

        # Python fallback
        compiled = re.compile(pattern)
        matches: list[str] = []
        for filepath in _iter_source_files(root):
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
            return f"No references to '{symbol}' found in {root}"
        return f"Found {len(matches)} reference(s) to '{symbol}':\n\n" + "\n".join(matches)
    except Exception as exc:
        return f"Error finding symbol '{symbol}': {exc}"
