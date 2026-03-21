"""AST-based code chunker using tree-sitter.

Splits source files into semantically meaningful chunks (functions, classes)
using tree-sitter for supported languages, with a line-window fallback for
everything else.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extension → language mapping
# ---------------------------------------------------------------------------
_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".swift": "swift",
    ".php": "php",
    ".scala": "scala",
    ".sh": "shell",
    ".bash": "shell",
    ".md": "markdown",
    ".mdx": "markdown",
}

# Languages with tree-sitter support in this module
_TREESITTER_LANGS: frozenset[str] = frozenset(
    ["python", "javascript", "typescript", "java", "go", "rust", "ruby", "c", "cpp"]
)

# tree-sitter query patterns per language.  Each entry is a list of
# (node_type_query_string, chunk_type_label) tuples.
_LANG_QUERIES: dict[str, str] = {
    "python": """
        (function_definition name: (identifier) @name) @definition.function
        (class_definition name: (identifier) @name) @definition.class
    """,
    "javascript": """
        (function_declaration name: (identifier) @name) @definition.function
        (method_definition name: (property_identifier) @name) @definition.function
        (arrow_function) @definition.function
        (class_declaration name: (identifier) @name) @definition.class
    """,
    "typescript": """
        (function_declaration name: (identifier) @name) @definition.function
        (method_definition name: (property_identifier) @name) @definition.function
        (arrow_function) @definition.function
        (class_declaration name: (identifier) @name) @definition.class
        (interface_declaration name: (type_identifier) @name) @definition.class
    """,
    "java": """
        (method_declaration name: (identifier) @name) @definition.function
        (class_declaration name: (identifier) @name) @definition.class
        (interface_declaration name: (identifier) @name) @definition.class
    """,
    "go": """
        (function_declaration name: (identifier) @name) @definition.function
        (method_declaration name: (field_identifier) @name) @definition.function
        (type_declaration (type_spec name: (type_identifier) @name)) @definition.class
    """,
    "rust": """
        (function_item name: (identifier) @name) @definition.function
        (impl_item) @definition.class
        (struct_item name: (type_identifier) @name) @definition.class
        (enum_item name: (type_identifier) @name) @definition.class
        (trait_item name: (type_identifier) @name) @definition.class
    """,
    "ruby": """
        (method name: (identifier) @name) @definition.function
        (singleton_method name: (identifier) @name) @definition.function
        (class name: (constant) @name) @definition.class
        (module name: (constant) @name) @definition.class
    """,
    "c": """
        (function_definition declarator: (function_declarator declarator: (identifier) @name)) @definition.function
        (struct_specifier name: (type_identifier) @name) @definition.class
    """,
    "cpp": """
        (function_definition declarator: (function_declarator declarator: (identifier) @name)) @definition.function
        (function_definition declarator: (function_declarator declarator: (qualified_identifier) @name)) @definition.function
        (class_specifier name: (type_identifier) @name) @definition.class
        (struct_specifier name: (type_identifier) @name) @definition.class
    """,
}

# Import node types by language (used to collect context)
_IMPORT_NODE_TYPES: dict[str, list[str]] = {
    "python": ["import_statement", "import_from_statement"],
    "javascript": ["import_statement", "import_declaration"],
    "typescript": ["import_statement", "import_declaration"],
    "java": ["import_declaration"],
    "go": ["import_declaration"],
    "rust": ["use_declaration"],
    "ruby": ["call"],  # require/include calls — handled specially
    "c": ["preproc_include"],
    "cpp": ["preproc_include"],
}

# File extensions that are always binary / never text
_BINARY_EXTENSIONS: frozenset[str] = frozenset(
    [
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg",
        ".pdf", ".doc", ".docx", ".xls", ".xlsx",
        ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
        ".exe", ".dll", ".so", ".dylib", ".a", ".lib",
        ".woff", ".woff2", ".ttf", ".otf", ".eot",
        ".mp3", ".mp4", ".wav", ".ogg", ".flac", ".avi", ".mov",
        ".pyc", ".pyo", ".class", ".jar",
    ]
)

_MAX_FILE_CHARS = 50_000
_FALLBACK_CHUNK_SIZE = 100
_FALLBACK_CHUNK_OVERLAP = 20


@dataclass
class CodeChunk:
    """A self-contained unit of source code ready for embedding."""

    content: str
    """The actual code text (may include import context prefix)."""

    file_path: str
    """Relative path within the repository."""

    language: str
    """Detected language (e.g. 'python', 'javascript')."""

    chunk_type: str
    """Structural type: 'function', 'class', 'module', or 'block'."""

    symbol_name: str
    """Function or class name, empty string for file-level chunks."""

    start_line: int
    """1-based start line in the original file."""

    end_line: int
    """1-based end line in the original file."""

    imports_context: str
    """Relevant import statements prepended for retrieval context."""

    signature: str
    """First line of the definition (e.g. 'def foo(x, y):')."""


class CodeChunker:
    """Splits source files into :class:`CodeChunk` objects for indexing.

    Uses tree-sitter to extract function and class definitions for
    supported languages.  Falls back to a sliding-window line splitter
    for unsupported languages.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_language(self, file_path: str) -> str | None:
        """Return the language name for *file_path* based on its extension.

        Returns ``None`` if the extension is not recognised or is binary.
        """
        suffix = Path(file_path).suffix.lower()
        if suffix in _BINARY_EXTENSIONS:
            return None
        return _EXT_TO_LANG.get(suffix)

    def chunk_file(self, file_path: str, content: str) -> list[CodeChunk]:
        """Split *content* into :class:`CodeChunk` objects.

        Args:
            file_path: Relative path used for metadata (not read from disk).
            content:   Full text of the file.

        Returns:
            List of chunks; empty if the file is too large or unreadable.
        """
        if len(content) > _MAX_FILE_CHARS:
            logger.debug("Skipping %s — exceeds %d char limit", file_path, _MAX_FILE_CHARS)
            return []

        language = self.detect_language(file_path)
        if language is None:
            logger.debug("Skipping %s — unsupported or binary extension", file_path)
            return []

        if language in _TREESITTER_LANGS:
            try:
                return self._chunk_with_treesitter(file_path, content, language)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "tree-sitter parse failed for %s (%s): %s — falling back to line chunks",
                    file_path, language, exc,
                )

        return self._chunk_by_lines(file_path, content, language)

    # ------------------------------------------------------------------
    # tree-sitter chunking
    # ------------------------------------------------------------------

    def _load_language(self, language: str):  # type: ignore[return]
        """Lazily import and return a tree-sitter Language object.

        Raises ImportError if the language binding is not installed.
        """
        if language == "python":
            import tree_sitter_python as ts_lang  # type: ignore[import]
        elif language == "javascript":
            import tree_sitter_javascript as ts_lang  # type: ignore[import]
        elif language == "typescript":
            import tree_sitter_typescript as ts_lang  # type: ignore[import]
            # tree-sitter-typescript exposes tsx and typescript sub-modules
            if hasattr(ts_lang, "language_typescript"):
                ts_lang = type("_M", (), {"language": ts_lang.language_typescript})()
            elif hasattr(ts_lang, "typescript"):
                ts_lang = type("_M", (), {"language": ts_lang.typescript})()
        elif language == "java":
            import tree_sitter_java as ts_lang  # type: ignore[import]
        elif language == "go":
            import tree_sitter_go as ts_lang  # type: ignore[import]
        elif language == "rust":
            import tree_sitter_rust as ts_lang  # type: ignore[import]
        elif language == "ruby":
            import tree_sitter_ruby as ts_lang  # type: ignore[import]
        elif language == "c":
            import tree_sitter_c as ts_lang  # type: ignore[import]
        elif language == "cpp":
            import tree_sitter_cpp as ts_lang  # type: ignore[import]
        else:
            raise ImportError(f"No tree-sitter binding for language: {language}")

        from tree_sitter import Language  # type: ignore[import]

        if callable(getattr(ts_lang, "language", None)):
            return Language(ts_lang.language())
        # Older API: Language(ts_lang.language, "python")
        raise ImportError(f"Cannot construct Language for {language}")

    def _chunk_with_treesitter(
        self, file_path: str, content: str, language: str
    ) -> list[CodeChunk]:
        """Parse *content* with tree-sitter and extract named definitions."""
        from tree_sitter import Parser  # type: ignore[import]

        ts_language = self._load_language(language)
        parser = Parser(ts_language)
        tree = parser.parse(content.encode("utf-8"))

        lines = content.splitlines()
        import_lines = self._collect_import_lines(tree.root_node, language, lines)

        query_src = _LANG_QUERIES.get(language, "")
        if not query_src.strip():
            return self._chunk_by_lines(file_path, content, language)

        try:
            query = ts_language.query(query_src)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Query build failed for %s: %s", language, exc)
            return self._chunk_by_lines(file_path, content, language)

        captures = query.captures(tree.root_node)

        # captures is dict[str, list[Node]] in newer tree-sitter
        # or list[(Node, str)] in older versions
        definition_nodes: list[tuple[object, str]] = []

        if isinstance(captures, dict):
            for capture_name, nodes in captures.items():
                if capture_name.startswith("definition."):
                    chunk_type = capture_name.split(".", 1)[1]  # "function" or "class"
                    for node in nodes:
                        definition_nodes.append((node, chunk_type))
        else:
            for node, capture_name in captures:
                if capture_name.startswith("definition."):
                    chunk_type = capture_name.split(".", 1)[1]
                    definition_nodes.append((node, chunk_type))

        if not definition_nodes:
            # File has no definitions — treat as single module chunk
            return self._make_module_chunk(file_path, content, language, lines)

        chunks: list[CodeChunk] = []
        seen_ranges: set[tuple[int, int]] = set()

        for node, chunk_type in sorted(definition_nodes, key=lambda x: x[0].start_point[0]):  # type: ignore[union-attr]
            start_row = node.start_point[0]  # type: ignore[union-attr]
            end_row = node.end_point[0]  # type: ignore[union-attr]

            range_key = (start_row, end_row)
            if range_key in seen_ranges:
                continue
            seen_ranges.add(range_key)

            start_line = start_row + 1  # 1-based
            end_line = end_row + 1

            node_text = node.text.decode("utf-8") if node.text else "\n".join(lines[start_row:end_row + 1])  # type: ignore[union-attr]
            first_line = lines[start_row] if start_row < len(lines) else ""
            symbol_name = self._extract_symbol_name(node, language)

            chunks.append(
                CodeChunk(
                    content=node_text,
                    file_path=file_path,
                    language=language,
                    chunk_type=chunk_type,
                    symbol_name=symbol_name,
                    start_line=start_line,
                    end_line=end_line,
                    imports_context=import_lines,
                    signature=first_line.strip(),
                )
            )

        return chunks if chunks else self._make_module_chunk(file_path, content, language, lines)

    def _collect_import_lines(self, root_node, language: str, lines: list[str]) -> str:
        """Return a string of all import/include lines in the file."""
        import_types = _IMPORT_NODE_TYPES.get(language, [])
        if not import_types:
            return ""

        collected: list[str] = []
        for child in root_node.children:
            if child.type in import_types:
                row = child.start_point[0]
                end_row = child.end_point[0]
                collected.extend(lines[row:end_row + 1])

        return "\n".join(collected)

    def _extract_symbol_name(self, node, language: str) -> str:  # type: ignore[return]
        """Best-effort extraction of the symbol name from a definition node."""
        # Walk direct children looking for a 'name' or identifier child
        name_types = {"identifier", "property_identifier", "type_identifier",
                      "field_identifier", "constant", "qualified_identifier"}
        for child in node.children:
            if child.type in name_types:
                return child.text.decode("utf-8") if child.text else ""
        # Recurse one level for compound declarators (C/C++)
        for child in node.children:
            for grandchild in child.children:
                if grandchild.type in name_types:
                    return grandchild.text.decode("utf-8") if grandchild.text else ""
        return ""

    def _make_module_chunk(
        self, file_path: str, content: str, language: str, lines: list[str]
    ) -> list[CodeChunk]:
        """Wrap an entire file as a single 'module' chunk."""
        return [
            CodeChunk(
                content=content,
                file_path=file_path,
                language=language,
                chunk_type="module",
                symbol_name="",
                start_line=1,
                end_line=len(lines),
                imports_context="",
                signature=lines[0].strip() if lines else "",
            )
        ]

    # ------------------------------------------------------------------
    # Line-window fallback
    # ------------------------------------------------------------------

    def _chunk_by_lines(
        self, file_path: str, content: str, language: str
    ) -> list[CodeChunk]:
        """Sliding-window chunking for unsupported languages.

        Produces windows of :data:`_FALLBACK_CHUNK_SIZE` lines with
        :data:`_FALLBACK_CHUNK_OVERLAP` lines of overlap.
        """
        lines = content.splitlines()
        if not lines:
            return []

        chunks: list[CodeChunk] = []
        step = _FALLBACK_CHUNK_SIZE - _FALLBACK_CHUNK_OVERLAP
        i = 0

        while i < len(lines):
            window = lines[i: i + _FALLBACK_CHUNK_SIZE]
            start_line = i + 1
            end_line = i + len(window)
            chunk_text = "\n".join(window)

            chunks.append(
                CodeChunk(
                    content=chunk_text,
                    file_path=file_path,
                    language=language,
                    chunk_type="block",
                    symbol_name="",
                    start_line=start_line,
                    end_line=end_line,
                    imports_context="",
                    signature=window[0].strip() if window else "",
                )
            )
            i += step

        return chunks
