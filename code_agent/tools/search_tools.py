"""Search tools — semantic, lexical, hybrid search, and repository indexing.

ADK tool functions wrapping the code_agent search and indexer layers.
All functions return plain strings consumed directly by LLM agents.
"""

from __future__ import annotations

import logging
from typing import Optional

from google.adk.tools import ToolContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_rag_store(collection_name: str | None = None):
    """Construct a RAGStore from application settings."""
    from code_agent.config import get_settings
    from code_agent.storage.rag_store import RAGStore

    cfg = get_settings()
    return RAGStore(
        mode=cfg.DEPLOYMENT_MODE,
        collection_name=collection_name or cfg.PGVECTOR_TABLE,
    )


def _build_vector_store(mode: str = "persistent"):
    """Construct a VectorStore from application settings (legacy ChromaDB fallback)."""
    from code_agent.config import get_settings
    from code_agent.search.vector_store import VectorStore

    cfg = get_settings()
    if mode == "memory":
        return VectorStore(mode="memory", collection_name=cfg.CHROMA_COLLECTION)
    return VectorStore(
        mode="persistent",
        path=cfg.CHROMA_PATH,
        collection_name=cfg.CHROMA_COLLECTION,
    )


def _build_embedder():
    """Construct a CodeEmbedder from application settings."""
    from code_agent.config import get_settings
    from code_agent.indexer.embedder import CodeEmbedder

    cfg = get_settings()
    if not cfg.GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY is not configured.  Set it in .env or as an "
            "environment variable before using semantic search."
        )
    return CodeEmbedder(
        api_key=cfg.GOOGLE_API_KEY,
        batch_size=cfg.EMBEDDING_BATCH_SIZE,
    )


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------

def semantic_search(
    query: str,
    repo_filter: str = "",
    language_filter: str = "",
    n_results: int = 10,
    tool_context: Optional[ToolContext] = None,
) -> str:
    """Search the global code knowledge base semantically.

    Uses vector embeddings to find code relevant to the query.
    Searches across all indexed repositories.
    Returns matching code snippets with file paths and line numbers.

    Best for: finding implementations, understanding design patterns, locating
    code by describing what it does rather than exact names.

    Args:
        query: Natural-language or code query describing what to find.
        repo_filter: Filter results to a specific repository slug
            (e.g. 'my-org/my-repo').  Empty means search all repos.
        language_filter: Filter results to a specific language
            (e.g. 'python', 'typescript').  Empty means all languages.
        n_results: Maximum number of results to return (default 10).
        tool_context: Optional ADK ToolContext for state access.

    Returns:
        Ranked list of matching code snippets with metadata, or an error
        message if the vector index is unavailable.
    """
    # Read active_repo from session state if repo_filter not explicitly set
    repo_filter = repo_filter or (tool_context.state.get("active_repo", "") if tool_context else "")

    from code_agent.config import get_settings
    cfg = get_settings()

    # Build optional metadata filters
    filters: dict | None = None
    filter_parts: dict = {}
    if repo_filter:
        filter_parts["repo"] = repo_filter
    if language_filter:
        filter_parts["language"] = language_filter.lower()
    if filter_parts:
        filters = filter_parts

    if cfg.RAG_BACKEND == "llamaindex":
        try:
            rag_store = _build_rag_store()
        except Exception as exc:
            return f"Error initialising RAGStore: {exc}"

        try:
            results = rag_store.query(
                query_text=query,
                n_results=max(1, n_results),
                filters=filters,
            )
        except Exception as exc:
            return f"Error querying RAGStore: {exc}"

        if not results:
            return f"No semantic matches found for: '{query}'"

        if tool_context is not None:
            tool_context.state["last_search"] = {"query": query, "results_count": len(results)}

        lines = [f"Semantic search results for '{query}' ({len(results)} hits):\n"]
        for i, r in enumerate(results, start=1):
            lines.append(f"[{i}] {r.file_path}  lines {r.start_line}–{r.end_line}"
                         f"  score={r.score:.3f}  lang={r.language}")
            if r.symbol_name:
                lines.append(f"     symbol: {r.symbol_name}")
            preview = r.content[:300].replace("\n", "\n     ")
            lines.append(f"     {preview}")
            lines.append("")
        return "\n".join(lines)

    # Legacy ChromaDB path
    try:
        store = _build_vector_store(mode="persistent")
        embedder = _build_embedder()
    except Exception as exc:
        return f"Error initialising search: {exc}"

    try:
        query_vec = embedder.embed_query(query)
    except Exception as exc:
        return f"Error embedding query: {exc}"

    # Build ChromaDB-style where filter
    where: dict | None = None
    chroma_filters: list[dict] = []
    if repo_filter:
        chroma_filters.append({"repo_id": {"$contains": repo_filter}})
    if language_filter:
        chroma_filters.append({"language": language_filter.lower()})
    if len(chroma_filters) == 1:
        where = chroma_filters[0]
    elif len(chroma_filters) > 1:
        where = {"$and": chroma_filters}

    try:
        results = store.query(
            query_embedding=query_vec,
            n_results=max(1, n_results),
            where=where,
        )
    except Exception as exc:
        return f"Error querying vector store: {exc}"

    if not results:
        return f"No semantic matches found for: '{query}'"

    if tool_context is not None:
        tool_context.state["last_search"] = {"query": query, "results_count": len(results)}

    lines = [f"Semantic search results for '{query}' ({len(results)} hits):\n"]
    for i, r in enumerate(results, start=1):
        lines.append(f"[{i}] {r.file_path}  lines {r.start_line}–{r.end_line}"
                     f"  score={r.score:.3f}  lang={r.language}")
        if r.symbol_name:
            lines.append(f"     symbol: {r.symbol_name}")
        preview = r.content[:300].replace("\n", "\n     ")
        lines.append(f"     {preview}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Lexical search
# ---------------------------------------------------------------------------

def lexical_search(
    pattern: str,
    local_path: str = ".",
    file_type: str = "",
    case_sensitive: bool = True,
    context_lines: int = 2,
    tool_context: Optional[ToolContext] = None,
) -> str:
    """Fast regex/literal search across files using ripgrep.

    Searches file contents for exact pattern matches.
    Use for: finding specific function names, variable names, strings, imports.
    Returns matching lines with file paths and line numbers.

    Args:
        pattern: Regular expression or literal string pattern to search for.
        local_path: Root directory to search (default: current directory).
        file_type: Restrict to this file type ('python', 'javascript', 'py',
            'js', etc.).  Empty means all files.
        case_sensitive: Use case-sensitive matching (default True).
        context_lines: Number of surrounding lines to include with each match
            (default 2).
        tool_context: Optional ADK ToolContext for state access.

    Returns:
        Matching lines with context and file locations, or an error message.
    """
    from pathlib import Path as _Path

    try:
        from code_agent.search.lexical_search import LexicalSearcher
    except ImportError as exc:
        return f"Error importing LexicalSearcher: {exc}"

    root = _Path(local_path).expanduser().resolve()
    if not root.exists():
        return f"Error: Path not found: {local_path}"

    try:
        searcher = LexicalSearcher()
        matches = searcher.search(
            pattern=pattern,
            path=str(root),
            file_type=file_type if file_type else None,
            case_sensitive=case_sensitive,
            max_results=100,
            context_lines=context_lines,
        )
    except Exception as exc:
        return f"Error during lexical search: {exc}"

    if not matches:
        return f"No matches for '{pattern}' in {root}"

    if tool_context is not None:
        tool_context.state["last_search"] = {"query": pattern, "results_count": len(matches)}

    lines = [f"Lexical search results for '{pattern}' ({len(matches)} matches):\n"]
    for m in matches:
        lines.append(f"{m.file_path}:{m.line_number}: {m.line_content.rstrip()}")
        for ctx in m.context_before:
            lines.append(f"  | {ctx}")
        lines.append(f"  > {m.line_content.rstrip()}")
        for ctx in m.context_after:
            lines.append(f"  | {ctx}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------

def hybrid_search(
    query: str,
    local_path: str = ".",
    repo_filter: str = "",
    n_results: int = 10,
) -> str:
    """Combined semantic + lexical search using Reciprocal Rank Fusion.

    Best of both worlds: finds semantically similar code AND exact matches.
    Use this as the default search for understanding code.
    Returns ranked results from both search methods.

    Semantic search finds code conceptually related to the query even when
    exact terms don't match.  Lexical search catches exact identifier matches.
    RRF merges and re-ranks both result lists.

    Args:
        query: Search query — can be natural language or code identifiers.
        local_path: Local repository path for lexical search component.
            Pass '.' or the repo root to enable lexical search.
        repo_filter: Filter semantic results to a specific repository slug.
            Empty means search all indexed repositories.
        n_results: Maximum results to return (default 10).

    Returns:
        Ranked code snippets from both search methods with source indicators,
        or an error message.
    """
    from pathlib import Path as _Path

    try:
        from code_agent.search.hybrid_search import HybridSearcher
        from code_agent.search.lexical_search import LexicalSearcher
    except ImportError as exc:
        return f"Error importing search modules: {exc}"

    try:
        store = _build_vector_store(mode="persistent")
        embedder = _build_embedder()
    except Exception as exc:
        return f"Error initialising search: {exc}"

    root = _Path(local_path).expanduser().resolve()
    path_arg = str(root) if root.exists() else None

    where: dict | None = None
    if repo_filter:
        where = {"repo_id": {"$contains": repo_filter}}

    try:
        searcher = HybridSearcher(
            vector_store=store,
            embedder=embedder,
            lexical_searcher=LexicalSearcher(),
        )
        results = searcher.search(
            query=query,
            local_path=path_arg,
            n_results=max(1, n_results),
            where=where,
        )
    except Exception as exc:
        return f"Error during hybrid search: {exc}"

    if not results:
        return f"No results for '{query}'"

    lines = [f"Hybrid search results for '{query}' ({len(results)} hits):\n"]
    for i, r in enumerate(results, start=1):
        lines.append(
            f"[{i}] {r.file_path}  lines {r.start_line}–{r.end_line}"
            f"  rrf={r.rrf_score:.4f}  source={r.source}  lang={r.language}"
        )
        if r.symbol_name:
            lines.append(f"     symbol: {r.symbol_name}")
        preview = r.content[:300].replace("\n", "\n     ")
        lines.append(f"     {preview}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Symbol reference finder
# ---------------------------------------------------------------------------

def find_symbol_references(
    symbol: str,
    local_path: str = ".",
    language: str = "",
) -> str:
    """Find all usages/references to a symbol (function, class, variable) across the codebase.

    Searches for all places where a symbol is called or referenced.
    Returns file paths, line numbers, and surrounding context.

    Uses word-boundary matching to avoid partial matches (e.g. searching
    'get_user' won't match 'get_user_by_id').

    Args:
        symbol: Identifier to find (function name, class name, variable name).
        local_path: Root directory to search (default: current directory).
        language: Optional language filter to restrict file types
            (e.g. 'python', 'javascript').

    Returns:
        All reference locations with line context, or an error message.
    """
    from pathlib import Path as _Path

    try:
        from code_agent.search.lexical_search import LexicalSearcher
    except ImportError as exc:
        return f"Error importing LexicalSearcher: {exc}"

    root = _Path(local_path).expanduser().resolve()
    if not root.exists():
        return f"Error: Path not found: {local_path}"

    try:
        searcher = LexicalSearcher()
        matches = searcher.find_symbol_references(
            symbol=symbol,
            path=str(root),
            language=language if language else None,
        )
    except Exception as exc:
        return f"Error finding references: {exc}"

    if not matches:
        return f"No references to '{symbol}' found in {root}"

    lines = [f"References to '{symbol}' ({len(matches)} found):\n"]
    for m in matches:
        lines.append(f"{m.file_path}:{m.line_number}: {m.line_content.rstrip()}")
        for ctx in m.context_before:
            lines.append(f"  | {ctx}")
        lines.append(f"  > {m.line_content.rstrip()}")
        for ctx in m.context_after:
            lines.append(f"  | {ctx}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Repository indexer
# ---------------------------------------------------------------------------

def index_local_repository(
    local_path: str,
    repo_id: str = "",
    mode: str = "memory",
    languages: str = "",
) -> str:
    """Index a local repository into memory for semantic search.

    Parses code files, creates embeddings, and stores in vector index.
    Uses RAGStore (LlamaIndex) when RAG_BACKEND=llamaindex (default),
    or ChromaDB when RAG_BACKEND=chromadb (legacy).
    Works in both local (in-memory) and EKS (pgvector) modes automatically.
    Use this before semantic_search when working with a freshly cloned repo.
    Returns indexing statistics.

    Args:
        local_path: Absolute or relative path to the local repository root.
        repo_id: Stable identifier for this repository used as the index key
            (e.g. 'github/my-org/my-repo').  Derived from the directory name
            if not provided.
        mode: Storage mode — 'memory' for a temporary in-session index or
            'persistent' to save to disk for reuse across sessions.
            In EKS mode, this parameter is ignored and pgvector is always used.
        languages: Comma-separated list of languages to index
            (e.g. 'python,javascript').  Empty means all supported languages.

    Returns:
        Indexing statistics including files processed, chunks created, and any
        errors encountered.
    """
    from pathlib import Path as _Path
    from code_agent.config import get_settings

    cfg = get_settings()

    root = _Path(local_path).expanduser().resolve()
    if not root.exists():
        return f"Error: Path not found: {local_path}"
    if not root.is_dir():
        return f"Error: Not a directory: {local_path}"

    # Derive repo_id from directory name if not given
    effective_repo_id = repo_id.strip() or root.name

    # Parse language filter
    lang_list: list[str] | None = None
    if languages.strip():
        lang_list = [l.strip().lower() for l in languages.split(",") if l.strip()]

    if cfg.RAG_BACKEND == "llamaindex":
        # Use RAGStore (LlamaIndex) path
        try:
            from code_agent.indexer.chunker import CodeChunker
        except ImportError as exc:
            return f"Error importing CodeChunker: {exc}"

        try:
            rag_store = _build_rag_store()
        except Exception as exc:
            return f"Error initialising RAGStore: {exc}"

        try:
            chunker = CodeChunker()
            chunks = chunker.chunk_repository(
                local_path=str(root),
                languages=lang_list,
            )
        except Exception as exc:
            return f"Error chunking repository: {exc}"

        try:
            rag_store.add_chunks(
                chunks=chunks,
                metadata_extra={
                    "repo": effective_repo_id,
                    "local_path": str(root),
                },
            )
        except Exception as exc:
            return f"Error adding chunks to RAGStore: {exc}"

        total_files = len({c.file_path for c in chunks})
        return "\n".join([
            f"Indexing complete for '{effective_repo_id}' (RAGStore / {cfg.DEPLOYMENT_MODE} mode):",
            f"  Path:           {root}",
            f"  Files indexed:  {total_files}",
            f"  Chunks created: {len(chunks)}",
            f"  Backend:        {cfg.RAG_BACKEND} ({cfg.DEPLOYMENT_MODE})",
        ])

    # Legacy ChromaDB path
    if mode not in ("memory", "persistent"):
        return f"Error: mode must be 'memory' or 'persistent', got '{mode}'"

    try:
        from code_agent.indexer.repository_indexer import RepositoryIndexer
    except ImportError as exc:
        return f"Error importing RepositoryIndexer: {exc}"

    try:
        store = _build_vector_store(mode=mode)
        embedder = _build_embedder()
    except Exception as exc:
        return f"Error initialising indexer: {exc}"

    try:
        indexer = RepositoryIndexer(vector_store=store, embedder=embedder)
        result = indexer.index_repository(
            local_path=str(root),
            repo_id=effective_repo_id,
            metadata={"local_path": str(root), "mode": mode},
            mode="full",
            languages=lang_list,
        )
    except Exception as exc:
        return f"Error indexing repository: {exc}"

    lines = [
        f"Indexing complete for '{effective_repo_id}' ({mode} mode):",
        f"  Path:          {root}",
        f"  Files found:   {result.total_files}",
        f"  Files indexed: {result.indexed_files}",
        f"  Chunks created: {result.total_chunks}",
        f"  Files skipped: {result.skipped_files}",
        f"  Duration:      {result.duration_seconds:.1f}s",
    ]
    if result.last_commit_sha:
        lines.append(f"  HEAD SHA:      {result.last_commit_sha}")
    if result.errors:
        lines.append(f"  Errors ({len(result.errors)}):")
        for err in result.errors[:10]:
            lines.append(f"    - {err}")
        if len(result.errors) > 10:
            lines.append(f"    ... and {len(result.errors) - 10} more")
    return "\n".join(lines)
