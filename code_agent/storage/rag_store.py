import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class RAGResult:
    content: str
    file_path: str
    symbol_name: str
    start_line: int
    end_line: int
    language: str
    score: float
    metadata: dict

class RAGStore:
    """
    LlamaIndex-based RAG store for code chunks.

    local mode:  SimpleVectorStore (in-memory, per-process)
    eks mode:    PGVectorStore (Aurora PostgreSQL + pgvector extension)

    Both modes use the same query/add interface so callers are deployment-agnostic.
    """

    def __init__(
        self,
        mode: str | None = None,           # None = read from DEPLOYMENT_MODE
        collection_name: str = "code_knowledge",
        embed_model=None,                   # LlamaIndex embedding model; None = build from config
    ):
        self._mode = mode or os.getenv("DEPLOYMENT_MODE", "local").lower()
        self._collection_name = collection_name
        self._index = None          # LlamaIndex VectorStoreIndex
        self._embed_model = embed_model
        self._vector_store = None   # underlying store
        self._setup_done = False

    def _ensure_setup(self):
        if self._setup_done:
            return
        if self._mode == "local":
            self._setup_local()
        else:
            self._setup_eks()
        self._setup_done = True

    def _setup_local(self):
        """In-memory LlamaIndex index with SimpleVectorStore."""
        from llama_index.core import VectorStoreIndex, StorageContext
        from llama_index.core.vector_stores import SimpleVectorStore

        embed = self._get_embed_model()
        vector_store = SimpleVectorStore()
        storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
        self._index = VectorStoreIndex(
            nodes=[],
            storage_context=storage_ctx,
            embed_model=embed,
        )
        self._vector_store = vector_store
        logger.info("RAGStore: local in-memory LlamaIndex index ready")

    def _setup_eks(self):
        """pgvector-backed LlamaIndex index on Aurora PostgreSQL."""
        from llama_index.core import VectorStoreIndex, StorageContext
        from llama_index.vector_stores.postgres import PGVectorStore

        db_url = os.getenv("DATABASE_URL", "")
        if not db_url:
            logger.warning("RAGStore EKS mode: DATABASE_URL not set, falling back to local")
            self._setup_local()
            return

        # Parse sync URL from asyncpg URL for PGVectorStore (it handles async internally)
        sync_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

        embed = self._get_embed_model()
        vector_store = PGVectorStore.from_params(
            database=_parse_db_name(sync_url),
            host=_parse_host(sync_url),
            password=_parse_password(sync_url),
            port=_parse_port(sync_url),
            user=_parse_user(sync_url),
            table_name=self._collection_name,
            embed_dim=768,   # gemini-embedding-001 default dim
        )
        storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
        self._index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_ctx,
            embed_model=embed,
        )
        self._vector_store = vector_store
        logger.info("RAGStore: pgvector LlamaIndex index ready (Aurora)")

    def _get_embed_model(self):
        """Build a LlamaIndex-compatible Google embedding model."""
        if self._embed_model:
            return self._embed_model
        api_key = os.getenv("GOOGLE_API_KEY", "")
        try:
            from llama_index.embeddings.google import GoogleGenerativeAIEmbedding
            return GoogleGenerativeAIEmbedding(
                model_name="models/gemini-embedding-001",
                api_key=api_key,
                embed_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "50")),
            )
        except ImportError:
            logger.warning("llama-index-embeddings-google not installed; using default embed model")
            from llama_index.core.embeddings import resolve_embed_model
            return resolve_embed_model("default")

    def add_chunks(self, chunks: list, metadata_extra: dict | None = None) -> None:
        """
        Index a list of CodeChunk objects into the RAG store.

        chunks: list of CodeChunk (from code_agent.indexer.chunker)
        metadata_extra: extra metadata applied to all chunks (e.g. repo, vcs, workspace)
        """
        self._ensure_setup()
        from llama_index.core.schema import TextNode

        nodes = []
        for chunk in chunks:
            meta = {
                "file_path": chunk.file_path,
                "language": chunk.language,
                "chunk_type": chunk.chunk_type,
                "symbol_name": chunk.symbol_name,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "signature": chunk.signature[:200] if chunk.signature else "",
            }
            if metadata_extra:
                meta.update(metadata_extra)

            node_id = f"{meta.get('repo', 'unknown')}::{chunk.file_path}::{chunk.start_line}"
            node = TextNode(
                text=f"{chunk.imports_context}\n{chunk.content}" if chunk.imports_context else chunk.content,
                id_=node_id,
                metadata=meta,
            )
            nodes.append(node)

        if nodes:
            self._index.insert_nodes(nodes)
            logger.debug("RAGStore: inserted %d nodes", len(nodes))

    def query(
        self,
        query_text: str,
        n_results: int = 10,
        filters: dict | None = None,
    ) -> list[RAGResult]:
        """
        Semantic search over indexed code chunks.

        filters: metadata key-value pairs to pre-filter results
                 e.g. {"language": "python", "repo": "auth-service"}
        """
        self._ensure_setup()
        from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter

        llama_filters = None
        if filters:
            llama_filters = MetadataFilters(
                filters=[ExactMatchFilter(key=k, value=v) for k, v in filters.items()]
            )

        retriever = self._index.as_retriever(
            similarity_top_k=n_results,
            filters=llama_filters,
        )
        nodes = retriever.retrieve(query_text)

        results = []
        for node in nodes:
            m = node.metadata
            results.append(RAGResult(
                content=node.text,
                file_path=m.get("file_path", ""),
                symbol_name=m.get("symbol_name", ""),
                start_line=int(m.get("start_line", 0)),
                end_line=int(m.get("end_line", 0)),
                language=m.get("language", ""),
                score=node.score or 0.0,
                metadata=m,
            ))
        return results

    def delete_by_repo(self, repo_id: str) -> None:
        """Remove all chunks for a given repo_id (for re-indexing)."""
        self._ensure_setup()
        # LlamaIndex doesn't have a bulk delete by metadata filter on all stores,
        # so we reconstruct. For PGVectorStore, use a direct SQL delete.
        if self._mode == "eks":
            try:
                self._vector_store.delete_nodes(
                    filters={"repo": repo_id}
                )
            except Exception as e:
                logger.warning("delete_by_repo failed: %s", e)
        # For local mode, in-memory store — rebuild index from scratch if needed

    def persist(self, path: str) -> None:
        """Persist the local in-memory index to disk (local mode only)."""
        if self._mode == "local" and self._index:
            self._index.storage_context.persist(persist_dir=path)
            logger.info("RAGStore: persisted to %s", path)

    def load(self, path: str) -> None:
        """Load a previously persisted local index from disk."""
        if self._mode != "local":
            return
        from llama_index.core import load_index_from_storage, StorageContext
        storage_ctx = StorageContext.from_defaults(persist_dir=path)
        embed = self._get_embed_model()
        self._index = load_index_from_storage(storage_ctx, embed_model=embed)
        self._setup_done = True
        logger.info("RAGStore: loaded from %s", path)


# ── URL parsing helpers ───────────────────────────────────────────────────────
# These parse postgresql://user:pass@host:port/dbname

def _parse_db_name(url: str) -> str:
    return url.rsplit("/", 1)[-1].split("?")[0]

def _parse_host(url: str) -> str:
    return url.split("@")[-1].split(":")[0].split("/")[0]

def _parse_port(url: str) -> int:
    try:
        part = url.split("@")[-1]
        if ":" in part.split("/")[0]:
            return int(part.split(":")[1].split("/")[0])
    except Exception:
        pass
    return 5432

def _parse_user(url: str) -> str:
    try:
        return url.split("://")[1].split(":")[0]
    except Exception:
        return ""

def _parse_password(url: str) -> str:
    try:
        return url.split("://")[1].split(":")[1].split("@")[0]
    except Exception:
        return ""
