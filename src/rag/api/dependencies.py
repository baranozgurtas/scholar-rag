"""FastAPI dependency providers.

These build the heavy singletons (BGE embedder, reranker, Qdrant client,
RAGChain) once at process start and inject them into routes via Depends.

Heavy components are lazy-initialized in `init_state()` called from the
app lifespan, so module-import doesn't pull in torch/transformers.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.config import Settings, get_settings
from rag.embeddings.bge_embedder import BGEEmbedder, build_embedder
from rag.generation.rag_chain import RAGChain, build_rag_chain_from_settings
from rag.logging_config import get_logger
from rag.observability.langfuse_tracer import LangfuseTracer
from rag.observability.token_counter import TokenLedger
from rag.retrieval.dense_retriever import DenseRetriever
from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.retrieval.reranker import CrossEncoderReranker
from rag.retrieval.sparse_retriever import SparseRetriever
from rag.vectorstore.qdrant_store import QdrantStore

logger = get_logger(__name__)


@dataclass
class AppState:
    """Process-wide singletons injected into routes."""

    settings: Settings
    embedder: BGEEmbedder
    store: QdrantStore
    reranker: CrossEncoderReranker
    rag_chain: RAGChain
    tracer: LangfuseTracer
    token_ledger: TokenLedger


_state: AppState | None = None


def init_state() -> AppState:
    """Build the AppState singleton. Called once from the FastAPI lifespan."""
    global _state
    if _state is not None:
        return _state

    settings = get_settings()
    logger.info("init_state_start")

    embedder = build_embedder(settings.embedding)
    store = QdrantStore(embedder=embedder, settings=settings.vectorstore)
    store.ensure_collection(recreate=False)

    dense_retr = DenseRetriever(store=store, embedder=embedder)
    sparse_retr = SparseRetriever(store=store, embedder=embedder)
    hybrid = HybridRetriever(dense=dense_retr, sparse=sparse_retr)

    reranker = CrossEncoderReranker(settings=settings.embedding)
    rag_chain = build_rag_chain_from_settings(
        hybrid=hybrid, reranker=reranker, use_reranker=True
    )

    tracer = LangfuseTracer(settings=settings.observability)
    token_ledger = TokenLedger()

    _state = AppState(
        settings=settings,
        embedder=embedder,
        store=store,
        reranker=reranker,
        rag_chain=rag_chain,
        tracer=tracer,
        token_ledger=token_ledger,
    )
    logger.info("init_state_complete")
    return _state


def get_state() -> AppState:
    """FastAPI dependency: return the initialized AppState."""
    if _state is None:
        # Defensive: should never happen in normal lifespan flow
        return init_state()
    return _state


def reset_state() -> None:
    """Test helper: tear down the singleton."""
    global _state
    _state = None


__all__ = ["AppState", "get_state", "init_state", "reset_state"]
