"""Sparse (lexical) retrieval using BGE-M3 sparse weights via Qdrant.

BGE-M3 produces BM25-style sparse vectors as part of its multi-functional
output. We store those alongside dense vectors in Qdrant and query the
sparse index directly — no separate BM25 service needed.

This is the lexical leg of hybrid retrieval, complementing dense:
- Dense catches paraphrases ("transformer architecture" ≈ "self-attention model")
- Sparse catches rare proper nouns ("BGE-M3", "TriviaQA", "Adam optimizer")
"""

from __future__ import annotations

from rag.embeddings.bge_embedder import BGEEmbedder
from rag.logging_config import get_logger
from rag.retrieval.types import RetrievedChunk
from rag.vectorstore.qdrant_store import QdrantStore

logger = get_logger(__name__)


class SparseRetriever:
    """Sparse retrieval via Qdrant sparse vectors (BGE-M3 lexical weights)."""

    def __init__(self, store: QdrantStore, embedder: BGEEmbedder) -> None:
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 30) -> list[RetrievedChunk]:
        """Return top-k sparse hits."""
        if not query or not query.strip():
            return []

        _, sparse_list = self.embedder.embed_dense_and_sparse([query], is_query=True)
        if not sparse_list or not sparse_list[0]:
            logger.debug("sparse_empty_query", query=query[:80])
            return []
        query_sparse = sparse_list[0]

        hits = self.store.search_sparse(query_sparse=query_sparse, top_k=top_k)
        out: list[RetrievedChunk] = []
        for hit in hits:
            payload = hit.payload or {}
            text = payload.get("text", "")
            chunk_id = payload.get("chunk_id", str(hit.id))
            metadata = {k: v for k, v in payload.items() if k != "text"}
            out.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    score=float(hit.score),
                    metadata=metadata,
                    score_breakdown={"sparse": float(hit.score)},
                )
            )
        logger.debug("sparse_retrieved", query_len=len(query), hits=len(out), top_k=top_k)
        return out


__all__ = ["SparseRetriever"]
