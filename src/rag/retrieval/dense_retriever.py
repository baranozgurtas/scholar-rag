"""Dense retrieval over the Qdrant `dense` named vector.

This is the bi-encoder retrieval leg of the hybrid pipeline. The query
is encoded by BGE-M3 (dense, 1024-d, L2-normalized) and we run cosine
similarity ANN search against the collection.
"""

from __future__ import annotations

from rag.embeddings.bge_embedder import BGEEmbedder
from rag.logging_config import get_logger
from rag.retrieval.types import RetrievedChunk
from rag.vectorstore.qdrant_store import QdrantStore

logger = get_logger(__name__)


class DenseRetriever:
    """Dense bi-encoder retrieval via Qdrant cosine ANN."""

    def __init__(self, store: QdrantStore, embedder: BGEEmbedder) -> None:
        self.store = store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 30,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """Return top-k dense hits as RetrievedChunk objects."""
        if not query or not query.strip():
            return []

        query_vec = self.embedder.embed_query(query)
        hits = self.store.search_dense(
            query_vector=query_vec,
            top_k=top_k,
            score_threshold=score_threshold,
        )

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
                    score_breakdown={"dense": float(hit.score)},
                )
            )
        logger.debug("dense_retrieved", query_len=len(query), hits=len(out), top_k=top_k)
        return out


__all__ = ["DenseRetriever"]
