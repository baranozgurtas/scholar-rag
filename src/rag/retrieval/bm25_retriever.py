"""In-memory BM25 retriever as a fallback / debugging tool.

The primary lexical retriever is `SparseRetriever` (BGE-M3 sparse via Qdrant),
which is more accurate and integrated. This `InMemoryBM25Retriever` exists for:

1. **Eval ablations** where we want a "pure BM25 baseline" untainted by
   BGE-M3's learned lexical weights.
2. **Debugging** when Qdrant sparse index returns unexpected results.
3. **Bootstrap** for environments where the sparse vector index isn't
   yet populated.

It builds the BM25 corpus by scrolling the entire Qdrant collection's
payloads — works fine for our 15-paper / few-thousand-chunk eval set.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from rag.logging_config import get_logger
from rag.retrieval.types import RetrievedChunk
from rag.vectorstore.qdrant_store import QdrantStore

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Lower-case word tokenizer (no stopword removal — keep behavior predictable)."""
    return _TOKEN_RE.findall(text.lower())


class InMemoryBM25Retriever:
    """BM25 retriever seeded from the Qdrant collection's payloads."""

    def __init__(self, store: QdrantStore) -> None:
        self.store = store
        self._bm25: BM25Okapi | None = None
        self._chunks: list[dict] = []

    def fit(self) -> InMemoryBM25Retriever:
        """Scroll Qdrant and build the BM25 index from chunk payloads."""
        self._chunks = self.store.scroll_all_payloads(batch=512)
        if not self._chunks:
            logger.warning("bm25_corpus_empty")
            self._bm25 = None
            return self
        tokenized = [_tokenize(c.get("text", "")) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("bm25_index_built", n_chunks=len(self._chunks))
        return self

    def retrieve(self, query: str, top_k: int = 30) -> list[RetrievedChunk]:
        """Return top-k BM25 hits."""
        if self._bm25 is None or not self._chunks:
            return []
        if not query or not query.strip():
            return []

        scores = self._bm25.get_scores(_tokenize(query))
        # top-k indices
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        out: list[RetrievedChunk] = []
        for i in top_idx:
            if scores[i] <= 0:
                continue
            payload = self._chunks[i]
            text = payload.get("text", "")
            chunk_id = payload.get("chunk_id", payload.get("id", str(i)))
            metadata = {k: v for k, v in payload.items() if k not in {"text", "id"}}
            out.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    score=float(scores[i]),
                    metadata=metadata,
                    score_breakdown={"bm25": float(scores[i])},
                )
            )
        logger.debug("bm25_retrieved", hits=len(out), top_k=top_k)
        return out


__all__ = ["InMemoryBM25Retriever"]
