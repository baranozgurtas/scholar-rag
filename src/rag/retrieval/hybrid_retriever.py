"""Hybrid retriever: Reciprocal Rank Fusion (RRF) of dense + sparse.

RRF is the de-facto standard for combining retrievers because it:
1. Requires no score normalization (scores from different retrievers can have
   wildly different scales — cosine in [-1, 1], BM25 unbounded).
2. Is parameter-light (single `k` constant, typically 60).
3. Is provably robust — small perturbations to either retriever's ranking
   change the fused result minimally.

Formula (per chunk c):
    rrf_score(c) = sum_over_retrievers( 1 / (k + rank_in_that_retriever(c)) )

We additionally support **per-retriever weighting** (dense_weight, sparse_weight)
for ablation studies — defaults to 1.0 each (standard RRF).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from rag.config import RetrievalSettings, get_settings
from rag.logging_config import get_logger
from rag.retrieval.dense_retriever import DenseRetriever
from rag.retrieval.sparse_retriever import SparseRetriever
from rag.retrieval.types import EmptySparseQuery, RetrievalFailure, RetrievedChunk

logger = get_logger(__name__)

RETRIEVAL_MODE_HYBRID = "hybrid"
# Sparse encoding was valid but had no positive weights; dense leg only.
RETRIEVAL_MODE_DENSE_ONLY = "dense_only"


@dataclass
class HybridRetrievalConfig:
    """Hybrid retrieval knobs, exposed for ablation."""

    dense_top_k: int
    sparse_top_k: int
    rrf_k: int
    final_top_k: int  # how many to pass downstream (to reranker)
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    use_dense: bool = True
    use_sparse: bool = True

    @classmethod
    def from_settings(cls, s: RetrievalSettings | None = None) -> HybridRetrievalConfig:
        s = s or get_settings().retrieval
        return cls(
            dense_top_k=s.dense_top_k,
            sparse_top_k=s.sparse_top_k,
            rrf_k=s.rrf_k,
            final_top_k=s.hybrid_top_k,
        )


class HybridRetriever:
    """Dense + sparse retrieval fused with Reciprocal Rank Fusion."""

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        config: HybridRetrievalConfig | None = None,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.config = config or HybridRetrievalConfig.from_settings()

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Run dense + sparse retrieval and fuse via RRF (see `retrieve_with_mode`)."""
        return self.retrieve_with_mode(query, top_k)[0]

    def retrieve_with_mode(
        self, query: str, top_k: int | None = None
    ) -> tuple[list[RetrievedChunk], str]:
        """Run dense + sparse retrieval, fuse via RRF, and report the mode used.

        Args:
            query: User query.
            top_k: Override the configured final_top_k.

        Returns:
            (chunks, mode). `mode` is RETRIEVAL_MODE_HYBRID, or
            RETRIEVAL_MODE_DENSE_ONLY when the query's sparse encoding was
            valid but empty, so only the dense leg could search.

        Raises:
            RetrievalFailure: an enabled leg raised, returned a non-finite
                score, or returned nothing while the other leg returned hits.
                Fusing the surviving leg would look like a normal result.
        """
        cfg = self.config
        final_top_k = top_k if top_k is not None else cfg.final_top_k
        mode = RETRIEVAL_MODE_HYBRID

        dense_hits = (
            self._run_leg("dense", self.dense.retrieve, query, cfg.dense_top_k)
            if cfg.use_dense else []
        )
        sparse_hits: list[RetrievedChunk] = []
        if cfg.use_sparse:
            try:
                sparse_hits = self._run_leg("sparse", self.sparse.retrieve, query, cfg.sparse_top_k)
            except EmptySparseQuery:
                logger.warning("sparse_query_empty_dense_only", query=query[:80])
                mode = RETRIEVAL_MODE_DENSE_ONLY
        if (
            mode == RETRIEVAL_MODE_HYBRID
            and cfg.use_dense
            and cfg.use_sparse
            and bool(dense_hits) != bool(sparse_hits)
        ):
            empty = "dense" if not dense_hits else "sparse"
            logger.error("retrieval_leg_empty", leg=empty, query=query[:80])
            raise RetrievalFailure(empty, "returned no hits while the other leg did")

        fused = self._rrf_fuse(
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            k=cfg.rrf_k,
            dense_weight=cfg.dense_weight,
            sparse_weight=cfg.sparse_weight,
        )
        result = fused[:final_top_k]
        logger.debug(
            "hybrid_retrieved",
            dense_hits=len(dense_hits),
            sparse_hits=len(sparse_hits),
            fused=len(fused),
            returned=len(result),
            mode=mode,
        )
        return result, mode

    @staticmethod
    def _run_leg(
        leg: str,
        fn: Callable[..., list[RetrievedChunk]],
        query: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Run one retrieval leg; raise RetrievalFailure instead of degrading."""
        try:
            hits = fn(query, top_k=top_k)
        except EmptySparseQuery:
            raise  # valid empty encoding: the caller records dense-only mode
        except Exception as e:
            logger.error("retrieval_leg_failed", leg=leg, error=f"{type(e).__name__}: {e}")
            raise RetrievalFailure(leg, f"{type(e).__name__}: {e}") from e
        n_bad = sum(1 for h in hits if not math.isfinite(h.score))
        if n_bad:
            logger.error("retrieval_leg_nonfinite", leg=leg, n_bad=n_bad)
            raise RetrievalFailure(leg, f"{n_bad} non-finite scores")
        return hits

    @staticmethod
    def _rrf_fuse(
        dense_hits: list[RetrievedChunk],
        sparse_hits: list[RetrievedChunk],
        k: int,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> list[RetrievedChunk]:
        """Reciprocal Rank Fusion. Returns chunks sorted by fused score desc."""
        # chunk_id → {hit, rrf_score, breakdown}
        merged: dict[str, dict] = {}

        for rank, hit in enumerate(dense_hits, start=1):
            contrib = dense_weight / (k + rank)
            entry = merged.setdefault(
                hit.chunk_id,
                {
                    "hit": hit,
                    "rrf": 0.0,
                    "breakdown": {"dense_score": 0.0, "dense_rank": None,
                                   "sparse_score": 0.0, "sparse_rank": None,
                                   "rrf_dense": 0.0, "rrf_sparse": 0.0},
                },
            )
            entry["rrf"] += contrib
            entry["breakdown"]["dense_score"] = hit.score
            entry["breakdown"]["dense_rank"] = rank
            entry["breakdown"]["rrf_dense"] = contrib

        for rank, hit in enumerate(sparse_hits, start=1):
            contrib = sparse_weight / (k + rank)
            if hit.chunk_id in merged:
                entry = merged[hit.chunk_id]
            else:
                entry = merged.setdefault(
                    hit.chunk_id,
                    {
                        "hit": hit,
                        "rrf": 0.0,
                        "breakdown": {"dense_score": 0.0, "dense_rank": None,
                                       "sparse_score": 0.0, "sparse_rank": None,
                                       "rrf_dense": 0.0, "rrf_sparse": 0.0},
                    },
                )
            entry["rrf"] += contrib
            entry["breakdown"]["sparse_score"] = hit.score
            entry["breakdown"]["sparse_rank"] = rank
            entry["breakdown"]["rrf_sparse"] = contrib

        # Materialize fused chunks with updated scores
        fused_list: list[RetrievedChunk] = []
        for entry in merged.values():
            hit: RetrievedChunk = entry["hit"]
            hit.score = float(entry["rrf"])
            hit.score_breakdown = {
                **hit.score_breakdown,
                **{k_: v for k_, v in entry["breakdown"].items() if v is not None and v != 0.0},
                "rrf_total": float(entry["rrf"]),
            }
            fused_list.append(hit)

        fused_list.sort(key=lambda c: c.score, reverse=True)
        return fused_list


__all__ = [
    "RETRIEVAL_MODE_DENSE_ONLY",
    "RETRIEVAL_MODE_HYBRID",
    "HybridRetrievalConfig",
    "HybridRetriever",
]
