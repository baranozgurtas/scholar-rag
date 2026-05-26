"""Cross-encoder reranker using BGE-reranker-v2-m3.

A cross-encoder jointly encodes the (query, document) pair and outputs
a single relevance score — much more accurate than bi-encoders for the
final ranking step, at the cost of higher latency (we only rerank the
top-20 candidates from hybrid retrieval, not the full corpus).

BGE-reranker-v2-m3 is the strongest open-source reranker as of late 2024
(top-tier on MTEB reranking, BEIR rerank), comparable to commercial
Cohere Rerank — but fully local and OSS, no API key, no data egress.
"""

from __future__ import annotations

import numpy as np
import torch
from FlagEmbedding import FlagReranker

from rag.config import EmbeddingSettings, get_settings
from rag.logging_config import get_logger
from rag.retrieval.types import RetrievedChunk

logger = get_logger(__name__)


class CrossEncoderReranker:
    """BGE-reranker-v2-m3 wrapper."""

    def __init__(self, settings: EmbeddingSettings | None = None) -> None:
        s = settings or get_settings().embedding
        self.settings = s
        device = self._resolve_device(s.reranker_device)
        logger.info(
            "reranker_loading",
            model=s.reranker_model,
            device=device,
            batch_size=s.reranker_batch_size,
        )
        use_fp16 = device in {"cuda", "mps"}
        # FlagReranker accepts a device string; falls back to CPU if unsupported.
        self._reranker = FlagReranker(
            s.reranker_model,
            use_fp16=use_fp16,
            device=device,
        )
        self._device = device

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested == "mps" and torch.backends.mps.is_available():
            return "mps"
        if requested == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """Score (query, candidate) pairs, return top-k sorted by reranker score.

        Args:
            query: User query.
            candidates: Candidate chunks (typically from hybrid retrieval).
            top_k: Number of final chunks to return.
            score_threshold: If set, drop candidates below this reranker score.
                Useful for abstention ("I couldn't find anything relevant").
        """
        if not candidates:
            return []

        pairs = [(query, c.text) for c in candidates]
        # compute_score returns raw logits; normalize=True applies sigmoid → (0, 1)
        scores = self._reranker.compute_score(
            pairs,
            batch_size=self.settings.reranker_batch_size,
            normalize=True,
        )
        # Handle scalar vs list returns from FlagReranker
        if isinstance(scores, float):
            scores = [scores]
        scores = np.asarray(scores, dtype=float)

        # Update each candidate's score & breakdown
        for c, s in zip(candidates, scores, strict=True):
            c.score_breakdown["reranker"] = float(s)
            c.score = float(s)  # final score → reranker score

        # Sort by reranker score desc; apply threshold; truncate
        ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
        if score_threshold is not None:
            ranked = [c for c in ranked if c.score >= score_threshold]
        result = ranked[:top_k]
        logger.debug(
            "reranked",
            n_candidates=len(candidates),
            n_above_threshold=len(ranked),
            returned=len(result),
            top_score=result[0].score if result else None,
        )
        return result


__all__ = ["CrossEncoderReranker"]
