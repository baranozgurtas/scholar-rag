"""Retrieval metrics computed against expected_sources ground truth.

A retrieval hit is defined at the **paper level**: a retrieved chunk
matches the ground truth if its `source` field equals one of the
question's `expected_sources` aliases. This is the right granularity
for our eval set because manual + adversarial ground truth is at the
paper level (we did not annotate chunk-level relevance).

For the synthetic subset (single source paper per question), Hit@k
and MRR@k reduce to "is the correct paper in the top-k". For manual
multi-source questions (e.g., BPR vs NCF), we treat the question as
correct if ANY expected source appears in the top-k; full precision
is also reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    """Aggregate retrieval metrics across a question set."""

    n: int
    hit_at_5: float
    hit_at_10: float
    mrr_at_10: float
    ndcg_at_10: float
    abstention_rate: float  # fraction returning no chunks

    def to_dict(self) -> dict[str, float]:
        return {
            "n": self.n,
            "hit_at_5": round(self.hit_at_5, 4),
            "hit_at_10": round(self.hit_at_10, 4),
            "mrr_at_10": round(self.mrr_at_10, 4),
            "ndcg_at_10": round(self.ndcg_at_10, 4),
            "abstention_rate": round(self.abstention_rate, 4),
        }


def _normalize_source(source: str) -> str:
    """`bge-m3.pdf` → `bge-m3` so we can compare against paper alias."""
    if source.lower().endswith(".pdf"):
        return source[:-4].lower()
    return source.lower()


def hit_at_k(retrieved_sources: list[str], expected: list[str], k: int) -> float:
    if not expected:
        return 0.0  # adversarial: by definition no expected source → handled separately
    top_k = {_normalize_source(s) for s in retrieved_sources[:k]}
    expected_norm = {_normalize_source(e) for e in expected}
    return 1.0 if (top_k & expected_norm) else 0.0


def mrr_at_k(retrieved_sources: list[str], expected: list[str], k: int) -> float:
    if not expected:
        return 0.0
    expected_norm = {_normalize_source(e) for e in expected}
    for rank, src in enumerate(retrieved_sources[:k], start=1):
        if _normalize_source(src) in expected_norm:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_sources: list[str], expected: list[str], k: int) -> float:
    """Binary relevance nDCG@k."""
    if not expected:
        return 0.0
    expected_norm = {_normalize_source(e) for e in expected}
    relevance = [
        1 if _normalize_source(s) in expected_norm else 0
        for s in retrieved_sources[:k]
    ]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance))
    # Ideal: all expected papers ranked first (capped at k)
    n_ideal = min(len(expected_norm), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_ideal))
    return (dcg / idcg) if idcg > 0 else 0.0


def compute_retrieval_metrics(
    per_question_sources: list[list[str]],
    per_question_expected: list[list[str]],
    abstentions: list[bool] | None = None,
) -> RetrievalMetrics:
    """Aggregate hit@5, hit@10, MRR@10, nDCG@10 over a question set.

    Adversarial questions (empty `expected`) are excluded from retrieval
    metric numerators but contribute to abstention_rate.
    """
    assert len(per_question_sources) == len(per_question_expected)
    n_total = len(per_question_sources)
    scored = [(s, e) for s, e in zip(per_question_sources, per_question_expected, strict=True) if e]
    n_scored = len(scored)

    if n_scored == 0:
        h5 = h10 = mrr = ndcg = 0.0
    else:
        h5 = sum(hit_at_k(s, e, 5) for s, e in scored) / n_scored
        h10 = sum(hit_at_k(s, e, 10) for s, e in scored) / n_scored
        mrr = sum(mrr_at_k(s, e, 10) for s, e in scored) / n_scored
        ndcg = sum(ndcg_at_k(s, e, 10) for s, e in scored) / n_scored

    abst_rate = 0.0
    if abstentions:
        abst_rate = sum(1 for a in abstentions if a) / max(1, len(abstentions))

    return RetrievalMetrics(
        n=n_total,
        hit_at_5=h5,
        hit_at_10=h10,
        mrr_at_10=mrr,
        ndcg_at_10=ndcg,
        abstention_rate=abst_rate,
    )


__all__ = [
    "RetrievalMetrics",
    "compute_retrieval_metrics",
    "hit_at_k",
    "mrr_at_k",
    "ndcg_at_k",
]
