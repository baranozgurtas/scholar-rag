"""Tests for hybrid retrieval (RRF) math and edge cases."""

from __future__ import annotations

import pytest

from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.retrieval.types import RetrievedChunk


def _mk(chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=f"text-{chunk_id}",
        score=score,
        metadata={"source": f"{chunk_id}.pdf"},
    )


class TestRRFMath:
    def test_winner_is_chunk_present_in_both_with_better_combined_rank(self) -> None:
        dense = [_mk("A", 0.9), _mk("B", 0.8), _mk("C", 0.7)]
        sparse = [_mk("B", 5.0), _mk("D", 4.0), _mk("A", 3.0)]
        fused = HybridRetriever._rrf_fuse(dense, sparse, k=60)
        # B: rank 2 dense + rank 1 sparse = 1/62 + 1/61 = 0.03252 (highest)
        # A: rank 1 dense + rank 3 sparse = 1/61 + 1/63 = 0.03227
        assert fused[0].chunk_id == "B"
        assert fused[1].chunk_id == "A"

    def test_dense_only_falls_back_correctly(self) -> None:
        dense = [_mk("A", 0.9), _mk("B", 0.8)]
        sparse = []
        fused = HybridRetriever._rrf_fuse(dense, sparse, k=60)
        assert fused[0].chunk_id == "A"
        assert fused[1].chunk_id == "B"

    def test_empty_both_returns_empty(self) -> None:
        fused = HybridRetriever._rrf_fuse([], [], k=60)
        assert fused == []

    def test_score_breakdown_populated(self) -> None:
        dense = [_mk("A", 0.9)]
        sparse = [_mk("A", 5.0)]
        fused = HybridRetriever._rrf_fuse(dense, sparse, k=60)
        bd = fused[0].score_breakdown
        assert "rrf_total" in bd
        assert "dense_score" in bd
        assert "sparse_score" in bd
        assert bd["rrf_total"] == pytest.approx(1 / 61 + 1 / 61, rel=1e-6)

    def test_weighted_rrf_changes_winner(self) -> None:
        dense = [_mk("A", 0.9), _mk("B", 0.8)]
        sparse = [_mk("B", 5.0), _mk("A", 3.0)]
        # Unweighted: A and B tied at 1/61 + 1/62
        # Heavily weight sparse → B wins
        fused = HybridRetriever._rrf_fuse(
            dense, sparse, k=60, dense_weight=0.1, sparse_weight=10.0
        )
        assert fused[0].chunk_id == "B"

    def test_preserves_only_in_one_retriever(self) -> None:
        dense = [_mk("A", 0.9), _mk("ONLY_DENSE", 0.5)]
        sparse = [_mk("ONLY_SPARSE", 3.0), _mk("A", 2.0)]
        fused = HybridRetriever._rrf_fuse(dense, sparse, k=60)
        ids = {c.chunk_id for c in fused}
        assert "ONLY_DENSE" in ids
        assert "ONLY_SPARSE" in ids
