"""Tests for retrieval metric calculations (hit@k, MRR@k, nDCG@k)."""

from __future__ import annotations

import math

import pytest

from eval.metrics import (
    compute_retrieval_metrics,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
)


class TestHitAtK:
    def test_hit_when_first_match(self) -> None:
        assert hit_at_k(["a.pdf", "b.pdf"], ["a"], k=5) == 1.0

    def test_no_hit(self) -> None:
        assert hit_at_k(["x.pdf", "y.pdf"], ["a"], k=5) == 0.0

    def test_k_smaller_than_match_position(self) -> None:
        # Match at position 3, k=2 → miss
        assert hit_at_k(["x.pdf", "y.pdf", "a.pdf"], ["a"], k=2) == 0.0
        # k=3 → hit
        assert hit_at_k(["x.pdf", "y.pdf", "a.pdf"], ["a"], k=3) == 1.0

    def test_normalizes_pdf_suffix(self) -> None:
        assert hit_at_k(["my-paper.pdf"], ["my-paper"], k=5) == 1.0

    def test_empty_expected_returns_zero(self) -> None:
        assert hit_at_k(["a.pdf"], [], k=5) == 0.0


class TestMRRAtK:
    def test_first_position(self) -> None:
        assert mrr_at_k(["a.pdf"], ["a"], k=10) == 1.0

    def test_second_position(self) -> None:
        assert mrr_at_k(["x.pdf", "a.pdf"], ["a"], k=10) == pytest.approx(0.5)

    def test_third_position(self) -> None:
        assert mrr_at_k(["x.pdf", "y.pdf", "a.pdf"], ["a"], k=10) == pytest.approx(1 / 3)

    def test_no_match(self) -> None:
        assert mrr_at_k(["x.pdf"], ["a"], k=10) == 0.0


class TestNDCGAtK:
    def test_perfect_ranking(self) -> None:
        # Single expected paper, first position → nDCG = 1.0
        v = ndcg_at_k(["a.pdf", "b.pdf"], ["a"], k=10)
        assert v == pytest.approx(1.0)

    def test_at_position_2(self) -> None:
        v = ndcg_at_k(["x.pdf", "a.pdf"], ["a"], k=10)
        # DCG = 1/log2(3) ≈ 0.631; IDCG = 1/log2(2) = 1 → nDCG ≈ 0.631
        assert v == pytest.approx(1 / math.log2(3), rel=1e-3)

    def test_no_match_returns_zero(self) -> None:
        assert ndcg_at_k(["x.pdf"], ["a"], k=10) == 0.0


class TestAggregate:
    def test_aggregate_across_questions(self) -> None:
        per_q_sources = [
            ["a.pdf", "b.pdf"],  # hit @ 1 for ["a"]
            ["x.pdf", "y.pdf", "c.pdf"],  # hit @ 3 for ["c"]
            ["a.pdf"],  # miss for ["z"]
        ]
        per_q_expected = [["a"], ["c"], ["z"]]
        m = compute_retrieval_metrics(per_q_sources, per_q_expected)
        # 2 of 3 hit@5
        assert m.hit_at_5 == pytest.approx(2 / 3)
        # MRR = (1 + 1/3 + 0) / 3 = 0.444
        assert m.mrr_at_10 == pytest.approx((1 + 1 / 3) / 3)

    def test_adversarial_excluded_from_retrieval_metrics(self) -> None:
        # Question 2 has empty expected → should be excluded from numerator
        per_q_sources = [["a.pdf"], ["x.pdf"]]
        per_q_expected = [["a"], []]
        m = compute_retrieval_metrics(per_q_sources, per_q_expected, abstentions=[False, True])
        # Only 1 scorable: hit@5 = 1.0
        assert m.hit_at_5 == 1.0
        assert m.abstention_rate == 0.5  # 1 of 2 abstained
