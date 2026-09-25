"""Eval harness: summaries, legacy regression, question files, threshold selection.

All deterministic; no model, Ollama or Qdrant needed.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from eval.harness import percentile, summarize, wilson_interval
from eval.questions import (
    QUESTIONS_PATH,
    QUESTIONS_V2_DRAFT_PATH,
    EvalQuestion,
    load_questions,
    verify_question,
)
from eval.reanalyze_legacy import reanalyze
from eval.select_threshold import apply_threshold, error_rates, select_threshold


def _rec(id_: str, expected: list[str], ranked: list[str], abstained: bool, **kw: Any) -> dict[str, Any]:
    return {
        "id": id_,
        "question": f"question {id_}",
        "split": kw.get("split", "dev"),
        "review_status": "unreviewed",
        "expected_sources": expected,
        "ranked_sources": ranked,
        "abstained": abstained,
        "outcome": kw.get("outcome", "model_abstained" if abstained else "answered"),
        "citations": kw.get("citations", []),
        "citation_check": kw.get("citation_check", {"n_extracted": 1, "n_valid": 1, "n_invalid": 0, "all_valid": True}),
        "latency_ms": {"total_ms": kw.get("ms", 1000.0)},
        "top_rerank_score": kw.get("score"),
        "error": kw.get("error"),
    }


class TestStats:
    def test_percentile_matches_numpy_linear(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0]
        assert percentile(xs, 50) == pytest.approx(2.5)
        assert percentile(xs, 95) == pytest.approx(3.85)
        assert percentile([], 50) is None

    def test_wilson_interval_zero_successes_is_not_zero_width(self) -> None:
        lo, hi = wilson_interval(0, 5)
        assert lo == 0.0
        assert 0.4 < hi < 0.45


class TestSummarize:
    def test_separates_answerable_and_unanswerable(self) -> None:
        recs = [
            _rec("a1", ["p1"], ["p1.pdf", "p2"], False, ms=100),
            _rec("a2", ["p1"], ["p2", "p1"], True, ms=200),  # false abstention
            _rec("u1", [], ["p3"], True, ms=300),
            _rec("u2", [], ["p3"], False, ms=400),  # false answer
        ]
        s = summarize(recs)
        assert s["retrieval"]["n"] == 2
        assert s["retrieval"]["mrr_at_10"] == pytest.approx((1 + 0.5) / 2)
        assert s["abstention"]["false_abstention_on_answerable"]["count"] == 1
        assert s["abstention"]["false_answer_on_unanswerable"]["count"] == 1
        assert s["latency"]["total_ms_p50"] == pytest.approx(250)
        ids = {f["id"]: f["failures"] for f in s["failures"]}
        assert ids["u2"] == ["false_answer"]
        assert "false_abstention:model_abstained" in ids["a2"]

    def test_errors_are_excluded_not_scored(self) -> None:
        recs = [
            _rec("a1", ["p1"], ["p1"], False),
            _rec("a2", ["p1"], [], False, error="RuntimeError: boom"),
        ]
        s = summarize(recs)
        assert s["counts"]["n_errors"] == 1
        assert s["retrieval"]["n"] == 1
        assert s["retrieval"]["hit_at_5"] == 1.0
        assert s["failures"][0]["failures"] == ["pipeline_error"]

    def test_citation_validity_is_reported_separately_from_grounding(self) -> None:
        bad = {"n_extracted": 2, "n_valid": 1, "n_invalid": 1, "all_valid": False}
        recs = [
            _rec("a1", ["p1"], ["p1"], False),
            _rec("a2", ["p1"], ["p1"], True, outcome="invalid_citation", citation_check=bad),
        ]
        c = summarize(recs)["citations"]
        assert c["tag_validity_over_all_generated_tags"]["count"] == 2
        assert c["tag_validity_over_all_generated_tags"]["n"] == 3
        assert c["withheld_by_guard"]["invalid_citation"] == 1
        assert c["factual_grounding"]["status"] == "not_measured"

    def test_shallow_rankings_are_flagged(self) -> None:
        s = summarize([_rec("a1", ["p1"], ["p1"] * 5, False)])
        assert s["retrieval"]["min_ranking_depth"] == 5
        assert "effectively @5" in s["retrieval"]["note"]


class TestLegacyRegression:
    """The committed results, re-derived. Pins the numbers the README quotes."""

    @pytest.fixture(scope="class")
    def legacy(self) -> dict[str, Any]:
        return reanalyze()

    def test_reported_retrieval_numbers_reproduce(self, legacy: dict[str, Any]) -> None:
        for cfg, s in legacy.items():
            rep = s["reported_metrics"]
            assert s["retrieval"]["hit_at_5"] == pytest.approx(rep["hit_at_5"], abs=1e-4), cfg
            assert s["retrieval"]["mrr_at_10"] == pytest.approx(rep["mrr_at_10"], abs=1e-4), cfg
            assert s["retrieval"]["ndcg_at_10"] == pytest.approx(rep["ndcg_at_10"], abs=1e-4), cfg

    def test_b_has_best_mrr_not_d(self, legacy: dict[str, Any]) -> None:
        assert legacy["B_dense_plus_rerank"]["retrieval"]["mrr_at_10"] == pytest.approx(0.9667, abs=1e-4)
        assert legacy["D_hybrid_plus_rerank"]["retrieval"]["mrr_at_10"] == pytest.approx(0.95)

    def test_false_abstention_counts(self, legacy: dict[str, Any]) -> None:
        got = {k: v["abstention"]["false_abstention_on_answerable"]["count"] for k, v in legacy.items()}
        assert got == {
            "A_dense_only": 3,
            "B_dense_plus_rerank": 1,
            "C_hybrid_no_rerank": 3,
            "D_hybrid_plus_rerank": 0,
        }

    def test_unanswerable_all_abstained_on_small_set(self, legacy: dict[str, Any]) -> None:
        for s in legacy.values():
            r = s["abstention"]["false_answer_on_unanswerable"]
            assert (r["count"], r["n"]) == (0, 5)

    def test_only_five_ranks_were_recorded(self, legacy: dict[str, Any]) -> None:
        for s in legacy.values():
            assert s["retrieval"]["min_ranking_depth"] == 5


class TestQuestionFiles:
    def test_legacy_set_has_no_templates_and_is_25(self) -> None:
        qs = load_questions(QUESTIONS_PATH)
        assert len(qs) == 25
        assert sum(q.answerable for q in qs) == 20
        assert {q.split for q in qs} == {"legacy_test"}

    def test_no_question_claims_human_review(self) -> None:
        # Flip this only after a person has actually reviewed questions.
        for path in (QUESTIONS_PATH, QUESTIONS_V2_DRAFT_PATH):
            assert all(q.review_status != "human_reviewed" for q in load_questions(path))

    def test_templates_are_rejected_by_loader(self, tmp_path) -> None:
        p = tmp_path / "q.jsonl"
        p.write_text(json.dumps({"question": "[FILL IN] x", "reference_answer": "y"}) + "\n")
        with pytest.raises(ValueError, match="unfilled template"):
            load_questions(p)

    def test_ids_unique_across_files(self) -> None:
        ids = [q.id for p in (QUESTIONS_PATH, QUESTIONS_V2_DRAFT_PATH) for q in load_questions(p)]
        assert len(ids) == len(set(ids))

    def test_draft_split_has_both_classes_in_dev_and_heldout(self) -> None:
        qs = load_questions(QUESTIONS_V2_DRAFT_PATH)
        for split in ("dev", "heldout"):
            sub = [q for q in qs if q.split == split]
            assert any(q.answerable for q in sub)
            assert any(not q.answerable for q in sub)

    @pytest.mark.parametrize(
        "q",
        [q for p in (QUESTIONS_PATH, QUESTIONS_V2_DRAFT_PATH) for q in load_questions(p)],
        ids=lambda q: q.id,
    )
    def test_question_passes_source_checks(self, q: EvalQuestion) -> None:
        assert verify_question(q) == []


class TestThresholdSelection:
    @staticmethod
    def _dev() -> list[dict[str, Any]]:
        return [
            _rec("a1", ["p"], ["p"], False, score=0.95),
            _rec("a2", ["p"], ["p"], False, score=0.80),
            _rec("a3", ["p"], ["p"], False, score=0.40),
            _rec("u1", [], ["p"], False, score=0.30),  # answered: false answer
            _rec("u2", [], ["p"], True, score=0.60),
        ]

    def test_gate_replay(self) -> None:
        assert apply_threshold(self._dev(), 0.5) == [False, False, True, True, True]
        assert apply_threshold(self._dev(), 0.0) == [False, False, False, False, True]

    def test_selects_lowest_threshold_that_fixes_false_answers_within_budget(self) -> None:
        sel = select_threshold(self._dev(), max_false_abstention=0.0)
        # Refusing u1 (0.30) needs t > 0.30; the next observed score, 0.40, does it
        # while still answering a3 (0.40 is not < 0.40).
        assert sel["chosen"]["threshold"] == pytest.approx(0.40)
        assert sel["chosen"]["false_answer"] == 0.0
        assert sel["chosen"]["false_abstention"] == 0.0

    def test_heldout_does_not_influence_selection(self) -> None:
        dev = self._dev()
        chosen = select_threshold(dev, 0.0)["chosen"]["threshold"]
        heldout = [_rec("h1", ["p"], ["p"], False, score=0.39, split="heldout")]
        assert select_threshold(dev, 0.0)["chosen"]["threshold"] == chosen
        assert error_rates(heldout, chosen)["false_abstention"] == 1.0

    def test_refuses_dev_without_both_classes(self) -> None:
        with pytest.raises(ValueError):
            select_threshold([_rec("a1", ["p"], ["p"], False, score=0.9)])
