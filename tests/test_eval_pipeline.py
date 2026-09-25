"""Retrieval-only ablation and resumable generation eval, with fakes (no models)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest

from eval import generation_eval
from eval.harness import summarize_retrieval
from eval.retrieval_ablation import (
    CONFIG_BY_NAME,
    PreflightError,
    append_jsonl,
    phase2_rank,
    read_jsonl,
)
from rag.config import get_settings
from rag.generation import rag_chain
from rag.guards.citation_checker import AnswerOutcome
from rag.retrieval.types import RetrievedChunk


def _chunk(i: int, source: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{source}_{i}",
        text=f"text {source} {i}",
        score=1.0 - 0.01 * i,
        metadata={"source": f"{source}.pdf", "paper_title": source.upper(), "page": 1, "section": "methods"},
        score_breakdown={"dense_score": 0.7 - 0.01 * i},
    )


def _candidate_record(qid: str, expected: list[str], sources: list[str]) -> dict[str, Any]:
    return {
        "id": qid,
        "question": f"question {qid}",
        "split": "dev",
        "review_status": "unreviewed",
        "source_kind": "drafted",
        "expected_sources": expected,
        "kind": "dense",
        "candidates": [_chunk(i, s).to_record() for i, s in enumerate(sources)],
        "top_dense_score": 0.7,
        "retrieval_ms": 12.0,
    }


class ReverseReranker:
    """Scores candidates in reverse input order; counts calls."""

    def __init__(self) -> None:
        self.calls = 0

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int = 5, **_: Any) -> list[RetrievedChunk]:
        self.calls += 1
        n = len(candidates)
        for i, c in enumerate(candidates):
            c.score = (i + 1) / n
        return sorted(candidates, key=lambda c: c.score, reverse=True)[:top_k]


class TestRetrievalOnly:
    def test_chunk_record_round_trip(self) -> None:
        c = _chunk(3, "adam")
        back = RetrievedChunk.from_record(c.to_record())
        assert back == c
        assert "text" not in c.to_record(include_text=False)

    def test_phase2_ranks_without_generator_and_resumes(self, tmp_path: Path) -> None:
        cand = tmp_path / "candidates_dense.jsonl"
        append_jsonl(cand, _candidate_record("q1", ["adam"], ["xgb"] * 11 + ["adam"]))
        append_jsonl(cand, _candidate_record("q2", [], ["xgb"] * 12))
        cfgs = [CONFIG_BY_NAME["A_dense_only"], CONFIG_BY_NAME["B_dense_plus_rerank"]]
        rr = ReverseReranker()
        phase2_rank(tmp_path, cfgs, get_settings(), rr)
        assert rr.calls == 2

        a = read_jsonl(tmp_path / "ranked_A_dense_only.jsonl")
        b = read_jsonl(tmp_path / "ranked_B_dense_plus_rerank.jsonl")
        assert len(a[0]["ranked_sources"]) == 10
        assert "adam" not in a[0]["ranked_sources"]  # candidate #12 is beyond depth 10
        assert b[0]["ranked_sources"][0] == "adam"  # reranker moved it to the top
        assert a[0]["top_rerank_score"] is None
        assert b[0]["top_rerank_score"] == pytest.approx(1.0)
        # Only the chunks the generator would see keep their text.
        assert all(("text" in d) == (i < 5) for i, d in enumerate(b[0]["ranked"]))

        # Resume: a third question is added; only it gets reranked.
        append_jsonl(cand, _candidate_record("q3", ["xgb"], ["xgb"] * 12))
        phase2_rank(tmp_path, cfgs, get_settings(), rr)
        assert rr.calls == 3
        assert [r["id"] for r in read_jsonl(tmp_path / "ranked_B_dense_plus_rerank.jsonl")] == ["q1", "q2", "q3"]

        s = summarize_retrieval(read_jsonl(tmp_path / "ranked_B_dense_plus_rerank.jsonl"))
        assert s["retrieval"]["n"] == 2
        assert s["retrieval"]["hit_at_5"] == 1.0
        assert s["top_rerank_score"]["unanswerable"]["n"] == 1

    def test_low_dense_score_is_flagged(self) -> None:
        rec = {
            "id": "q", "question": "q", "expected_sources": ["adam"],
            "ranked_sources": ["causal"] * 10, "top_dense_score": 0.02,
            "latency_ms": {"retrieval_ms": 1, "rerank_ms": 0}, "error": None,
        }
        f = summarize_retrieval([rec])["failures"][0]["failures"]
        assert "retrieval_miss_at_5" in f
        assert any(x.startswith("low_top_dense_score") for x in f)


# ─── Generation eval ──────────────────────────────────────────────


class FakeGenerator:
    """Stands in for AnswerGenerator; replies are consumed in order."""

    replies: ClassVar[list[str | None]] = []
    calls = 0

    def __init__(self, llm: Any = None) -> None:
        pass

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> rag_chain.GenerationResult:
        FakeGenerator.calls += 1
        reply = FakeGenerator.replies.pop(0)
        if reply is None:
            return rag_chain.GenerationResult(
                AnswerOutcome.GENERATION_ERROR, "abst", True, [], {}, "[generation error] down", 5.0
            )
        from rag.guards.citation_checker import apply_answer_policy

        d = apply_answer_policy(reply, [c.to_citation_tag() for c in chunks])
        return rag_chain.GenerationResult(
            d.outcome, d.released_answer, d.abstained, d.citations, d.citation_check.to_dict(), reply, 5.0
        )


def _retrieval_run(tmp_path: Path, n: int) -> Path:
    run = tmp_path / "retrieval_run"
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps({"models": {"embedding": "e", "reranker": "r"}, "questions": {"review_status": {}}, "corpus": {}})
    )
    for i in range(1, n + 1):
        ranked = [_chunk(j, "adam") for j in range(10)]
        append_jsonl(
            run / "ranked_D_hybrid_plus_rerank.jsonl",
            {
                "id": f"q{i}", "question": f"question {i}", "split": "dev", "review_status": "unreviewed",
                "source_kind": "drafted", "expected_sources": ["adam"], "config": "D_hybrid_plus_rerank",
                "ranked_sources": ["adam"] * 10, "ranked": [c.to_record() for c in ranked],
                "top_rerank_score": 0.9, "top_dense_score": 0.7,
                "latency_ms": {"retrieval_ms": 10.0, "rerank_ms": 20.0}, "error": None,
            },
        )
    return run


@pytest.fixture
def fake_generation(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    info = {"digest": "sha256:aaa"}
    monkeypatch.setattr(generation_eval, "ollama_model_info", lambda *_a, **_k: dict(info))
    monkeypatch.setattr(rag_chain, "AnswerGenerator", FakeGenerator)
    FakeGenerator.calls = 0
    return info


class TestGenerationEval:
    def test_saves_each_question_and_resumes_after_generator_failure(
        self, tmp_path: Path, fake_generation: dict[str, Any]
    ) -> None:
        run = _retrieval_run(tmp_path, 3)
        tag = "[Paper: ADAM | p.1 | §methods]"
        FakeGenerator.replies = [f"Cited answer {tag}.", "Uncited answer.", None]

        with pytest.raises(PreflightError, match="Nothing written"):
            generation_eval.run_generation_eval(run, "D_hybrid_plus_rerank")
        out = run / "generation_D_hybrid_plus_rerank"
        recs = read_jsonl(out / "records.jsonl")
        assert [r["outcome"] for r in recs] == ["answered", "uncited_answer"]
        assert recs[1]["raw_answer"] == "Uncited answer."
        assert recs[0]["latency_ms"]["total_ms"] == pytest.approx(35.0)

        FakeGenerator.replies = [f"Third {tag}."]
        generation_eval.run_generation_eval(run, "D_hybrid_plus_rerank")
        assert FakeGenerator.calls == 4  # q3 retried once; q1, q2 not re-generated
        recs = read_jsonl(out / "records.jsonl")
        assert [r["id"] for r in recs] == ["q1", "q2", "q3"]
        summary = json.loads((out / "summary.json").read_text())
        assert summary["abstention"]["false_abstention_on_answerable"]["count"] == 1

    def test_refuses_to_mix_generator_versions(self, tmp_path: Path, fake_generation: dict[str, Any]) -> None:
        run = _retrieval_run(tmp_path, 2)
        FakeGenerator.replies = [None]
        with pytest.raises(PreflightError):
            generation_eval.run_generation_eval(run, "D_hybrid_plus_rerank")
        fake_generation["digest"] = "sha256:different"
        with pytest.raises(PreflightError, match="generator changed"):
            generation_eval.run_generation_eval(run, "D_hybrid_plus_rerank")

    def test_rerank_gate_abstains_without_calling_generator(
        self, tmp_path: Path, fake_generation: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = _retrieval_run(tmp_path, 1)
        monkeypatch.setattr(get_settings().retrieval, "rerank_score_threshold", 0.95)
        FakeGenerator.replies = []
        generation_eval.run_generation_eval(run, "D_hybrid_plus_rerank")
        rec = read_jsonl(run / "generation_D_hybrid_plus_rerank" / "records.jsonl")[0]
        assert FakeGenerator.calls == 0
        assert rec["outcome"] == AnswerOutcome.LOW_RERANK_SCORE
