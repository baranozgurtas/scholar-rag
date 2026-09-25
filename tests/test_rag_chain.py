"""RAGChain behaviour with fake retriever, reranker and LLM (no Ollama, no models)."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from rag.config import RetrievalSettings
from rag.generation.rag_chain import RAGChain
from rag.guards.citation_checker import ABSTENTION_TEXT, AnswerOutcome
from rag.retrieval.types import RetrievedChunk

TITLE = "M3-Embedding: Multi-Linguality"
VALID_TAG = f"[Paper: {TITLE} | p.3 | §methods]"


def _chunk(i: int, source: str = "bge-m3-chen-2024.pdf") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"c{i}",
        text=f"chunk text {i}",
        score=1.0 / (i + 1),
        metadata={"source": source, "paper_title": TITLE, "page": 3, "section": "methods"},
    )


class FakeHybrid:
    def __init__(self, n: int = 12) -> None:
        self.n = n

    def retrieve(self, question: str, top_k: int = 20) -> list[RetrievedChunk]:
        return [_chunk(i) for i in range(min(self.n, top_k))]


class FakeReranker:
    """Assigns fixed descending scores starting at `top`."""

    def __init__(self, top: float) -> None:
        self.top = top

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int = 5, **_: Any) -> list[RetrievedChunk]:
        for i, c in enumerate(candidates):
            c.score = self.top - 0.01 * i
            c.score_breakdown["reranker"] = c.score
        return candidates[:top_k]


class FakeLLM:
    """Returns a fixed reply and counts calls."""

    def __init__(self, reply: str | Exception) -> None:
        self.reply = reply
        self.calls = 0

    def runnable(self) -> RunnableLambda:
        def _call(_prompt: Any) -> AIMessage:
            self.calls += 1
            if isinstance(self.reply, Exception):
                raise self.reply
            return AIMessage(content=self.reply)

        return RunnableLambda(_call)


def _chain(reply: str | Exception, threshold: float = 0.0, top: float = 0.9, n: int = 12) -> tuple[RAGChain, FakeLLM]:
    llm = FakeLLM(reply)
    chain = RAGChain(
        hybrid=FakeHybrid(n),  # type: ignore[arg-type]
        reranker=FakeReranker(top),  # type: ignore[arg-type]
        retrieval_settings=RetrievalSettings(RETRIEVAL_RERANK_SCORE_THRESHOLD=threshold),
        llm=llm.runnable(),
    )
    return chain, llm


class TestAnswerPolicyInChain:
    def test_valid_cited_answer_is_released(self) -> None:
        chain, _ = _chain(f"It uses [CLS] for dense retrieval {VALID_TAG}.")
        r = chain.answer("q")
        assert r.outcome == AnswerOutcome.ANSWERED
        assert not r.abstained
        assert r.citation_check["all_valid"]
        assert VALID_TAG in r.answer

    def test_uncited_answer_is_withheld_and_raw_output_kept(self) -> None:
        raw = "It uses the CLS token for dense retrieval."
        chain, _ = _chain(raw)
        r = chain.answer("q")
        assert r.outcome == AnswerOutcome.UNCITED_ANSWER
        assert r.abstained
        assert r.answer == ABSTENTION_TEXT
        assert r.raw_answer == raw
        assert r.citation_check["all_valid"] is False

    def test_citation_outside_supplied_context_is_withheld(self) -> None:
        chain, _ = _chain("Dropout halves weights [Paper: Dropout | p.2 | §methods].")
        r = chain.answer("q")
        assert r.outcome == AnswerOutcome.INVALID_CITATION
        assert r.answer == ABSTENTION_TEXT
        assert r.citation_check["n_invalid"] == 1

    def test_generation_exception_is_not_shown_as_an_answer(self) -> None:
        chain, _ = _chain(RuntimeError("ollama down"))
        r = chain.answer("q")
        assert r.outcome == AnswerOutcome.GENERATION_ERROR
        assert r.answer == ABSTENTION_TEXT
        assert "ollama down" in r.raw_answer

    def test_model_abstention(self) -> None:
        chain, _ = _chain(ABSTENTION_TEXT)
        r = chain.answer("q")
        assert r.outcome == AnswerOutcome.MODEL_ABSTAINED
        assert r.abstained


class TestRerankGate:
    def test_threshold_zero_disables_gate(self) -> None:
        chain, llm = _chain(ABSTENTION_TEXT, threshold=0.0, top=0.01)
        r = chain.answer("q")
        assert llm.calls == 1
        assert r.outcome == AnswerOutcome.MODEL_ABSTAINED

    def test_low_top_score_abstains_before_generation(self) -> None:
        chain, llm = _chain(f"answer {VALID_TAG}", threshold=0.5, top=0.3)
        r = chain.answer("q")
        assert llm.calls == 0
        assert r.outcome == AnswerOutcome.LOW_RERANK_SCORE
        assert r.abstained
        assert r.top_rerank_score == pytest.approx(0.3)

    def test_top_score_above_threshold_generates_with_full_context(self) -> None:
        chain, llm = _chain(f"answer {VALID_TAG}", threshold=0.5, top=0.9)
        r = chain.answer("q")
        assert llm.calls == 1
        # Gate is on the top-1 score only; it does not drop lower-scored chunks.
        assert len(r.retrieved_chunks) == 5

    def test_no_candidates_abstains_with_no_context(self) -> None:
        chain, llm = _chain("unused", n=0)
        r = chain.answer("q")
        assert llm.calls == 0
        assert r.outcome == AnswerOutcome.NO_CONTEXT


class TestRankingDepth:
    def test_ranked_sources_go_deeper_than_llm_context(self) -> None:
        chain, _ = _chain(f"answer {VALID_TAG}")
        r = chain.answer("q")
        assert len(r.retrieved_chunks) == 5
        assert len(r.ranked_sources) == 10
