"""A failed retrieval stage must end in `retrieval_failed`, never an ordinary answer.

Regression for the live demo: a broken query forward pass returned NaN
lexical weights (silently dropped → empty sparse leg) or a scrambled dense
vector, and the surviving leg was fused and answered as if nothing happened.
A NaN rerank score reached the JSON encoder and produced an HTTP 500.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from rag.config import EmbeddingSettings, RetrievalSettings
from rag.embeddings.output_checks import check_dense, to_sparse
from rag.generation.rag_chain import RAGChain
from rag.guards.citation_checker import ABSTENTION_TEXT, AnswerOutcome
from rag.retrieval.hybrid_retriever import (
    RETRIEVAL_MODE_DENSE_ONLY,
    RETRIEVAL_MODE_HYBRID,
    HybridRetrievalConfig,
    HybridRetriever,
)
from rag.retrieval.sparse_retriever import SparseRetriever
from rag.retrieval.types import (
    EmptySparseQuery,
    ModelOutputError,
    RetrievalFailure,
    RetrievedChunk,
)

NAN = float("nan")


def _mk(chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=f"text-{chunk_id}",
        score=score,
        metadata={"source": "p.pdf", "paper_title": "Paper", "page": 1, "section": "abstract"},
    )


class FakeLeg:
    def __init__(self, result: list[RetrievedChunk] | Exception) -> None:
        self.result = result

    def retrieve(self, query: str, top_k: int = 30) -> list[RetrievedChunk]:
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _hybrid(dense: Any, sparse: Any, **cfg: Any) -> HybridRetriever:
    config = HybridRetrievalConfig(dense_top_k=30, sparse_top_k=30, rrf_k=60, final_top_k=20, **cfg)
    return HybridRetriever(FakeLeg(dense), FakeLeg(sparse), config)  # type: ignore[arg-type]


class TestHybridLegFailures:
    def test_healthy_legs_are_fused(self) -> None:
        out = _hybrid([_mk("A", 0.9)], [_mk("A", 5.0)]).retrieve("q")
        assert [c.chunk_id for c in out] == ["A"]

    def test_leg_exception_raises_retrieval_failure(self) -> None:
        with pytest.raises(RetrievalFailure) as ei:
            _hybrid(ModelOutputError("dense embedding contains NaN/inf"), [_mk("A", 5.0)]).retrieve("q")
        assert ei.value.stage == "dense"
        assert "NaN" in ei.value.reason

    def test_non_finite_leg_score_raises(self) -> None:
        with pytest.raises(RetrievalFailure) as ei:
            _hybrid([_mk("A", 0.9)], [_mk("B", NAN)]).retrieve("q")
        assert ei.value.stage == "sparse"

    @pytest.mark.parametrize(("dense_empty", "failed"), [(True, "dense"), (False, "sparse")])
    def test_one_empty_leg_raises(self, dense_empty: bool, failed: str) -> None:
        hits = [_mk("A", 0.9)]
        dense, sparse = ([], hits) if dense_empty else (hits, [])
        with pytest.raises(RetrievalFailure) as ei:
            _hybrid(dense, sparse).retrieve("q")
        assert ei.value.stage == failed

    def test_empty_sparse_encoding_is_recorded_dense_only(self) -> None:
        # Regression: a valid but empty sparse encoding (e.g. a stopword-only
        # query) must not be treated as a broken leg.
        hybrid = _hybrid([_mk("A", 0.9), _mk("B", 0.8)], EmptySparseQuery("no weights"))
        out, mode = hybrid.retrieve_with_mode("q")
        assert mode == RETRIEVAL_MODE_DENSE_ONLY
        assert [c.chunk_id for c in out] == ["A", "B"]
        assert hybrid.retrieve("q") == out  # list API unchanged

    def test_healthy_retrieval_reports_hybrid_mode(self) -> None:
        _, mode = _hybrid([_mk("A", 0.9)], [_mk("A", 5.0)]).retrieve_with_mode("q")
        assert mode == RETRIEVAL_MODE_HYBRID

    def test_nan_sparse_weights_still_fail_closed(self) -> None:
        # Non-finite weights raise in the embedder, not EmptySparseQuery.
        with pytest.raises(RetrievalFailure) as ei:
            _hybrid([_mk("A", 0.9)], ModelOutputError("sparse lexical weights contain NaN/inf")).retrieve("q")
        assert ei.value.stage == "sparse"

    def test_both_empty_is_not_a_failure(self) -> None:
        # An empty collection is "no context", not a broken leg.
        assert _hybrid([], []).retrieve("q") == []

    def test_disabled_leg_is_not_checked(self) -> None:
        out = _hybrid([_mk("A", 0.9)], [], use_sparse=False).retrieve("q")
        assert [c.chunk_id for c in out] == ["A"]


class TestSparseRetrieverSignals:
    def _retriever(self, weights: list[dict[int, float]] | Exception) -> SparseRetriever:
        embedder = MagicMock()
        if isinstance(weights, Exception):
            embedder.embed_dense_and_sparse.side_effect = weights
        else:
            embedder.embed_dense_and_sparse.return_value = (np.ones((1, 4)), weights)
        return SparseRetriever(store=MagicMock(), embedder=embedder)

    def test_empty_encoding_raises_empty_sparse_query(self) -> None:
        with pytest.raises(EmptySparseQuery):
            self._retriever([{}]).retrieve("the of and")

    def test_embedder_error_propagates(self) -> None:
        with pytest.raises(ModelOutputError):
            self._retriever(ModelOutputError("sparse lexical weights contain NaN/inf")).retrieve("q")


class TestOutputChecks:
    def test_nan_lexical_weight_raises_instead_of_being_dropped(self) -> None:
        with pytest.raises(ModelOutputError):
            to_sparse({"5": 0.3, "9": NAN})

    def test_finite_nonpositive_weights_are_dropped(self) -> None:
        assert to_sparse({"5": 0.3, "9": 0.0, "11": -0.1}) == {5: 0.3}

    @pytest.mark.parametrize("row", [[NAN, 1.0], [np.inf, 1.0], [0.0, 0.0]])
    def test_bad_dense_rows_raise(self, row: list[float]) -> None:
        with pytest.raises(ModelOutputError):
            check_dense(np.array([row]))

    def test_good_dense_passes(self) -> None:
        check_dense(np.array([[0.6, 0.8]]))


# ─── Chain: explicit outcome, no LLM call, JSON-safe ────────────────


class FailingHybrid:
    def retrieve(self, question: str, top_k: int = 20) -> list[RetrievedChunk]:
        raise RetrievalFailure("dense", "ModelOutputError: dense embedding contains NaN/inf")


class OkHybrid:
    def retrieve(self, question: str, top_k: int = 20) -> list[RetrievedChunk]:
        return [_mk(f"c{i}", 1.0 / (i + 1)) for i in range(8)]


class NaNReranker:
    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int = 5, **_: Any) -> list[RetrievedChunk]:
        raise ModelOutputError("reranker returned NaN/inf scores")


class CountingLLM:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, *_: Any, **__: Any) -> str:
        self.calls += 1
        return "should not be called [Paper: Paper | p.1 | §abstract]"


def _chain(hybrid: Any, reranker: Any) -> tuple[RAGChain, CountingLLM]:
    from langchain_core.runnables import RunnableLambda

    llm = CountingLLM()
    chain = RAGChain(
        hybrid=hybrid,
        reranker=reranker,
        retrieval_settings=RetrievalSettings(),
        llm=RunnableLambda(llm.invoke),
    )
    return chain, llm


class DenseOnlyHybrid:
    def retrieve_with_mode(self, question: str, top_k: int = 20) -> tuple[list[RetrievedChunk], str]:
        return [_mk(f"c{i}", 1.0 / (i + 1)) for i in range(8)], RETRIEVAL_MODE_DENSE_ONLY


class TestChainDenseOnly:
    def test_dense_only_answer_is_released_and_recorded(self) -> None:
        from langchain_core.runnables import RunnableLambda

        chain = RAGChain(
            hybrid=DenseOnlyHybrid(),  # type: ignore[arg-type]
            reranker=None,
            retrieval_settings=RetrievalSettings(),
            llm=RunnableLambda(lambda _: "Answer [Paper: Paper | p.1 | §abstract]."),
        )
        r = chain.answer("q")
        assert r.outcome == AnswerOutcome.ANSWERED
        assert r.retrieval_mode == RETRIEVAL_MODE_DENSE_ONLY
        assert r.to_dict()["retrieval_mode"] == "dense_only"


class TestChainRetrievalFailure:
    @pytest.mark.parametrize(
        ("hybrid", "reranker", "stage"),
        [(FailingHybrid(), None, "dense"), (OkHybrid(), NaNReranker(), "rerank")],
    )
    def test_failure_is_explicit_and_not_answered(self, hybrid: Any, reranker: Any, stage: str) -> None:
        chain, llm = _chain(hybrid, reranker)
        r = chain.answer("q")
        assert r.outcome == AnswerOutcome.RETRIEVAL_FAILED
        assert r.outcome_detail.startswith(f"{stage}:")
        assert r.abstained and r.answer == ABSTENTION_TEXT
        assert r.retrieved_chunks == []
        assert llm.calls == 0
        # The HTTP 500 came from NaN reaching the JSON encoder.
        json.dumps(r.to_dict(), allow_nan=False)


# ─── Device wiring: FlagEmbedding reads `devices`, not `device` ─────


@pytest.fixture
def stubbed_flagembedding(monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    """Import the embedder/reranker wrappers against stub torch/FlagEmbedding/diskcache."""
    ns = SimpleNamespace(calls={}, encode_out=None, scores=[0.5])

    class FakeM3:
        def __init__(self, name: str, **kw: Any) -> None:
            ns.calls["embedder"] = kw

        def encode(self, texts: list[str], **_: Any) -> dict[str, Any]:
            return ns.encode_out

    class FakeReranker:
        def __init__(self, name: str, **kw: Any) -> None:
            ns.calls["reranker"] = kw

        def compute_score(self, pairs: Any, **_: Any) -> list[float]:
            return ns.scores

    stubs = {
        "FlagEmbedding": types.ModuleType("FlagEmbedding"),
        "torch": types.ModuleType("torch"),
        "diskcache": types.ModuleType("diskcache"),
    }
    stubs["FlagEmbedding"].BGEM3FlagModel = FakeM3  # type: ignore[attr-defined]
    stubs["FlagEmbedding"].FlagReranker = FakeReranker  # type: ignore[attr-defined]
    stubs["torch"].backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True))  # type: ignore[attr-defined]
    stubs["torch"].cuda = SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
    stubs["diskcache"].Cache = object  # type: ignore[attr-defined]
    for name, mod in stubs.items():
        monkeypatch.setitem(sys.modules, name, mod)

    wrappers = ("rag.embeddings.bge_embedder", "rag.retrieval.reranker")
    saved = {n: sys.modules.pop(n, None) for n in wrappers}
    ns.bge = importlib.import_module("rag.embeddings.bge_embedder")
    ns.rr = importlib.import_module("rag.retrieval.reranker")
    yield ns
    for n, mod in saved.items():
        if mod is None:
            sys.modules.pop(n, None)
        else:
            sys.modules[n] = mod


class TestDeviceWiring:
    @pytest.mark.parametrize(("device", "fp16"), [("cpu", False), ("mps", True)])
    def test_configured_device_reaches_flagembedding(
        self, stubbed_flagembedding: SimpleNamespace, device: str, fp16: bool
    ) -> None:
        s = EmbeddingSettings(EMBEDDING_DEVICE=device, RERANKER_DEVICE=device)
        stubbed_flagembedding.bge.BGEEmbedder(settings=s)
        stubbed_flagembedding.rr.CrossEncoderReranker(settings=s)
        for kw in stubbed_flagembedding.calls.values():
            assert kw["devices"] == device
            assert "device" not in kw
            assert kw["use_fp16"] is fp16

    def test_nan_query_embedding_raises(self, stubbed_flagembedding: SimpleNamespace) -> None:
        stubbed_flagembedding.encode_out = {
            "dense_vecs": np.array([[NAN] * 4]),
            "lexical_weights": [{"5": NAN}],
        }
        e = stubbed_flagembedding.bge.BGEEmbedder(settings=EmbeddingSettings(EMBEDDING_DEVICE="cpu"))
        with pytest.raises(ModelOutputError):
            e.embed_query("q")
        with pytest.raises(ModelOutputError):
            e.embed_dense_and_sparse(["q"], is_query=True)

    def test_nan_rerank_score_raises(self, stubbed_flagembedding: SimpleNamespace) -> None:
        stubbed_flagembedding.scores = [NAN]
        r = stubbed_flagembedding.rr.CrossEncoderReranker(settings=EmbeddingSettings(RERANKER_DEVICE="cpu"))
        with pytest.raises(ModelOutputError):
            r.rerank("q", [_mk("A", 0.1)])
