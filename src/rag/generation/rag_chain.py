"""End-to-end RAG chain: query → hybrid retrieval → rerank → generate → cite.

This is the single entry point used by the FastAPI service and the eval
harness. It exposes one method:

    chain.answer(question: str) -> RAGResponse

with full structured output (answer, citations, retrieved chunks, latencies,
prompt version, abstention flag and the reason for it).

Abstention can happen at three points:
1. No candidates survive retrieval                  → `no_context`
2. Top reranker score < RETRIEVAL_RERANK_SCORE_THRESHOLD (a gate on the
   top-1 score; disabled when the threshold is <= 0, the default)
                                                     → `low_rerank_score`
3. After generation, the answer policy in `rag.guards.citation_checker`
   (model abstained, uncited answer, invalid citation tag, ...).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from langchain_core.output_parsers import StrOutputParser

from rag.config import RetrievalSettings, get_settings
from rag.generation.llm import build_generator_llm
from rag.generation.prompts import (
    RAG_ANSWER_PROMPT_VERSION,
    build_rag_answer_prompt,
    format_context,
)
from rag.guards.citation_checker import (
    ABSTENTION_TEXT,
    AnswerOutcome,
    apply_answer_policy,
)
from rag.logging_config import get_logger
from rag.retrieval.types import RetrievedChunk

if TYPE_CHECKING:  # heavy imports (torch, FlagEmbedding) only for type hints
    from rag.retrieval.hybrid_retriever import HybridRetriever
    from rag.retrieval.reranker import CrossEncoderReranker

logger = get_logger(__name__)

# How many ranked chunks are kept in the response for retrieval metrics
# (Hit@k / MRR@10 / nDCG@10). Only `final_top_k` of them go to the LLM.
RANKING_DEPTH = 10


@dataclass
class RAGResponse:
    """Structured RAG output, also serializable for API + Langfuse."""

    question: str
    answer: str  # what the user is shown (ABSTENTION_TEXT if withheld)
    abstained: bool
    citations: list[str]
    citation_check: dict[str, Any]
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)
    prompt_version: str = ""
    config_summary: dict[str, Any] = field(default_factory=dict)
    # One of AnswerOutcome.*: why the answer was released or withheld.
    outcome: str = AnswerOutcome.ANSWERED
    # Raw generator output before the answer policy ran ("" if no LLM call).
    raw_answer: str = ""
    # Highest reranker score among candidates (None when no reranker ran).
    top_rerank_score: float | None = None
    # Sources of the top-RANKING_DEPTH ranked chunks, for retrieval metrics.
    ranked_sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def pre_generation_gate(
    final: list[RetrievedChunk], top_rerank_score: float | None, threshold: float
) -> str | None:
    """Outcome if the question must abstain before generation, else None.

    The rerank gate acts on the top-1 score only and is disabled for
    threshold <= 0 or when no reranker ran (score is None).
    """
    if not final:
        return AnswerOutcome.NO_CONTEXT
    if top_rerank_score is not None and threshold > 0 and top_rerank_score < threshold:
        return AnswerOutcome.LOW_RERANK_SCORE
    return None


@dataclass
class GenerationResult:
    """Output of `AnswerGenerator.generate` after the answer policy ran."""

    outcome: str
    answer: str
    abstained: bool
    citations: list[str]
    citation_check: dict[str, Any]
    raw_answer: str
    generation_ms: float


class AnswerGenerator:
    """Prompt → LLM → answer policy, for a fixed list of context chunks.

    Shared by `RAGChain.answer` (serving) and `eval.generation_eval`, which
    feeds it chunks saved by a retrieval-only run so that no embedder or
    reranker is loaded while the generator runs.
    """

    def __init__(self, llm: Any | None = None) -> None:
        self._llm = llm if llm is not None else build_generator_llm()
        self._chain = build_rag_answer_prompt() | self._llm | StrOutputParser()

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> GenerationResult:
        tagged = [(c.to_citation_tag(), c.text) for c in chunks]
        t0 = time.perf_counter()
        failed = False
        try:
            raw = self._chain.invoke(
                {"context": format_context(tagged), "question": question}
            ).strip()
        except Exception as e:
            logger.error("generation_failed", error=str(e))
            raw = f"[generation error] {e}"
            failed = True
        gen_ms = (time.perf_counter() - t0) * 1000

        decision = apply_answer_policy(
            raw, allowed_tags=[tag for tag, _ in tagged], generation_failed=failed
        )
        if decision.withheld_by_guard:
            logger.warning(
                "answer_withheld",
                outcome=decision.outcome,
                invalid_tags=decision.citation_check.invalid_tags,
            )
        return GenerationResult(
            outcome=decision.outcome,
            answer=decision.released_answer,
            abstained=decision.abstained,
            citations=decision.citations,
            citation_check=decision.citation_check.to_dict(),
            raw_answer=raw,
            generation_ms=gen_ms,
        )


class RAGChain:
    """Hybrid retrieval + reranker + Qwen2.5 generator with citation guards.

    Args:
        hybrid: HybridRetriever already wired with dense + sparse.
        reranker: Cross-encoder reranker (or None for ablations).
        retrieval_settings: Knobs for top-k, abstention threshold.
        use_reranker: If False, skip reranking — used for ablation configs.
        llm: Chat model / runnable to generate with. Defaults to the Ollama
            generator from settings; tests pass a fake.
    """

    def __init__(
        self,
        hybrid: HybridRetriever,
        reranker: CrossEncoderReranker | None,
        retrieval_settings: RetrievalSettings | None = None,
        use_reranker: bool = True,
        llm: Any | None = None,
    ) -> None:
        self.hybrid = hybrid
        self.reranker = reranker if use_reranker else None
        self.use_reranker = use_reranker and reranker is not None
        self.retrieval_settings = retrieval_settings or get_settings().retrieval

        self.generator = AnswerGenerator(llm=llm)

    # ─── Public API ────────────────────────────────────────────────
    def answer(self, question: str, debug: bool = False) -> RAGResponse:
        """Execute the full pipeline for one question."""
        t0 = time.perf_counter()
        timings: dict[str, float] = {}
        final_top_k = self.retrieval_settings.final_top_k

        # 1) Hybrid retrieval
        t_retr = time.perf_counter()
        hybrid_top_k = self.retrieval_settings.hybrid_top_k
        candidates = self.hybrid.retrieve(question, top_k=hybrid_top_k)
        timings["retrieval_ms"] = (time.perf_counter() - t_retr) * 1000

        # 2) Rerank (optional). Rank deeper than final_top_k so retrieval
        # metrics @10 are real; only the first final_top_k reach the LLM.
        t_rr = time.perf_counter()
        depth = max(final_top_k, RANKING_DEPTH)
        top_rerank_score: float | None = None
        if self.use_reranker and self.reranker is not None and candidates:
            ranked = self.reranker.rerank(query=question, candidates=candidates, top_k=depth)
            top_rerank_score = ranked[0].score if ranked else None
        else:
            ranked = candidates[:depth]
        timings["rerank_ms"] = (time.perf_counter() - t_rr) * 1000
        final = ranked[:final_top_k]
        ranked_sources = [c.source for c in ranked]

        # 3) Pre-generation abstention gates
        gate = pre_generation_gate(
            final, top_rerank_score, self.retrieval_settings.rerank_score_threshold
        )
        if gate is not None:
            return self._build_abstention_response(
                question, timings, t0, final, gate, top_rerank_score, ranked_sources
            )

        # 4) Generate + enforce the answer policy
        g = self.generator.generate(question, final)
        timings["generation_ms"] = g.generation_ms
        timings["total_ms"] = (time.perf_counter() - t0) * 1000

        return RAGResponse(
            question=question,
            answer=g.answer,
            abstained=g.abstained,
            citations=g.citations,
            citation_check=g.citation_check,
            retrieved_chunks=[self._chunk_to_dict(c, debug=debug) for c in final],
            latency_ms=timings,
            prompt_version=self._prompt_version(),
            config_summary=self._config_summary(),
            outcome=g.outcome,
            raw_answer=g.raw_answer,
            top_rerank_score=top_rerank_score,
            ranked_sources=ranked_sources,
        )

    # ─── Internal helpers ──────────────────────────────────────────
    @staticmethod
    def _prompt_version() -> str:
        return f"{RAG_ANSWER_PROMPT_VERSION.name}@{RAG_ANSWER_PROMPT_VERSION.version}"

    def _config_summary(self) -> dict[str, Any]:
        return {
            "use_reranker": self.use_reranker,
            "hybrid_top_k": self.retrieval_settings.hybrid_top_k,
            "final_top_k": self.retrieval_settings.final_top_k,
            "rerank_threshold": self.retrieval_settings.rerank_score_threshold,
        }

    def _build_abstention_response(
        self,
        question: str,
        timings: dict[str, float],
        t0: float,
        chunks: list[RetrievedChunk],
        outcome: str,
        top_rerank_score: float | None,
        ranked_sources: list[str],
    ) -> RAGResponse:
        timings["generation_ms"] = 0.0
        timings["total_ms"] = (time.perf_counter() - t0) * 1000
        return RAGResponse(
            question=question,
            answer=ABSTENTION_TEXT,
            abstained=True,
            citations=[],
            citation_check={
                "n_extracted": 0,
                "n_valid": 0,
                "n_invalid": 0,
                "invalid_tags": [],
                "all_valid": False,
            },
            retrieved_chunks=[self._chunk_to_dict(c) for c in chunks],
            latency_ms=timings,
            prompt_version=self._prompt_version(),
            config_summary=self._config_summary(),
            outcome=outcome,
            top_rerank_score=top_rerank_score,
            ranked_sources=ranked_sources,
        )

    @staticmethod
    def _chunk_to_dict(c: RetrievedChunk, debug: bool = False) -> dict[str, Any]:
        d = {
            "chunk_id": c.chunk_id,
            "source": c.source,
            "page": c.page,
            "section": c.section,
            "paper_title": c.paper_title,
            "score": c.score,
            "score_breakdown": c.score_breakdown,
        }
        if debug:
            d["text"] = c.text
        else:
            d["text_preview"] = (c.text[:240] + "…") if len(c.text) > 240 else c.text
        return d


def build_rag_chain_from_settings(
    hybrid: HybridRetriever,
    reranker: CrossEncoderReranker | None,
    use_reranker: bool = True,
) -> RAGChain:
    """Convenience factory pulling retrieval settings from env."""
    return RAGChain(
        hybrid=hybrid,
        reranker=reranker,
        retrieval_settings=get_settings().retrieval,
        use_reranker=use_reranker,
    )


__all__ = [
    "ABSTENTION_TEXT",
    "RANKING_DEPTH",
    "AnswerGenerator",
    "GenerationResult",
    "RAGChain",
    "RAGResponse",
    "build_rag_chain_from_settings",
    "pre_generation_gate",
]
