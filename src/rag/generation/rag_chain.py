"""End-to-end RAG chain: query → hybrid retrieval → rerank → generate → cite.

This is the single entry point used by the FastAPI service, the Streamlit UI,
and the eval harness. It exposes one method:

    chain.answer(question: str) -> RAGResponse

with full structured output (answer, citations, retrieved chunks, latencies,
prompt version, abstention flag).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from rag.config import RetrievalSettings, get_settings
from rag.generation.llm import build_generator_llm
from rag.generation.prompts import (
    RAG_ANSWER_PROMPT_VERSION,
    build_rag_answer_prompt,
    format_context,
)
from rag.guards.citation_checker import (
    CitationCheckResult,
    extract_citation_tags,
    validate_citations_against_context,
)
from rag.logging_config import get_logger
from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.retrieval.reranker import CrossEncoderReranker
from rag.retrieval.types import RetrievedChunk

logger = get_logger(__name__)

ABSTENTION_TEXT = (
    "I could not find sufficient information in the indexed papers to answer this question."
)


@dataclass
class RAGResponse:
    """Structured RAG output, also serializable for API + Langfuse."""

    question: str
    answer: str
    abstained: bool
    citations: list[str]
    citation_check: dict[str, Any]
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)
    prompt_version: str = ""
    config_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RAGChain:
    """Hybrid retrieval + reranker + Qwen2.5 generator with citation guards.

    Args:
        hybrid: HybridRetriever already wired with dense + sparse.
        reranker: Cross-encoder reranker (or None for ablations).
        retrieval_settings: Knobs for top-k, abstention threshold.
        use_reranker: If False, skip reranking — used for ablation configs.
    """

    def __init__(
        self,
        hybrid: HybridRetriever,
        reranker: CrossEncoderReranker | None,
        retrieval_settings: RetrievalSettings | None = None,
        use_reranker: bool = True,
    ) -> None:
        self.hybrid = hybrid
        self.reranker = reranker if use_reranker else None
        self.use_reranker = use_reranker and reranker is not None
        self.retrieval_settings = retrieval_settings or get_settings().retrieval

        self._prompt = build_rag_answer_prompt()
        self._llm = build_generator_llm()
        # LCEL chain: prompt → llm → str
        self._answer_chain = self._prompt | self._llm | StrOutputParser()

    # ─── Public API ────────────────────────────────────────────────
    def answer(self, question: str, debug: bool = False) -> RAGResponse:
        """Execute the full pipeline for one question."""
        t0 = time.perf_counter()
        timings: dict[str, float] = {}

        # 1) Hybrid retrieval
        t_retr = time.perf_counter()
        hybrid_top_k = self.retrieval_settings.hybrid_top_k
        candidates = self.hybrid.retrieve(question, top_k=hybrid_top_k)
        timings["retrieval_ms"] = (time.perf_counter() - t_retr) * 1000

        # 2) Rerank (optional)
        t_rr = time.perf_counter()
        if self.use_reranker and self.reranker is not None and candidates:
            final = self.reranker.rerank(
                query=question,
                candidates=candidates,
                top_k=self.retrieval_settings.final_top_k,
                score_threshold=self.retrieval_settings.rerank_score_threshold or None,
            )
        else:
            final = candidates[: self.retrieval_settings.final_top_k]
        timings["rerank_ms"] = (time.perf_counter() - t_rr) * 1000

        # 3) Abstention check: no chunks survived
        if not final:
            return self._build_abstention_response(question, timings, t0, candidates)

        # 4) Build prompt context with citation tags
        tagged_chunks: list[tuple[str, str]] = [(c.to_citation_tag(), c.text) for c in final]
        context_block = format_context(tagged_chunks)

        # 5) Generate
        t_gen = time.perf_counter()
        try:
            answer_text = self._answer_chain.invoke(
                {"context": context_block, "question": question}
            ).strip()
        except Exception as e:
            logger.error("generation_failed", error=str(e))
            answer_text = f"[generation error] {e}"
        timings["generation_ms"] = (time.perf_counter() - t_gen) * 1000

        # 6) Post-generation guards: did the model abstain? are citations valid?
        abstained = ABSTENTION_TEXT.lower() in answer_text.lower()
        citations = extract_citation_tags(answer_text)
        citation_check: CitationCheckResult = validate_citations_against_context(
            citations=citations,
            allowed_tags=[tag for tag, _ in tagged_chunks],
        )

        timings["total_ms"] = (time.perf_counter() - t0) * 1000

        return RAGResponse(
            question=question,
            answer=answer_text,
            abstained=abstained,
            citations=citations,
            citation_check=citation_check.to_dict(),
            retrieved_chunks=[self._chunk_to_dict(c, debug=debug) for c in final],
            latency_ms=timings,
            prompt_version=f"{RAG_ANSWER_PROMPT_VERSION.name}@{RAG_ANSWER_PROMPT_VERSION.version}",
            config_summary={
                "use_reranker": self.use_reranker,
                "hybrid_top_k": hybrid_top_k,
                "final_top_k": self.retrieval_settings.final_top_k,
                "rerank_threshold": self.retrieval_settings.rerank_score_threshold,
            },
        )

    # ─── Internal helpers ──────────────────────────────────────────
    def _build_abstention_response(
        self,
        question: str,
        timings: dict[str, float],
        t0: float,
        candidates: list[RetrievedChunk],
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
                "all_valid": True,
            },
            retrieved_chunks=[self._chunk_to_dict(c) for c in candidates[:5]],
            latency_ms=timings,
            prompt_version=f"{RAG_ANSWER_PROMPT_VERSION.name}@{RAG_ANSWER_PROMPT_VERSION.version}",
            config_summary={
                "use_reranker": self.use_reranker,
                "abstained_reason": "no_chunks_above_threshold",
            },
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


# Re-export RunnableLambda usage for users wanting to compose further
__all__ = [
    "ABSTENTION_TEXT",
    "RAGChain",
    "RAGResponse",
    "RunnableLambda",
    "build_rag_chain_from_settings",
]
