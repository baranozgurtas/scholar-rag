"""RAGAS evaluation with local Qwen2.5:14b judge.

We run four RAGAS metrics on the production config (Hybrid + Reranker):
- **faithfulness**: every claim in the answer is grounded in the retrieved context
- **answer_relevancy**: the answer addresses the question
- **context_precision**: ratio of relevant chunks among retrieved
- **context_recall**: ratio of relevant info in retrieved vs ground truth

Both the judge LLM and the embeddings used by RAGAS are local (Qwen via
Ollama + BGE-M3 via sentence-transformers) — no OpenAI keys, no data egress,
fully reproducible on a developer's Mac.

Adversarial questions are evaluated separately: we report the abstention
rate (should be 1.0 = all 5 abstained), not RAGAS metrics, because RAGAS
metrics aren't meaningful for "I don't know" answers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from eval.generate_questions import EvalQuestion
from eval.retrieval_ablation import QUESTIONS_PATH, RESULTS_DIR, load_questions
from rag.config import get_settings
from rag.embeddings.bge_embedder import build_embedder
from rag.generation.llm import build_judge_llm
from rag.generation.rag_chain import build_rag_chain_from_settings
from rag.logging_config import configure_logging, get_logger
from rag.retrieval.dense_retriever import DenseRetriever
from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.retrieval.reranker import CrossEncoderReranker
from rag.retrieval.sparse_retriever import SparseRetriever
from rag.vectorstore.qdrant_store import QdrantStore

logger = get_logger(__name__)


def _run_pipeline_over_questions(
    questions: list[EvalQuestion],
) -> list[dict[str, Any]]:
    """Run the production RAG pipeline over each question and collect outputs."""
    settings = get_settings()
    embedder = build_embedder(settings.embedding)
    store = QdrantStore(embedder=embedder, settings=settings.vectorstore)
    store.ensure_collection(recreate=False)
    dense = DenseRetriever(store=store, embedder=embedder)
    sparse = SparseRetriever(store=store, embedder=embedder)
    hybrid = HybridRetriever(dense=dense, sparse=sparse)
    reranker = CrossEncoderReranker(settings=settings.embedding)
    chain = build_rag_chain_from_settings(hybrid=hybrid, reranker=reranker, use_reranker=True)

    out: list[dict[str, Any]] = []
    for i, q in enumerate(questions, start=1):
        resp = chain.answer(q.question, debug=True)
        out.append(
            {
                "question": q.question,
                "answer": resp.answer,
                "abstained": resp.abstained,
                "contexts": [c.get("text") or c.get("text_preview", "") for c in resp.retrieved_chunks],
                "ground_truth": q.reference_answer,
                "source_kind": q.source_kind,
                "expected_sources": q.expected_sources,
            }
        )
        if i % 5 == 0:
            logger.info("ragas_pipeline_progress", done=i, total=len(questions))
    return out


def _run_ragas(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Run RAGAS metrics on records, returning aggregated + per-row scores.

    Returns dict with structure:
        {
          "aggregate": {faithfulness, answer_relevancy, context_precision, context_recall},
          "per_question": [{question, faithfulness, ..., ground_truth}, ...]
        }
    """
    # Lazy imports — RAGAS is a heavy dependency
    try:
        from datasets import Dataset
        from langchain_ollama import OllamaEmbeddings
        from ragas import evaluate as ragas_evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as e:
        logger.error("ragas_import_failed", error=str(e))
        raise

    # Filter out adversarial questions for RAGAS (handled separately)
    scorable = [r for r in records if r.get("source_kind") != "adversarial"]
    if not scorable:
        return {"aggregate": {}, "per_question": [], "n_scored": 0}

    ds = Dataset.from_list(
        [
            {
                "question": r["question"],
                "answer": r["answer"],
                "contexts": r["contexts"],
                "ground_truth": r["ground_truth"],
            }
            for r in scorable
        ]
    )

    settings = get_settings()
    judge = build_judge_llm(settings.llm)
    # For RAGAS context_precision/recall we need embeddings — reuse local Ollama
    ragas_embed = OllamaEmbeddings(
        model=settings.embedding.model_name if "nomic" in settings.embedding.model_name.lower() else "nomic-embed-text",
        base_url=settings.llm.ollama_base_url,
    )
    ragas_llm = LangchainLLMWrapper(judge)
    ragas_emb_wrap = LangchainEmbeddingsWrapper(ragas_embed)

    logger.info("ragas_evaluate_start", n=len(scorable))
    result = ragas_evaluate(
        dataset=ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_emb_wrap,
        raise_exceptions=False,
    )
    logger.info("ragas_evaluate_done")

    # ragas returns a dataframe-like Result; aggregate via to_pandas if possible
    try:
        df = result.to_pandas()
        aggregate = {
            "faithfulness": float(df["faithfulness"].mean()),
            "answer_relevancy": float(df["answer_relevancy"].mean()),
            "context_precision": float(df["context_precision"].mean()),
            "context_recall": float(df["context_recall"].mean()),
        }
        per_question = []
        for r, (_, row) in zip(scorable, df.iterrows(), strict=True):
            per_question.append(
                {
                    "question": r["question"],
                    "source_kind": r["source_kind"],
                    "faithfulness": float(row.get("faithfulness", 0.0) or 0.0),
                    "answer_relevancy": float(row.get("answer_relevancy", 0.0) or 0.0),
                    "context_precision": float(row.get("context_precision", 0.0) or 0.0),
                    "context_recall": float(row.get("context_recall", 0.0) or 0.0),
                }
            )
    except Exception as e:
        logger.warning("ragas_to_pandas_failed_fallback", error=str(e))
        # Fallback: try dict()
        aggregate = dict(result) if hasattr(result, "__iter__") else {}
        per_question = []

    return {
        "aggregate": {k: round(v, 4) for k, v in aggregate.items()},
        "per_question": per_question,
        "n_scored": len(scorable),
    }


def _write_outputs(
    pipeline_records: list[dict[str, Any]],
    ragas_result: dict[str, Any],
    out_dir: Path = RESULTS_DIR,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON dump (full data)
    (out_dir / "ragas_eval.json").write_text(
        json.dumps(
            {"records": pipeline_records, "ragas": ragas_result},
            indent=2,
            ensure_ascii=False,
        )
    )

    # Markdown summary
    agg = ragas_result.get("aggregate", {})
    n_scored = ragas_result.get("n_scored", 0)
    adv_records = [r for r in pipeline_records if r.get("source_kind") == "adversarial"]
    adv_abst = sum(1 for r in adv_records if r["abstained"]) / max(1, len(adv_records))

    md = [
        "# RAGAS Evaluation Summary\n",
        "**Production config**: Hybrid retrieval (BGE-M3 dense + sparse, RRF) + "
        "BGE-reranker-v2-m3 cross-encoder + Qwen2.5:14b generator.\n",
        "**Judge LLM**: Qwen2.5:14b (local, T=0).\n",
        f"**Eval set size**: {n_scored} scorable (excluding {len(adv_records)} adversarial).\n",
        "",
        "## RAGAS metrics\n",
        "| Metric | Score | What it measures |",
        "|---|---|---|",
        f"| **Faithfulness** | {agg.get('faithfulness', 0):.3f} | Fraction of answer claims grounded in retrieved context |",
        f"| **Answer relevancy** | {agg.get('answer_relevancy', 0):.3f} | Whether the answer addresses the question |",
        f"| **Context precision** | {agg.get('context_precision', 0):.3f} | Fraction of retrieved chunks that are relevant |",
        f"| **Context recall** | {agg.get('context_recall', 0):.3f} | Fraction of ground-truth info captured by retrieval |",
        "",
        "## Adversarial abstention",
        f"- Adversarial questions: **{len(adv_records)}**",
        f"- Correctly abstained: **{int(adv_abst * len(adv_records))}** ({adv_abst*100:.0f}%)",
        "",
    ]
    (out_dir / "ragas_summary.md").write_text("\n".join(md) + "\n")
    logger.info("ragas_outputs_written", dir=str(out_dir))


def run_ragas_eval(
    questions_path: Path = QUESTIONS_PATH,
    out_dir: Path = RESULTS_DIR,
) -> dict[str, Any]:
    questions = load_questions(questions_path)
    if not questions:
        logger.error("no_questions_for_ragas")
        return {}
    records = _run_pipeline_over_questions(questions)
    ragas_result = _run_ragas(records)
    _write_outputs(records, ragas_result, out_dir)
    return ragas_result


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    p = argparse.ArgumentParser(description="RAGAS evaluation with local Qwen judge.")
    p.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = p.parse_args(argv)
    result = run_ragas_eval(args.questions, args.out_dir)
    agg = result.get("aggregate", {})
    print(f"RAGAS aggregate: {agg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main", "run_ragas_eval"]
