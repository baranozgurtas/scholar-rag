"""4-config retrieval ablation.

Compares:
    A) Dense-only            (baseline)
    B) Dense + Reranker
    C) Hybrid (Dense + Sparse via RRF)
    D) Hybrid + Reranker     (production config)

Each config is run against the loaded question set. We report:
    Hit@5, Hit@10, MRR@10, nDCG@10, abstention_rate

The ablation table is the bread and butter of the CV bullet — it shows
the precision uplift from adding hybrid retrieval AND a cross-encoder
reranker, both in numerical terms a recruiter or interviewer can ask about.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.generate_questions import EvalQuestion
from eval.metrics import compute_retrieval_metrics
from rag.config import get_settings
from rag.embeddings.bge_embedder import build_embedder
from rag.generation.rag_chain import RAGChain
from rag.logging_config import configure_logging, get_logger
from rag.retrieval.dense_retriever import DenseRetriever
from rag.retrieval.hybrid_retriever import HybridRetrievalConfig, HybridRetriever
from rag.retrieval.reranker import CrossEncoderReranker
from rag.retrieval.sparse_retriever import SparseRetriever
from rag.vectorstore.qdrant_store import QdrantStore

logger = get_logger(__name__)

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ConfigResult:
    """Ablation result for one configuration."""

    name: str
    description: str
    metrics: dict[str, float]
    raw_per_question: list[dict[str, Any]]


def load_questions(path: Path = QUESTIONS_PATH) -> list[EvalQuestion]:
    """Load eval questions from JSONL, skipping un-filled manual templates."""
    if not path.exists():
        raise FileNotFoundError(f"Question set not found: {path}. Run `make generate-questions` first.")
    questions: list[EvalQuestion] = []
    skipped = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Skip placeholders the user hasn't filled in
        if d.get("question", "").startswith("[FILL IN]"):
            skipped += 1
            continue
        questions.append(EvalQuestion(**d))
    logger.info("questions_loaded", n=len(questions), skipped_placeholders=skipped)
    return questions


def _build_chain(
    store: QdrantStore,
    embedder,
    use_dense: bool,
    use_sparse: bool,
    use_reranker: bool,
    reranker: CrossEncoderReranker | None,
) -> RAGChain:
    """Build a RAGChain with a custom hybrid config (for ablation)."""
    settings = get_settings()
    dense = DenseRetriever(store=store, embedder=embedder)
    sparse = SparseRetriever(store=store, embedder=embedder)
    hybrid_cfg = HybridRetrievalConfig(
        dense_top_k=settings.retrieval.dense_top_k,
        sparse_top_k=settings.retrieval.sparse_top_k,
        rrf_k=settings.retrieval.rrf_k,
        final_top_k=settings.retrieval.hybrid_top_k,
        use_dense=use_dense,
        use_sparse=use_sparse,
    )
    hybrid = HybridRetriever(dense=dense, sparse=sparse, config=hybrid_cfg)
    return RAGChain(
        hybrid=hybrid,
        reranker=reranker,
        retrieval_settings=settings.retrieval,
        use_reranker=use_reranker,
    )


def run_one_config(
    name: str,
    description: str,
    chain: RAGChain,
    questions: list[EvalQuestion],
) -> ConfigResult:
    """Run one config over all questions and compute metrics."""
    per_q_sources: list[list[str]] = []
    per_q_expected: list[list[str]] = []
    abstentions: list[bool] = []
    raw: list[dict[str, Any]] = []

    for i, q in enumerate(questions, start=1):
        resp = chain.answer(q.question, debug=False)
        retrieved_sources = [c["source"] for c in resp.retrieved_chunks]
        per_q_sources.append(retrieved_sources)
        per_q_expected.append(q.expected_sources)
        abstentions.append(resp.abstained)
        raw.append(
            {
                "question": q.question,
                "expected_sources": q.expected_sources,
                "source_kind": q.source_kind,
                "retrieved_sources": retrieved_sources,
                "abstained": resp.abstained,
                "n_chunks": len(resp.retrieved_chunks),
                "latency_ms": resp.latency_ms,
            }
        )
        if i % 10 == 0:
            logger.info("ablation_progress", config=name, done=i, total=len(questions))

    metrics = compute_retrieval_metrics(per_q_sources, per_q_expected, abstentions)
    # Adversarial abstention rate: fraction of adversarial questions that abstained
    adv_indices = [i for i, q in enumerate(questions) if q.source_kind == "adversarial"]
    if adv_indices:
        adv_abstention = sum(abstentions[i] for i in adv_indices) / len(adv_indices)
        metrics_dict = metrics.to_dict() | {
            "adversarial_abstention_rate": round(adv_abstention, 4),
            "n_adversarial": len(adv_indices),
        }
    else:
        metrics_dict = metrics.to_dict()

    return ConfigResult(
        name=name, description=description, metrics=metrics_dict, raw_per_question=raw
    )


def run_ablation(questions: list[EvalQuestion]) -> list[ConfigResult]:
    """Run all 4 ablation configs and return their results."""
    settings = get_settings()
    embedder = build_embedder(settings.embedding)
    store = QdrantStore(embedder=embedder, settings=settings.vectorstore)
    store.ensure_collection(recreate=False)
    reranker = CrossEncoderReranker(settings=settings.embedding)

    configs = [
        ("A_dense_only", "Dense-only baseline (no sparse, no reranker)",
         {"use_dense": True, "use_sparse": False, "use_reranker": False}),
        ("B_dense_plus_rerank", "Dense + cross-encoder reranker",
         {"use_dense": True, "use_sparse": False, "use_reranker": True}),
        ("C_hybrid_no_rerank", "Hybrid (dense + sparse, RRF) without reranker",
         {"use_dense": True, "use_sparse": True, "use_reranker": False}),
        ("D_hybrid_plus_rerank", "Hybrid + cross-encoder reranker (production)",
         {"use_dense": True, "use_sparse": True, "use_reranker": True}),
    ]

    results: list[ConfigResult] = []
    for name, desc, kwargs in configs:
        logger.info("ablation_config_start", name=name)
        chain = _build_chain(store=store, embedder=embedder, reranker=reranker, **kwargs)
        res = run_one_config(name=name, description=desc, chain=chain, questions=questions)
        results.append(res)
        logger.info("ablation_config_done", name=name, metrics=res.metrics)

    return results


def write_results(results: list[ConfigResult], out_dir: Path = RESULTS_DIR) -> None:
    """Write per-config raw JSON + a single markdown summary table."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-config JSON dumps (raw + metrics)
    for r in results:
        path = out_dir / f"ablation_{r.name}.json"
        path.write_text(
            json.dumps(
                {
                    "name": r.name,
                    "description": r.description,
                    "metrics": r.metrics,
                    "per_question": r.raw_per_question,
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    # Markdown summary
    md_lines: list[str] = []
    md_lines.append("# Retrieval Ablation Results\n")
    md_lines.append("Four configurations evaluated on the same question set. "
                     "Bold = best per column.\n")

    header_cols = ["Config", "Description", "Hit@5", "Hit@10", "MRR@10",
                   "nDCG@10", "Abstain%", "Adv.Abstain%"]
    md_lines.append("| " + " | ".join(header_cols) + " |")
    md_lines.append("|" + "|".join(["---"] * len(header_cols)) + "|")

    # Find best per metric for bolding
    metric_keys = ["hit_at_5", "hit_at_10", "mrr_at_10", "ndcg_at_10"]
    best_per_metric = {k: max(r.metrics.get(k, 0.0) for r in results) for k in metric_keys}
    best_adv = max(r.metrics.get("adversarial_abstention_rate", 0.0) for r in results)

    def fmt(value: float, key: str, best: float) -> str:
        s = f"{value:.3f}"
        if abs(value - best) < 1e-6:
            return f"**{s}**"
        return s

    for r in results:
        row = [
            f"`{r.name}`",
            r.description,
            fmt(r.metrics.get("hit_at_5", 0), "hit_at_5", best_per_metric["hit_at_5"]),
            fmt(r.metrics.get("hit_at_10", 0), "hit_at_10", best_per_metric["hit_at_10"]),
            fmt(r.metrics.get("mrr_at_10", 0), "mrr_at_10", best_per_metric["mrr_at_10"]),
            fmt(r.metrics.get("ndcg_at_10", 0), "ndcg_at_10", best_per_metric["ndcg_at_10"]),
            f"{r.metrics.get('abstention_rate', 0)*100:.1f}%",
            fmt(r.metrics.get("adversarial_abstention_rate", 0), "adv", best_adv),
        ]
        md_lines.append("| " + " | ".join(row) + " |")

    # Uplift section (D vs A)
    a = next((r for r in results if r.name == "A_dense_only"), None)
    d = next((r for r in results if r.name == "D_hybrid_plus_rerank"), None)
    if a and d:
        md_lines.append("\n## Production config (D) vs Dense-only baseline (A)\n")
        for key, label in [
            ("hit_at_5", "Hit@5"),
            ("hit_at_10", "Hit@10"),
            ("mrr_at_10", "MRR@10"),
            ("ndcg_at_10", "nDCG@10"),
        ]:
            a_v = a.metrics.get(key, 0.0)
            d_v = d.metrics.get(key, 0.0)
            if a_v > 0:
                uplift = (d_v - a_v) / a_v * 100
                md_lines.append(f"- **{label}**: {a_v:.3f} → {d_v:.3f}  ({uplift:+.1f}%)")

    out_path = out_dir / "ablation_results.md"
    out_path.write_text("\n".join(md_lines) + "\n")
    logger.info("ablation_md_written", path=str(out_path))


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    p = argparse.ArgumentParser(description="Run retrieval ablation across 4 configs.")
    p.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = p.parse_args(argv)

    questions = load_questions(args.questions)
    if not questions:
        logger.error("no_questions_to_eval")
        return 1
    results = run_ablation(questions)
    write_results(results, out_dir=args.out_dir)
    print(f"Ablation complete. Results: {args.out_dir / 'ablation_results.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "ConfigResult",
    "load_questions",
    "main",
    "run_ablation",
    "run_one_config",
    "write_results",
]
