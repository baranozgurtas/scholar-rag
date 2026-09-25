"""RAGAS evaluation with a local Ollama judge (JUDGE_MODEL).

Scores the saved output of `eval.generation_eval` (one config); it does not
re-run retrieval or generation, so only the judge (and the RAGAS embedding
model) is loaded. Four RAGAS metrics:
- **faithfulness**: every claim in the answer is grounded in the retrieved context
- **answer_relevancy**: the answer addresses the question
- **context_precision**: ratio of relevant chunks among retrieved
- **context_recall**: ratio of relevant info in retrieved vs ground truth

Both the judge LLM and the embeddings used by RAGAS are local (JUDGE_MODEL
and nomic-embed-text, both via Ollama): no OpenAI keys, no data egress.

Only answerable questions whose answer was released (not abstained or
withheld) are scored; RAGAS metrics are not meaningful for abstentions.
Unanswerable-question behaviour is reported by `eval.generation_eval`.

Caveats: the judge is the same local model family as the generator, and a
judge call that fails is reported as missing (None), never as 0 or 1.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from eval.questions import PROJECT_ROOT, load_questions
from rag.config import get_settings
from rag.generation.llm import build_judge_llm
from rag.logging_config import configure_logging, get_logger

logger = get_logger(__name__)

RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def _num(x: Any) -> float | None:
    """float, or None for missing / NaN judge outputs."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def load_generation_records(records_path: Path) -> list[dict[str, Any]]:
    """Generation records + reference answers from the run's question file."""
    manifest = json.loads((records_path.parent / "manifest.json").read_text())
    qpath = PROJECT_ROOT / manifest["questions"]["path"]
    refs = {q.id: q.reference_answer for q in load_questions(qpath)}
    out = []
    for line in records_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out.append(
            {
                "id": r["id"],
                "question": r["question"],
                "answer": r["answer"],
                "abstained": r["abstained"],
                "contexts": [c["text"] for c in r["context_chunks"]],
                "ground_truth": refs[r["id"]],
                "source_kind": r["source_kind"],
                "expected_sources": r["expected_sources"],
                "outcome": r["outcome"],
            }
        )
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

    # Score only answerable questions with a released answer
    scorable = [r for r in records if r.get("expected_sources") and not r.get("abstained")]
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
    df = result.to_pandas()
    per_question = []
    for r, (_, row) in zip(scorable, df.iterrows(), strict=True):
        per_question.append(
            {"question": r["question"], "source_kind": r["source_kind"]}
            | {m: _num(row.get(m)) for m in RAGAS_METRICS}
        )
    aggregate: dict[str, float] = {}
    n_valid: dict[str, int] = {}
    for m in RAGAS_METRICS:
        vals = [pq[m] for pq in per_question if pq[m] is not None]
        n_valid[m] = len(vals)
        if vals:
            aggregate[m] = sum(vals) / len(vals)

    return {
        "aggregate": {k: round(v, 4) for k, v in aggregate.items()},
        "n_valid_per_metric": n_valid,
        "per_question": per_question,
        "n_scored": len(scorable),
    }


def _write_outputs(
    pipeline_records: list[dict[str, Any]],
    ragas_result: dict[str, Any],
    out_dir: Path,
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
    settings = get_settings()
    agg = ragas_result.get("aggregate", {})
    n_valid = ragas_result.get("n_valid_per_metric", {})
    n_scored = ragas_result.get("n_scored", 0)
    unans = [r for r in pipeline_records if not r.get("expected_sources")]
    unans_abst = sum(1 for r in unans if r["abstained"])

    def cell(m: str) -> str:
        v = agg.get(m)
        return "n/a" if v is None else f"{v:.3f} (n={n_valid.get(m, 0)})"

    md = [
        "# RAGAS Evaluation Summary\n",
        f"**Generator**: {settings.llm.generator_model}. "
        f"**Judge**: {settings.llm.judge_model} (T={settings.llm.judge_temperature}); "
        "same model family, so absolute scores may be biased.\n",
        f"**Scored**: {n_scored} answerable questions with a released answer.\n",
        "",
        "## RAGAS metrics (LLM-judged, not human-verified)\n",
        "| Metric | Score | What it estimates |",
        "|---|---|---|",
        f"| Faithfulness | {cell('faithfulness')} | Share of answer statements the judge infers from the context |",
        f"| Answer relevancy | {cell('answer_relevancy')} | Whether the answer addresses the question |",
        f"| Context precision | {cell('context_precision')} | Share of retrieved chunks judged relevant |",
        f"| Context recall | {cell('context_recall')} | Share of reference-answer content found in context |",
        "",
        f"Unanswerable questions abstained: {unans_abst}/{len(unans)}",
        "",
    ]
    (out_dir / "ragas_summary.md").write_text("\n".join(md) + "\n")
    logger.info("ragas_outputs_written", dir=str(out_dir))


def run_ragas_eval(records_path: Path, out_dir: Path | None = None) -> dict[str, Any]:
    records = load_generation_records(records_path)
    ragas_result = _run_ragas(records)
    _write_outputs(records, ragas_result, out_dir or records_path.parent)
    return ragas_result


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    p = argparse.ArgumentParser(description="RAGAS (LLM judge) on saved generation records.")
    p.add_argument("records", type=Path, help="records.jsonl from eval.generation_eval")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args(argv)
    result = run_ragas_eval(args.records, args.out_dir)
    print(f"RAGAS aggregate (LLM-judged): {result.get('aggregate', {})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["load_generation_records", "main", "run_ragas_eval"]
