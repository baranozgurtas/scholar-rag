"""Answer-generation eval for ONE retrieval config, one question at a time.

Reads `ranked_<config>.jsonl` from a finished retrieval-only run
(`eval.retrieval_ablation`), takes the saved top-`final_top_k` chunks, applies
the same pre-generation gate and `AnswerGenerator` (prompt → LLM → answer
policy) as the API, and appends one record per question, flushed to disk.

No embedder or reranker is loaded here; only the Ollama generator is called.
Re-running the same command resumes: finished question IDs are skipped. It
refuses to resume if the generator model, its Ollama digest, the prompt
version or the retrieval run differ from the manifest, so one output
directory never mixes models.

If the generator call fails (e.g. Ollama stops), the run stops without
writing that question; nothing is recorded as a score for it.

    python -m eval.generation_eval eval/results/runs/<retrieval run> \\
        --config D_hybrid_plus_rerank

LLM-judged grounding is a separate step: `python -m eval.ragas_eval <records>`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.harness import render_markdown, round_floats, summarize
from eval.retrieval_ablation import (
    CONFIG_BY_NAME,
    PreflightError,
    append_jsonl,
    git_state,
    read_jsonl,
)

DEFAULT_CONFIG = "D_hybrid_plus_rerank"


def ollama_model_info(base_url: str, model: str) -> dict[str, Any]:
    """Ollama's metadata for `model`; raise PreflightError if unavailable."""
    import httpx

    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        r.raise_for_status()
    except Exception as e:
        raise PreflightError(f"Ollama not reachable at {base_url}: {e}") from e
    models = {m["name"]: m for m in r.json().get("models", [])}
    if model not in models:
        raise PreflightError(f"Generator {model!r} not pulled in Ollama (available: {sorted(models)})")
    return models[model]


def generation_record(ranked: dict[str, Any], result: Any | None, gate: str | None, final: list[Any]) -> dict[str, Any]:
    """Merge a retrieval record with a generation result (or a gate abstention)."""
    from rag.guards.citation_checker import ABSTENTION_TEXT

    rec = {
        k: ranked[k]
        for k in (
            "id", "question", "split", "review_status", "source_kind", "expected_sources",
            "ranked_sources", "top_rerank_score", "top_dense_score", "config",
        )
    }
    rec["context_chunks"] = [{"chunk_id": c.chunk_id, "text": c.text} for c in final]
    lat = dict(ranked["latency_ms"])
    if gate is not None:
        rec |= {
            "outcome": gate,
            "abstained": True,
            "answer": ABSTENTION_TEXT,
            "raw_answer": "",
            "citations": [],
            "citation_check": {"n_extracted": 0, "n_valid": 0, "n_invalid": 0, "invalid_tags": [], "all_valid": False},
        }
        lat["generation_ms"] = 0.0
    else:
        rec |= {
            "outcome": result.outcome,
            "abstained": result.abstained,
            "answer": result.answer,
            "raw_answer": result.raw_answer,
            "citations": result.citations,
            "citation_check": result.citation_check,
        }
        lat["generation_ms"] = result.generation_ms
    # Retrieval and generation ran in separate processes; total is their sum.
    lat["total_ms"] = sum(float(lat.get(k, 0.0)) for k in ("retrieval_ms", "rerank_ms", "generation_ms"))
    rec["latency_ms"] = lat
    rec["error"] = None
    return rec


def run_generation_eval(retrieval_run: Path, config: str, out_dir: Path | None = None, limit: int | None = None) -> Path:
    from rag.config import get_settings
    from rag.generation.prompts import RAG_ANSWER_PROMPT_VERSION
    from rag.generation.rag_chain import AnswerGenerator, pre_generation_gate
    from rag.guards.citation_checker import AnswerOutcome
    from rag.retrieval.types import RetrievedChunk

    settings = get_settings()
    ranked_path = retrieval_run / f"ranked_{config}.jsonl"
    ranked = read_jsonl(ranked_path)[:limit]
    if not ranked:
        raise PreflightError(f"No retrieval records at {ranked_path}; run eval.retrieval_ablation first.")
    retrieval_manifest = json.loads((retrieval_run / "manifest.json").read_text())

    info = ollama_model_info(settings.llm.ollama_base_url, settings.llm.generator_model)
    manifest = {
        "kind": "generation",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_state(),
        "retrieval_run": retrieval_run.name,
        "retrieval_config": config,
        "retrieval_manifest_models": retrieval_manifest["models"],
        "questions": retrieval_manifest["questions"],
        "corpus": retrieval_manifest["corpus"],
        "generator": {
            "model": settings.llm.generator_model,
            "ollama_digest": info.get("digest"),
            "ollama_details": info.get("details"),
            "temperature": settings.llm.generator_temperature,
            "max_tokens": settings.llm.generator_max_tokens,
        },
        "prompt_version": f"{RAG_ANSWER_PROMPT_VERSION.name}@{RAG_ANSWER_PROMPT_VERSION.version}",
        "final_top_k": settings.retrieval.final_top_k,
        "rerank_score_threshold": settings.retrieval.rerank_score_threshold,
    }

    out_dir = out_dir or retrieval_run / f"generation_{config}"
    records_path = out_dir / "records.jsonl"
    if (out_dir / "manifest.json").exists():
        old = json.loads((out_dir / "manifest.json").read_text())
        for key in ("generator", "prompt_version", "retrieval_run", "retrieval_config", "final_top_k", "rerank_score_threshold"):
            if old.get(key) != manifest.get(key):
                raise PreflightError(f"Cannot resume {out_dir}: {key} changed ({old.get(key)!r} -> {manifest.get(key)!r}).")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    finished = {r["id"] for r in read_jsonl(records_path)}
    generator = AnswerGenerator()
    final_top_k = settings.retrieval.final_top_k
    for i, rr in enumerate(ranked, start=1):
        if rr["id"] in finished:
            continue
        final = [RetrievedChunk.from_record(d) for d in rr["ranked"][:final_top_k]]
        gate = pre_generation_gate(final, rr.get("top_rerank_score"), settings.retrieval.rerank_score_threshold)
        result = None
        if gate is None:
            t0 = time.perf_counter()
            result = generator.generate(rr["question"], final)
            if result.outcome == AnswerOutcome.GENERATION_ERROR:
                raise PreflightError(
                    f"Generator failed on {rr['id']} after {time.perf_counter() - t0:.0f}s: "
                    f"{result.raw_answer[:200]}. Nothing written for it; re-run to resume."
                )
        append_jsonl(records_path, generation_record(rr, result, gate, final))
        print(f"[generation:{config}] {i}/{len(ranked)} {rr['id']} outcome={gate or result.outcome}", flush=True)

    records = read_jsonl(records_path)
    summary = summarize(records)
    (out_dir / "summary.json").write_text(json.dumps(round_floats(summary), indent=2, ensure_ascii=False) + "\n")
    preamble = (
        f"Generation run for `{config}` from retrieval run `{retrieval_run.name}` · generator "
        f"`{manifest['generator']['model']}` (digest `{(manifest['generator']['ollama_digest'] or '')[:12]}`) · "
        f"{len(records)}/{len(ranked)} questions complete · review status {manifest['questions']['review_status']}."
    )
    (out_dir / "report.md").write_text(render_markdown({config: summary}, "Generation evaluation", preamble))
    return out_dir


def main(argv: list[str] | None = None) -> int:
    from rag.logging_config import configure_logging

    configure_logging()
    p = argparse.ArgumentParser(description="Generation eval for one config, resumable.")
    p.add_argument("retrieval_run", type=Path)
    p.add_argument("--config", default=DEFAULT_CONFIG, choices=sorted(CONFIG_BY_NAME))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)
    try:
        out = run_generation_eval(args.retrieval_run, args.config, args.out_dir, args.limit)
    except PreflightError as e:
        print(f"Generation evaluation stopped: {e}", file=sys.stderr)
        return 2
    print(f"Generation report: {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["generation_record", "main", "run_generation_eval"]
