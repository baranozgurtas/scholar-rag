"""Retrieval-only 4-config ablation. No generator (Qwen) is called.

    A) Dense-only            B) Dense + reranker
    C) Hybrid (dense+sparse) D) Hybrid + reranker (API default)

B reranks A's candidates and D reranks C's, so the run has two phases and
never holds the embedder and the reranker in memory together:

  Phase 1 (embedder only): for each question, the top `RETRIEVAL_HYBRID_TOP_K`
      candidates for "dense" and "hybrid", with text and scores
      → candidates_dense.jsonl, candidates_hybrid.jsonl
  Phase 2 (reranker only, embedder released): rerank saved candidates
      → ranked_<config>.jsonl (top RANKING_DEPTH=10 per question)

Every file is appended one question at a time and flushed, so an interrupted
run resumes with `--run-dir <existing dir>`; finished questions are skipped.
Output also holds manifest.json, summary.json and report.md.

    python -m eval.retrieval_ablation                                   # legacy 25
    python -m eval.retrieval_ablation --questions eval/questions_v2_draft.jsonl
    python -m eval.retrieval_ablation --run-dir eval/results/runs/<dir>   # resume

Generation is evaluated separately for one config: `eval.generation_eval`.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from eval.harness import render_retrieval_markdown, summarize_retrieval
from eval.questions import PDF_DIR, QUESTIONS_PATH, EvalQuestion, file_sha256, load_questions

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RUNS_DIR = RESULTS_DIR / "runs"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANKING_DEPTH = 10


@dataclass(frozen=True)
class AblationConfig:
    name: str
    description: str
    candidates: str  # "dense" | "hybrid"
    use_reranker: bool


CONFIGS: list[AblationConfig] = [
    AblationConfig("A_dense_only", "Dense only", "dense", False),
    AblationConfig("B_dense_plus_rerank", "Dense + cross-encoder reranker", "dense", True),
    AblationConfig("C_hybrid_no_rerank", "Hybrid (dense + sparse, RRF)", "hybrid", False),
    AblationConfig("D_hybrid_plus_rerank", "Hybrid + cross-encoder reranker", "hybrid", True),
]
CONFIG_BY_NAME = {c.name: c for c in CONFIGS}


class PreflightError(RuntimeError):
    """A required service, model or index is unavailable; no metrics are produced."""


# ─── Small I/O helpers ────────────────────────────────────────────


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one record and force it to disk (crash-safe resume)."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def done_ids(path: Path) -> set[str]:
    return {r["id"] for r in read_jsonl(path)}


def source_alias(source: str) -> str:
    return source[:-4] if source.lower().endswith(".pdf") else source


# ─── Manifest ─────────────────────────────────────────────────────


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions(names: list[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def git_state() -> dict[str, Any]:
    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(status),
        "n_modified_paths": len(status.splitlines()) if status else 0,
    }


def build_manifest(
    settings: Any,
    questions_path: Path,
    questions: list[EvalQuestion],
    collection_stats: dict[str, Any],
    configs: list[AblationConfig],
) -> dict[str, Any]:
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    qpath = questions_path.resolve()
    return {
        "kind": "retrieval_only",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_state(),
        "models": {
            "embedding": settings.embedding.model_name,
            "reranker": settings.embedding.reranker_model,
            "embedding_device": settings.embedding.device,
            "reranker_device": settings.embedding.reranker_device,
            "generator": None,  # not used by this run
        },
        "retrieval_settings": settings.retrieval.model_dump(),
        "chunking_settings": settings.chunking.model_dump(),
        "ranking_depth": RANKING_DEPTH,
        "configs": [c.__dict__ for c in configs],
        "corpus": {
            "n_pdfs": len(pdfs),
            "pdfs": {p.stem: file_sha256(p)[:16] for p in pdfs},
            "qdrant_backend": (
                f"embedded:{settings.vectorstore.path}" if settings.vectorstore.path else settings.vectorstore.url
            ),
            "qdrant_collection": collection_stats.get("collection"),
            "qdrant_points": collection_stats.get("points_count"),
            "qdrant_unique_sources": collection_stats.get("unique_sources"),
        },
        "questions": {
            "path": str(qpath.relative_to(PROJECT_ROOT)) if qpath.is_relative_to(PROJECT_ROOT) else str(qpath),
            "sha256": file_sha256(questions_path),
            "n": len(questions),
            "n_answerable": sum(q.answerable for q in questions),
            "n_unanswerable": sum(not q.answerable for q in questions),
            "review_status": {
                s: sum(q.review_status == s for q in questions) for s in sorted({q.review_status for q in questions})
            },
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions(
                ["torch", "FlagEmbedding", "transformers", "qdrant-client", "langchain-ollama"]
            ),
        },
    }


def check_resume_compatible(old: dict[str, Any], new: dict[str, Any]) -> None:
    """Refuse to resume if anything that affects results changed."""
    keys = [
        ("questions", "sha256"),
        ("models", "embedding"),
        ("models", "reranker"),
        ("corpus", "qdrant_points"),
        ("corpus", "pdfs"),
        ("retrieval_settings",),
        ("chunking_settings",),
    ]
    for path in keys:
        a, b = old, new
        for k in path:
            a, b = a.get(k), b.get(k)
        if a != b:
            raise PreflightError(f"Cannot resume: {'.'.join(path)} changed ({a!r} -> {b!r}).")


# ─── Records ──────────────────────────────────────────────────────


def question_fields(q: EvalQuestion) -> dict[str, Any]:
    return {
        "id": q.id,
        "question": q.question,
        "split": q.split,
        "review_status": q.review_status,
        "source_kind": q.source_kind,
        "expected_sources": q.expected_sources,
    }


def ranked_record(
    cand: dict[str, Any], config: AblationConfig, ranked: list[Any], rerank_ms: float, final_top_k: int
) -> dict[str, Any]:
    """Per-question ranking record. `ranked` is a list of RetrievedChunk."""
    rec = {k: cand[k] for k in ("id", "question", "split", "review_status", "source_kind", "expected_sources")}
    rec |= {
        "config": config.name,
        "ranked_sources": [source_alias(c.source) for c in ranked],
        # Text is kept only for the chunks the generator would see.
        "ranked": [c.to_record(include_text=i < final_top_k) for i, c in enumerate(ranked)],
        "top_rerank_score": ranked[0].score if (config.use_reranker and ranked) else None,
        "top_dense_score": cand.get("top_dense_score"),
        "latency_ms": {"retrieval_ms": cand["retrieval_ms"], "rerank_ms": rerank_ms},
        "error": None,
    }
    return rec


# ─── Phases ───────────────────────────────────────────────────────


def phase1_candidates(
    run_dir: Path, questions: list[EvalQuestion], kinds: list[str], settings: Any, store: Any, embedder: Any
) -> None:
    """Embedder only: save top-k candidates for each candidate kind."""
    from rag.retrieval.dense_retriever import DenseRetriever
    from rag.retrieval.hybrid_retriever import HybridRetrievalConfig, HybridRetriever
    from rag.retrieval.sparse_retriever import SparseRetriever

    r = settings.retrieval
    for kind in kinds:
        hybrid = HybridRetriever(
            dense=DenseRetriever(store=store, embedder=embedder),
            sparse=SparseRetriever(store=store, embedder=embedder),
            config=HybridRetrievalConfig(
                dense_top_k=r.dense_top_k,
                sparse_top_k=r.sparse_top_k,
                rrf_k=r.rrf_k,
                final_top_k=r.hybrid_top_k,
                use_dense=True,
                use_sparse=(kind == "hybrid"),
            ),
        )
        path = run_dir / f"candidates_{kind}.jsonl"
        finished = done_ids(path)
        for i, q in enumerate(questions, start=1):
            if q.id in finished:
                continue
            t0 = time.perf_counter()
            cands = hybrid.retrieve(q.question, top_k=r.hybrid_top_k)
            ms = (time.perf_counter() - t0) * 1000
            dense_scores = [c.score_breakdown.get("dense_score") for c in cands]
            dense_scores = [s for s in dense_scores if s]
            append_jsonl(
                path,
                question_fields(q)
                | {
                    "kind": kind,
                    "candidates": [c.to_record() for c in cands],
                    "top_dense_score": max(dense_scores) if dense_scores else None,
                    "retrieval_ms": ms,
                },
            )
            print(f"[candidates:{kind}] {i}/{len(questions)} {q.id}", flush=True)


def phase2_rank(run_dir: Path, configs: list[AblationConfig], settings: Any, reranker: Any | None) -> None:
    """Reranker only (or no model): turn saved candidates into top-10 rankings."""
    from rag.retrieval.types import RetrievedChunk

    final_top_k = settings.retrieval.final_top_k
    for cfg in configs:
        cands = read_jsonl(run_dir / f"candidates_{cfg.candidates}.jsonl")
        path = run_dir / f"ranked_{cfg.name}.jsonl"
        finished = done_ids(path)
        for i, cand in enumerate(cands, start=1):
            if cand["id"] in finished:
                continue
            chunks = [RetrievedChunk.from_record(d) for d in cand["candidates"]]
            t0 = time.perf_counter()
            if cfg.use_reranker and chunks:
                assert reranker is not None
                ranked = reranker.rerank(query=cand["question"], candidates=chunks, top_k=RANKING_DEPTH)
            else:
                ranked = chunks[:RANKING_DEPTH]
            ms = (time.perf_counter() - t0) * 1000 if cfg.use_reranker else 0.0
            append_jsonl(path, ranked_record(cand, cfg, ranked, ms, final_top_k))
            print(f"[{cfg.name}] {i}/{len(cands)} {cand['id']}", flush=True)


def release_torch_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def write_summary(run_dir: Path, configs: list[AblationConfig], manifest: dict[str, Any]) -> dict[str, Any]:
    summaries = {c.name: summarize_retrieval(read_jsonl(run_dir / f"ranked_{c.name}.jsonl")) for c in configs}
    (run_dir / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n")
    preamble = (
        f"Retrieval-only run `{run_dir.name}` · no generator called · embedding "
        f"`{manifest['models']['embedding']}` · reranker `{manifest['models']['reranker']}` · questions "
        f"`{manifest['questions']['path']}` (n={manifest['questions']['n']}, review status "
        f"{manifest['questions']['review_status']}) · commit `{manifest['git']['commit']}`"
        f"{' (dirty tree)' if manifest['git']['dirty'] else ''} · {manifest['corpus']['n_pdfs']} PDFs / "
        f"{manifest['corpus']['qdrant_points']} chunks."
    )
    (run_dir / "report.md").write_text(render_retrieval_markdown(summaries, "Retrieval ablation", preamble))
    return summaries


def run_retrieval_ablation(
    questions_path: Path = QUESTIONS_PATH,
    configs: list[AblationConfig] = CONFIGS,
    runs_dir: Path = RUNS_DIR,
    run_dir: Path | None = None,
    limit: int | None = None,
) -> Path:
    from rag.config import get_settings
    from rag.vectorstore.qdrant_store import QdrantStore

    settings = get_settings()
    questions = load_questions(questions_path)[:limit]

    # Collection check before loading any model.
    store = QdrantStore(embedder=None, settings=settings.vectorstore)  # type: ignore[arg-type]
    try:
        stats = store.collection_stats()
    except Exception as e:
        raise PreflightError(f"Qdrant collection unavailable: {e}") from e
    if not stats.get("points_count"):
        raise PreflightError("Qdrant collection is empty; run `make ingest` first.")

    manifest = build_manifest(settings, questions_path, questions, stats, configs)
    if run_dir is not None and (run_dir / "manifest.json").exists():
        check_resume_compatible(json.loads((run_dir / "manifest.json").read_text()), manifest)
        print(f"Resuming {run_dir}", flush=True)
    else:
        sha = (manifest["git"]["commit"] or "nogit")[:8] + ("-dirty" if manifest["git"]["dirty"] else "")
        run_dir = run_dir or runs_dir / f"retrieval_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{sha}"
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # Phase 1: embedder only.
    kinds = sorted({c.candidates for c in configs})
    if any(len(done_ids(run_dir / f"candidates_{k}.jsonl")) < len(questions) for k in kinds):
        from rag.embeddings.bge_embedder import build_embedder

        embedder = build_embedder(settings.embedding)
        store.embedder = embedder
        phase1_candidates(run_dir, questions, kinds, settings, store, embedder)
        store.embedder = None
        del embedder
        release_torch_memory()

    # Phase 2: reranker only.
    reranker = None
    if any(c.use_reranker and len(done_ids(run_dir / f"ranked_{c.name}.jsonl")) < len(questions) for c in configs):
        from rag.retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(settings=settings.embedding)
    phase2_rank(run_dir, configs, settings, reranker)
    del reranker
    release_torch_memory()

    write_summary(run_dir, configs, json.loads((run_dir / "manifest.json").read_text()))
    return run_dir


def main(argv: list[str] | None = None) -> int:
    from rag.logging_config import configure_logging

    configure_logging()
    p = argparse.ArgumentParser(description="Retrieval-only 4-config ablation (no generator).")
    p.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    p.add_argument("--configs", nargs="*", default=[c.name for c in CONFIGS])
    p.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    p.add_argument("--run-dir", type=Path, default=None, help="Resume (or name) a run directory.")
    p.add_argument("--limit", type=int, default=None, help="Only the first N questions (smoke runs).")
    args = p.parse_args(argv)

    unknown = set(args.configs) - set(CONFIG_BY_NAME)
    if unknown:
        p.error(f"unknown configs: {sorted(unknown)}")
    selected = [CONFIG_BY_NAME[n] for n in args.configs]
    try:
        run_dir = run_retrieval_ablation(args.questions, selected, args.runs_dir, args.run_dir, args.limit)
    except PreflightError as e:
        print(f"Retrieval evaluation did not run: {e}", file=sys.stderr)
        return 2
    print(f"Run complete: {run_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "CONFIGS",
    "RANKING_DEPTH",
    "AblationConfig",
    "PreflightError",
    "append_jsonl",
    "build_manifest",
    "phase2_rank",
    "ranked_record",
    "read_jsonl",
    "run_retrieval_ablation",
]
