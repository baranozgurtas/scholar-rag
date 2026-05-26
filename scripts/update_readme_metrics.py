"""Replace placeholder metrics in README.md with real eval results.

Reads:
    eval/results/ablation_<config>.json   (per-config metrics)
    eval/results/ragas_eval.json          (RAGAS aggregate)

Updates the README's ablation table and RAGAS panel between the markers:
    <!-- METRICS:ABLATION:START -->
    ...
    <!-- METRICS:ABLATION:END -->

and:
    <!-- METRICS:RAGAS:START -->
    ...
    <!-- METRICS:RAGAS:END -->

Run with:
    python scripts/update_readme_metrics.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"
README = ROOT / "README.md"

ABLATION_MARK_START = "<!-- METRICS:ABLATION:START -->"
ABLATION_MARK_END = "<!-- METRICS:ABLATION:END -->"
RAGAS_MARK_START = "<!-- METRICS:RAGAS:START -->"
RAGAS_MARK_END = "<!-- METRICS:RAGAS:END -->"
BULLET_MARK_START = "<!-- METRICS:CV_BULLET:START -->"
BULLET_MARK_END = "<!-- METRICS:CV_BULLET:END -->"


def _load_ablation() -> list[dict]:
    """Load per-config ablation JSON files in canonical order A → D."""
    canonical = [
        "A_dense_only",
        "B_dense_plus_rerank",
        "C_hybrid_no_rerank",
        "D_hybrid_plus_rerank",
    ]
    out = []
    for name in canonical:
        path = RESULTS / f"ablation_{name}.json"
        if path.exists():
            out.append(json.loads(path.read_text()))
    return out


def _render_ablation_table(configs: list[dict]) -> str:
    if not configs:
        return "_No ablation results found. Run `make eval` first._"

    lines = [
        "| Config | Description | Hit@5 | Hit@10 | MRR@10 | nDCG@10 | Abstain% | Adv.Abstain% |",
        "|---|---|---|---|---|---|---|---|",
    ]
    # Find best per metric for bolding
    metric_keys = ["hit_at_5", "hit_at_10", "mrr_at_10", "ndcg_at_10"]
    best = {k: max(c["metrics"].get(k, 0.0) for c in configs) for k in metric_keys}
    best_adv = max(c["metrics"].get("adversarial_abstention_rate", 0.0) for c in configs)

    def fmt(value: float, best_value: float) -> str:
        s = f"{value:.3f}"
        return f"**{s}**" if abs(value - best_value) < 1e-6 else s

    for c in configs:
        m = c["metrics"]
        lines.append(
            "| `{name}` | {desc} | {h5} | {h10} | {mrr} | {ndcg} | {abst} | {adv} |".format(
                name=c["name"],
                desc=c["description"],
                h5=fmt(m.get("hit_at_5", 0), best["hit_at_5"]),
                h10=fmt(m.get("hit_at_10", 0), best["hit_at_10"]),
                mrr=fmt(m.get("mrr_at_10", 0), best["mrr_at_10"]),
                ndcg=fmt(m.get("ndcg_at_10", 0), best["ndcg_at_10"]),
                abst=f"{m.get('abstention_rate', 0)*100:.1f}%",
                adv=fmt(m.get("adversarial_abstention_rate", 0), best_adv),
            )
        )

    # Uplift block
    a = next((c for c in configs if c["name"] == "A_dense_only"), None)
    d = next((c for c in configs if c["name"] == "D_hybrid_plus_rerank"), None)
    if a and d:
        lines.append("")
        lines.append("**Production (D) vs Dense-only baseline (A):**")
        for key, label in [
            ("hit_at_5", "Hit@5"),
            ("hit_at_10", "Hit@10"),
            ("mrr_at_10", "MRR@10"),
            ("ndcg_at_10", "nDCG@10"),
        ]:
            av = a["metrics"].get(key, 0.0)
            dv = d["metrics"].get(key, 0.0)
            uplift = (dv - av) / av * 100 if av > 0 else 0
            lines.append(f"- **{label}**: {av:.3f} → {dv:.3f}  (**{uplift:+.1f}%**)")
    return "\n".join(lines)


def _render_ragas() -> str:
    path = RESULTS / "ragas_eval.json"
    if not path.exists():
        return "_No RAGAS results found. Run `make eval-ragas` first._"
    data = json.loads(path.read_text())
    agg = data.get("ragas", {}).get("aggregate", {})
    n = data.get("ragas", {}).get("n_scored", 0)
    records = data.get("records", [])
    adv = [r for r in records if r.get("source_kind") == "adversarial"]
    adv_correct = sum(1 for r in adv if r.get("abstained"))

    lines = [
        f"_Evaluated on **{n}** scorable questions (judge: Qwen2.5:14b local, T=0)._",
        "",
        "| Metric | Score | What it measures |",
        "|---|---|---|",
        f"| **Faithfulness** | {agg.get('faithfulness', 0):.3f} | Fraction of answer claims grounded in retrieved context |",
        f"| **Answer relevancy** | {agg.get('answer_relevancy', 0):.3f} | Whether the answer addresses the question |",
        f"| **Context precision** | {agg.get('context_precision', 0):.3f} | Fraction of retrieved chunks that are relevant |",
        f"| **Context recall** | {agg.get('context_recall', 0):.3f} | Fraction of ground-truth info captured by retrieval |",
        "",
        f"**Adversarial abstention**: {adv_correct}/{len(adv)} ({(adv_correct/max(1,len(adv)))*100:.0f}%) — out-of-corpus questions correctly refused.",
    ]
    return "\n".join(lines)


def _render_cv_bullet() -> str:
    configs = _load_ablation()
    ragas_path = RESULTS / "ragas_eval.json"
    if not configs or not ragas_path.exists():
        return "_Run `make eval` to populate._"
    a = next((c for c in configs if c["name"] == "A_dense_only"), None)
    d = next((c for c in configs if c["name"] == "D_hybrid_plus_rerank"), None)
    ragas_data = json.loads(ragas_path.read_text())
    agg = ragas_data.get("ragas", {}).get("aggregate", {})
    n_scored = ragas_data.get("ragas", {}).get("n_scored", 0)

    if a and d:
        a_cp_proxy = a["metrics"].get("mrr_at_10", 0)
        d_cp_proxy = d["metrics"].get("mrr_at_10", 0)
        uplift = (d_cp_proxy - a_cp_proxy) / a_cp_proxy * 100 if a_cp_proxy > 0 else 0
    else:
        uplift = 0
    faith = agg.get("faithfulness", 0)

    bullet = (
        f"> **Production-grade RAG system for academic literature QA with hybrid retrieval "
        f"(BGE-M3 dense + sparse + Reciprocal Rank Fusion) and BGE cross-encoder reranking; "
        f"achieves **{faith:.2f} RAGAS faithfulness** and **{uplift:+.0f}% MRR@10** over "
        f"dense-only baseline on {n_scored}-question eval set, served via FastAPI + Qdrant + "
        f"Ollama (Qwen2.5:14b) with Docker, CI, and Langfuse observability.**\n"
        f"> _Stack: Python, LangChain, LangGraph, Qdrant, BGE-M3, BGE-reranker-v2, Ollama, "
        f"Qwen2.5, FastAPI, RAGAS, Langfuse, Docker, GitHub Actions._"
    )
    return bullet


def _patch_between(text: str, start: str, end: str, replacement: str) -> str:
    pattern = re.compile(
        f"({re.escape(start)})(.*?)({re.escape(end)})",
        flags=re.DOTALL,
    )
    if not pattern.search(text):
        # Markers absent — append at end with markers
        return text + f"\n\n{start}\n{replacement}\n{end}\n"
    return pattern.sub(lambda _m: f"{start}\n{replacement}\n{end}", text)


def main() -> int:
    if not README.exists():
        print(f"ERROR: README not found at {README}", file=sys.stderr)
        return 1
    text = README.read_text()
    text = _patch_between(text, ABLATION_MARK_START, ABLATION_MARK_END, _render_ablation_table(_load_ablation()))
    text = _patch_between(text, RAGAS_MARK_START, RAGAS_MARK_END, _render_ragas())
    text = _patch_between(text, BULLET_MARK_START, BULLET_MARK_END, _render_cv_bullet())
    README.write_text(text)
    print(f"✓ README updated with eval results from {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
