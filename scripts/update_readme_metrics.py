"""Regenerate the README results table from a computed summary file.

    python scripts/update_readme_metrics.py                       # legacy re-analysis
    python scripts/update_readme_metrics.py eval/results/runs/<run>/summary.json

Only the block between these markers is replaced; nothing is appended:
    <!-- METRICS:ABLATION:START -->
    <!-- METRICS:ABLATION:END -->

Every number comes from the summary file. If the file or the markers are
missing the script fails instead of writing placeholders.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DEFAULT_SUMMARY = ROOT / "eval" / "results" / "legacy_reanalysis.json"
START = "<!-- METRICS:ABLATION:START -->"
END = "<!-- METRICS:ABLATION:END -->"

DESCRIPTIONS = {
    "A_dense_only": "Dense only",
    "B_dense_plus_rerank": "Dense + rerank",
    "C_hybrid_no_rerank": "Hybrid (RRF), no rerank",
    "D_hybrid_plus_rerank": "Hybrid + rerank (API default)",
}


def _rate(r: dict) -> str:
    return f"{r['count']}/{r['n']}" if r["n"] else "n/a"


def render(summaries: dict) -> str:
    depth = min(s["retrieval"]["min_ranking_depth"] for s in summaries.values())
    at = "@5" if depth < 10 else "@10"
    lines = [
        f"| Config | Hit@5 | MRR{at} | nDCG{at} | False abstention (answerable) | Answered unanswerable | p50 / p95 latency (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    best_mrr = max(s["retrieval"]["mrr_at_10"] for s in summaries.values())
    for name, s in summaries.items():
        r, a, lt = s["retrieval"], s["abstention"], s["latency"]
        mrr = f"{r['mrr_at_10']:.3f}"
        if abs(r["mrr_at_10"] - best_mrr) < 1e-9:
            mrr = f"**{mrr}**"
        lines.append(
            f"| `{name}` {DESCRIPTIONS.get(name, '')} | {r['hit_at_5']:.3f} | {mrr} | {r['ndcg_at_10']:.3f} | "
            f"{_rate(a['false_abstention_on_answerable'])} | {_rate(a['false_answer_on_unanswerable'])} | "
            f"{lt['total_ms_p50'] / 1000:.1f} / {lt['total_ms_p95'] / 1000:.1f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0]) if argv else DEFAULT_SUMMARY
    if not path.exists():
        print(f"ERROR: summary not found: {path}", file=sys.stderr)
        return 1
    text = README.read_text()
    if START not in text or END not in text:
        print("ERROR: README markers missing; not modifying README.", file=sys.stderr)
        return 1
    table = render(json.loads(path.read_text()))
    pattern = re.compile(f"{re.escape(START)}.*?{re.escape(END)}", flags=re.DOTALL)
    README.write_text(pattern.sub(lambda _m: f"{START}\n{table}\n{END}", text))
    print(f"README table updated from {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
