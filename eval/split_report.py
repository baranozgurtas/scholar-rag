"""Per-split (dev / held-out) summaries of an existing run. No models.

    python -m eval.split_report <retrieval run dir>
    python -m eval.split_report <retrieval run dir>/generation_D_hybrid_plus_rerank

Writes `summary_by_split.json` and `report_by_split.md` next to the records.
Every table is labelled INDICATIVE while any question in the run is not
human-reviewed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from eval.harness import (
    render_markdown,
    render_retrieval_markdown,
    round_floats,
    summarize,
    summarize_retrieval,
)
from eval.retrieval_ablation import read_jsonl

SPLITS = ("dev", "heldout")


def indicative_label(records: list[dict[str, Any]]) -> str:
    status = {r["id"]: r.get("review_status") for r in records}
    unreviewed = sum(1 for s in status.values() if s != "human_reviewed")
    if unreviewed:
        return (
            f"**INDICATIVE** — {unreviewed}/{len(status)} questions are not human-reviewed "
            "(LLM-drafted; evidence machine-checked only). Do not cite as a benchmark result."
        )
    return ""


def split_report(run_dir: Path) -> dict[str, Any]:
    generation = (run_dir / "records.jsonl").exists()
    if generation:
        files = {run_dir.name.removeprefix("generation_"): run_dir / "records.jsonl"}
    else:
        files = {p.stem.removeprefix("ranked_"): p for p in sorted(run_dir.glob("ranked_*.jsonl"))}
    if not files:
        raise FileNotFoundError(f"No ranked_*.jsonl or records.jsonl in {run_dir}")

    out: dict[str, Any] = {}
    md: list[str] = []
    for split in SPLITS:
        per_cfg = {}
        all_recs: list[dict[str, Any]] = []
        for cfg, path in files.items():
            recs = [r for r in read_jsonl(path) if r.get("split") == split]
            all_recs += recs
            per_cfg[cfg] = summarize(recs) if generation else summarize_retrieval(recs)
        out[split] = per_cfg
        render = render_markdown if generation else render_retrieval_markdown
        md.append(render(per_cfg, f"{split} split", indicative_label(all_recs)))
    (run_dir / "summary_by_split.json").write_text(
        json.dumps(round_floats(out), indent=2, ensure_ascii=False) + "\n"
    )
    (run_dir / "report_by_split.md").write_text("\n".join(md))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Dev / held-out summaries for a run.")
    p.add_argument("run_dir", type=Path)
    args = p.parse_args(argv)
    split_report(args.run_dir)
    print(f"Wrote {args.run_dir / 'report_by_split.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["indicative_label", "split_report"]
