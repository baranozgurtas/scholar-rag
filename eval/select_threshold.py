"""Choose a rerank-score abstention threshold on dev, report it on held-out.

The chain's threshold is a gate on the top-1 reranker score applied *before*
generation (see `RAGChain.answer`). Given a run's records (which store
`top_rerank_score` and the generator's outcome), the effect of any threshold
t can therefore be replayed offline: a question abstains if it already
abstained, or if its top score < t. Outputs above t are the recorded ones.

Selection uses only `split == "dev"` records. Held-out records are scored
once, at the chosen threshold. `legacy_test` records are refused, because
that is the set the README's numbers come from.

    python -m eval.select_threshold eval/results/runs/<run>/records_D_hybrid_plus_rerank.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def apply_threshold(records: list[dict[str, Any]], t: float) -> list[bool]:
    """Abstention decision per record if the top-1 gate were set to `t`."""
    out = []
    for r in records:
        s = r.get("top_rerank_score")
        out.append(bool(r.get("abstained")) or (t > 0 and s is not None and s < t))
    return out


def error_rates(records: list[dict[str, Any]], t: float) -> dict[str, Any]:
    abst = apply_threshold(records, t)
    ans = [a for r, a in zip(records, abst, strict=True) if r.get("expected_sources")]
    unans = [a for r, a in zip(records, abst, strict=True) if not r.get("expected_sources")]
    fa = sum(ans) / len(ans) if ans else None
    fans = sum(1 for a in unans if not a) / len(unans) if unans else None
    return {
        "threshold": t,
        "n_answerable": len(ans),
        "n_unanswerable": len(unans),
        "false_abstention": fa,
        "false_answer": fans,
    }


def select_threshold(
    dev: list[dict[str, Any]], max_false_abstention: float = 0.10
) -> dict[str, Any]:
    """Largest-benefit threshold on dev.

    Among candidates (0 = gate off, and every observed dev top score, i.e.
    each point where a decision flips) with dev false abstention <=
    `max_false_abstention`, pick the one with the lowest dev false-answer
    rate; ties go to the lower threshold (fewer refusals).
    """
    if not dev or not any(r.get("expected_sources") for r in dev) or all(r.get("expected_sources") for r in dev):
        raise ValueError("dev split needs both answerable and unanswerable questions")
    scores = sorted({r["top_rerank_score"] for r in dev if r.get("top_rerank_score") is not None})
    if not scores:
        raise ValueError("records have no top_rerank_score (run a config with the reranker)")
    candidates = [0.0, *scores]
    rows = [error_rates(dev, t) for t in candidates]
    feasible = [r for r in rows if r["false_abstention"] <= max_false_abstention]
    best = min(feasible, key=lambda r: (r["false_answer"], r["threshold"]))
    return {"chosen": best, "max_false_abstention": max_false_abstention, "dev_curve": rows}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("records", type=Path, help="records_<config>.jsonl from a run")
    p.add_argument("--max-false-abstention", type=float, default=0.10)
    args = p.parse_args(argv)

    records = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]
    records = [r for r in records if not r.get("error")]
    if any(r.get("split") == "legacy_test" for r in records):
        print("Refusing: records include the legacy_test split (the reported test set).", file=sys.stderr)
        return 2
    dev = [r for r in records if r.get("split") == "dev"]
    heldout = [r for r in records if r.get("split") == "heldout"]
    if not heldout:
        print("No held-out records; nothing to evaluate the threshold on.", file=sys.stderr)
        return 2

    sel = select_threshold(dev, args.max_false_abstention)
    t = sel["chosen"]["threshold"]
    result = {
        "records": str(args.records),
        "selection_on_dev": sel,
        "heldout_at_chosen": error_rates(heldout, t),
        "heldout_gate_off": error_rates(heldout, 0.0),
        "review_status_note": "Unreviewed questions: treat as indicative only.",
    }
    out = args.records.with_name(args.records.stem + "_threshold.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("heldout_at_chosen", "heldout_gate_off")}, indent=2))
    print(f"Chosen on dev: {t:.4f}. Full output: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["apply_threshold", "error_rates", "select_threshold"]
