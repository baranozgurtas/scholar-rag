"""Re-analyse the committed ablation results with the current metric definitions.

Input:  eval/results/ablation_{A,B,C,D}_*.json   (committed, produced on Colab)
Output: eval/results/legacy_reanalysis.json / .md

Deterministic and model-free: runs in CI. It does not re-run any model.

What the legacy records can and cannot support:
- They store only the 5 chunks passed to the LLM, so "@10" metrics are @5.
- `abstained` was a substring test for the abstention sentence, so an
  output that contained the sentence *and* other content counted as an
  abstention. Several adversarial records have citation tags, which
  suggests such mixed outputs; the answer text was not saved, so this
  cannot be checked.
- Allowed context tags were not saved, so citation tag validity cannot be
  recomputed. Only the number of extracted tags is available.
- Latency is total wall-clock per question (Colab T4).
- No chunk ever went through the "no chunks above threshold" branch: the
  default threshold 0.0 disabled filtering, and every record has 5 chunks.

    python -m eval.reanalyze_legacy
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from eval.harness import SERIALIZED_FLOAT_DIGITS, render_markdown, round_floats, summarize
from eval.questions import QUESTIONS_PATH, load_questions

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CONFIGS = [
    "A_dense_only",
    "B_dense_plus_rerank",
    "C_hybrid_no_rerank",
    "D_hybrid_plus_rerank",
]
# Aliases corrected after the legacy run (see eval/papers.txt).
LEGACY_ALIAS_RENAMES = {
    "ml-tips-domingos-2012": "data-discontents-paullada-2020",
    "dropout-srivastava-2014": "dropout-hinton-2012",
}


def _alias(source: str) -> str:
    stem = source[:-4] if source.lower().endswith(".pdf") else source
    return LEGACY_ALIAS_RENAMES.get(stem, stem)


def legacy_records(config: str, results_dir: Path = RESULTS_DIR) -> list[dict[str, Any]]:
    """Convert one committed ablation JSON into harness records."""
    data = json.loads((results_dir / f"ablation_{config}.json").read_text())
    by_text = {q.question: q for q in load_questions(QUESTIONS_PATH)}
    records = []
    for row in data["per_question"]:
        q = by_text[row["question"]]
        latency = row["latency_ms"]
        total = latency.get("total_ms") if isinstance(latency, dict) else latency
        records.append(
            {
                "id": q.id,
                "question": row["question"],
                "split": q.split,
                "review_status": q.review_status,
                "expected_sources": [_alias(s) for s in row["expected_sources"]],
                "ranked_sources": [_alias(s) for s in row["retrieved_sources"]],
                "abstained": bool(row["abstained"]),
                "outcome": "legacy_abstained_substring" if row["abstained"] else "legacy_answered",
                "citations": row.get("citations", []),
                # Validity unknown: allowed tags were not recorded.
                "citation_check": {},
                "latency_ms": {"total_ms": total},
                "error": None,
            }
        )
    return records


def legacy_extras(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Facts only the legacy records can show."""
    unans = [r for r in records if not r["expected_sources"]]
    ans = [r for r in records if r["expected_sources"]]
    return {
        "unanswerable_abstained_but_emitted_citation_tags": sum(
            1 for r in unans if r["abstained"] and r["citations"]
        ),
        "answerable_released_with_zero_citation_tags": sum(
            1 for r in ans if not r["abstained"] and not r["citations"]
        ),
        "records_with_fewer_than_5_chunks": sum(1 for r in records if len(r["ranked_sources"]) < 5),
    }


def reanalyze(results_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cfg in CONFIGS:
        recs = legacy_records(cfg, results_dir)
        s = summarize(recs)
        s["legacy_extras"] = legacy_extras(recs)
        s["reported_metrics"] = json.loads((results_dir / f"ablation_{cfg}.json").read_text())["metrics"]
        out[cfg] = s
    return out


PREAMBLE = """Re-analysis of the committed 25-question results (`eval/results/ablation_*.json`,
generated on a Colab T4 before this harness existed) with the current metric
definitions. No model was re-run. Limitations:

- Only the 5 chunks sent to the LLM were stored, so MRR@10 / nDCG@10 are really @5
  and Hit@10 = Hit@5.
- `abstained` was a substring test; outputs containing the abstention sentence plus
  other text counted as abstentions. Answer text was not stored.
- Citation tag validity cannot be recomputed (allowed tags not stored).
- The 20 answerable questions are LLM-generated and not human-reviewed; the 5
  unanswerable ones are obvious out-of-domain questions. Small n: see the 95% CIs.
- The pre-generation rerank threshold was 0.0 (disabled). Every abstention in these
  runs came from the generator emitting the abstention sentence."""


def main(argv: list[str] | None = None) -> int:
    results = reanalyze()
    (RESULTS_DIR / "legacy_reanalysis.json").write_text(
        # Floats rounded to SERIALIZED_FLOAT_DIGITS so the file is identical
        # across Python versions / platforms (checked in CI).
        json.dumps(round_floats(results, SERIALIZED_FLOAT_DIGITS), indent=2, ensure_ascii=False) + "\n"
    )
    md = render_markdown(
        dict(results),
        title="Legacy results, re-analysed",
        preamble=PREAMBLE,
    )
    extras = ["## Legacy-only observations", "", "| Config | Unanswerable 'abstained' but emitted citation tags | Answerable released with 0 tags |", "|---|---|---|"]
    for k, v in results.items():
        e = v["legacy_extras"]
        extras.append(
            f"| `{k}` | {e['unanswerable_abstained_but_emitted_citation_tags']}/5 | "
            f"{e['answerable_released_with_zero_citation_tags']} |"
        )
    (RESULTS_DIR / "legacy_reanalysis.md").write_text(md + "\n".join(extras) + "\n")
    print(f"Wrote {RESULTS_DIR / 'legacy_reanalysis.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["CONFIGS", "LEGACY_ALIAS_RENAMES", "legacy_records", "reanalyze"]
