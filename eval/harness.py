"""Summaries over per-question eval records (pure Python, no model imports).

A *record* is one question answered by one pipeline config, as written to
`records_<config>.jsonl` by `eval.retrieval_ablation`. Required keys:

    id, question, split, expected_sources, review_status,
    ranked_sources   (top-10 ranking, or fewer for legacy runs),
    outcome, abstained, citations, citation_check, latency_ms, error

`summarize()` turns a list of records into separately reported metrics:

- retrieval (answerable questions only): Hit@5, MRR@10, nDCG@10
- false abstention rate on answerable questions
- false answer rate on unanswerable questions
- citation *tag validity* (structural) — kept apart from factual grounding,
  which this module does not measure
- p50 / p95 latency and a per-question failure list

Questions whose pipeline call raised are counted as errors and excluded from
every rate; they are never scored as zeros or ones.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from eval.metrics import all_sources_at_k, hit_at_k, mrr_at_k, ndcg_at_k

WITHHELD_OUTCOMES = {"mixed_abstention", "uncited_answer", "invalid_citation"}

FACTUAL_GROUNDING_NOTE = (
    "Not measured by this harness. Citation tag validity only shows that a "
    "cited (title, page) was among the supplied passages, not that the passage "
    "supports the claim. Use `python -m eval.ragas_eval` (LLM judge) or human "
    "review for grounding."
)


# Serialized summaries round every float to this many decimal places.
# Summation order and libm differences across Python builds change the last
# digits (e.g. 0.9666666666666666 vs 0.9666666666666668); 6 places is far
# finer than any reported precision (3) and makes committed JSON reproducible.
SERIALIZED_FLOAT_DIGITS = 6


def round_floats(obj: Any, ndigits: int = SERIALIZED_FLOAT_DIGITS) -> Any:
    """Recursively round floats in dicts/lists for stable JSON output."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [round_floats(v, ndigits) for v in obj]
    return obj


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile (same convention as numpy's default)."""
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _rate(k: int, n: int) -> dict[str, Any]:
    ci = wilson_interval(k, n)
    return {
        "count": k,
        "n": n,
        "rate": (k / n) if n else None,
        "wilson95": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
    }


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


# Statuses for released answers to false-premise items.
PREMISE_CORRECTED = "premise_corrected"          # set only by a human label
PREMISE_UNSUPPORTED = "unsupported_answer"       # counts as a false answer
PREMISE_NEEDS_REVIEW = "needs_manual_review"     # neither credited nor penalised
MANUAL_PREMISE_LABELS = {PREMISE_CORRECTED, PREMISE_UNSUPPORTED}


def premise_status(r: dict[str, Any]) -> str | None:
    """Status of a released answer to a false-premise item (else None).

    No answer is ever credited automatically. A human label in
    `manual_premise_label` (from `premise_reviews.json`, see
    `apply_manual_premise_labels`) decides. Without one:
    - the answer mentions none of `premise_correction.review_trigger_terms_any`
      → `unsupported_answer` (it cannot be stating the correction);
    - it mentions one → `needs_manual_review` (a mention of "Claude-1.3" is not
      evidence of a correct correction; it may still assert the false premise).
    """
    pc = r.get("premise_correction") or {}
    if not pc or r.get("abstained") or r.get("error"):
        return None
    label = r.get("manual_premise_label")
    if label is not None:
        if label not in MANUAL_PREMISE_LABELS:
            raise ValueError(f"{r.get('id')}: invalid manual_premise_label {label!r}")
        return label
    answer = (r.get("answer") or "").lower()
    triggers = pc.get("review_trigger_terms_any", [])
    return PREMISE_NEEDS_REVIEW if any(t.lower() in answer for t in triggers) else PREMISE_UNSUPPORTED


PREMISE_REVIEWS_FILE = "premise_reviews.json"


def load_premise_reviews(run_dir: Path) -> dict[str, str]:
    """Human labels written next to a generation run's records, if any.

    Format: {"H12": "premise_corrected" | "unsupported_answer", ...}. The file
    is written by a person after reading the answer; nothing generates it.
    """
    path = run_dir / PREMISE_REVIEWS_FILE
    return json.loads(path.read_text()) if path.exists() else {}


def apply_manual_premise_labels(records: list[dict[str, Any]], labels: dict[str, str]) -> None:
    """Attach human labels {question_id: premise_corrected|unsupported_answer}."""
    for r in records:
        if r.get("id") in labels:
            if labels[r["id"]] not in MANUAL_PREMISE_LABELS:
                raise ValueError(f"{r['id']}: invalid label {labels[r['id']]!r}")
            r["manual_premise_label"] = labels[r["id"]]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the separately reported metrics for one config's records."""
    errors = [r for r in records if r.get("error")]
    ok = [r for r in records if not r.get("error")]
    answerable = [r for r in ok if r.get("expected_sources")]
    unanswerable = [r for r in ok if not r.get("expected_sources")]

    # ── Retrieval (answerable only) ─────────────────────────────
    retrieval = _retrieval_block(answerable)

    # ── Abstention ──────────────────────────────────────────────
    false_abst = [r for r in answerable if r.get("abstained")]
    # False-premise answers: only a human label can credit a correction;
    # answers awaiting review are neither false nor correct (see premise_status).
    pending = [r for r in unanswerable if premise_status(r) == PREMISE_NEEDS_REVIEW]
    corrected = [r for r in unanswerable if premise_status(r) == PREMISE_CORRECTED]
    false_ans = [
        r for r in unanswerable
        if not r.get("abstained") and premise_status(r) not in (PREMISE_NEEDS_REVIEW, PREMISE_CORRECTED)
    ]
    false_premise = [r for r in unanswerable if r.get("premise_correction")]
    n_scored_unans = len(unanswerable) - len(pending)
    abstention = {
        "false_abstention_on_answerable": _rate(len(false_abst), len(answerable)),
        "false_abstention_by_outcome": dict(Counter(r.get("outcome", "?") for r in false_abst)),
        # Pending-review items are left out of the denominator; the upper bound
        # counts every pending item as a false answer.
        "false_answer_on_unanswerable": _rate(len(false_ans), n_scored_unans),
        "false_answer_upper_bound_if_pending_are_false": _rate(len(false_ans) + len(pending), len(unanswerable)),
        "pending_manual_review": [r.get("id") for r in pending],
        "abstention_by_outcome_on_unanswerable": dict(
            Counter(r.get("outcome", "?") for r in unanswerable if r.get("abstained"))
        ),
        "false_premise": {
            "n": len(false_premise),
            "abstained": sum(1 for r in false_premise if r.get("abstained")),
            "premise_corrected_human_labelled": len(corrected),
            "needs_manual_review": len(pending),
            "unsupported_answer": sum(1 for r in false_premise if premise_status(r) == PREMISE_UNSUPPORTED),
            "note": (
                "No correction is credited automatically. Released answers that mention a "
                "review-trigger term are needs_manual_review until a human labels them in "
                "premise_reviews.json; answers without one are unsupported (false) answers."
            ),
        },
    }

    # ── Citations: structural tag validity ──────────────────────
    released = [r for r in ok if not r.get("abstained")]
    # Records without a citation_check (legacy runs) have unknown validity.
    released_known = [r for r in released if "all_valid" in (r.get("citation_check") or {})]
    released_all_valid = sum(1 for r in released_known if r["citation_check"]["all_valid"])
    tagged = [
        r for r in ok
        if (r.get("citation_check") or {}).get("n_extracted", 0) > 0
        and r["citation_check"].get("n_valid") is not None
    ]
    tags_total = sum(r["citation_check"]["n_extracted"] for r in tagged)
    tags_valid = sum(r["citation_check"]["n_valid"] for r in tagged)
    outcomes = Counter(r.get("outcome", "?") for r in ok)
    citations = {
        "released_answers": len(released),
        "released_with_all_tags_valid": _rate(released_all_valid, len(released_known)),
        "tag_validity_over_all_generated_tags": _rate(tags_valid, tags_total),
        "withheld_by_guard": {k: outcomes.get(k, 0) for k in sorted(WITHHELD_OUTCOMES)},
        "validity_recorded": len(released_known) == len(released),
        "factual_grounding": {"status": "not_measured", "note": FACTUAL_GROUNDING_NOTE},
    }

    # ── Latency ─────────────────────────────────────────────────
    totals = [float(r["latency_ms"]["total_ms"]) for r in ok if r.get("latency_ms", {}).get("total_ms") is not None]
    latency = {
        "n": len(totals),
        "total_ms_p50": percentile(totals, 50),
        "total_ms_p95": percentile(totals, 95),
    }

    # ── Per-question failures ───────────────────────────────────
    failures: list[dict[str, Any]] = []
    for r in records:
        kinds: list[str] = []
        if r.get("error"):
            kinds.append("pipeline_error")
        elif r.get("expected_sources"):
            if not hit_at_k(r.get("ranked_sources") or [], r["expected_sources"], 5):
                kinds.append("retrieval_miss_at_5")
            elif not all_sources_at_k(r.get("ranked_sources") or [], r["expected_sources"], 5):
                kinds.append("missing_required_source_at_5")
            if r.get("abstained"):
                kinds.append(f"false_abstention:{r.get('outcome', '?')}")
        elif premise_status(r) == PREMISE_NEEDS_REVIEW:
            kinds.append("needs_manual_review:false_premise")
        elif premise_status(r) == PREMISE_CORRECTED:
            pass  # human-labelled correct correction
        elif not r.get("abstained"):
            kinds.append("false_answer")
        if not r.get("error") and r.get("outcome") in WITHHELD_OUTCOMES:
            kinds.append(f"withheld:{r['outcome']}")
        if kinds:
            failures.append(
                {
                    "id": r.get("id", ""),
                    "question": r.get("question", "")[:120],
                    "failures": kinds,
                    "expected_sources": r.get("expected_sources", []),
                    "top_sources": (r.get("ranked_sources") or [])[:5],
                    "error": r.get("error"),
                }
            )

    return {
        "counts": {
            "n_records": len(records),
            "n_errors": len(errors),
            "n_answerable": len(answerable),
            "n_unanswerable": len(unanswerable),
            "review_status": dict(Counter(r.get("review_status", "?") for r in records)),
            "splits": dict(Counter(r.get("split", "?") for r in records)),
        },
        "retrieval": retrieval,
        "abstention": abstention,
        "citations": citations,
        "latency": latency,
        "failures": failures,
    }


def _retrieval_block(answerable: list[dict[str, Any]]) -> dict[str, Any]:
    depths = [len(r.get("ranked_sources") or []) for r in answerable]
    min_depth = min(depths) if depths else 0
    multi = [r for r in answerable if len(r["expected_sources"]) > 1]
    return {
        "n": len(answerable),
        "hit_at_5": _mean([hit_at_k(r["ranked_sources"], r["expected_sources"], 5) for r in answerable]),
        "all_sources_at_5": _mean(
            [all_sources_at_k(r["ranked_sources"], r["expected_sources"], 5) for r in answerable]
        ),
        "multi_paper": {
            "n": len(multi),
            "hit_at_5": _mean([hit_at_k(r["ranked_sources"], r["expected_sources"], 5) for r in multi]),
            "all_sources_at_5": _mean(
                [all_sources_at_k(r["ranked_sources"], r["expected_sources"], 5) for r in multi]
            ),
        },
        "mrr_at_10": _mean([mrr_at_k(r["ranked_sources"], r["expected_sources"], 10) for r in answerable]),
        "ndcg_at_10": _mean([ndcg_at_k(r["ranked_sources"], r["expected_sources"], 10) for r in answerable]),
        "min_ranking_depth": min_depth,
        "note": (
            f"Only {min_depth} ranked sources were recorded per question, so the "
            f"@10 metrics are effectively @{min_depth}."
            if 0 < min_depth < 10
            else ""
        ),
    }


def _score_stats(xs: list[float]) -> dict[str, Any]:
    return {
        "n": len(xs),
        "min": min(xs) if xs else None,
        "p50": percentile(xs, 50),
        "max": max(xs) if xs else None,
    }


# Dense cosine of the best candidate below this is suspicious for BGE-M3 on
# this corpus (fresh-process Adam query: 0.73; figure-debris chunks: ~0.34).
LOW_TOP_DENSE_SCORE = 0.4


def summarize_retrieval(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Retrieval-only summary (records from `eval.retrieval_ablation`)."""
    errors = [r for r in records if r.get("error")]
    ok = [r for r in records if not r.get("error")]
    answerable = [r for r in ok if r.get("expected_sources")]
    unanswerable = [r for r in ok if not r.get("expected_sources")]

    def top_scores(rs: list[dict[str, Any]]) -> list[float]:
        return [r["top_rerank_score"] for r in rs if r.get("top_rerank_score") is not None]

    lat = [
        float(r["latency_ms"].get("retrieval_ms", 0)) + float(r["latency_ms"].get("rerank_ms", 0))
        for r in ok
        if r.get("latency_ms")
    ]
    failures = []
    for r in records:
        kinds = []
        if r.get("error"):
            kinds.append("pipeline_error")
        elif r.get("expected_sources") and not hit_at_k(r["ranked_sources"], r["expected_sources"], 5):
            kinds.append("retrieval_miss_at_5")
        elif r.get("expected_sources") and not all_sources_at_k(r["ranked_sources"], r["expected_sources"], 5):
            kinds.append("missing_required_source_at_5")
        tds = r.get("top_dense_score")
        if tds is not None and tds < LOW_TOP_DENSE_SCORE:
            kinds.append(f"low_top_dense_score:{tds:.3f}")
        if kinds:
            failures.append(
                {
                    "id": r.get("id", ""),
                    "question": r.get("question", "")[:120],
                    "failures": kinds,
                    "expected_sources": r.get("expected_sources", []),
                    "top_sources": (r.get("ranked_sources") or [])[:5],
                    "error": r.get("error"),
                }
            )
    return {
        "counts": {
            "n_records": len(records),
            "n_errors": len(errors),
            "n_answerable": len(answerable),
            "n_unanswerable": len(unanswerable),
            "review_status": dict(Counter(r.get("review_status", "?") for r in records)),
            "splits": dict(Counter(r.get("split", "?") for r in records)),
        },
        "retrieval": _retrieval_block(answerable),
        "top_rerank_score": {
            "answerable": _score_stats(top_scores(answerable)),
            "unanswerable": _score_stats(top_scores(unanswerable)),
        },
        "latency": {
            "n": len(lat),
            "retrieval_plus_rerank_ms_p50": percentile(lat, 50),
            "retrieval_plus_rerank_ms_p95": percentile(lat, 95),
        },
        "failures": failures,
    }


def render_retrieval_markdown(summaries: dict[str, dict[str, Any]], title: str, preamble: str = "") -> str:
    lines = [f"# {title}", ""]
    if preamble:
        lines += [preamble, ""]
    lines += [
        "## Retrieval (answerable questions)",
        "",
        "| Config | n | Hit@5 | All-sources@5 | Multi-paper n / Hit@5 / All-sources@5 | MRR@10 | nDCG@10 | Retrieval+rerank p50 / p95 (ms) | Errors |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in summaries.items():
        r, lt = s["retrieval"], s["latency"]
        mp = r["multi_paper"]
        lines.append(
            f"| `{name}` | {r['n']} | {_fmt(r['hit_at_5'])} | {_fmt(r['all_sources_at_5'])} | "
            f"{mp['n']} / {_fmt(mp['hit_at_5'])} / {_fmt(mp['all_sources_at_5'])} | "
            f"{_fmt(r['mrr_at_10'])} | {_fmt(r['ndcg_at_10'])} | "
            f"{_fmt(lt['retrieval_plus_rerank_ms_p50'], 0)} / {_fmt(lt['retrieval_plus_rerank_ms_p95'], 0)} | "
            f"{s['counts']['n_errors']} |"
        )
    lines += [
        "",
        "## Top-1 rerank score by answerability (input for a dev-split threshold; not a result)",
        "",
        "| Config | answerable min / p50 / max | unanswerable min / p50 / max |",
        "|---|---|---|",
    ]
    for name, s in summaries.items():
        a, u = s["top_rerank_score"]["answerable"], s["top_rerank_score"]["unanswerable"]
        if not a["n"] and not u["n"]:
            continue
        lines.append(
            f"| `{name}` | {_fmt(a['min'])} / {_fmt(a['p50'])} / {_fmt(a['max'])} | "
            f"{_fmt(u['min'])} / {_fmt(u['p50'])} / {_fmt(u['max'])} |"
        )
    lines += ["", "## Per-question failures", ""]
    for name, s in summaries.items():
        lines.append(f"### `{name}`")
        if not s["failures"]:
            lines += ["", "None.", ""]
            continue
        lines += ["", "| id | failures | expected | top-5 sources | question |", "|---|---|---|---|---|"]
        for f in s["failures"]:
            lines.append(
                f"| {f['id']} | {', '.join(f['failures'])} | {', '.join(f['expected_sources']) or '—'} | "
                f"{', '.join(f['top_sources'])} | {f['question']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _fmt(x: float | None, digits: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def _fmt_rate(r: dict[str, Any]) -> str:
    if not r["n"]:
        return "n/a (n=0)"
    lo, hi = r["wilson95"]
    return f"{r['count']}/{r['n']} ({r['rate']:.1%}; 95% CI {lo:.0%}-{hi:.0%})"


def render_markdown(summaries: dict[str, dict[str, Any]], title: str, preamble: str = "") -> str:
    """Render {config_name: summary} as a Markdown report."""
    lines = [f"# {title}", ""]
    if preamble:
        lines += [preamble, ""]
    lines += [
        "## Retrieval (answerable questions)",
        "",
        "| Config | n | Hit@5 | All-sources@5 | Multi-paper n / All-sources@5 | MRR@10 | nDCG@10 | Note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, s in summaries.items():
        r = s["retrieval"]
        lines.append(
            f"| `{name}` | {r['n']} | {_fmt(r['hit_at_5'])} | {_fmt(r['all_sources_at_5'])} | "
            f"{r['multi_paper']['n']} / {_fmt(r['multi_paper']['all_sources_at_5'])} | {_fmt(r['mrr_at_10'])} | "
            f"{_fmt(r['ndcg_at_10'])} | {r['note']} |"
        )
    lines += [
        "",
        "## Abstention",
        "",
        "| Config | False abstention (answerable) | False answer (unanswerable; pending review excluded) | False-premise items: abstained / human-labelled corrected / pending review / unsupported | Errors |",
        "|---|---|---|---|---|",
    ]
    for name, s in summaries.items():
        a = s["abstention"]
        fp = a["false_premise"]
        fp_cell = (
            f"{fp['abstained']} / {fp['premise_corrected_human_labelled']} / {fp['needs_manual_review']} / "
            f"{fp['unsupported_answer']} (n={fp['n']})"
            if fp["n"]
            else "—"
        )
        lines.append(
            f"| `{name}` | {_fmt_rate(a['false_abstention_on_answerable'])} | "
            f"{_fmt_rate(a['false_answer_on_unanswerable'])} | {fp_cell} | {s['counts']['n_errors']} |"
        )
    lines += [
        "",
        "## Citations (tag validity is structural, not factual grounding)",
        "",
        "| Config | Released answers | Released with all tags valid | Tag validity (all generated tags) | Withheld by guard |",
        "|---|---|---|---|---|",
    ]
    for name, s in summaries.items():
        c = s["citations"]
        withheld = (
            ", ".join(f"{k}={v}" for k, v in c["withheld_by_guard"].items())
            if c["validity_recorded"]
            else "not recorded (no guard)"
        )
        lines.append(
            f"| `{name}` | {c['released_answers']} | {_fmt_rate(c['released_with_all_tags_valid'])} | "
            f"{_fmt_rate(c['tag_validity_over_all_generated_tags'])} | {withheld} |"
        )
    lines += ["", f"Factual grounding: {FACTUAL_GROUNDING_NOTE}", ""]
    lines += ["## Latency", "", "| Config | n | p50 total (s) | p95 total (s) |", "|---|---|---|---|"]
    for name, s in summaries.items():
        lt = s["latency"]
        p50 = lt["total_ms_p50"] / 1000 if lt["total_ms_p50"] is not None else None
        p95 = lt["total_ms_p95"] / 1000 if lt["total_ms_p95"] is not None else None
        lines.append(f"| `{name}` | {lt['n']} | {_fmt(p50, 1)} | {_fmt(p95, 1)} |")
    lines += ["", "## Per-question failures", ""]
    for name, s in summaries.items():
        lines.append(f"### `{name}`")
        if not s["failures"]:
            lines += ["", "None.", ""]
            continue
        lines += ["", "| id | failures | expected | top-5 sources | question |", "|---|---|---|---|---|"]
        for f in s["failures"]:
            lines.append(
                f"| {f['id']} | {', '.join(f['failures'])} | {', '.join(f['expected_sources']) or '—'} | "
                f"{', '.join(f['top_sources'])} | {f['question']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


__all__ = [
    "FACTUAL_GROUNDING_NOTE",
    "PREMISE_REVIEWS_FILE",
    "WITHHELD_OUTCOMES",
    "apply_manual_premise_labels",
    "load_premise_reviews",
    "percentile",
    "premise_status",
    "render_markdown",
    "render_retrieval_markdown",
    "round_floats",
    "summarize",
    "summarize_retrieval",
    "wilson_interval",
]
