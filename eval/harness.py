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

import math
from collections import Counter
from typing import Any

from eval.metrics import hit_at_k, mrr_at_k, ndcg_at_k

WITHHELD_OUTCOMES = {"mixed_abstention", "uncited_answer", "invalid_citation"}

FACTUAL_GROUNDING_NOTE = (
    "Not measured by this harness. Citation tag validity only shows that a "
    "cited (title, page) was among the supplied passages, not that the passage "
    "supports the claim. Use `python -m eval.ragas_eval` (LLM judge) or human "
    "review for grounding."
)


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
    false_ans = [r for r in unanswerable if not r.get("abstained")]
    abstention = {
        "false_abstention_on_answerable": _rate(len(false_abst), len(answerable)),
        "false_abstention_by_outcome": dict(Counter(r.get("outcome", "?") for r in false_abst)),
        "false_answer_on_unanswerable": _rate(len(false_ans), len(unanswerable)),
        "abstention_by_outcome_on_unanswerable": dict(
            Counter(r.get("outcome", "?") for r in unanswerable if r.get("abstained"))
        ),
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
            if r.get("abstained"):
                kinds.append(f"false_abstention:{r.get('outcome', '?')}")
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
    return {
        "n": len(answerable),
        "hit_at_5": _mean([hit_at_k(r["ranked_sources"], r["expected_sources"], 5) for r in answerable]),
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
        "| Config | n | Hit@5 | MRR@10 | nDCG@10 | Retrieval+rerank p50 / p95 (ms) | Errors |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, s in summaries.items():
        r, lt = s["retrieval"], s["latency"]
        lines.append(
            f"| `{name}` | {r['n']} | {_fmt(r['hit_at_5'])} | {_fmt(r['mrr_at_10'])} | {_fmt(r['ndcg_at_10'])} | "
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
        "| Config | n | Hit@5 | MRR@10 | nDCG@10 | Note |",
        "|---|---|---|---|---|---|",
    ]
    for name, s in summaries.items():
        r = s["retrieval"]
        lines.append(
            f"| `{name}` | {r['n']} | {_fmt(r['hit_at_5'])} | {_fmt(r['mrr_at_10'])} | "
            f"{_fmt(r['ndcg_at_10'])} | {r['note']} |"
        )
    lines += [
        "",
        "## Abstention",
        "",
        "| Config | False abstention (answerable) | False answer (unanswerable) | Errors |",
        "|---|---|---|---|",
    ]
    for name, s in summaries.items():
        a = s["abstention"]
        lines.append(
            f"| `{name}` | {_fmt_rate(a['false_abstention_on_answerable'])} | "
            f"{_fmt_rate(a['false_answer_on_unanswerable'])} | {s['counts']['n_errors']} |"
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
    "WITHHELD_OUTCOMES",
    "percentile",
    "render_markdown",
    "render_retrieval_markdown",
    "summarize",
    "summarize_retrieval",
    "wilson_interval",
]
