"""Build a Markdown review packet for a question file (no models).

For each question: split, expected source(s), reference answer, the
supporting page + quote (answerable) or the reason it is unanswerable
(absent terms and scope), open concerns for the reviewer, and blank
reviewer fields. The packet never marks anything as reviewed; a person
edits `review_status` in the JSONL after reviewing.

    python -m eval.review_packet eval/questions_v2_draft.jsonl \\
        --out eval/review/questions_v2_draft_review.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from eval.questions import EvalQuestion, file_sha256, load_questions, verify_question


def _md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def question_block(q: EvalQuestion) -> str:
    kind = "answerable" if q.answerable else ("FALSE PREMISE" if q.premise_correction else "UNANSWERABLE")
    exposure = "outcomes already inspected" if q.inspected_before_freeze else "not yet inspected"
    lines = [f"### {q.id} · {q.split} · {kind} · rev {q.revision} · {exposure}", ""]
    lines.append(f"**Question:** {q.question}")
    lines.append("")
    if q.answerable:
        lines.append(f"**Expected source(s):** {', '.join(f'`{s}`' for s in q.expected_sources)}")
        lines.append("")
        lines.append(f"**Reference answer:** {q.reference_answer}")
        lines.append("")
        lines.append("**Supporting evidence (machine-checked verbatim on the stated page):**")
        for ev in q.evidence:
            lines.append(f"- `{ev['source']}` p.{ev['page']}: “{ev['quote']}”")
    else:
        scope = q.absent_terms.get("scope", ["*"])
        scope_txt = "the whole corpus" if scope == ["*"] else ", ".join(f"`{s}`" for s in scope)
        terms = ", ".join(f"“{t}”" for t in q.absent_terms.get("terms", []))
        lines.append(f"**Why unanswerable:** none of {terms} occurs anywhere in {scope_txt} (machine-checked).")
        note = q.notes.split("Not human-reviewed.")[-1].strip()
        if note:
            lines.append(f"**Distractor / note:** {note}")
        if q.premise_correction:
            pc = q.premise_correction
            lines.append("")
            lines.append(
                f"**Premise correction (manual review):** acceptable answer: {pc.get('acceptable_answer', '')} "
                f"A released answer mentioning any of "
                f"{', '.join(f'“{t}”' for t in pc.get('review_trigger_terms_any', []))} is flagged "
                f"needs_manual_review; it counts as corrected only if a person labels it so in "
                f"premise_reviews.json. Released answers without those terms are false answers."
            )
            for ev in pc.get("evidence", []):
                lines.append(f"- `{ev['source']}` p.{ev['page']}: “{ev['quote']}”")
    problems = verify_question(q)
    concerns = q.review_notes + q.label_issues
    lines.append("")
    lines.append(f"**Mechanical checks:** {'pass' if not problems else '; '.join(problems)}")
    if concerns:
        lines.append("")
        lines.append("**Notes for the reviewer:**")
        lines += [f"- {c}" for c in concerns]
    lines += [
        "",
        "Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______",
        "",
    ]
    return "\n".join(lines)


def build_packet(path: Path) -> str:
    qs = load_questions(path)
    n_ans = sum(q.answerable for q in qs)
    header = [
        f"# Review packet: `{path.name}`",
        "",
        f"SHA-256 `{file_sha256(path)[:16]}` · {len(qs)} questions ({n_ans} answerable, "
        f"{len(qs) - n_ans} unanswerable) · review status: "
        f"{', '.join(sorted({q.review_status for q in qs}))}.",
        "",
        "All questions were drafted by an LLM. Quotes and absent terms are machine-checked "
        "against the PDF text; that does not establish that a question is well-posed, "
        "unambiguous, or that the reference answer is complete. Nothing here is "
        "human-reviewed. After review, set `review_status` to `human_reviewed` in the "
        "JSONL for the questions you accept (and update "
        "`tests/test_eval_harness.py::TestQuestionFiles::test_no_question_claims_human_review`).",
        "",
        "Held-out caveat: items marked *outcomes already inspected* (D01-D10 and "
        "H01-H12, including revised ones) had their retrieval results looked at before this revision, so the "
        "held-out split is not pristine for them. Only the new hard items (D11-D15, H13-H20) "
        "have not been run or inspected.",
        "",
        "| id | split | type | rev | expected source(s) | inspected | kind |",
        "|---|---|---|---|---|---|---|",
    ]
    for q in qs:
        kind = "answerable" if q.answerable else ("false premise" if q.premise_correction else "unanswerable")
        header.append(
            f"| {q.id} | {q.split} | {q.question_type} | {q.revision} | "
            f"{', '.join(q.expected_sources) or '—'} | {'yes' if q.inspected_before_freeze else 'no'} | {kind} |"
        )
    header.append("")
    return "\n".join(header) + "\n" + "\n".join(question_block(q) for q in qs)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build a question review packet.")
    p.add_argument("questions", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_packet(args.questions))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
