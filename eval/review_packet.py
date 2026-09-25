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

# Concerns found while drafting / auditing, for the human reviewer to rule on.
REVIEW_CONCERNS: dict[str, list[str]] = {
    "D02": [
        "Evidence quote covers only the single-vector half of the comparison; the "
        "late-interaction half is on the same page but split by a line-break hyphen "
        "('multi-vector repre- sentations'), so it was not quoted verbatim.",
    ],
    "D05": ["Needs two papers; check both quotes support the stated contrast."],
    "D10": [
        "Ambiguous: NCF (ncf-he-2017) reports BPR's hit ratio on MovieLens as a "
        "baseline. The question restricts to 'the original BPR paper', so it is "
        "labelled unanswerable, but an answer citing NCF's BPR baseline is defensible. "
        "Decide: keep as unanswerable, relabel as answerable from NCF, or rephrase.",
    ],
    "H06": [
        "Evidence quote shows the first-moment correction only; the second-moment line "
        "(same algorithm box, p.2) should also be checked against the reference answer.",
    ],
    "H07": [
        "Passage (p.5) is about the ImageNet re-creation by Recht et al. 2019; the question "
        "does not name ImageNet. Confirm it is unambiguous or add the dataset name.",
    ],
    "H12": [
        "False-premise style: the paper evaluates Claude-1.3, not Claude 3. An answer "
        "that corrects the premise is arguably correct but is scored as a false answer "
        "by the current metric. Decide whether to keep, rephrase, or score separately.",
    ],
}


def _md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def question_block(q: EvalQuestion) -> str:
    lines = [f"### {q.id} · {q.split} · {'answerable' if q.answerable else 'UNANSWERABLE'}", ""]
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
    problems = verify_question(q)
    concerns = REVIEW_CONCERNS.get(q.id, []) + q.label_issues
    lines.append("")
    lines.append(f"**Mechanical checks:** {'pass' if not problems else '; '.join(problems)}")
    if concerns:
        lines.append("")
        lines.append("**Open concerns:**")
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
        "| id | split | answerable | expected source(s) | open concerns |",
        "|---|---|---|---|---|",
    ]
    for q in qs:
        n_concerns = len(REVIEW_CONCERNS.get(q.id, [])) + len(q.label_issues)
        header.append(
            f"| {q.id} | {q.split} | {'yes' if q.answerable else 'no'} | "
            f"{', '.join(q.expected_sources) or '—'} | {n_concerns or ''} |"
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
