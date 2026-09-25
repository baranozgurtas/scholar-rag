"""Eval question schema, loading, and source-evidence verification.

Kept free of model / LLM imports so CI can validate question files without
torch, Ollama or Qdrant.

Question files:
- `eval/questions.jsonl`          — the 25 questions behind the committed
  results (20 LLM-generated + 5 hand-written out-of-corpus), split
  `legacy_test`. Kept as-is so the committed numbers stay reproducible.
- `eval/questions_v2_draft.jsonl` — a proposed dev / held-out set with
  answerable and near-miss unanswerable questions. Drafted with an LLM,
  every evidence quote machine-checked against the PDF text, **not yet
  human-reviewed** (`review_status: "unreviewed"`).

Nothing in this module marks a question as human-validated; only a person
editing `review_status` to `"human_reviewed"` does that.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
PDF_DIR = PROJECT_ROOT / "data" / "pdfs"
QUESTIONS_PATH = EVAL_DIR / "questions.jsonl"
QUESTIONS_V2_DRAFT_PATH = EVAL_DIR / "questions_v2_draft.jsonl"

PLACEHOLDER_PREFIX = "[FILL IN]"
SPLITS = {"legacy_test", "dev", "heldout"}
REVIEW_STATUSES = {"unreviewed", "human_reviewed", "handwritten"}


@dataclass
class EvalQuestion:
    """One eval question with ground truth.

    `expected_sources` holds PDF aliases (file stem under data/pdfs/). An
    empty list means the question is unanswerable from the corpus.
    """

    question: str
    reference_answer: str
    expected_sources: list[str] = field(default_factory=list)
    expected_clusters: list[str] = field(default_factory=list)
    question_type: str = "factoid"
    source_kind: str = "synthetic"  # synthetic | drafted | adversarial | near_miss
    notes: str = ""
    id: str = ""
    split: str = "legacy_test"
    review_status: str = "unreviewed"
    # [{"source": alias, "page": int, "quote": str}] — must occur on that page
    evidence: list[dict[str, Any]] = field(default_factory=list)
    # For unanswerable questions: {"scope": [aliases] or ["*"], "terms": [...]}
    # — none of the terms may occur anywhere in the scoped papers.
    absent_terms: dict[str, list[str]] = field(default_factory=dict)
    # Known problems with this question or its label (kept, not hidden).
    label_issues: list[str] = field(default_factory=list)

    @property
    def answerable(self) -> bool:
        return bool(self.expected_sources)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_questions(path: Path = QUESTIONS_PATH) -> list[EvalQuestion]:
    """Load questions from JSONL. Unfilled `[FILL IN]` templates are rejected."""
    if not path.exists():
        raise FileNotFoundError(f"Question set not found: {path}")
    questions: list[EvalQuestion] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("question", "").startswith(PLACEHOLDER_PREFIX):
            raise ValueError(
                f"{path.name}:{lineno} is an unfilled template; complete it or remove it."
            )
        questions.append(EvalQuestion(**d))
    return questions


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ─── Evidence verification against PDF text ──────────────────────


def normalize_text(text: str) -> str:
    """NFKC (expands ligatures like 'ﬁ'), lower-case, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip().lower()


@lru_cache(maxsize=64)
def pdf_pages(alias: str, pdf_dir: Path = PDF_DIR) -> tuple[str, ...]:
    """Normalized text of each page (1-indexed via position + 1)."""
    import fitz  # PyMuPDF; imported lazily

    path = pdf_dir / f"{alias}.pdf"
    with fitz.open(path) as doc:
        return tuple(normalize_text(page.get_text()) for page in doc)


def verify_question(q: EvalQuestion, pdf_dir: Path = PDF_DIR) -> list[str]:
    """Return a list of problems (empty = passes the mechanical checks).

    Checks: sources exist as PDFs; answerable questions carry evidence whose
    quotes occur verbatim (after normalization) on the stated page; each
    unanswerable question lists absent terms that indeed never occur in the
    scoped papers. Passing does not make a question human-validated.
    """
    problems: list[str] = []
    available = {p.stem for p in pdf_dir.glob("*.pdf")}
    for s in q.expected_sources:
        if s not in available:
            problems.append(f"expected source {s!r} has no PDF in {pdf_dir}")
    if q.split not in SPLITS:
        problems.append(f"unknown split {q.split!r}")
    if q.review_status not in REVIEW_STATUSES:
        problems.append(f"unknown review_status {q.review_status!r}")

    if q.answerable and q.split != "legacy_test":
        if not q.evidence:
            problems.append("answerable question has no evidence quotes")
        for ev in q.evidence:
            alias, page, quote = ev["source"], int(ev["page"]), ev["quote"]
            if alias not in available:
                problems.append(f"evidence source {alias!r} missing")
                continue
            pages = pdf_pages(alias, pdf_dir)
            if not 1 <= page <= len(pages):
                problems.append(f"{alias} has no page {page}")
            elif normalize_text(quote) not in pages[page - 1]:
                problems.append(f"quote not found on {alias} p.{page}: {quote[:60]!r}")

    if not q.answerable and q.absent_terms:
        scope = q.absent_terms.get("scope", ["*"])
        aliases = sorted(available) if scope == ["*"] else scope
        for alias in aliases:
            full = " ".join(pdf_pages(alias, pdf_dir))
            for term in q.absent_terms.get("terms", []):
                if normalize_text(term) in full:
                    problems.append(f"'absent' term {term!r} occurs in {alias}")
    return problems


__all__ = [
    "PLACEHOLDER_PREFIX",
    "QUESTIONS_PATH",
    "QUESTIONS_V2_DRAFT_PATH",
    "EvalQuestion",
    "file_sha256",
    "load_questions",
    "normalize_text",
    "pdf_pages",
    "verify_question",
]
