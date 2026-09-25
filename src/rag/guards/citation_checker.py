"""Citation guards for RAG output.

Three responsibilities:
1. **Extract** citation tags from the LLM's answer (regex-based, robust to
   formatting drift).
2. **Validate** each extracted tag against the set of citation tags the
   LLM was given in context. Any tag not in the allowed set points to a
   passage the model was never shown.
3. **Enforce** an answer policy (`apply_answer_policy`): an answer is only
   released if it cites at least one tag and every tag it cites refers to a
   supplied passage. Otherwise the answer is withheld and replaced by the
   abstention sentence, with a machine-readable reason.

Scope: tag validity is a *structural* check. It shows that the cited
(title, page) was part of the supplied context; it does NOT show that the
cited passage supports the claim next to it, nor that every factual claim
carries a citation. Factual grounding needs a separate judgement (human
review or an NLI / LLM-judge faithfulness check) and is reported
separately by the eval harness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Accepted citation formats (anything else is not a citation):
#   [Paper: TITLE | p.NUM | §SECTION]   canonical, requested by the prompt
#   (Paper: TITLE | p.NUM | §SECTION)   same fields in parentheses; models
#                                       sometimes emit this (eval item D01)
# - Title: anything except `|` and the closing delimiter
# - Page: digits, optionally a dash range like 3-4
# - Section: word chars / underscores / dashes
# Both forms are normalized to the canonical bracket form before validation,
# so they pass or fail exactly the same checks. Looser variants (commas
# instead of pipes, missing page or section, "p. 3") are NOT accepted.
_FIELDS = r"Paper:\s*(?P<title>[^|{close}]+?)\s*\|\s*p\.(?P<page>[\d\-]+)\s*\|\s*§(?P<section>[\w\-]+)\s*"
_CITATION_RE = re.compile(r"\[" + _FIELDS.format(close=r"\]") + r"\]", re.IGNORECASE)
_PAREN_CITATION_RE = re.compile(r"\(" + _FIELDS.format(close=r"\)") + r"\)", re.IGNORECASE)


@dataclass
class CitationCheckResult:
    """Result of validating extracted citations against allowed context tags."""

    n_extracted: int
    n_valid: int
    n_invalid: int
    invalid_tags: list[str] = field(default_factory=list)
    # True only if at least one tag was extracted AND none were invalid.
    # Zero citations is not "all valid" — there is nothing to check.
    all_valid: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_extracted": self.n_extracted,
            "n_valid": self.n_valid,
            "n_invalid": self.n_invalid,
            "invalid_tags": self.invalid_tags,
            "all_valid": self.all_valid,
        }


def extract_citation_tags(text: str) -> list[str]:
    """Return every distinct citation tag (in order of first appearance) from text.

    Bracketed and parenthesized tags are both returned in canonical bracket
    form, so a parenthesized tag is validated by the same rules.
    """
    seen: list[str] = []
    seen_set: set[str] = set()
    matches = sorted(
        [*_CITATION_RE.finditer(text), *_PAREN_CITATION_RE.finditer(text)],
        key=lambda m: m.start(),
    )
    for m in matches:
        # Reconstruct in canonical form to make comparison robust to whitespace
        tag = (
            f"[Paper: {m.group('title').strip()} "
            f"| p.{m.group('page').strip()} "
            f"| §{m.group('section').strip()}]"
        )
        if tag not in seen_set:
            seen.append(tag)
            seen_set.add(tag)
    return seen


def _normalize_tag(tag: str) -> str:
    """Lower-case + collapse whitespace for fuzzy comparison."""
    return re.sub(r"\s+", " ", tag.strip().lower())


# A shortened cited title must be at least this long to be matched as a
# substring of a supplied title. Stops tags like "[Paper: a | p.3 | §x]"
# from validating against any title on page 3 that contains the letter "a".
MIN_SHORT_TITLE_CHARS = 6


def _short_title_matches(cited_title: str, allowed_title: str) -> bool:
    """True if `cited_title` is a whole-word substring of `allowed_title`."""
    if len(cited_title) < MIN_SHORT_TITLE_CHARS:
        return False
    return re.search(rf"(?<!\w){re.escape(cited_title)}(?!\w)", allowed_title) is not None


def validate_citations_against_context(
    citations: list[str], allowed_tags: list[str]
) -> CitationCheckResult:
    """Validate each extracted citation against the tags provided in context.

    A citation is valid if (in order of strictness):
    1. Normalized form exactly matches an allowed tag, OR
    2. (title, page) tuple matches an allowed prefix (drops section), OR
    3. The citation title is a whole-word substring (>= MIN_SHORT_TITLE_CHARS
       chars) of an allowed title on the same page — accepts the common LLM
       behavior of shortening long paper titles (e.g. "M3-Embedding" instead
       of "M3-Embedding: Multi-Linguality, ...").

    `all_valid` is False when `citations` is empty. A valid tag says nothing
    about whether the cited passage supports the claim (see module docstring).
    """
    allowed_norm = {_normalize_tag(t) for t in allowed_tags}
    # (title_lower, page) prefix set for fuzzy match
    allowed_prefix: set[tuple[str, str]] = set()
    # (title_lower, page) for substring match
    allowed_titles_by_page: dict[str, list[str]] = {}
    for t in allowed_tags:
        m = _CITATION_RE.search(t)
        if m:
            title_lower = m.group("title").strip().lower()
            page = m.group("page").strip()
            allowed_prefix.add((title_lower, page))
            allowed_titles_by_page.setdefault(page, []).append(title_lower)

    n_valid = 0
    invalid: list[str] = []
    for c in citations:
        if _normalize_tag(c) in allowed_norm:
            n_valid += 1
            continue
        m = _CITATION_RE.search(c)
        if m:
            cited_title = m.group("title").strip().lower()
            cited_page = m.group("page").strip()
            if (cited_title, cited_page) in allowed_prefix:
                n_valid += 1
                continue
            # Substring match: cited title appears in any allowed title for the same page
            candidates = allowed_titles_by_page.get(cited_page, [])
            if any(_short_title_matches(cited_title, t) for t in candidates):
                n_valid += 1
                continue
        invalid.append(c)

    return CitationCheckResult(
        n_extracted=len(citations),
        n_valid=n_valid,
        n_invalid=len(invalid),
        invalid_tags=invalid,
        all_valid=len(citations) > 0 and len(invalid) == 0,
    )


# ─── Answer policy ─────────────────────────────────────────────────

ABSTENTION_TEXT = (
    "I could not find sufficient information in the indexed papers to answer this question."
)


class AnswerOutcome:
    """Machine-readable outcomes of `apply_answer_policy` (string constants)."""

    ANSWERED = "answered"
    # The model's whole output is the abstention sentence.
    MODEL_ABSTAINED = "model_abstained"
    # The model wrote the abstention sentence *and* other content. The two
    # contradict each other, so the content is withheld.
    MIXED_ABSTENTION = "mixed_abstention"
    # Substantive answer with zero citation tags.
    UNCITED_ANSWER = "uncited_answer"
    # At least one tag does not refer to a supplied passage.
    INVALID_CITATION = "invalid_citation"
    GENERATION_ERROR = "generation_error"
    # Set by the chain before generation, not by this function:
    NO_CONTEXT = "no_context"
    LOW_RERANK_SCORE = "low_rerank_score"

    WITHHELD_BY_GUARD = frozenset({MIXED_ABSTENTION, UNCITED_ANSWER, INVALID_CITATION})


@dataclass
class AnswerDecision:
    """What the user is shown, and why."""

    outcome: str
    released_answer: str
    abstained: bool
    citations: list[str]
    citation_check: CitationCheckResult

    @property
    def withheld_by_guard(self) -> bool:
        return self.outcome in AnswerOutcome.WITHHELD_BY_GUARD


def _strip_abstention(text: str) -> str:
    """Remove the abstention sentence and leftover punctuation/whitespace."""
    rest = re.sub(re.escape(ABSTENTION_TEXT), " ", text, flags=re.IGNORECASE)
    return re.sub(r"[\s.,;:*\"'`-]+", " ", rest).strip()


def apply_answer_policy(
    answer_text: str,
    allowed_tags: list[str],
    generation_failed: bool = False,
) -> AnswerDecision:
    """Decide whether a generated answer may be released.

    Policy (evaluated in order):
    1. Generation raised                         → withheld, `generation_error`.
    2. Output is exactly the abstention sentence → `model_abstained`.
    3. Abstention sentence + other content       → withheld, `mixed_abstention`.
    4. No citation tags                          → withheld, `uncited_answer`.
    5. Any tag outside the supplied context      → withheld, `invalid_citation`.
    6. Otherwise                                 → released, `answered`.

    "Withheld" means the user sees `ABSTENTION_TEXT`; callers keep the raw
    model output for debugging and evaluation. Passing this policy does not
    establish that the answer is factually supported by the cited passages.
    """
    citations = extract_citation_tags(answer_text) if not generation_failed else []
    check = validate_citations_against_context(citations, allowed_tags)

    def _decision(outcome: str, released: str, abstained: bool) -> AnswerDecision:
        return AnswerDecision(
            outcome=outcome,
            released_answer=released,
            abstained=abstained,
            citations=citations,
            citation_check=check,
        )

    if generation_failed:
        return _decision(AnswerOutcome.GENERATION_ERROR, ABSTENTION_TEXT, True)
    if ABSTENTION_TEXT.lower() in answer_text.lower():
        if not _strip_abstention(answer_text):
            return _decision(AnswerOutcome.MODEL_ABSTAINED, ABSTENTION_TEXT, True)
        return _decision(AnswerOutcome.MIXED_ABSTENTION, ABSTENTION_TEXT, True)
    if check.n_extracted == 0:
        return _decision(AnswerOutcome.UNCITED_ANSWER, ABSTENTION_TEXT, True)
    if check.n_invalid > 0:
        return _decision(AnswerOutcome.INVALID_CITATION, ABSTENTION_TEXT, True)
    return _decision(AnswerOutcome.ANSWERED, answer_text, False)


__all__ = [
    "ABSTENTION_TEXT",
    "AnswerDecision",
    "AnswerOutcome",
    "CitationCheckResult",
    "apply_answer_policy",
    "extract_citation_tags",
    "validate_citations_against_context",
]
