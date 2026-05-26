"""Citation guards for RAG output.

Two responsibilities:
1. **Extract** citation tags from the LLM's answer (regex-based, robust to
   formatting drift).
2. **Validate** each extracted tag against the set of citation tags the
   LLM was given in context. Any tag not in the allowed set is a
   *fabricated citation* — the LLM hallucinated a source.

Fabricated citations are the most dangerous failure mode of RAG systems
in production, because they look authoritative but are wrong. We surface
them explicitly in the API response and the Langfuse trace, so they can
be monitored and alerted on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Matches: [Paper: TITLE | p.NUM | §SECTION]
# - Title can contain almost anything except `|` and `]`
# - Page is one or more digits (allow optional dash range like 3-4)
# - Section is one or more word chars / underscores / dashes
_CITATION_RE = re.compile(
    r"\[Paper:\s*(?P<title>[^|\]]+?)\s*\|\s*p\.(?P<page>[\d\-]+)\s*\|\s*§(?P<section>[\w\-]+)\s*\]",
    re.IGNORECASE,
)


@dataclass
class CitationCheckResult:
    """Result of validating extracted citations against allowed context tags."""

    n_extracted: int
    n_valid: int
    n_invalid: int
    invalid_tags: list[str] = field(default_factory=list)
    all_valid: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_extracted": self.n_extracted,
            "n_valid": self.n_valid,
            "n_invalid": self.n_invalid,
            "invalid_tags": self.invalid_tags,
            "all_valid": self.all_valid,
        }


def extract_citation_tags(text: str) -> list[str]:
    """Return every distinct citation tag (in order of first appearance) from text."""
    seen: list[str] = []
    seen_set: set[str] = set()
    for m in _CITATION_RE.finditer(text):
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


def validate_citations_against_context(
    citations: list[str], allowed_tags: list[str]
) -> CitationCheckResult:
    """Validate each extracted citation against the tags provided in context.

    A citation is valid if (in order of strictness):
    1. Normalized form exactly matches an allowed tag, OR
    2. (title, page) tuple matches an allowed prefix (drops section), OR
    3. The citation title is a substring of any allowed title AND page matches
       — accepts the common LLM behavior of shortening long paper titles
       (e.g. "M3-Embedding" instead of "M3-Embedding: Multi-Linguality, ...").

    For strict eval mode, downstream code can require `all_valid=True`.
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
            if any(cited_title in t or t.startswith(cited_title) for t in candidates):
                n_valid += 1
                continue
        invalid.append(c)

    return CitationCheckResult(
        n_extracted=len(citations),
        n_valid=n_valid,
        n_invalid=len(invalid),
        invalid_tags=invalid,
        all_valid=len(invalid) == 0,
    )


__all__ = [
    "CitationCheckResult",
    "extract_citation_tags",
    "validate_citations_against_context",
]
