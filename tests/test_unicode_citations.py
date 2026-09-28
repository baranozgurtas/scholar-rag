"""Unicode (NFKC) normalization in ingestion, citation tags and tag matching.

Regression for demo question D07: the indexed title "Eﬀects using Random
Forests" carries the PDF ligature U+FB00. The model cited "Effects ...", the
tag failed to match, and a correctly cited answer was withheld.
"""

from __future__ import annotations

from pathlib import Path

import fitz

from rag.guards.citation_checker import (
    AnswerOutcome,
    apply_answer_policy,
    validate_citations_against_context,
)
from rag.ingestion.pdf_loader import load_pdf
from rag.retrieval.types import RetrievedChunk

LIGATURE_TITLE = "Eﬀects using Random Forests"  # "Eﬀects", as stored in the index
PLAIN_TITLE = "Effects using Random Forests"


def _tag(title: str, page: int = 3, section: str = "related_work") -> str:
    return f"[Paper: {title} | p.{page} | §{section}]"


class TestLigatureMatching:
    def test_d07_plain_citation_matches_ligature_context_tag(self) -> None:
        r = validate_citations_against_context([_tag(PLAIN_TITLE)], [_tag(LIGATURE_TITLE)])
        assert r.all_valid and r.n_valid == 1

    def test_ligature_citation_matches_plain_context_tag(self) -> None:
        r = validate_citations_against_context([_tag(LIGATURE_TITLE)], [_tag(PLAIN_TITLE)])
        assert r.all_valid

    def test_title_and_page_rule_uses_normalized_title(self) -> None:
        # Different section: rule 2 (title + page) must match after NFKC.
        r = validate_citations_against_context(
            [_tag(PLAIN_TITLE, section="abstract")], [_tag(LIGATURE_TITLE)]
        )
        assert r.all_valid

    def test_shortened_title_with_ligature_matches_same_page(self) -> None:
        r = validate_citations_against_context(
            ["[Paper: Random Forests | p.3 | §related_work]"],
            [_tag("Estimation of Heterogeneous Treatment Eﬀects using Random Forests")],
        )
        assert r.all_valid

    def test_d07_answer_is_released(self) -> None:
        answer = f"A tree is honest if each example is used for splits or estimates {_tag(PLAIN_TITLE)}."
        d = apply_answer_policy(answer, allowed_tags=[_tag(LIGATURE_TITLE)])
        assert d.outcome == AnswerOutcome.ANSWERED
        assert not d.abstained


class TestNormalizationDoesNotWeakenChecks:
    def test_wrong_page_still_rejected(self) -> None:
        r = validate_citations_against_context([_tag(PLAIN_TITLE, page=4)], [_tag(LIGATURE_TITLE)])
        assert r.n_invalid == 1

    def test_other_paper_on_same_page_still_rejected(self) -> None:
        r = validate_citations_against_context(
            [_tag("Effects of Dropout")], [_tag(LIGATURE_TITLE)]
        )
        assert r.n_invalid == 1

    def test_ligature_short_title_below_floor_rejected(self) -> None:
        # "ﬀ" normalizes to "ff": 2 chars, below MIN_SHORT_TITLE_CHARS.
        r = validate_citations_against_context(["[Paper: ﬀ | p.3 | §x]"], [_tag(LIGATURE_TITLE)])
        assert r.n_invalid == 1

    def test_expanded_ligatures_must_still_be_a_whole_word(self) -> None:
        # "ﬁﬁﬁ" → "fififi": 6 chars after NFKC, but not a word of the title.
        r = validate_citations_against_context(
            ["[Paper: ﬁﬁﬁ | p.3 | §x]"], [_tag(LIGATURE_TITLE)]
        )
        assert r.n_invalid == 1

    def test_partial_word_still_rejected(self) -> None:
        r = validate_citations_against_context(["[Paper: ffects using | p.3 | §x]"], [_tag(LIGATURE_TITLE)])
        assert r.n_invalid == 1


class TestTagsAndIngestion:
    def test_chunk_tag_uses_plain_title(self) -> None:
        c = RetrievedChunk(
            chunk_id="c1",
            text="t",
            score=1.0,
            metadata={"paper_title": LIGATURE_TITLE, "page": 3, "section": "related_work"},
        )
        assert c.paper_title == PLAIN_TITLE
        assert c.to_citation_tag() == _tag(PLAIN_TITLE)

    def test_load_pdf_normalizes_title(self, tmp_path: Path) -> None:
        path = tmp_path / "ligature.pdf"
        doc = fitz.open()
        doc.new_page().insert_text((72, 72), "Some body text for the page.")
        doc.set_metadata({"title": LIGATURE_TITLE})
        doc.save(path)
        doc.close()
        assert load_pdf(path).title == PLAIN_TITLE
