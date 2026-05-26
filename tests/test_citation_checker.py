"""Tests for citation extraction and fabrication detection."""

from __future__ import annotations

from rag.guards.citation_checker import (
    extract_citation_tags,
    validate_citations_against_context,
)


class TestExtraction:
    def test_extracts_single_citation(self) -> None:
        text = "BGE-M3 is multi-functional [Paper: BGE-M3 | p.1 | §abstract]."
        tags = extract_citation_tags(text)
        assert tags == ["[Paper: BGE-M3 | p.1 | §abstract]"]

    def test_extracts_multiple_citations(self) -> None:
        text = (
            "It supports dense [Paper: BGE-M3 | p.2 | §methods] "
            "and sparse [Paper: BGE-M3 | p.3 | §methods]."
        )
        tags = extract_citation_tags(text)
        assert len(tags) == 2

    def test_deduplicates_repeated_citations(self) -> None:
        text = "[Paper: BGE-M3 | p.1 | §abstract] some text [Paper: BGE-M3 | p.1 | §abstract]"
        tags = extract_citation_tags(text)
        assert len(tags) == 1

    def test_handles_no_citations(self) -> None:
        assert extract_citation_tags("plain text") == []

    def test_extracts_with_extra_whitespace(self) -> None:
        text = "[Paper:  BGE-M3   |  p.1  |  §abstract  ]"
        tags = extract_citation_tags(text)
        assert tags == ["[Paper: BGE-M3 | p.1 | §abstract]"]


class TestValidation:
    def test_all_valid(self) -> None:
        citations = ["[Paper: BGE-M3 | p.1 | §abstract]"]
        allowed = ["[Paper: BGE-M3 | p.1 | §abstract]"]
        r = validate_citations_against_context(citations, allowed)
        assert r.n_extracted == 1
        assert r.n_valid == 1
        assert r.n_invalid == 0
        assert r.all_valid

    def test_detects_fabrication(self) -> None:
        citations = ["[Paper: Fake Paper | p.99 | §intro]"]
        allowed = ["[Paper: BGE-M3 | p.1 | §abstract]"]
        r = validate_citations_against_context(citations, allowed)
        assert r.n_invalid == 1
        assert not r.all_valid
        assert "Fake Paper" in r.invalid_tags[0]

    def test_fuzzy_match_drops_section(self) -> None:
        # LLM kept title + page but dropped section — still considered valid (fuzzy)
        citations = ["[Paper: BGE-M3 | p.1 | §intro]"]
        allowed = ["[Paper: BGE-M3 | p.1 | §abstract]"]
        r = validate_citations_against_context(citations, allowed)
        assert r.n_valid == 1
        assert r.all_valid

    def test_mixed_valid_invalid(self) -> None:
        citations = [
            "[Paper: BGE-M3 | p.1 | §abstract]",
            "[Paper: Unknown | p.5 | §results]",
        ]
        allowed = ["[Paper: BGE-M3 | p.1 | §abstract]"]
        r = validate_citations_against_context(citations, allowed)
        assert r.n_valid == 1
        assert r.n_invalid == 1
        assert not r.all_valid
