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

    def test_zero_citations_is_not_all_valid(self) -> None:
        r = validate_citations_against_context([], ["[Paper: BGE-M3 | p.1 | §abstract]"])
        assert r.n_extracted == 0
        assert not r.all_valid

    def test_short_title_spoof_is_rejected(self) -> None:
        # A one-letter "title" must not validate by substring against a real title.
        citations = ["[Paper: M | p.1 | §abstract]"]
        allowed = ["[Paper: M3-Embedding: Multi-Linguality | p.1 | §abstract]"]
        r = validate_citations_against_context(citations, allowed)
        assert r.n_invalid == 1

    def test_shortened_title_on_same_page_is_accepted(self) -> None:
        citations = ["[Paper: M3-Embedding | p.16 | §methods]"]
        allowed = ["[Paper: M3-Embedding: Multi-Linguality, Multi-Functionality | p.16 | §methods]"]
        assert validate_citations_against_context(citations, allowed).all_valid

    def test_shortened_title_on_wrong_page_is_rejected(self) -> None:
        citations = ["[Paper: M3-Embedding | p.99 | §methods]"]
        allowed = ["[Paper: M3-Embedding: Multi-Linguality, Multi-Functionality | p.16 | §methods]"]
        assert not validate_citations_against_context(citations, allowed).all_valid


class TestAnswerPolicy:
    """`apply_answer_policy`: what is released and why.

    These tests pin down structural behaviour only. Passing the policy does
    not mean the cited passage supports the claim.
    """

    ALLOWED = (
        "[Paper: BGE-M3 | p.1 | §abstract]",
        "[Paper: Adam: A Method for Stochastic Optimization | p.2 | §methods]",
    )

    def _apply(self, text: str, **kw):
        from rag.guards.citation_checker import apply_answer_policy

        return apply_answer_policy(text, list(self.ALLOWED), **kw)

    def test_cited_answer_is_released(self) -> None:
        from rag.guards.citation_checker import AnswerOutcome

        text = "BGE-M3 supports dense retrieval [Paper: BGE-M3 | p.1 | §abstract]."
        d = self._apply(text)
        assert d.outcome == AnswerOutcome.ANSWERED
        assert d.released_answer == text
        assert not d.abstained
        assert d.citation_check.all_valid

    def test_substantive_answer_without_citations_is_withheld(self) -> None:
        from rag.guards.citation_checker import ABSTENTION_TEXT, AnswerOutcome

        d = self._apply("BGE-M3 was trained on 1.2 billion text pairs.")
        assert d.outcome == AnswerOutcome.UNCITED_ANSWER
        assert d.abstained
        assert d.released_answer == ABSTENTION_TEXT
        assert d.withheld_by_guard
        assert not d.citation_check.all_valid

    def test_citation_outside_context_is_withheld(self) -> None:
        from rag.guards.citation_checker import ABSTENTION_TEXT, AnswerOutcome

        text = "It uses dropout [Paper: Dropout | p.3 | §methods]."
        d = self._apply(text)
        assert d.outcome == AnswerOutcome.INVALID_CITATION
        assert d.released_answer == ABSTENTION_TEXT
        assert d.citation_check.invalid_tags == ["[Paper: Dropout | p.3 | §methods]"]

    def test_one_invalid_among_valid_citations_withholds_whole_answer(self) -> None:
        from rag.guards.citation_checker import AnswerOutcome

        text = (
            "Dense retrieval [Paper: BGE-M3 | p.1 | §abstract]; "
            "and 12 layers [Paper: BGE-M3 | p.9 | §methods]."
        )
        d = self._apply(text)
        assert d.outcome == AnswerOutcome.INVALID_CITATION
        assert d.citation_check.n_valid == 1
        assert d.citation_check.n_invalid == 1

    def test_malformed_tag_counts_as_uncited(self) -> None:
        from rag.guards.citation_checker import AnswerOutcome

        # Missing page field: not a parseable tag, so the answer is uncited.
        d = self._apply("BGE-M3 is multilingual [Paper: BGE-M3 | §abstract].")
        assert d.outcome == AnswerOutcome.UNCITED_ANSWER

    def test_pure_abstention(self) -> None:
        from rag.guards.citation_checker import ABSTENTION_TEXT, AnswerOutcome

        d = self._apply(f"  {ABSTENTION_TEXT}  ")
        assert d.outcome == AnswerOutcome.MODEL_ABSTAINED
        assert d.abstained
        assert not d.withheld_by_guard

    def test_abstention_plus_content_is_withheld_as_mixed(self) -> None:
        from rag.guards.citation_checker import ABSTENTION_TEXT, AnswerOutcome

        text = f"{ABSTENTION_TEXT} However, Adam uses bias correction [Paper: Adam: A Method for Stochastic Optimization | p.2 | §methods]."
        d = self._apply(text)
        assert d.outcome == AnswerOutcome.MIXED_ABSTENTION
        assert d.released_answer == ABSTENTION_TEXT
        assert d.withheld_by_guard

    def test_generation_error_is_not_released(self) -> None:
        from rag.guards.citation_checker import ABSTENTION_TEXT, AnswerOutcome

        d = self._apply("[generation error] connection refused", generation_failed=True)
        assert d.outcome == AnswerOutcome.GENERATION_ERROR
        assert d.released_answer == ABSTENTION_TEXT
        assert d.citations == []


class TestParenthesizedCitations:
    """`(Paper: … | p.N | §S)` is accepted only under the bracket-tag rules."""

    RAG_TAG = "[Paper: Retrieval-Augmented Generation for | p.2 | §methods]"
    # Verbatim shape of the qwen2.5:7b output for eval item D01.
    D01_OUTPUT = (
        "The parametric memory is a pre-trained seq2seq transformer and the non-parametric "
        "memory is a dense vector index of Wikipedia (Paper: Retrieval-Augmented Generation "
        "for | p.2 | §methods)."
    )

    def _apply(self, text: str, allowed: list[str] | None = None):
        from rag.guards.citation_checker import apply_answer_policy

        return apply_answer_policy(text, allowed if allowed is not None else [self.RAG_TAG])

    def test_d01_parenthesized_tag_matching_context_is_released(self) -> None:
        from rag.guards.citation_checker import AnswerOutcome

        d = self._apply(self.D01_OUTPUT)
        assert d.outcome == AnswerOutcome.ANSWERED
        assert d.citations == [self.RAG_TAG]  # normalized to canonical form
        assert d.citation_check.all_valid

    def test_parenthesized_and_bracketed_same_tag_deduplicate(self) -> None:
        text = f"A (Paper: Retrieval-Augmented Generation for | p.2 | §methods). B {self.RAG_TAG}."
        assert self._apply(text).citations == [self.RAG_TAG]

    def test_fabricated_parenthesized_tag_is_withheld(self) -> None:
        from rag.guards.citation_checker import AnswerOutcome

        d = self._apply("RAG uses BM25 (Paper: Retrieval-Augmented Generation for | p.9 | §methods).")
        assert d.outcome == AnswerOutcome.INVALID_CITATION
        d = self._apply("It was trained on C4 (Paper: Some Other Paper | p.2 | §methods).")
        assert d.outcome == AnswerOutcome.INVALID_CITATION

    def test_parenthesized_short_title_spoof_is_rejected(self) -> None:
        from rag.guards.citation_checker import AnswerOutcome

        d = self._apply("Claim (Paper: R | p.2 | §methods).")
        assert d.outcome == AnswerOutcome.INVALID_CITATION

    def test_loose_parenthetical_reference_is_not_a_citation(self) -> None:
        from rag.guards.citation_checker import AnswerOutcome

        for text in (
            "RAG uses DPR (Paper: Retrieval-Augmented Generation for, p.2).",
            "RAG uses DPR (Retrieval-Augmented Generation for | p.2 | §methods).",
            "RAG uses DPR (see page 2).",
        ):
            assert self._apply(text).outcome == AnswerOutcome.UNCITED_ANSWER, text

    def test_uncited_answer_still_withheld(self) -> None:
        from rag.guards.citation_checker import AnswerOutcome

        assert self._apply("RAG combines a retriever with a generator.").outcome == AnswerOutcome.UNCITED_ANSWER

    def test_mixed_refusal_with_parenthesized_tag_is_withheld(self) -> None:
        from rag.guards.citation_checker import ABSTENTION_TEXT, AnswerOutcome

        d = self._apply(f"{ABSTENTION_TEXT} But it uses DPR (Paper: Retrieval-Augmented Generation for | p.2 | §methods).")
        assert d.outcome == AnswerOutcome.MIXED_ABSTENTION
        assert d.released_answer == ABSTENTION_TEXT
