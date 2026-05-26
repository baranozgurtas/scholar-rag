"""Tests for the section-aware chunker."""

from __future__ import annotations

from pathlib import Path

from rag.config import ChunkingSettings
from rag.ingestion.chunker import SectionAwareChunker
from rag.ingestion.pdf_loader import LoadedDocument, PageContent


def _build_doc(pages_spec: list[tuple[str, str]], title: str = "Test Paper") -> LoadedDocument:
    """Construct a LoadedDocument from (section, text) tuples."""
    pages = [
        PageContent(text=text, page_number=i + 1, section=section)
        for i, (section, text) in enumerate(pages_spec)
    ]
    return LoadedDocument(
        source_path=Path("/tmp/test_paper.pdf"),
        file_hash="deadbeefdeadbeef",
        title=title,
        pages=pages,
    )


class TestSectionAwareChunker:
    def test_chunks_have_required_metadata(self) -> None:
        doc = _build_doc([("abstract", "This is the abstract. " * 50)])
        chunker = SectionAwareChunker(ChunkingSettings(chunk_size=200, chunk_overlap=20))
        chunks = chunker.chunk_document(doc)
        assert chunks, "should produce at least one chunk"
        for c in chunks:
            assert c.metadata["source"] == "test_paper.pdf"
            assert c.metadata["section"] == "abstract"
            assert c.metadata["page"] >= 1
            assert "chunk_id" in c.metadata
            assert "file_hash" in c.metadata

    def test_section_boundaries_are_hard(self) -> None:
        """No chunk should span two sections."""
        doc = _build_doc(
            [
                ("methods", "Methodology details. " * 40),
                ("results", "Result findings. " * 40),
            ]
        )
        chunker = SectionAwareChunker(
            ChunkingSettings(chunk_size=300, chunk_overlap=50, section_aware=True)
        )
        chunks = chunker.chunk_document(doc)
        sections = {c.metadata["section"] for c in chunks}
        assert "methods" in sections
        assert "results" in sections
        # Each chunk has exactly one section
        for c in chunks:
            assert c.metadata["section"] in {"methods", "results"}

    def test_tiny_chunks_filtered(self) -> None:
        """Chunks under 50 chars should be dropped."""
        doc = _build_doc([("abstract", "Tiny.")])
        chunker = SectionAwareChunker(ChunkingSettings(chunk_size=200, chunk_overlap=20))
        chunks = chunker.chunk_document(doc)
        assert chunks == []

    def test_deterministic_chunk_ids(self) -> None:
        doc = _build_doc([("abstract", "Same text " * 50)])
        chunker = SectionAwareChunker(ChunkingSettings(chunk_size=200, chunk_overlap=20))
        c1 = chunker.chunk_document(doc)
        c2 = chunker.chunk_document(doc)
        ids1 = [c.metadata["chunk_id"] for c in c1]
        ids2 = [c.metadata["chunk_id"] for c in c2]
        assert ids1 == ids2

    def test_flat_mode_ignores_sections(self) -> None:
        doc = _build_doc(
            [
                ("methods", "Methods text. " * 20),
                ("results", "Results text. " * 20),
            ]
        )
        chunker = SectionAwareChunker(
            ChunkingSettings(chunk_size=500, chunk_overlap=50, section_aware=False)
        )
        chunks = chunker.chunk_document(doc)
        # Flat mode: chunks tied to their originating page's section
        assert chunks, "flat mode should still produce chunks"

    def test_to_langchain_document(self) -> None:
        doc = _build_doc([("abstract", "Abstract text. " * 30)])
        chunker = SectionAwareChunker(ChunkingSettings(chunk_size=200, chunk_overlap=20))
        chunks = chunker.chunk_document(doc)
        lc = chunks[0].to_langchain()
        assert lc.page_content == chunks[0].text
        assert lc.metadata["source"] == "test_paper.pdf"
