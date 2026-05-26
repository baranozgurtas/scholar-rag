"""Section-aware document chunking.

Strategy:
1. If `section_aware=True` (default), each section becomes a sub-document
   before chunking. Section boundaries become hard chunk boundaries —
   no chunk crosses a section, preserving topical coherence.
2. Within each section, fall back to `RecursiveCharacterTextSplitter`
   with the configured chunk_size and chunk_overlap.
3. Each chunk carries rich metadata: source, page, section, chunk_idx,
   file_hash, paper_title — used downstream for citation and filtering.

This is the elite touch academic RAG papers (Self-RAG, RAFT) call out:
section-aware chunking improves both retrieval precision and citation
faithfulness on long-form scientific documents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import ChunkingSettings, get_settings
from rag.ingestion.pdf_loader import LoadedDocument, PageContent
from rag.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """A retrieval-ready text chunk with metadata."""

    text: str
    metadata: dict[str, Any]

    def to_langchain(self) -> Document:
        return Document(page_content=self.text, metadata=self.metadata)


class SectionAwareChunker:
    """Splits documents into chunks while respecting section boundaries.

    Falls back to plain recursive splitting if `section_aware=False` or
    a document has no detected sections.
    """

    def __init__(self, settings: ChunkingSettings | None = None) -> None:
        self.settings = settings or get_settings().chunking
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_document(self, doc: LoadedDocument) -> list[Chunk]:
        """Split a LoadedDocument into chunks with full metadata."""
        if self.settings.section_aware:
            return self._chunk_section_aware(doc)
        return self._chunk_flat(doc)

    def _chunk_flat(self, doc: LoadedDocument) -> list[Chunk]:
        """Plain recursive chunking, ignoring section structure."""
        chunks: list[Chunk] = []
        for page in doc.pages:
            for piece in self._splitter.split_text(page.text):
                chunks.append(
                    Chunk(
                        text=piece,
                        metadata=self._build_metadata(
                            doc, page=page, chunk_idx=len(chunks), section=page.section
                        ),
                    )
                )
        return self._post_process(chunks)

    def _chunk_section_aware(self, doc: LoadedDocument) -> list[Chunk]:
        """Group pages by section, then recursively split each section."""
        # Group consecutive pages with same section together
        grouped: list[tuple[str, list[PageContent]]] = []
        for page in doc.pages:
            if grouped and grouped[-1][0] == page.section:
                grouped[-1][1].append(page)
            else:
                grouped.append((page.section, [page]))

        chunks: list[Chunk] = []
        for section, pages in grouped:
            section_text = "\n\n".join(p.text for p in pages)
            if not section_text.strip():
                continue
            pieces = self._splitter.split_text(section_text)
            primary_page = pages[0].page_number
            for piece in pieces:
                chunks.append(
                    Chunk(
                        text=piece,
                        metadata=self._build_metadata(
                            doc,
                            page=pages[0],
                            chunk_idx=len(chunks),
                            section=section,
                            page_start=primary_page,
                            page_end=pages[-1].page_number,
                        ),
                    )
                )

        return self._post_process(chunks)

    @staticmethod
    def _build_metadata(
        doc: LoadedDocument,
        page: PageContent,
        chunk_idx: int,
        section: str,
        page_start: int | None = None,
        page_end: int | None = None,
    ) -> dict[str, Any]:
        return {
            "source": doc.source_path.name,
            "source_path": str(doc.source_path),
            "file_hash": doc.file_hash,
            "paper_title": doc.title,
            "page": page.page_number,
            "page_start": page_start if page_start is not None else page.page_number,
            "page_end": page_end if page_end is not None else page.page_number,
            "section": section,
            "chunk_idx": chunk_idx,
        }

    @staticmethod
    def _post_process(chunks: list[Chunk]) -> list[Chunk]:
        """Drop empty / tiny chunks and reindex."""
        filtered = [c for c in chunks if len(c.text.strip()) >= 50]
        for i, c in enumerate(filtered):
            c.metadata["chunk_idx"] = i
            c.metadata["chunk_id"] = f"{c.metadata['file_hash'][:8]}_{i:04d}"
        return filtered

    def chunk_documents(self, docs: list[LoadedDocument]) -> list[Chunk]:
        """Chunk a batch of documents and log totals."""
        all_chunks: list[Chunk] = []
        for doc in docs:
            doc_chunks = self.chunk_document(doc)
            all_chunks.extend(doc_chunks)
            logger.info(
                "doc_chunked",
                source=doc.source_path.name,
                chunks=len(doc_chunks),
                sections=len({c.metadata["section"] for c in doc_chunks}),
            )
        logger.info("chunking_complete", total_chunks=len(all_chunks), docs=len(docs))
        return all_chunks


__all__ = ["Chunk", "SectionAwareChunker"]
