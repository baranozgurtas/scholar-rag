"""PDF loader using PyMuPDF (fitz) with section-aware extraction.

We use PyMuPDF over pypdf because it gives much better text extraction
quality on academic PDFs (preserves layout, handles columns, exposes
font info we use for heading detection).

Each loaded document is a list of `PageContent` records carrying page
number, raw text, and detected section heading (Abstract / Introduction /
Methods / Results / Discussion / Conclusion / References / Other).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

from rag.logging_config import get_logger

logger = get_logger(__name__)

# Common section headings in academic papers (order matters for regex precedence)
_SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("abstract", re.compile(r"^\s*abstract\s*$", re.IGNORECASE)),
    ("introduction", re.compile(r"^\s*(?:\d+\.?\s*)?introduction\s*$", re.IGNORECASE)),
    ("related_work", re.compile(r"^\s*(?:\d+\.?\s*)?related\s+work\s*$", re.IGNORECASE)),
    ("background", re.compile(r"^\s*(?:\d+\.?\s*)?background\s*$", re.IGNORECASE)),
    ("methods", re.compile(r"^\s*(?:\d+\.?\s*)?(?:methods?|methodology|approach)\s*$", re.IGNORECASE)),
    ("experiments", re.compile(r"^\s*(?:\d+\.?\s*)?experiments?(?:\s+setup)?\s*$", re.IGNORECASE)),
    ("results", re.compile(r"^\s*(?:\d+\.?\s*)?results?\s*$", re.IGNORECASE)),
    ("evaluation", re.compile(r"^\s*(?:\d+\.?\s*)?evaluation\s*$", re.IGNORECASE)),
    ("discussion", re.compile(r"^\s*(?:\d+\.?\s*)?discussion\s*$", re.IGNORECASE)),
    ("conclusion", re.compile(r"^\s*(?:\d+\.?\s*)?conclusions?\s*$", re.IGNORECASE)),
    ("limitations", re.compile(r"^\s*(?:\d+\.?\s*)?limitations?\s*$", re.IGNORECASE)),
    ("references", re.compile(r"^\s*references\s*$", re.IGNORECASE)),
    ("appendix", re.compile(r"^\s*appendix\s*(?:[a-z]\b)?\s*$", re.IGNORECASE)),
]


@dataclass
class PageContent:
    """A single page of a loaded PDF."""

    text: str
    page_number: int  # 1-indexed
    section: str = "other"


@dataclass
class LoadedDocument:
    """A loaded PDF document with metadata."""

    source_path: Path
    file_hash: str
    title: str
    pages: list[PageContent]
    metadata: dict = field(default_factory=dict)

    @property
    def num_pages(self) -> int:
        return len(self.pages)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


def _detect_section(line: str) -> str | None:
    """Return canonical section name if `line` looks like a heading, else None."""
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return None
    for name, pattern in _SECTION_PATTERNS:
        if pattern.match(stripped):
            return name
    return None


def _compute_file_hash(path: Path) -> str:
    """MD5 hash of file content for deduplication."""
    h = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_title(doc: fitz.Document, fallback: str) -> str:
    """Best-effort title extraction from PDF metadata or first-page heuristic."""
    meta_title = doc.metadata.get("title", "") if doc.metadata else ""
    if meta_title and len(meta_title) > 5 and not meta_title.lower().startswith("untitled"):
        return meta_title.strip()

    # Fallback: largest-font line on page 1 (academic papers convention)
    if len(doc) > 0:
        page = doc[0]
        blocks = page.get_text("dict").get("blocks", [])
        candidates: list[tuple[float, str]] = []
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    size = span.get("size", 0.0)
                    if text and len(text) > 10 and size > 0:
                        candidates.append((size, text))
        if candidates:
            candidates.sort(reverse=True)
            # Pick top-font line that doesn't look like an artifact
            for _size, text in candidates[:5]:
                if not re.search(r"\d{4}|arxiv|doi", text, re.IGNORECASE) and len(text) < 200:
                    return text
    return fallback


def load_pdf(path: Path) -> LoadedDocument:
    """Load a single PDF and detect section per page.

    Section detection is sticky: once a section heading is found on a page,
    all subsequent text belongs to that section until the next heading.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    file_hash = _compute_file_hash(path)
    pages: list[PageContent] = []
    current_section = "other"

    with fitz.open(path) as doc:
        title = _extract_title(doc, fallback=path.stem)
        metadata = {
            "author": (doc.metadata or {}).get("author", ""),
            "subject": (doc.metadata or {}).get("subject", ""),
            "creator": (doc.metadata or {}).get("creator", ""),
            "producer": (doc.metadata or {}).get("producer", ""),
        }

        for page_idx, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            text = text.strip()
            if not text:
                continue

            # Sticky section: scan lines for heading, update on hit
            for line in text.splitlines():
                detected = _detect_section(line)
                if detected is not None:
                    current_section = detected
                    break  # one section change per page is enough

            pages.append(
                PageContent(text=text, page_number=page_idx, section=current_section)
            )

    logger.info(
        "pdf_loaded",
        source=path.name,
        pages=len(pages),
        title=title[:80],
        file_hash=file_hash[:8],
    )
    return LoadedDocument(
        source_path=path,
        file_hash=file_hash,
        title=title,
        pages=pages,
        metadata=metadata,
    )


def load_pdfs(pdf_dir: Path) -> list[LoadedDocument]:
    """Load all PDFs from a directory (non-recursive)."""
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        logger.warning("no_pdfs_found", dir=str(pdf_dir))
        return []

    docs: list[LoadedDocument] = []
    for p in pdf_paths:
        try:
            docs.append(load_pdf(p))
        except Exception as e:
            logger.error("pdf_load_failed", source=p.name, error=str(e))
    logger.info("pdfs_loaded", count=len(docs), dir=str(pdf_dir))
    return docs


__all__ = ["LoadedDocument", "PageContent", "load_pdf", "load_pdfs"]
