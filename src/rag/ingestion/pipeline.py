"""End-to-end ingestion pipeline: PDF dir → chunks → embeddings → Qdrant.

Run via:
    python -m rag.ingestion.pipeline --pdf-dir ./data/pdfs

Or, from another module:
    from rag.ingestion.pipeline import ingest_directory
    ingest_directory(Path("./data/pdfs"))
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.config import get_settings
from rag.embeddings.bge_embedder import build_embedder
from rag.ingestion.chunker import SectionAwareChunker
from rag.ingestion.pdf_loader import load_pdfs
from rag.logging_config import configure_logging, get_logger
from rag.vectorstore.qdrant_store import QdrantStore

logger = get_logger(__name__)


def ingest_directory(
    pdf_dir: Path,
    recreate_collection: bool = False,
    skip_existing: bool = True,
) -> dict[str, int]:
    """Ingest all PDFs from a directory into the configured Qdrant collection.

    Args:
        pdf_dir: Directory containing PDF files.
        recreate_collection: If True, drop and recreate the collection first.
        skip_existing: If True, skip PDFs whose file_hash already exists.

    Returns:
        Dict with counts: docs_loaded, chunks_created, points_upserted.
    """
    settings = get_settings()
    pdf_dir = Path(pdf_dir) if pdf_dir else settings.ingestion.pdf_dir
    pdf_dir.mkdir(parents=True, exist_ok=True)

    logger.info("ingestion_start", pdf_dir=str(pdf_dir))

    # 1) Load PDFs
    docs = load_pdfs(pdf_dir)
    if not docs:
        logger.warning("no_documents_to_ingest")
        return {"docs_loaded": 0, "chunks_created": 0, "points_upserted": 0}

    # 2) Chunk
    chunker = SectionAwareChunker(settings.chunking)
    chunks = chunker.chunk_documents(docs)
    if not chunks:
        logger.warning("no_chunks_produced")
        return {"docs_loaded": len(docs), "chunks_created": 0, "points_upserted": 0}

    # 3) Embed + upsert into Qdrant
    embedder = build_embedder(settings.embedding)
    store = QdrantStore(embedder=embedder, settings=settings.vectorstore)
    store.ensure_collection(recreate=recreate_collection)
    upserted = store.upsert_chunks(chunks, skip_existing=skip_existing)

    stats = store.collection_stats()
    logger.info(
        "ingestion_complete",
        docs_loaded=len(docs),
        chunks_created=len(chunks),
        points_upserted=upserted,
        collection_total_points=stats["points_count"],
        unique_sources=stats["unique_sources"],
    )
    return {
        "docs_loaded": len(docs),
        "chunks_created": len(chunks),
        "points_upserted": upserted,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rag.ingestion.pipeline",
        description="Ingest PDFs into the Qdrant vector store.",
    )
    p.add_argument(
        "--pdf-dir",
        type=Path,
        default=None,
        help="Directory containing PDF files (default: from env / settings).",
    )
    p.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the collection before ingesting.",
    )
    p.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-ingest PDFs even if their file_hash is already present.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = _build_parser().parse_args(argv)
    pdf_dir = args.pdf_dir or get_settings().ingestion.pdf_dir
    result = ingest_directory(
        pdf_dir=pdf_dir,
        recreate_collection=args.recreate,
        skip_existing=not args.no_skip_existing,
    )
    print(
        f"Ingested {result['docs_loaded']} documents, "
        f"created {result['chunks_created']} chunks, "
        f"upserted {result['points_upserted']} points."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["ingest_directory", "main"]
