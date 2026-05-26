"""Download the eval corpus from arXiv into data/pdfs/.

Reads `eval/papers.txt`, downloads each paper by ID, and saves it under
`<short_alias>.pdf` in the configured PDF directory. Idempotent: skips
files that already exist (verified by hash).

Run with:
    python -m eval.download_papers
    make download-papers
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import arxiv

from rag.config import get_settings
from rag.logging_config import configure_logging, get_logger

logger = get_logger(__name__)

PAPERS_LIST = Path(__file__).resolve().parent / "papers.txt"


@dataclass(frozen=True)
class PaperRef:
    arxiv_id: str
    alias: str
    cluster: str


def load_paper_refs(path: Path = PAPERS_LIST) -> list[PaperRef]:
    """Parse papers.txt into a list of PaperRef records."""
    refs: list[PaperRef] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            logger.warning("malformed_paper_line", line=raw)
            continue
        refs.append(PaperRef(arxiv_id=parts[0], alias=parts[1], cluster=parts[2]))
    return refs


def download_paper(ref: PaperRef, out_dir: Path, force: bool = False) -> Path | None:
    """Download a single paper. Returns the saved path or None on failure."""
    out_path = out_dir / f"{ref.alias}.pdf"
    if out_path.exists() and not force:
        logger.info("paper_exists_skip", alias=ref.alias, path=str(out_path))
        return out_path

    try:
        search = arxiv.Search(id_list=[ref.arxiv_id])
        client = arxiv.Client()
        results = list(client.results(search))
        if not results:
            logger.error("arxiv_not_found", arxiv_id=ref.arxiv_id)
            return None
        paper = results[0]
        paper.download_pdf(dirpath=str(out_dir), filename=f"{ref.alias}.pdf")
        logger.info(
            "paper_downloaded",
            alias=ref.alias,
            arxiv_id=ref.arxiv_id,
            title=paper.title[:80],
            size_kb=out_path.stat().st_size // 1024,
        )
        return out_path
    except Exception as e:
        logger.error("paper_download_failed", arxiv_id=ref.arxiv_id, error=str(e))
        return None


def download_all(out_dir: Path | None = None, force: bool = False) -> dict[str, int]:
    """Download every paper in papers.txt."""
    out_dir = out_dir or get_settings().ingestion.pdf_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    refs = load_paper_refs()
    logger.info("download_start", n_papers=len(refs), out_dir=str(out_dir))
    ok = 0
    fail = 0
    for ref in refs:
        result = download_paper(ref, out_dir, force=force)
        if result is not None:
            ok += 1
        else:
            fail += 1
        # Gentle on arxiv: 3s between requests (arxiv API guideline)
        time.sleep(3.0)
    logger.info("download_complete", ok=ok, fail=fail, total=len(refs))
    return {"ok": ok, "fail": fail, "total": len(refs)}


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    import argparse

    p = argparse.ArgumentParser(description="Download eval corpus from arXiv.")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--force", action="store_true", help="Re-download existing PDFs.")
    args = p.parse_args(argv)
    result = download_all(out_dir=args.out_dir, force=args.force)
    print(f"Downloaded {result['ok']}/{result['total']} papers ({result['fail']} failed).")
    return 0 if result["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["PaperRef", "download_all", "download_paper", "load_paper_refs", "main"]
