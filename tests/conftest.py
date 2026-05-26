"""Pytest fixtures shared across the test suite."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Ensure tests don't read the developer's .env file or touch real services."""
    monkeypatch.chdir(tmp_path)
    # Block obvious external services by default
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:0")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:0")
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    monkeypatch.setenv("EMBEDDING_DEVICE", "cpu")
    monkeypatch.setenv("RERANKER_DEVICE", "cpu")
    monkeypatch.setenv("EMBEDDING_CACHE_ENABLED", "false")
    yield


@pytest.fixture
def fake_chunk_metadata() -> dict[str, Any]:
    return {
        "source": "bge-m3-chen-2024.pdf",
        "source_path": "/data/pdfs/bge-m3-chen-2024.pdf",
        "file_hash": "abcd1234abcd1234",
        "paper_title": "BGE-M3: Multi-Functional Embedding",
        "page": 3,
        "page_start": 3,
        "page_end": 3,
        "section": "methods",
        "chunk_idx": 7,
        "chunk_id": "abcd1234_0007",
    }


@pytest.fixture
def fake_loaded_document(fake_chunk_metadata: dict[str, Any]):
    """Construct a LoadedDocument with two pages across two sections."""
    from rag.ingestion.pdf_loader import LoadedDocument, PageContent

    pages = [
        PageContent(
            text="Abstract\nThis paper introduces BGE-M3, a multi-functional embedding model "
            * 5,
            page_number=1,
            section="abstract",
        ),
        PageContent(
            text="Methods\nWe train the model with multi-task learning across dense, sparse, "
            "and multi-vector retrieval objectives. " * 6,
            page_number=2,
            section="methods",
        ),
        PageContent(
            text="Results\nBGE-M3 achieves state-of-the-art performance on MTEB. " * 6,
            page_number=3,
            section="results",
        ),
    ]
    return LoadedDocument(
        source_path=Path(fake_chunk_metadata["source_path"]),
        file_hash=fake_chunk_metadata["file_hash"],
        title=fake_chunk_metadata["paper_title"],
        pages=pages,
    )


@pytest.fixture
def mock_qdrant_store(fake_chunk_metadata: dict[str, Any]) -> MagicMock:
    """A MagicMock that mimics QdrantStore's interface."""
    from qdrant_client.http import models

    store = MagicMock()
    store.collection_name = "test_collection"
    # search_dense / search_sparse return ScoredPoint-like objects
    point_payload = {**fake_chunk_metadata, "text": "BGE-M3 supports dense and sparse retrieval."}

    def _scored(point_id: str, score: float) -> Any:
        sp = MagicMock(spec=models.ScoredPoint)
        sp.id = point_id
        sp.score = score
        sp.payload = point_payload
        return sp

    store.search_dense.return_value = [_scored("p1", 0.92), _scored("p2", 0.81)]
    store.search_sparse.return_value = [_scored("p2", 8.5), _scored("p3", 4.2)]
    store.collection_stats.return_value = {
        "collection": "test_collection",
        "points_count": 100,
        "unique_sources": 3,
        "unique_file_hashes": 3,
        "sources_sample": ["bge-m3-chen-2024.pdf", "ragas-es-2023.pdf"],
    }
    store.scroll_all_payloads.return_value = [
        {"chunk_id": f"c{i}", "text": f"Chunk {i} about machine learning {'embedding' if i % 2 else 'optimizer'}",
         "source": "fake.pdf"}
        for i in range(10)
    ]
    return store


@pytest.fixture
def mock_embedder() -> MagicMock:
    """A MagicMock embedder returning deterministic dense + sparse outputs."""
    import numpy as np

    emb = MagicMock()
    emb.embed_query.return_value = [0.1] * 1024

    def _dense_sparse(texts: list[str], is_query: bool = False):
        n = len(texts)
        dense = np.tile(np.linspace(0.0, 1.0, 1024), (n, 1))
        sparse = [dict.fromkeys(range(5), 0.5) for _ in range(n)]
        return dense, sparse

    emb.embed_dense_and_sparse.side_effect = _dense_sparse
    return emb
