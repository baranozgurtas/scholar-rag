"""FastAPI smoke tests using TestClient and a mocked AppState."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from rag.api.dependencies import AppState, get_state
from rag.config import get_settings


@pytest.fixture
def mock_app_state() -> AppState:
    """Build a minimal AppState with all components mocked."""
    settings = get_settings()
    rag_chain = MagicMock()
    # Mock answer returns a serializable RAGResponse-like object
    response_obj = MagicMock()
    response_obj.question = "q"
    response_obj.answer = "mocked answer"
    response_obj.abstained = False
    response_obj.citations = ["[Paper: BGE-M3 | p.1 | §abstract]"]
    response_obj.citation_check = {
        "n_extracted": 1, "n_valid": 1, "n_invalid": 0, "invalid_tags": [], "all_valid": True
    }
    response_obj.retrieved_chunks = [
        {
            "chunk_id": "c1", "source": "bge-m3.pdf", "page": 1, "section": "abstract",
            "paper_title": "BGE-M3", "score": 0.9, "score_breakdown": {"reranker": 0.9},
            "text_preview": "Some text",
        }
    ]
    response_obj.latency_ms = {"retrieval_ms": 50, "rerank_ms": 100, "generation_ms": 800,
                                "total_ms": 950}
    response_obj.prompt_version = "rag_answer@1.2"
    response_obj.config_summary = {"use_reranker": True}
    response_obj.to_dict.return_value = {
        "question": "q", "answer": "mocked answer", "abstained": False,
        "citations": response_obj.citations, "citation_check": response_obj.citation_check,
        "retrieved_chunks": response_obj.retrieved_chunks, "latency_ms": response_obj.latency_ms,
        "prompt_version": response_obj.prompt_version, "config_summary": response_obj.config_summary,
    }
    rag_chain.answer.return_value = response_obj

    store = MagicMock()
    store.client.get_collection.return_value = MagicMock(points_count=42)
    store.collection_stats.return_value = {
        "collection": settings.vectorstore.collection,
        "points_count": 42,
        "unique_sources": 3,
        "unique_file_hashes": 3,
        "sources_sample": ["a.pdf", "b.pdf"],
    }

    tracer = MagicMock()
    tracer.enabled = False

    token_ledger = MagicMock()
    token_ledger.aggregate.return_value = {"queries": 0}
    token_ledger.record.return_value = None

    return AppState(
        settings=settings,
        embedder=MagicMock(),
        store=store,
        reranker=MagicMock(),
        rag_chain=rag_chain,
        tracer=tracer,
        token_ledger=token_ledger,
    )


@pytest.fixture
def client(mock_app_state: AppState) -> TestClient:
    """TestClient with `get_state` overridden to return the mocked state."""
    from rag.api.main import create_app

    app = create_app()
    app.dependency_overrides[get_state] = lambda: mock_app_state
    with TestClient(app) as c:
        yield c


class TestRootAndDocs:
    def test_root(self, client: TestClient) -> None:
        r = client.get("/")
        assert r.status_code == 200
        body = r.json()
        assert body["service"]
        assert body["docs"] == "/docs"

    def test_openapi_schema(self, client: TestClient) -> None:
        r = client.get("/openapi.json")
        assert r.status_code == 200
        schema = r.json()
        # Verify all 4 endpoints are present
        assert "/query" in schema["paths"]
        assert "/ingest" in schema["paths"]
        assert "/health" in schema["paths"]
        assert "/stats" in schema["paths"]


class TestQuery:
    def test_query_happy_path(self, client: TestClient) -> None:
        r = client.post("/query", json={"question": "What is BGE-M3?"})
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == "mocked answer"
        assert body["abstained"] is False
        assert body["citation_check"]["all_valid"]
        assert body["prompt_version"] == "rag_answer@1.2"

    def test_query_rejects_empty(self, client: TestClient) -> None:
        r = client.post("/query", json={"question": ""})
        assert r.status_code == 422

    def test_query_rejects_unknown_field(self, client: TestClient) -> None:
        # extra="forbid" enforced
        r = client.post("/query", json={"question": "ok", "unknown_field": "x"})
        assert r.status_code == 422


class TestHealth:
    def test_health_ok(self, client: TestClient, mock_app_state: AppState) -> None:
        # Patch httpx call to Ollama: avoid real network in the test
        from unittest.mock import patch

        import httpx

        async def _mock_get(*_args, **_kwargs):
            class R:
                status_code = 200
            return R()

        with patch.object(httpx.AsyncClient, "get", new=_mock_get):
            r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["qdrant_reachable"] is True
        assert body["points_count"] == 42

    def test_health_degraded_when_ollama_down(self, client: TestClient) -> None:
        # By default in mocked-state setup, Ollama is unreachable → status=degraded
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        # qdrant_reachable=True (mocked) but ollama unreachable → status='degraded'
        assert body["ollama_reachable"] is False
        assert body["status"] == "degraded"


class TestStats:
    def test_stats_returns_collection_info(self, client: TestClient) -> None:
        r = client.get("/stats")
        assert r.status_code == 200
        body = r.json()
        assert body["points_count"] == 42
        assert body["unique_sources"] == 3


class TestMetrics:
    def test_metrics_endpoint_exposes_prometheus(self, client: TestClient) -> None:
        r = client.get("/metrics")
        assert r.status_code == 200
        body = r.text
        assert "rag_query_total" in body or "rag_query_latency_seconds" in body
