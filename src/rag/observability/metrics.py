"""Prometheus metrics for the RAG service.

Exposed at /metrics by the FastAPI app. Three high-signal metric families:

- `rag_query_total{status}` — counter, status ∈ {ok, abstained, error}
- `rag_query_latency_seconds{stage}` — histogram, stage ∈
   {retrieval, rerank, generation, total}
- `rag_retrieved_chunks` — histogram, distribution of chunks reaching the LLM
- `rag_citation_fabrication_total` — counter, hallucinated citations seen
- `rag_token_usage_total{kind}` — counter, kind ∈ {prompt, completion}
"""

from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from prometheus_client.openmetrics.exposition import (
    CONTENT_TYPE_LATEST as OPENMETRICS_CONTENT_TYPE,
)

# Use a dedicated registry so tests / multiple imports don't double-register
REGISTRY = CollectorRegistry()

QUERY_TOTAL = Counter(
    "rag_query_total",
    "Total RAG queries handled, partitioned by status.",
    labelnames=("status",),
    registry=REGISTRY,
)

QUERY_LATENCY = Histogram(
    "rag_query_latency_seconds",
    "Latency per pipeline stage.",
    labelnames=("stage",),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
    registry=REGISTRY,
)

RETRIEVED_CHUNKS = Histogram(
    "rag_retrieved_chunks",
    "Number of chunks reaching the LLM after rerank.",
    buckets=(0, 1, 2, 3, 5, 8, 13, 21),
    registry=REGISTRY,
)

CITATION_FABRICATION = Counter(
    "rag_citation_fabrication_total",
    "Citations the LLM produced that were not in the provided context.",
    registry=REGISTRY,
)

TOKEN_USAGE = Counter(
    "rag_token_usage_total",
    "Token usage by kind.",
    labelnames=("kind",),
    registry=REGISTRY,
)

INGEST_DOCS = Counter(
    "rag_ingest_documents_total",
    "Documents successfully ingested.",
    registry=REGISTRY,
)

INGEST_CHUNKS = Counter(
    "rag_ingest_chunks_total",
    "Chunks upserted to the vector store.",
    registry=REGISTRY,
)


def record_response(response_dict: dict) -> None:
    """Update metrics from a serialized RAGResponse."""
    if response_dict.get("abstained"):
        QUERY_TOTAL.labels(status="abstained").inc()
    else:
        QUERY_TOTAL.labels(status="ok").inc()

    latency = response_dict.get("latency_ms") or {}
    for stage_key, stage_label in [
        ("retrieval_ms", "retrieval"),
        ("rerank_ms", "rerank"),
        ("generation_ms", "generation"),
        ("total_ms", "total"),
    ]:
        if stage_key in latency:
            QUERY_LATENCY.labels(stage=stage_label).observe(latency[stage_key] / 1000.0)

    chunks = response_dict.get("retrieved_chunks") or []
    RETRIEVED_CHUNKS.observe(len(chunks))

    cc = response_dict.get("citation_check") or {}
    if cc.get("n_invalid"):
        CITATION_FABRICATION.inc(cc["n_invalid"])


def record_error() -> None:
    QUERY_TOTAL.labels(status="error").inc()


def render_metrics() -> tuple[bytes, str]:
    """Return (body, content_type) for the /metrics endpoint."""
    return generate_latest(REGISTRY), OPENMETRICS_CONTENT_TYPE


__all__ = [
    "CITATION_FABRICATION",
    "INGEST_CHUNKS",
    "INGEST_DOCS",
    "QUERY_LATENCY",
    "QUERY_TOTAL",
    "REGISTRY",
    "RETRIEVED_CHUNKS",
    "TOKEN_USAGE",
    "record_error",
    "record_response",
    "render_metrics",
]
