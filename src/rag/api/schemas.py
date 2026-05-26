"""Pydantic schemas for the FastAPI service.

These are the public contract: any breaking change here is a breaking
change for the Streamlit UI and any external clients. Keep them stable
and additive.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ─── /query ──────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    """Incoming query request."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1, max_length=2000, description="User question.")
    debug: bool = Field(default=False, description="Include full chunk text in response.")


class ScoreBreakdown(BaseModel):
    """Per-chunk score breakdown for introspection."""

    model_config = ConfigDict(extra="allow")

    dense: float | None = None
    sparse: float | None = None
    reranker: float | None = None
    rrf_total: float | None = None


class RetrievedChunkDTO(BaseModel):
    """One chunk that made it to the LLM."""

    model_config = ConfigDict(extra="allow")

    chunk_id: str
    source: str
    page: int
    section: str
    paper_title: str
    score: float
    score_breakdown: dict[str, Any] = Field(default_factory=dict)
    text_preview: str | None = None
    text: str | None = None  # only when debug=True


class CitationCheckDTO(BaseModel):
    n_extracted: int
    n_valid: int
    n_invalid: int
    invalid_tags: list[str] = Field(default_factory=list)
    all_valid: bool


class LatencyMs(BaseModel):
    model_config = ConfigDict(extra="allow")

    retrieval_ms: float | None = None
    rerank_ms: float | None = None
    generation_ms: float | None = None
    total_ms: float | None = None


class QueryResponse(BaseModel):
    """Outgoing query response."""

    question: str
    answer: str
    abstained: bool
    citations: list[str] = Field(default_factory=list)
    citation_check: CitationCheckDTO
    retrieved_chunks: list[RetrievedChunkDTO] = Field(default_factory=list)
    latency_ms: LatencyMs
    prompt_version: str
    config_summary: dict[str, Any] = Field(default_factory=dict)


# ─── /ingest ─────────────────────────────────────────────────────


class IngestRequest(BaseModel):
    """Trigger ingestion of a directory of PDFs."""

    model_config = ConfigDict(extra="forbid")

    pdf_dir: str | None = Field(
        default=None,
        description="Override PDF directory (default: from settings).",
    )
    recreate_collection: bool = False
    skip_existing: bool = True


class IngestResponse(BaseModel):
    docs_loaded: int
    chunks_created: int
    points_upserted: int


# ─── /health ─────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
    version: str
    qdrant_reachable: bool
    ollama_reachable: bool
    collection: str
    points_count: int


# ─── /stats ──────────────────────────────────────────────────────


class StatsResponse(BaseModel):
    collection: str
    points_count: int
    unique_sources: int
    unique_file_hashes: int
    sources_sample: list[str]
    token_ledger: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "CitationCheckDTO",
    "HealthResponse",
    "IngestRequest",
    "IngestResponse",
    "LatencyMs",
    "QueryRequest",
    "QueryResponse",
    "RetrievedChunkDTO",
    "ScoreBreakdown",
    "StatsResponse",
]
