"""HTTP routes for the RAG API.

All endpoints are async; the heavy CPU/GPU work (embedding, reranking,
LLM call) is offloaded to `asyncio.to_thread` so the event loop stays
responsive for concurrent requests.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, HTTPException, status

from rag.api.dependencies import AppState, get_state
from rag.api.schemas import (
    CitationCheckDTO,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    LatencyMs,
    QueryRequest,
    QueryResponse,
    RetrievedChunkDTO,
    StatsResponse,
)
from rag.config import get_settings
from rag.logging_config import get_logger
from rag.observability import metrics
from rag.observability.token_counter import count_tokens

logger = get_logger(__name__)

router = APIRouter()


# ─── /query ──────────────────────────────────────────────────────


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question against the indexed papers.",
)
async def query(req: QueryRequest, state: AppState = Depends(get_state)) -> QueryResponse:
    """Run the full RAG pipeline for a question."""
    try:
        # Offload to thread pool (LLM call is blocking)
        response = await asyncio.to_thread(
            state.rag_chain.answer, req.question, debug=req.debug
        )
        response_dict = response.to_dict()

        # Observability: metrics + Langfuse + token ledger
        metrics.record_response(response_dict)
        state.tracer.trace_query(
            question=req.question,
            response=response_dict,
            prompt_version=response.prompt_version,
        )

        prompt_tokens = count_tokens(req.question) + sum(
            count_tokens(c.get("text") or c.get("text_preview") or "")
            for c in response_dict["retrieved_chunks"]
        )
        completion_tokens = count_tokens(response.answer)
        state.token_ledger.record(
            question=req.question,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            extra={
                "abstained": response.abstained,
                "prompt_version": response.prompt_version,
            },
        )
        metrics.TOKEN_USAGE.labels(kind="prompt").inc(prompt_tokens)
        metrics.TOKEN_USAGE.labels(kind="completion").inc(completion_tokens)

        return QueryResponse(
            question=response.question,
            answer=response.answer,
            abstained=response.abstained,
            citations=response.citations,
            citation_check=CitationCheckDTO(**response.citation_check),
            retrieved_chunks=[RetrievedChunkDTO(**c) for c in response.retrieved_chunks],
            latency_ms=LatencyMs(**response.latency_ms),
            prompt_version=response.prompt_version,
            config_summary=response.config_summary,
        )
    except HTTPException:
        raise
    except Exception as e:
        metrics.record_error()
        logger.error("query_failed", error=str(e), question=req.question[:120])
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {e}",
        ) from e


# ─── /ingest ─────────────────────────────────────────────────────


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest PDFs from a directory into the vector store.",
)
async def ingest(
    req: IngestRequest, state: AppState = Depends(get_state)
) -> IngestResponse:
    """Trigger ingestion. Long-running but synchronous (no background tasks)."""
    from rag.ingestion.pipeline import ingest_directory

    pdf_dir = Path(req.pdf_dir) if req.pdf_dir else state.settings.ingestion.pdf_dir
    try:
        result = await asyncio.to_thread(
            ingest_directory,
            pdf_dir=pdf_dir,
            recreate_collection=req.recreate_collection,
            skip_existing=req.skip_existing,
        )
        metrics.INGEST_DOCS.inc(result["docs_loaded"])
        metrics.INGEST_CHUNKS.inc(result["points_upserted"])
        return IngestResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e
    except Exception as e:
        logger.error("ingest_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {e}",
        ) from e


# ─── /health ─────────────────────────────────────────────────────


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health probe (used by Docker, k8s, load balancers).",
)
async def health(state: AppState = Depends(get_state)) -> HealthResponse:
    """Check Qdrant + Ollama connectivity. Returns 200 always; status field
    captures the actual state so external probes can decide what to do."""
    settings = get_settings()
    qdrant_ok = False
    points = 0
    try:
        info = state.store.client.get_collection(state.settings.vectorstore.collection)
        qdrant_ok = True
        points = info.points_count or 0
    except Exception as e:
        logger.warning("qdrant_unreachable", error=str(e))

    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{settings.llm.ollama_base_url}/api/tags")
            ollama_ok = r.status_code == 200
    except Exception as e:
        logger.warning("ollama_unreachable", error=str(e))

    overall = "ok" if (qdrant_ok and ollama_ok) else "degraded"
    return HealthResponse(
        status=overall,
        version=settings.app_version,
        qdrant_reachable=qdrant_ok,
        ollama_reachable=ollama_ok,
        collection=settings.vectorstore.collection,
        points_count=points,
    )


# ─── /stats ──────────────────────────────────────────────────────


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Vector store + token ledger summary.",
)
async def stats(state: AppState = Depends(get_state)) -> StatsResponse:
    """Aggregate stats: collection size, distinct sources, token usage history."""
    try:
        coll_stats = await asyncio.to_thread(state.store.collection_stats)
        ledger = state.token_ledger.aggregate()
        return StatsResponse(
            collection=coll_stats["collection"],
            points_count=coll_stats["points_count"],
            unique_sources=coll_stats["unique_sources"],
            unique_file_hashes=coll_stats["unique_file_hashes"],
            sources_sample=coll_stats["sources_sample"],
            token_ledger=ledger,
        )
    except Exception as e:
        logger.error("stats_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stats failed: {e}",
        ) from e


__all__ = ["router"]
