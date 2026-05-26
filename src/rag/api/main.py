"""FastAPI application entry point.

Run with:
    uvicorn rag.api.main:app --host 0.0.0.0 --port 8000

Or via Makefile:
    make api

Lifespan:
- On startup: configure logging, initialize the AppState (loads embedder,
  reranker, opens Qdrant client, builds RAG chain).
- On shutdown: flush Langfuse buffer.
"""
from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from rag.api.dependencies import get_state, init_state
from rag.api.routes import router
from rag.config import get_settings
from rag.logging_config import configure_logging, get_logger
from rag.observability import metrics

logger = get_logger(__name__)

# Static UI directory (vanilla HTML+CSS+JS frontend)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_STATIC_DIR = _PROJECT_ROOT / "static"


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup: build singletons. Shutdown: flush tracers."""
    configure_logging()
    logger.info("api_starting", version=get_settings().app_version)
    state = init_state()
    logger.info(
        "api_ready",
        collection=state.settings.vectorstore.collection,
        generator=state.settings.llm.generator_model,
    )
    try:
        yield
    finally:
        logger.info("api_shutting_down")
        try:
            state.tracer.flush()
        except Exception as e:
            logger.warning("shutdown_flush_failed", error=str(e))


def create_app() -> FastAPI:
    """App factory (lets tests build isolated app instances)."""
    settings = get_settings()
    app = FastAPI(
        title="Research RAG Assistant",
        description=(
            "Production-grade RAG system for academic literature QA with "
            "hybrid retrieval (BGE-M3 dense + sparse) and BGE cross-encoder "
            "reranking, served by Qwen2.5:7b via Ollama."
        ),
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    # ---- Static frontend (vanilla HTML + JS) -------------------------------
    if _STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        async def ui_root() -> FileResponse:
            return FileResponse(str(_STATIC_DIR / "index.html"))

        @app.get("/api", include_in_schema=False)
        async def api_index() -> dict[str, str]:
            return {
                "service": settings.app_name,
                "version": settings.app_version,
                "docs": "/docs",
                "health": "/health",
                "metrics": "/metrics",
                "ui": "/",
            }

        logger.info("static_ui_mounted", path=str(_STATIC_DIR))
    else:
        @app.get("/", include_in_schema=False)
        async def root() -> dict[str, str]:
            return {
                "service": settings.app_name,
                "version": settings.app_version,
                "docs": "/docs",
                "health": "/health",
                "metrics": "/metrics",
                "ui": "(no static/ folder found)",
            }
        logger.warning("static_ui_missing", path=str(_STATIC_DIR))

    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint() -> Response:
        body, content_type = metrics.render_metrics()
        return Response(content=body, media_type=content_type)

    @app.exception_handler(ValueError)
    async def value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app


app = create_app()

__all__ = ["app", "create_app", "get_state", "lifespan"]
