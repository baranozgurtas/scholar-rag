"""Langfuse tracing integration with graceful no-op fallback.

If `LANGFUSE_ENABLED=false` or the public/secret keys are missing, every
function in this module is a no-op — the app runs identically without
Langfuse, just without trace persistence.

This pattern keeps Langfuse fully optional: the developer experience
of running a local Qdrant + Ollama + no-cloud-keys is one command,
while production deployments can wire in a self-hosted Langfuse instance.
"""

from __future__ import annotations

from typing import Any

from rag.config import ObservabilitySettings, get_settings
from rag.logging_config import get_logger

logger = get_logger(__name__)

try:
    from langfuse import Langfuse  # type: ignore[import-untyped]

    LANGFUSE_AVAILABLE = True
except ImportError:  # pragma: no cover
    Langfuse = None  # type: ignore[assignment,misc]
    LANGFUSE_AVAILABLE = False


class LangfuseTracer:
    """Thin wrapper that no-ops when Langfuse is disabled or unconfigured."""

    def __init__(self, settings: ObservabilitySettings | None = None) -> None:
        s = settings or get_settings().observability
        self.settings = s
        self.client: Any | None = None

        if not s.langfuse_enabled:
            logger.info("langfuse_disabled_by_config")
            return
        if not LANGFUSE_AVAILABLE:
            logger.warning("langfuse_package_not_installed")
            return
        if not (s.langfuse_public_key and s.langfuse_secret_key):
            logger.info("langfuse_keys_missing_skip")
            return

        try:
            self.client = Langfuse(
                public_key=s.langfuse_public_key,
                secret_key=s.langfuse_secret_key,
                host=s.langfuse_host,
            )
            logger.info("langfuse_initialized", host=s.langfuse_host)
        except Exception as e:
            logger.warning("langfuse_init_failed", error=str(e))
            self.client = None

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def trace_query(
        self,
        question: str,
        response: dict[str, Any],
        prompt_version: str,
    ) -> None:
        """Persist a query → response trace to Langfuse (no-op if disabled)."""
        if not self.enabled or self.client is None:
            return
        try:
            trace = self.client.trace(
                name="rag_query",
                input={"question": question},
                output={
                    "answer": response.get("answer"),
                    "abstained": response.get("abstained"),
                    "citations": response.get("citations"),
                },
                metadata={
                    "prompt_version": prompt_version,
                    "config_summary": response.get("config_summary"),
                    "citation_check": response.get("citation_check"),
                    "latency_ms": response.get("latency_ms"),
                    "n_retrieved": len(response.get("retrieved_chunks") or []),
                },
                tags=["rag", "prod" if not response.get("abstained") else "abstained"],
            )
            # Sub-span for retrieval (so latency per stage is visible)
            latency = response.get("latency_ms") or {}
            for stage_key, stage_name in [
                ("retrieval_ms", "retrieval"),
                ("rerank_ms", "rerank"),
                ("generation_ms", "generation"),
            ]:
                if stage_key in latency:
                    trace.span(
                        name=stage_name,
                        metadata={"latency_ms": latency[stage_key]},
                    )
        except Exception as e:
            logger.warning("langfuse_trace_failed", error=str(e))

    def flush(self) -> None:
        if self.enabled and self.client is not None:
            try:
                self.client.flush()
            except Exception as e:
                logger.warning("langfuse_flush_failed", error=str(e))


__all__ = ["LANGFUSE_AVAILABLE", "LangfuseTracer"]
