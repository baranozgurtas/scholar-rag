"""Observability primitives: structured logs (in `rag.logging_config`),
Prometheus metrics, token counting, and Langfuse tracing.

Submodules (metrics, langfuse_tracer) are imported lazily via dotted
access — keeps heavy deps (prometheus_client, langfuse) out of the
import path until actually used.
"""

from rag.observability.token_counter import TokenLedger, count_tokens

__all__ = ["TokenLedger", "count_tokens"]
