"""Structured logging via structlog.

Outputs either JSON (for production / log aggregators) or rich console (for dev),
controlled by `LOG_FORMAT` env var. All log records carry the app name + version
in their context.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.typing import EventDict, Processor

from rag.config import get_settings


def _add_app_context(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Inject app name + version into every log record."""
    settings = get_settings()
    event_dict.setdefault("app", settings.app_name)
    event_dict.setdefault("version", settings.app_version)
    return event_dict


def configure_logging() -> None:
    """Initialize structlog with JSON or console renderer.

    Idempotent: safe to call multiple times (e.g. once at app startup,
    once in test fixtures).
    """
    settings = get_settings()
    log_level = getattr(logging, settings.observability.log_level)

    # Configure stdlib logging (third-party libs log into this)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Shared processors for structlog + stdlib bridge
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _add_app_context,
    ]

    if settings.observability.log_format == "json":
        renderer: Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Bridge stdlib loggers (uvicorn, httpx, etc.) into structlog
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    root_logger = logging.getLogger()
    # Replace default handler to avoid duplicate formatting
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # Quiet noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "qdrant_client.http"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound to the given name (default: caller module)."""
    return structlog.get_logger(name)


__all__ = ["configure_logging", "get_logger"]
