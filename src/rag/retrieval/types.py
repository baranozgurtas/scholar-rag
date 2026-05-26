"""Shared data types for the retrieval pipeline.

A single `RetrievedChunk` flows through the entire pipeline:
    Qdrant search → hybrid fusion → reranker → LLM context

Each stage can update `score` and add an entry to `score_breakdown`,
which makes the whole pipeline introspectable (used by /query API
debug output and by Langfuse traces).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievedChunk:
    """One chunk retrieved from the vector store with metadata + scores."""

    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]
    score_breakdown: dict[str, float] = field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")

    @property
    def page(self) -> int:
        return int(self.metadata.get("page", 0))

    @property
    def section(self) -> str:
        return self.metadata.get("section", "other")

    @property
    def paper_title(self) -> str:
        return self.metadata.get("paper_title", self.source)

    def to_citation_tag(self) -> str:
        """Render citation tag used inside LLM prompts.

        Format: [Paper: <title>, p.<page>, §<section>]
        Stable, parseable by the citation_checker downstream.
        """
        return (
            f"[Paper: {self.paper_title} | p.{self.page} | §{self.section}]"
        )


__all__ = ["RetrievedChunk"]
