"""Ollama-served Qwen2.5 LLM wrapper.

Provides a thin factory around `langchain_ollama.ChatOllama` so the same
configuration flows through both the generator and the judge (for RAGAS).
The judge uses temperature=0 + a different system prompt — same model,
different role.
"""

from __future__ import annotations

from langchain_ollama import ChatOllama

from rag.config import LLMSettings, get_settings
from rag.logging_config import get_logger

logger = get_logger(__name__)


def build_generator_llm(settings: LLMSettings | None = None) -> ChatOllama:
    """Construct the generator LLM (high-quality RAG answers)."""
    s = settings or get_settings().llm
    logger.info(
        "generator_llm_init",
        model=s.generator_model,
        base_url=s.ollama_base_url,
        temperature=s.generator_temperature,
    )
    return ChatOllama(
        base_url=s.ollama_base_url,
        model=s.generator_model,
        temperature=s.generator_temperature,
        num_predict=s.generator_max_tokens,
        # Disable streaming at LLM level; we stream at API layer if needed.
        # Keep top_p at default; temperature alone controls determinism here.
    )


def build_judge_llm(settings: LLMSettings | None = None) -> ChatOllama:
    """Construct the judge LLM (RAGAS evaluation; deterministic)."""
    s = settings or get_settings().llm
    logger.info(
        "judge_llm_init",
        model=s.judge_model,
        base_url=s.ollama_base_url,
        temperature=s.judge_temperature,
    )
    return ChatOllama(
        base_url=s.ollama_base_url,
        model=s.judge_model,
        temperature=s.judge_temperature,
        num_predict=512,  # judge outputs are short
    )


__all__ = ["build_generator_llm", "build_judge_llm"]
