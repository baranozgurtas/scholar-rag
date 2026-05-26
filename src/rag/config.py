"""Centralized configuration using Pydantic Settings.

All configuration is read from environment variables (or a `.env` file).
Settings are immutable once loaded and are accessed via `get_settings()`
which returns a cached singleton — easy to override in tests via
`get_settings.cache_clear()`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ─── Module-level paths ───────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR_DEFAULT = DATA_DIR / "pdfs"
CACHE_DIR_DEFAULT = PROJECT_ROOT / ".cache"
LOGS_DIR_DEFAULT = PROJECT_ROOT / "logs"
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval" / "results"


class LLMSettings(BaseSettings):
    """Generator + judge LLM configuration (Ollama-served)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    generator_model: str = Field(default="qwen2.5:14b", alias="GENERATOR_MODEL")
    generator_temperature: float = Field(default=0.1, alias="GENERATOR_TEMPERATURE")
    generator_max_tokens: int = Field(default=1024, alias="GENERATOR_MAX_TOKENS")
    judge_model: str = Field(default="qwen2.5:14b", alias="JUDGE_MODEL")
    judge_temperature: float = Field(default=0.0, alias="JUDGE_TEMPERATURE")


class EmbeddingSettings(BaseSettings):
    """Dense embedding + cross-encoder reranker configuration."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    model_name: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL")
    device: Literal["cpu", "cuda", "mps"] = Field(default="mps", alias="EMBEDDING_DEVICE")
    batch_size: int = Field(default=16, alias="EMBEDDING_BATCH_SIZE")
    normalize: bool = Field(default=True, alias="EMBEDDING_NORMALIZE")

    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3", alias="RERANKER_MODEL")
    reranker_device: Literal["cpu", "cuda", "mps"] = Field(default="mps", alias="RERANKER_DEVICE")
    reranker_batch_size: int = Field(default=8, alias="RERANKER_BATCH_SIZE")


class VectorStoreSettings(BaseSettings):
    """Qdrant vector store configuration."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    collection: str = Field(default="research_papers", alias="QDRANT_COLLECTION")
    vector_size: int = Field(default=1024, alias="QDRANT_VECTOR_SIZE")
    prefer_grpc: bool = Field(default=False, alias="QDRANT_PREFER_GRPC")

    @field_validator("api_key", mode="before")
    @classmethod
    def _empty_to_none(cls, v: str | None) -> str | None:
        if isinstance(v, str) and not v.strip():
            return None
        return v


class RetrievalSettings(BaseSettings):
    """Retrieval pipeline configuration: dense + sparse → RRF → rerank."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    dense_top_k: int = Field(default=30, alias="RETRIEVAL_DENSE_TOP_K")
    sparse_top_k: int = Field(default=30, alias="RETRIEVAL_SPARSE_TOP_K")
    rrf_k: int = Field(default=60, alias="RETRIEVAL_RRF_K")
    hybrid_top_k: int = Field(default=20, alias="RETRIEVAL_HYBRID_TOP_K")
    final_top_k: int = Field(default=5, alias="RETRIEVAL_FINAL_TOP_K")
    rerank_score_threshold: float = Field(default=0.0, alias="RETRIEVAL_RERANK_SCORE_THRESHOLD")


class ChunkingSettings(BaseSettings):
    """Document chunking configuration."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    section_aware: bool = Field(default=True, alias="CHUNK_SECTION_AWARE")


class IngestionSettings(BaseSettings):
    """Ingestion pipeline configuration."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    pdf_dir: Path = Field(default=PDF_DIR_DEFAULT, alias="PDF_DIR")
    batch_size: int = Field(default=32, alias="INGESTION_BATCH_SIZE")


class APISettings(BaseSettings):
    """FastAPI server configuration."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    host: str = Field(default="0.0.0.0", alias="API_HOST")
    port: int = Field(default=8000, alias="API_PORT")
    workers: int = Field(default=1, alias="API_WORKERS")
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:8501", "http://localhost:3000"],
        alias="CORS_ORIGINS",
    )


class ObservabilitySettings(BaseSettings):
    """Logging + Langfuse tracing configuration."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", alias="LOG_LEVEL"
    )
    log_format: Literal["json", "console"] = Field(default="json", alias="LOG_FORMAT")

    langfuse_enabled: bool = Field(default=True, alias="LANGFUSE_ENABLED")
    langfuse_host: str = Field(default="http://localhost:3001", alias="LANGFUSE_HOST")
    langfuse_public_key: str | None = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, alias="LANGFUSE_SECRET_KEY")

    @field_validator("langfuse_public_key", "langfuse_secret_key", mode="before")
    @classmethod
    def _empty_to_none(cls, v: str | None) -> str | None:
        if isinstance(v, str) and not v.strip():
            return None
        return v


class CacheSettings(BaseSettings):
    """Caching configuration (DiskCache-based embedding cache)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    cache_dir: Path = Field(default=CACHE_DIR_DEFAULT, alias="CACHE_DIR")
    embedding_cache_enabled: bool = Field(default=True, alias="EMBEDDING_CACHE_ENABLED")
    embedding_cache_size_gb: int = Field(default=2, alias="EMBEDDING_CACHE_SIZE_GB")


class Settings(BaseSettings):
    """Top-level settings composed from domain-specific sections."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="research-rag-assistant", alias="APP_NAME")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    random_seed: int = Field(default=42, alias="RANDOM_SEED")

    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vectorstore: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    api: APISettings = Field(default_factory=APISettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)

    # Paths (not settable via env, derived)
    project_root: Path = Field(default=PROJECT_ROOT, exclude=True)
    logs_dir: Path = Field(default=LOGS_DIR_DEFAULT, exclude=True)
    eval_results_dir: Path = Field(default=EVAL_RESULTS_DIR, exclude=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton Settings instance.

    Use `get_settings.cache_clear()` in tests to force reload after env override.
    """
    return Settings()


__all__ = [
    "PROJECT_ROOT",
    "APISettings",
    "CacheSettings",
    "ChunkingSettings",
    "EmbeddingSettings",
    "IngestionSettings",
    "LLMSettings",
    "ObservabilitySettings",
    "RetrievalSettings",
    "Settings",
    "VectorStoreSettings",
    "get_settings",
]
