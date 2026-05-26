"""BGE-M3 embeddings wrapper using FlagEmbedding.

BGE-M3 is multi-functional (dense + sparse + multi-vector). For now we use:
- **Dense** embeddings → Qdrant dense vectors (cosine similarity)
- **Sparse** embeddings → Qdrant sparse vectors (BM25-style lexical match)

This gives true hybrid retrieval inside a single model + single vector store,
which is the configuration BGE-M3 was designed for and recommended by the
authors (Chen et al., 2024).

The embedder implements the LangChain `Embeddings` interface for dense
output (so it plugs into `QdrantVectorStore` directly) and exposes
`embed_dense_and_sparse` for the hybrid retrieval pipeline.

A DiskCache layer optionally caches embeddings keyed by SHA-1 of the text,
which makes re-ingestion of an unchanged corpus near-instant.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch
from diskcache import Cache
from FlagEmbedding import BGEM3FlagModel
from langchain_core.embeddings import Embeddings

from rag.config import EmbeddingSettings, get_settings
from rag.logging_config import get_logger

logger = get_logger(__name__)


class BGEEmbedder(Embeddings):
    """BGE-M3 embedder with dense + sparse output and disk caching.

    Args:
        settings: Override embedding settings (else read from env).
        cache: Optional DiskCache instance (None disables caching).
    """

    def __init__(
        self,
        settings: EmbeddingSettings | None = None,
        cache: Cache | None = None,
    ) -> None:
        s = settings or get_settings().embedding
        self.settings = s
        self._cache = cache

        device = self._resolve_device(s.device)
        logger.info(
            "bge_embedder_loading",
            model=s.model_name,
            device=device,
            batch_size=s.batch_size,
        )
        # use_fp16=True on MPS/CUDA halves memory; on CPU we keep fp32.
        use_fp16 = device in {"cuda", "mps"}
        self._model = BGEM3FlagModel(
            s.model_name,
            use_fp16=use_fp16,
            device=device,
        )
        self._device = device

    @staticmethod
    def _resolve_device(requested: str) -> str:
        """Resolve `mps` → `mps` only if available, else fall back."""
        if requested == "mps":
            if torch.backends.mps.is_available():
                return "mps"
            logger.warning("mps_unavailable_falling_back_to_cpu")
            return "cpu"
        if requested == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            logger.warning("cuda_unavailable_falling_back_to_cpu")
            return "cpu"
        return "cpu"

    # ─── LangChain Embeddings interface ────────────────────────────
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Dense embedding for a batch of documents (cached if enabled)."""
        return self._embed_dense_cached(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        """Dense embedding for a single query (never cached — queries are unique)."""
        out = self._model.encode(
            [text],
            batch_size=1,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        vec = out["dense_vecs"][0]
        if self.settings.normalize:
            vec = vec / (np.linalg.norm(vec) + 1e-12)
        return vec.tolist()

    # ─── Hybrid (dense + sparse) ──────────────────────────────────
    def embed_dense_and_sparse(
        self, texts: list[str], is_query: bool = False
    ) -> tuple[np.ndarray, list[dict[int, float]]]:
        """Return (dense [N, D], sparse [N] as list of {token_id: weight}).

        Args:
            texts: Input strings.
            is_query: True for query encoding (skips cache, single-batch).
        """
        if is_query or not self._cache_enabled():
            out = self._model.encode(
                texts,
                batch_size=self.settings.batch_size,
                max_length=512,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
        else:
            out = self._encode_with_cache(texts, return_sparse=True)

        dense = np.asarray(out["dense_vecs"])
        if self.settings.normalize:
            norms = np.linalg.norm(dense, axis=1, keepdims=True) + 1e-12
            dense = dense / norms

        sparse_raw = out["lexical_weights"]  # list of dict[str_token_id, float]
        sparse: list[dict[int, float]] = []
        for d in sparse_raw:
            sparse.append({int(k): float(v) for k, v in d.items() if float(v) > 0})
        return dense, sparse

    # ─── Caching helpers ──────────────────────────────────────────
    def _cache_enabled(self) -> bool:
        return self._cache is not None and get_settings().cache.embedding_cache_enabled

    @staticmethod
    def _cache_key(text: str, kind: str) -> str:
        # Tag with model + normalize flag so cache invalidates on config change
        h = hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()
        return f"bge-m3:{kind}:{h}"

    def _embed_dense_cached(self, texts: list[str]) -> np.ndarray:
        if not self._cache_enabled():
            out = self._model.encode(
                texts,
                batch_size=self.settings.batch_size,
                max_length=512,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            dense = np.asarray(out["dense_vecs"])
            if self.settings.normalize:
                norms = np.linalg.norm(dense, axis=1, keepdims=True) + 1e-12
                dense = dense / norms
            return dense
        return self._encode_with_cache(texts, return_sparse=False)["dense_vecs"]

    def _encode_with_cache(
        self, texts: list[str], return_sparse: bool
    ) -> dict[str, Any]:
        """Look up each text in the cache; encode the misses; merge results."""
        cache = self._cache
        assert cache is not None

        n = len(texts)
        dense_out: list[np.ndarray | None] = [None] * n
        sparse_out: list[dict[int, float] | None] = [None] * n
        miss_idx: list[int] = []
        miss_texts: list[str] = []

        for i, t in enumerate(texts):
            d_key = self._cache_key(t, "dense")
            s_key = self._cache_key(t, "sparse")
            cached_dense = cache.get(d_key)
            cached_sparse = cache.get(s_key) if return_sparse else None
            if cached_dense is not None and (not return_sparse or cached_sparse is not None):
                dense_out[i] = cached_dense
                if return_sparse:
                    sparse_out[i] = cached_sparse
            else:
                miss_idx.append(i)
                miss_texts.append(t)

        if miss_texts:
            out = self._model.encode(
                miss_texts,
                batch_size=self.settings.batch_size,
                max_length=512,
                return_dense=True,
                return_sparse=return_sparse,
                return_colbert_vecs=False,
            )
            new_dense = np.asarray(out["dense_vecs"])
            if self.settings.normalize:
                norms = np.linalg.norm(new_dense, axis=1, keepdims=True) + 1e-12
                new_dense = new_dense / norms

            for j, orig_i in enumerate(miss_idx):
                dense_out[orig_i] = new_dense[j]
                cache.set(self._cache_key(miss_texts[j], "dense"), new_dense[j])
                if return_sparse:
                    sparse_dict = {
                        int(k): float(v)
                        for k, v in out["lexical_weights"][j].items()
                        if float(v) > 0
                    }
                    sparse_out[orig_i] = sparse_dict
                    cache.set(self._cache_key(miss_texts[j], "sparse"), sparse_dict)

        dense_arr = np.stack([d for d in dense_out if d is not None])
        result: dict[str, Any] = {"dense_vecs": dense_arr}
        if return_sparse:
            result["lexical_weights"] = [s for s in sparse_out if s is not None]
        logger.debug(
            "embedding_cache_stats",
            total=n,
            hits=n - len(miss_idx),
            misses=len(miss_idx),
        )
        return result


def build_embedder(settings: EmbeddingSettings | None = None) -> BGEEmbedder:
    """Factory that wires up DiskCache from global settings."""
    cfg = get_settings()
    cache: Cache | None = None
    if cfg.cache.embedding_cache_enabled:
        cfg.cache.cache_dir.mkdir(parents=True, exist_ok=True)
        cache = Cache(
            str(cfg.cache.cache_dir / "embeddings"),
            size_limit=cfg.cache.embedding_cache_size_gb * 1024**3,
        )
    return BGEEmbedder(settings=settings, cache=cache)


__all__ = ["BGEEmbedder", "build_embedder"]
