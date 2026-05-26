"""Qdrant vector store wrapper supporting native hybrid (dense + sparse).

This module owns the Qdrant collection schema. The collection has:
- One **dense** vector named `dense` (cosine, BGE-M3 1024-d)
- One **sparse** vector named `sparse` (BGE-M3 lexical weights)
- Rich **payload** for every point: chunk text, source, page, section, etc.

By using Qdrant's named-vectors feature we avoid running a separate BM25
index alongside the vector DB — one store, two retrieval modes, atomic
upserts. This is the configuration BGE-M3 was designed for.
"""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient, models

from rag.config import VectorStoreSettings, get_settings
from rag.embeddings.bge_embedder import BGEEmbedder
from rag.ingestion.chunker import Chunk
from rag.logging_config import get_logger

logger = get_logger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


class QdrantStore:
    """Qdrant collection manager for hybrid retrieval."""

    def __init__(
        self,
        embedder: BGEEmbedder,
        settings: VectorStoreSettings | None = None,
    ) -> None:
        self.settings = settings or get_settings().vectorstore
        self.embedder = embedder
        self.client = QdrantClient(
            url=self.settings.url,
            api_key=self.settings.api_key,
            prefer_grpc=self.settings.prefer_grpc,
            timeout=60,
        )
        self.collection_name = self.settings.collection

    # ─── Collection lifecycle ──────────────────────────────────────
    def ensure_collection(self, recreate: bool = False) -> None:
        """Create the collection if it doesn't exist (or recreate when asked)."""
        exists = self.client.collection_exists(self.collection_name)
        if exists and recreate:
            logger.warning("recreating_collection", name=self.collection_name)
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    DENSE_VECTOR_NAME: models.VectorParams(
                        size=self.settings.vector_size,
                        distance=models.Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False),
                    ),
                },
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128),
            )
            # Useful payload indexes for filtering
            for field, schema in [
                ("file_hash", models.PayloadSchemaType.KEYWORD),
                ("source", models.PayloadSchemaType.KEYWORD),
                ("section", models.PayloadSchemaType.KEYWORD),
                ("chunk_idx", models.PayloadSchemaType.INTEGER),
            ]:
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field,
                        field_schema=schema,
                    )
                except Exception as e:
                    logger.debug("payload_index_skip", field=field, reason=str(e))
            logger.info(
                "collection_created",
                name=self.collection_name,
                vector_size=self.settings.vector_size,
            )
        else:
            logger.info("collection_exists", name=self.collection_name)

    def collection_stats(self) -> dict[str, Any]:
        """Return point count and distinct file hashes in the collection."""
        info = self.client.get_collection(self.collection_name)
        # Sample distinct sources via scroll (cheap on small collections)
        sources: set[str] = set()
        file_hashes: set[str] = set()
        offset = None
        scanned = 0
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                payload = p.payload or {}
                if "source" in payload:
                    sources.add(payload["source"])
                if "file_hash" in payload:
                    file_hashes.add(payload["file_hash"])
            scanned += len(points)
            if offset is None or scanned > 100_000:
                break
        return {
            "collection": self.collection_name,
            "points_count": info.points_count or 0,
            "unique_sources": len(sources),
            "unique_file_hashes": len(file_hashes),
            "sources_sample": sorted(sources)[:20],
        }

    def file_hash_exists(self, file_hash: str) -> bool:
        """Check whether a file hash is already indexed (dedup helper)."""
        result, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="file_hash", match=models.MatchValue(value=file_hash)
                    )
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return len(result) > 0

    def delete_by_file_hash(self, file_hash: str) -> int:
        """Delete all points for a given file hash; return deleted count."""
        before = self.collection_stats()["points_count"]
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_hash", match=models.MatchValue(value=file_hash)
                        )
                    ]
                )
            ),
        )
        after = self.collection_stats()["points_count"]
        deleted = before - after
        logger.info("deleted_by_file_hash", file_hash=file_hash[:8], deleted=deleted)
        return deleted

    # ─── Upsert ────────────────────────────────────────────────────
    def upsert_chunks(self, chunks: list[Chunk], skip_existing: bool = True) -> int:
        """Embed (dense + sparse) and upsert chunks into Qdrant.

        Args:
            chunks: Chunks to add.
            skip_existing: If True, skip chunks whose file_hash is already indexed.

        Returns:
            Number of points actually upserted.
        """
        if not chunks:
            return 0

        if skip_existing:
            existing_hashes = {
                c.metadata["file_hash"]
                for c in chunks
                if self.file_hash_exists(c.metadata["file_hash"])
            }
            if existing_hashes:
                chunks = [c for c in chunks if c.metadata["file_hash"] not in existing_hashes]
                logger.info(
                    "skipped_existing_files",
                    n_skipped_hashes=len(existing_hashes),
                    chunks_remaining=len(chunks),
                )
            if not chunks:
                return 0

        texts = [c.text for c in chunks]
        dense, sparse = self.embedder.embed_dense_and_sparse(texts, is_query=False)

        points = []
        for chunk, dvec, svec in zip(chunks, dense, sparse, strict=True):
            point_id = self._deterministic_id(chunk.metadata["chunk_id"])
            sparse_vec = models.SparseVector(
                indices=list(svec.keys()),
                values=list(svec.values()),
            )
            payload = {**chunk.metadata, "text": chunk.text}
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        DENSE_VECTOR_NAME: dvec.tolist(),
                        SPARSE_VECTOR_NAME: sparse_vec,
                    },
                    payload=payload,
                )
            )

        # Batch upsert in groups of 64 for safety on slow networks
        batch_size = 64
        upserted = 0
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=self.collection_name, points=batch, wait=True)
            upserted += len(batch)
        logger.info("upserted_chunks", count=upserted)
        return upserted

    # ─── Search primitives (used by retrievers) ───────────────────
    def search_dense(
        self,
        query_vector: list[float],
        top_k: int,
        score_threshold: float | None = None,
    ) -> list[models.ScoredPoint]:
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=models.NamedVector(name=DENSE_VECTOR_NAME, vector=query_vector),
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )

    def search_sparse(
        self,
        query_sparse: dict[int, float],
        top_k: int,
    ) -> list[models.ScoredPoint]:
        sparse_vec = models.SparseVector(
            indices=list(query_sparse.keys()),
            values=list(query_sparse.values()),
        )
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=models.NamedSparseVector(name=SPARSE_VECTOR_NAME, vector=sparse_vec),
            limit=top_k,
            with_payload=True,
        )

    def scroll_all_payloads(self, batch: int = 256) -> list[dict[str, Any]]:
        """Scroll the entire collection; payload-only. Used by BM25 retriever to
        seed an in-memory fallback corpus if needed during eval."""
        out: list[dict[str, Any]] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=batch,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                if p.payload:
                    out.append({**p.payload, "id": str(p.id)})
            if offset is None:
                break
        return out

    @staticmethod
    def _deterministic_id(chunk_id: str) -> str:
        """Generate a stable UUID5 from chunk_id so re-upserts overwrite cleanly."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


__all__ = ["DENSE_VECTOR_NAME", "SPARSE_VECTOR_NAME", "QdrantStore"]
