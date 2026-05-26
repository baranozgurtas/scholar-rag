# Architecture

## High-level view

```mermaid
flowchart LR
    User([User]) -->|HTTP| API[FastAPI<br/>/query, /ingest, /health, /stats, /metrics]
    UI[Streamlit UI<br/>app.py] -.->|HTTP| API
    EVAL[Eval harness<br/>ablation + RAGAS] -.->|in-process| RAGChain

    API --> RAGChain
    RAGChain --> HR[Hybrid Retriever]
    HR --> Dense[Dense Retriever<br/>BGE-M3]
    HR --> Sparse[Sparse Retriever<br/>BGE-M3 lexical]
    Dense --> Qdrant[(Qdrant<br/>named vectors<br/>dense + sparse)]
    Sparse --> Qdrant
    HR --> RRF[RRF Fusion<br/>k=60]
    RRF --> RR[Cross-Encoder Reranker<br/>BGE-reranker-v2-m3]
    RR --> LLM[Qwen2.5:7b<br/>via Ollama]
    LLM --> Guards[Citation Guard<br/>regex + fuzzy match]
    Guards --> API

    API -->|traces| LF[(Langfuse)]
    API -->|metrics| Prom[/metrics endpoint/]
    API -->|tokens| Ledger[(JSONL ledger)]

    classDef store fill:#1e3a5f,stroke:#4a90e2,color:#fff
    classDef model fill:#5f1e3a,stroke:#e24a90,color:#fff
    classDef api fill:#3a5f1e,stroke:#90e24a,color:#fff
    class Qdrant,LF,Ledger store
    class Dense,Sparse,LLM,RR,RRF model
    class API,UI,EVAL,RAGChain,HR,Guards api
```

## Request lifecycle: `POST /query`

```mermaid
sequenceDiagram
    participant U as Client
    participant A as FastAPI
    participant R as RAGChain
    participant H as HybridRetriever
    participant Q as Qdrant
    participant X as Reranker
    participant L as Qwen2.5
    participant G as Citation Guard

    U->>A: POST /query {"question": "..."}
    A->>R: chain.answer(question)
    R->>H: retrieve(question, top_k=20)
    par parallel retrieval
        H->>Q: search(dense, k=30)
        H->>Q: search(sparse, k=30)
    end
    H->>H: RRF fuse, top-20
    H-->>R: 20 candidates
    R->>X: rerank(question, candidates, top_k=5)
    X-->>R: 5 chunks with reranker scores
    alt no chunks above threshold
        R-->>A: abstain (no LLM call)
    else
        R->>L: generate(prompt + context)
        L-->>R: answer with citations
        R->>G: validate(answer, allowed_tags)
        G-->>R: CitationCheckResult
    end
    R-->>A: RAGResponse (answer, citations, breakdown, latencies)
    A->>A: emit metrics + Langfuse trace + token ledger
    A-->>U: JSON response
```

## Ingestion lifecycle

```mermaid
flowchart LR
    PDF[PDFs<br/>data/pdfs/*.pdf] --> L[PyMuPDF Loader<br/>section detection]
    L --> C[Section-Aware Chunker<br/>1000/200 + recursive]
    C --> E[BGE-M3 Embedder<br/>dense + sparse]
    E --> Cache[(DiskCache<br/>SHA-1 keyed)]
    E --> U[QdrantStore.upsert_chunks<br/>UUID5 deterministic IDs]
    U --> Q[(Qdrant Collection<br/>HNSW M=16<br/>payload indexes)]
```

Key invariants:
- `file_hash` (MD5) attached to every chunk → identical PDFs are no-ops
- `chunk_id` = `{file_hash[:8]}_{seq:04d}` is deterministic → re-ingest
  is a clean overwrite, not duplicated
- Section boundaries are hard chunk boundaries → no chunk spans
  Abstract + Introduction

## Eval pipeline

```mermaid
flowchart LR
    Papers[15 arXiv papers] -->|make download-papers| PDFs
    PDFs -->|make ingest| Q[(Qdrant)]
    PDFs -->|make generate-questions| QS[questions.jsonl<br/>30 synth + 15 manual + 5 adversarial]

    QS --> A[Ablation Runner<br/>4 configs]
    QS --> R[RAGAS Runner<br/>faithfulness, relevancy,<br/>precision, recall]
    Q --> A
    Q --> R

    A --> AR[ablation_results.md]
    R --> RR[ragas_summary.md]
    AR --> UM[update_readme_metrics.py]
    RR --> UM
    UM --> README[README.md<br/>with real numbers]
```

## Data flow at the storage layer

Each Qdrant point has:

| Field | Type | Purpose |
|---|---|---|
| `id` | UUID5 from `chunk_id` | Stable identity for re-upsert |
| `vector["dense"]` | `float32[1024]` | BGE-M3 dense, L2-normalized |
| `vector["sparse"]` | sparse | BGE-M3 lexical weights |
| `payload.text` | str | The chunk text |
| `payload.source` | str | PDF filename (e.g., `bge-m3-chen-2024.pdf`) |
| `payload.file_hash` | keyword-indexed | Dedup + cascade delete |
| `payload.paper_title` | str | For citation tags |
| `payload.page` | int | Page number |
| `payload.section` | keyword-indexed | Section name (abstract/methods/etc.) |
| `payload.chunk_idx` | int-indexed | Position within paper |
| `payload.chunk_id` | str | Stable string ID `<hash>_<seq>` |

Payload indexes let us cheaply scroll a single paper, count chunks by
section, or filter retrieval to specific clusters (`section=results`)
without re-scanning the whole vector index.

## Concurrency model

- FastAPI routes are `async`. The blocking work (LLM generation,
  embedding, Qdrant search) is offloaded with `asyncio.to_thread`.
  The event loop stays responsive — concurrent `/query` requests
  are interleaved through the BGE / Ollama backends.
- BGE embedder + reranker hold their own internal threading via
  PyTorch. We never call them concurrently from multiple threads
  (FlagEmbedding is not fully reentrant on MPS).
- Qdrant client is thread-safe; multiple goroutines share one client.
- The `TokenLedger` and `LangfuseTracer` are thread-safe by design
  (lock for the ledger, queue for Langfuse).

## Configuration cascade

All runtime behavior flows from environment variables → Pydantic
Settings classes → injected into the code:

```
.env / process env
        │
        ▼
Settings (rag.config.Settings)
        │
        ├── LLMSettings           → llm.py (Ollama URL, model)
        ├── EmbeddingSettings     → bge_embedder.py, reranker.py
        ├── VectorStoreSettings   → qdrant_store.py
        ├── RetrievalSettings     → hybrid_retriever.py, rag_chain.py
        ├── ChunkingSettings      → chunker.py
        ├── IngestionSettings     → pipeline.py
        ├── APISettings           → api/main.py
        ├── ObservabilitySettings → metrics.py, langfuse_tracer.py
        └── CacheSettings         → bge_embedder.py disk cache
```

`get_settings()` is `@lru_cache`-d so a single load per process; tests
override with `monkeypatch.setenv(...)` + `get_settings.cache_clear()`.
