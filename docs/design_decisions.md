# Design Decisions

This document is the "why" companion to the code: every non-obvious choice
made in this project, with the trade-offs considered and the alternatives
rejected. It exists primarily so any reader — recruiter, interviewer, or
future me — can audit the engineering reasoning without spelunking through
git blame.

## 1. Vector store: Qdrant (not Chroma, pgvector, Pinecone, Weaviate)

**Choice**: Qdrant 1.12, self-hosted in Docker.

**Rejected alternatives**:
- *Chroma*: fine for prototypes but I was reaching into `vectorstore._collection`
  private APIs for stats and deletion in the original code — a signal that the
  abstraction was wrong for production needs. No first-class sparse vector support.
- *pgvector*: solid if Postgres is already in your stack. Lacks ergonomic
  hybrid search; sparse vectors require a parallel column or extension.
- *Pinecone*: managed-only, defeats the OSS / self-hosted goal and adds
  data-egress concerns for EU customers.
- *Weaviate*: credible competitor. Qdrant won on (a) simpler operational
  footprint (single Rust binary), (b) cleaner sparse-vector ergonomics,
  (c) wider European enterprise adoption (Bosch, Deloitte, BMW).

**Why Qdrant won**: native named vectors let us store **dense + sparse in
the same collection, indexed atomically** — one upsert writes both, one
delete removes both. Payload filtering is first-class and indexed. Rust
core means consistent latency under load. Docker image is ~50 MB.

## 2. Embeddings + reranker: BGE-M3 + BGE-reranker-v2-m3 (not OpenAI / Cohere)

**Choice**: `BAAI/bge-m3` for embeddings (dense + sparse + multi-vector
output from a single model), `BAAI/bge-reranker-v2-m3` for cross-encoder
reranking.

**Rejected alternatives**:
- *OpenAI `text-embedding-3-small`*: strong, but requires API key and
  ships data to OpenAI. Disqualifying for GDPR-sensitive enterprise
  deployments — and an unnecessary dependency for a project meant to
  demonstrate OSS-first deployability.
- *Cohere Rerank*: top-tier reranker but again API-only.
- *`nomic-embed-text`*: good, but BGE-M3 is the only model that produces
  dense **and** sparse weights from a single pass, which is exactly
  the configuration our Qdrant collection is built around.

**Why BGE-M3 won**: built for hybrid. Sparse weights are learned (not
plain BM25 IDF), so the lexical leg is itself ML-tuned. MTEB top-tier.
1024 dense dim is a good RAM/quality trade-off. Apple Silicon MPS
acceleration works out of the box via PyTorch.

## 3. Retrieval: hybrid (dense + sparse) with RRF, then cross-encoder rerank

**Why hybrid over dense-only**: academic papers contain rare proper
nouns (model names like "LLaMA-2", dataset names like "TriviaQA",
formula symbols, citation keys) that dense embeddings smooth over. A
dense-only system can rank a chunk discussing "transformers" higher
than one containing the literal token "BERT" when queried for "BERT".
Hybrid catches both.

**Why RRF over weighted score sum**: cosine scores live in `[-1, 1]`,
BM25/sparse scores are unbounded. Summing them naïvely is meaningless;
normalizing them introduces hyperparameters. RRF is rank-based, has one
parameter (`k=60`), and is provably robust to score-scale differences.

**Why cross-encoder rerank on top**: bi-encoder retrieval optimizes
*recall in top-50* — it gets the right chunk into the candidate pool.
The top-5 that the LLM actually sees needs *precision*. A cross-encoder
jointly encodes (query, chunk) pairs and captures interaction signals
bi-encoders cannot. Cost: ~150ms on 20 candidates with BGE-reranker-v2-m3
on Apple Silicon — well below LLM generation time, so a free win.

## 4. Chunking: section-aware with recursive fallback

**Why section-aware**: academic papers have hard topical boundaries
(Abstract / Methods / Results / Discussion). A chunk that crosses
section boundaries blends topics and hurts both retrieval precision
(noisy embedding) and answer faithfulness (LLM may stitch claims
across unrelated sections). The original RAFT paper (Zhang et al.,
2024) and Self-RAG explicitly recommend this.

**Why 1000/200**: empirically tuned. 500 fragments methodology sections;
1500 mixes topics within a single section. 1000 with 200 overlap is the
local optimum on academic PDFs in this corpus. A v2 enhancement would
be semantic chunking, but the gain is marginal at 3x the ingestion cost
on this corpus size.

## 5. Generator + judge: Qwen2.5:7b via Ollama (not GPT-4 / Claude)

**Why local OSS for both roles**:
- **GDPR / data-egress**: a recurring concern for EU companies in
  fintech, healthcare, government. Demonstrating that the full system
  runs without sending data to any external API is a strong signal.
- **Reproducibility**: the eval results commit-pinned with `make eval`
  reproduce on any developer's Mac. No "API drift" between runs.
- **Judge-of-self conflict-of-interest**: using GPT-4 to grade GPT-4
  outputs in RAGAS is a known eval bias. Local Qwen judging local Qwen
  is at least transparent about its limits (and the model card is
  cited in `docs/architecture.md`).

**Why Qwen2.5:7b specifically over Llama 3.1, Mistral, DeepSeek**:
strongest open-source 14B class as of late 2024 on Chinese-English
multilingual + reasoning + instruction following benchmarks. Apple
Metal acceleration via Ollama works smoothly. 14B fits comfortably in
24GB unified memory (M1/M2/M3 Pro/Max).

**Why same model for generator and judge**: it's a deliberately
controlled bias. Differences between configs in ablation are still
meaningful — both configs are evaluated by the same judge, so absolute
faithfulness numbers may be biased but **deltas** are not. A multi-judge
setup is a v2 improvement.

## 6. Eval: hybrid synthetic + manual + adversarial

**Why not pure synthetic**: LLM-generated questions are LLM-friendly.
Faithfulness measured against LLM-generated ground truth circles back
on itself. Pure synthetic is the most common RAG eval failure mode —
the numbers look great because the test was built by the system being
tested.

**Why not pure manual**: too small (we can credibly produce ~15
high-quality manually). Statistical power suffers; the eval becomes
a vibe check.

**Why include 5 adversarial**: hallucination handling is the failure
mode that makes RAG systems dangerous in production. The eval should
measure it explicitly. Adversarial questions force the abstention path
and provide a hard-to-fake metric: "5/5 correctly abstained" is a
single number that defends an entire failure mode.

## 7. Serving: FastAPI as the source of truth, Streamlit as a thin client

**Why not Streamlit-only**: the original codebase had Streamlit calling
the retrieval / LLM stack directly. That's fine for a demo but
- couples UI cold-start time to torch / transformers imports (slow),
- makes the same logic non-callable from curl, the eval harness, or
  any future integration,
- breaks the deployment story (Streamlit servers are not API servers).

**Why FastAPI**: standard, async-native, generates OpenAPI schema for
free, plugs into Prometheus + Langfuse + structlog cleanly, runs in a
~200MB container. The Streamlit UI now calls `POST /query` over HTTP —
one source of truth.

## 8. Observability: Langfuse + structured JSON logs + Prometheus

**Why all three**: they answer different questions.
- *structlog JSON logs*: per-event, indexed by Datadog / Loki / etc.
- *Prometheus metrics*: aggregate rates and histograms for Grafana
  dashboards and alerting (e.g., alert on `rag_citation_fabrication_total`
  rate).
- *Langfuse traces*: per-query waterfall view of retrieval → rerank →
  generation, with prompt versioning and human-in-the-loop quality
  rating. Indispensable for prompt iteration.

All three are wired such that disabling any one is a no-op (Langfuse
gracefully degrades if keys are missing, Prometheus is just an
endpoint, logs default to console if `LOG_FORMAT=console`).

## 9. Apple Silicon: containers for stores, host for compute

**Why Ollama on the host, not in a container**: Docker Desktop for Mac
does not expose Metal GPU acceleration to containers. Running Ollama
in a container would force CPU inference (~10x slower). On Mac the
idiomatic pattern is: stores (Qdrant, Langfuse, Postgres) in Docker,
compute (Ollama, the API itself during dev) on the host. The API
container reaches the host's Ollama via `host.docker.internal:11434`.

**Why this still ships to production**: the `Dockerfile` is ARM64/AMD64
multi-arch and assumes Ollama is reachable at `OLLAMA_BASE_URL`. In
production that URL points to a separate Ollama instance (or any
OpenAI-compatible endpoint) — the deployment topology shifts but the
code does not.

## 10. License: Apache 2.0, not MIT

Apache 2.0 includes a patent grant that MIT lacks. For an OSS project
intended to be re-used by enterprises (this is a portfolio piece
demonstrating I would write production code), Apache is the safer
default. The original repo was MIT; this rebuild upgrades.
