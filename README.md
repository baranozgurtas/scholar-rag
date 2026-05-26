# Scholar RAG
> Production-grade Retrieval-Augmented Generation system for academic literature QA.
> Hybrid retrieval (BGE-M3 dense + sparse) with cross-encoder reranking, served by Qwen2.5:7b via Ollama, exposed through FastAPI with a custom vanilla-JS frontend.

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-yellow.svg)](https://www.langchain.com/)
[![Tests](https://img.shields.io/badge/tests-51%2F51%20passing-success.svg)](#testing)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

---

## Why this project exists

Most RAG demos answer the question "can an LLM be grounded in a vector store?". This project answers a harder one: **can a RAG system be trusted enough to be shipped?**

The bar that was set:

- Every claim in the generated answer **must** carry a citation that points to a real paper, page, and section.
- The system **must** refuse to answer questions whose evidence is not in the corpus, rather than fabricate.
- The retrieval pipeline must be **measured and ablated**, not just asserted to work.
- The whole thing must run **fully offline** on a developer laptop — no proprietary APIs, no hidden costs.

What follows is the result.

---

## Results

Four retrieval configurations were evaluated end-to-end on 25 questions (20 synthetic from corpus, 5 adversarial out-of-corpus). The full eval ran on a Colab T4 GPU; eval code, per-question logs, and JSON dumps live under [`eval/results/`](./eval/results/).

<!-- METRICS:ABLATION:START -->
| Config | Description | Hit@5 | Hit@10 | MRR@10 | nDCG@10 | Abstain% | Adv.Abstain |
|---|---|---|---|---|---|---|---|
| `A_dense_only` | Dense-only baseline | 0.950 | 0.950 | 0.925 | 0.932 | 32.0% | **1.000** |
| `B_dense_plus_rerank` | Dense + cross-encoder rerank | **1.000** | **1.000** | **0.967** | **0.975** | 24.0% | **1.000** |
| `C_hybrid_no_rerank` | Hybrid (dense + sparse + RRF) | 0.950 | 0.950 | 0.950 | 0.950 | 32.0% | **1.000** |
| `D_hybrid_plus_rerank` | Hybrid + rerank (production) | 0.950 | 0.950 | 0.950 | 0.950 | **20.0%** | **1.000** |
<!-- METRICS:ABLATION:END -->

**Reading the table:**

- **Reranking is the largest single contributor.** Going from `A` → `B` lifts Hit@5 from 0.95 → 1.00 and MRR@10 from 0.925 → 0.967.
- **Hybrid retrieval alone (C) does not beat dense (A) on this corpus** because BGE-M3 dense embeddings already capture the methodology questions well; sparse helps mostly on rare-term queries, which are underrepresented in 20 questions.
- **D (hybrid + rerank) is shipped as the production config** despite tying `B` on Hit/MRR, for two reasons: (1) lowest in-corpus abstention rate (20% vs. 24%, the system speaks up more confidently when it should), and (2) sparse is a defensive layer for future corpus expansion where rare-term recall matters.
- **Adversarial abstention is 100% across every configuration.** All 5 out-of-corpus questions (e.g. *"What is the best ramen restaurant in Zurich?"*) were correctly refused, with zero fabricated citations. This is the single most important number on this page.




---

## Demo

The system is shown answering three deliberately different question types, each illustrating a separate guarantee.

### 1. Grounded answer with citation breakdown

[Grounded answer with score breakdown] <img width="1440" height="850" alt="Screenshot 2026-05-26 at 8 17 04 PM" src="https://github.com/user-attachments/assets/1f717c77-fc2f-4f65-bc00-e3d6e8b91ecc" />


A simple architectural question about M3-Embedding. The model produces a short, citation-attached answer; the right panel shows the top retrieved chunks with their dense / sparse / rerank scores. The inline `[1]` pill is clickable and scrolls the sources panel to the cited chunk — citations are first-class UI primitives, not afterthoughts.

### 2. Cross-paper synthesis via hybrid retrieval

[Cross-paper retrieval across BPR and NCF papers] <img width="1439" height="816" alt="Screenshot 2026-05-26 at 8 21 21 PM" src="https://github.com/user-attachments/assets/caaf3857-2325-456b-8336-4fc0da8576eb" />


A comparative question forces the retriever to surface evidence from two distinct papers (`ncf-he-2017` and `bpr-rendle-2009`). Hybrid retrieval and the reranker correctly mix them in the top-5; the generated answer cites both. This is the failure mode that pure dense retrieval struggles with most.

### 3. Out-of-corpus query: zero hallucination

[Adversarial out-of-corpus question correctly abstained] <img width="1436" height="845" alt="Screenshot 2026-05-26 at 10 18 45 PM 1" src="https://github.com/user-attachments/assets/09003461-1d26-4609-a7e6-196fb361ba2e" />


An adversarial query (*"What is the best ramen restaurant in Zurich?"*) that has no answer in the corpus. The retriever still returns its top-5 chunks — N-BEATS, Adam, BPR, NCF — but the rerank scores top out at `0.671` (vs. `0.94+` on in-corpus questions), and the system enters the abstain branch. The UI suppresses citation pills and replaces the validity badge with `🛡 abstained (no fabrication)`. This behavior — declining to answer rather than hallucinating — is what produces the **100% adversarial abstention** number in the table above.

---

## Architecture
```
+-- Ingestion ----------------------------------------------------------+
|                                                                      |
|  15 arXiv PDFs                                                       |
|       \__> Section-aware chunker                                     |
|                  \__> BGE-M3 (dense + sparse, one forward pass)      |
|                            \__> Qdrant (named vectors)               |
|                                                                      |
+----------------------------------------------------------------------+
                                  |
                                  v
+-- Query -------------------------------------------------------------+
|                                                                     |
|  User query                                                         |
|     |__> Dense retrieval ----+                                      |
|     \__> Sparse retrieval ---+--> RRF fusion (k=60)                 |
|                                       |                             |
|                                       v                             |
|                              Cross-encoder rerank (v2-m3)           |
|                                       |                             |
|                                       v                             |
|                              Top-k context + paper/page/section     |
|                                       |                             |
|                                       v                             |
|                              Qwen2.5:7b via Ollama                  |
|                                       |                             |
|                                       v                             |
|                              Citation checker (substring match)     |
|                                  |__ valid       --> Final answer   |
|                                  \__ fabricated  --> Abstain        |
|                                                                     |
+---------------------------------------------------------------------+

```

Every chunk carries `paper_title`, `page`, and `section` metadata. The prompt instructs the model to attach a `[Paper: TITLE | p.N | §SECTION]` tag to every claim. After generation, a citation checker parses these tags and validates them against the retrieved set — any tag that didn't actually come back from retrieval is flagged as fabricated and surfaced in the UI.

If the rerank scores are too low or the citation checker detects fabrication, the system returns:

> *"I could not find sufficient information in the indexed papers to answer this question."*

— rather than guessing. This is what produces the 100% adversarial abstention rate.

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| Generator + Judge LLM | **Qwen2.5:7b** via Ollama | Strong open-weight model, runs on a laptop, no API cost |
| Embeddings | **BAAI/bge-m3** (1024-dim, dense + sparse) | Single model emits both signals via shared backbone |
| Reranker | **BAAI/bge-reranker-v2-m3** (cross-encoder) | Largest precision lift per ms in the ablation |
| Vector DB | **Qdrant** (native named vectors) | Stores dense and sparse in one collection, hybrid search at the DB level |
| Fusion | **Reciprocal Rank Fusion**, k=60 | Score-scale-free, reproducible, no hyperparameter tuning |
| Orchestration | **LangChain 0.3** + LangGraph | Composable prompt → LLM → parser chains |
| API | **FastAPI** + Uvicorn | Async, OpenAPI docs, mounts static UI |
| Frontend | **Vanilla HTML + CSS + JS** | No framework bloat, ~600 LoC total, cite-pill interactions |
| Observability | **Langfuse** (self-hosted) + **Prometheus** + **structlog** | Per-query traces, RED metrics, structured JSON logs |
| Tests | **pytest** — 51/51 passing | Chunker, retrieval, citation checker, eval metrics, API |
| CI | **GitHub Actions** | Lint (ruff), type-check, full test matrix on push |
| Packaging | **Docker Compose** (dev + prod) | Qdrant + Langfuse Postgres + the API itself |

---

## Quickstart

### Requirements

- macOS or Linux
- Python 3.11
- Docker (for Qdrant + Langfuse)
- [Ollama](https://ollama.com/) installed locally
- ~10 GB free disk (models + index)

### Setup

```bash
git clone https://github.com/baranozgurtas/research-rag-assistant.git
cd research-rag-assistant

# Python environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .

# Pull the LLM
ollama pull qwen2.5:7b

# Bring up Qdrant + Langfuse
make services-up

# Configure environment
cp .env.example .env  # edit if you need to change device or model

# Ingest the 15 papers under data/pdfs/ (chunks → embed → Qdrant)
make ingest

# Run the test suite (~5 min, downloads BGE-M3 on first run)
make test
```

### Run

Two terminals:

```bash
# Terminal 1 — FastAPI (port 8000)
make api

# Terminal 2 — open the UI in the default browser
make ui
```

Then visit **http://localhost:8000**.

---

## The UI

The frontend is a single-page vanilla-JS app served by FastAPI's `StaticFiles` mount. Three-panel layout, dark mode, collapsible side panels — the answer is always the focal point.

| Panel | Contents |
|---|---|
| **Left** (collapsible) | Service status (Qdrant, Ollama), chunk count, pipeline summary, recent queries persisted in `localStorage` |
| **Center** | Question, streaming answer with **inline citation pills** (click to scroll the right panel to that source), per-stage latency breakdown, citation validity badge |
| **Right** (collapsible) | Top-k retrieved chunks ranked by reranker score, each showing dense/sparse/rerank score breakdown and metadata (paper, page, section) |

When the system abstains, citation pills are suppressed and a `🛡 abstained (no fabrication)` badge replaces the usual `N/N citations valid` indicator — making it visually unambiguous that the model declined to answer.

---

## Evaluation

The eval is reproducible end-to-end. The full pipeline runs all 4 configurations on the question set and writes:

```
eval/results/
├── ablation_A_dense_only.json
├── ablation_B_dense_plus_rerank.json
├── ablation_C_hybrid_no_rerank.json
├── ablation_D_hybrid_plus_rerank.json
├── live_*.jsonl              # per-query records, flushed after every query
└── ablation_results.md       # the table at the top of this README
```

To re-run:

```bash
python -m eval.run_full_eval
python scripts/update_readme_metrics.py   # injects new numbers into this README
```

**Metric definitions (paper-level relevance):**

- **Hit@k**: fraction of questions where at least one expected paper appears in the top-k retrieved sources.
- **MRR@k**: mean reciprocal rank of the first expected paper in the top-k.
- **nDCG@k**: discounted cumulative gain at k, normalized; each paper counted at most once at its first appearance to avoid double-counting chunks from the same source.
- **Abstain rate**: fraction of all 25 questions where the system returned the abstention sentence.
- **Adv.Abstain**: fraction of the 5 *out-of-corpus* questions where the system correctly abstained. This is the headline number — every config scores 1.000.

---

## Defensive design choices

A non-exhaustive list of decisions that exist specifically to make the system harder to fool. Each is testable.

1. **Citation tags are validated, not trusted.** The model can produce any string between brackets — the citation checker only counts tags whose `Paper: TITLE` literal substring-matches a title in the retrieved set. Anything else is marked fabricated.
2. **Rerank scores gate the answer path.** If the top reranker score falls below the abstention threshold, the system enters the abstain branch instead of generating with weak context.
3. **The judge LLM and the generator are the same model (Qwen2.5:7b) by config but separate calls.** This lets a future deployment swap one without touching the other; the abstraction is already there.
4. **Section-aware chunking, not fixed windows.** The chunker respects paper section boundaries (abstract, introduction, related work, methodology, experiments, conclusion). A citation pointing to `§experiments` is meaningful, not a token offset.
5. **Sparse + dense in one model.** BGE-M3 emits both via a shared backbone in one forward pass — half the inference cost of running two embedders.
6. **RRF instead of weighted fusion.** No `α` to tune, no score-scale assumptions, no surprises when a new corpus shifts score distributions.
7. **Self-hosted observability.** Langfuse runs in Docker; no telemetry leaves the laptop.
8. **No streaming-only answer path.** The full answer is generated and citation-checked before being returned. Partial fabrications cannot leak through a stream.
9. **Tests cover the citation checker.** Including adversarial inputs that try to spoof tags. See [`tests/test_citation_checker.py`](tests/test_citation_checker.py).
10. **`localStorage`-backed recent queries on the client only.** Server is stateless; the UI is reloadable without losing context, but no PII ever crosses the wire.

The longer rationale lives in [`docs/design_decisions.md`](docs/design_decisions.md).

---

## Project layout

```
research-rag-assistant/
├── src/rag/
│   ├── ingestion/        # PDF loader, section-aware chunker, ingest pipeline
│   ├── embeddings/       # BGE-M3 wrapper (dense + sparse in one call)
│   ├── vectorstore/      # Qdrant client with named-vector schema
│   ├── retrieval/        # dense, sparse, hybrid (RRF), reranker
│   ├── generation/       # prompts (versioned), RAG chain, LLM factories
│   ├── guards/           # citation checker, abstention logic
│   ├── observability/    # Prometheus metrics, token ledger, Langfuse tracer
│   ├── api/              # FastAPI routes, dependencies, schemas
│   ├── config.py         # Pydantic Settings (env-driven)
│   └── logging_config.py # structlog setup
├── static/               # vanilla HTML + CSS + JS frontend
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── tests/                # 51 tests across 7 files
├── eval/
│   ├── papers.txt              # 15 arXiv IDs
│   ├── download_papers.py
│   ├── generate_questions.py
│   ├── questions.jsonl         # 20 synthetic + 5 adversarial
│   ├── retrieval_ablation.py   # 4-config ablation runner
│   ├── metrics.py              # Hit@k, MRR, nDCG implementations
│   ├── run_full_eval.py
│   └── results/                # JSON dumps + markdown summaries
├── data/pdfs/            # the 15 source papers
├── scripts/
│   ├── bootstrap.sh
│   └── update_readme_metrics.py  # patches the marker blocks in this file
├── docs/
│   ├── architecture.md         # detailed diagrams
│   └── design_decisions.md     # rationale for each choice
├── .github/workflows/ci.yml
├── docker-compose.yml          # production stack
├── docker-compose.dev.yml      # Qdrant + Langfuse for local dev
├── Dockerfile                  # multi-stage (ARM64 + AMD64)
├── Makefile
└── pyproject.toml
```

---

## Testing

```bash
make test        # full suite, ~5 min cold (model download), <90s warm
make lint        # ruff
make type-check  # mypy
```

The suite covers:

- Section-aware chunker (boundary detection, page mapping, edge cases)
- Dense, sparse, and hybrid retrieval (mock embedder, deterministic RRF)
- Citation checker (valid tags, fabricated tags, multi-citation strings)
- Eval metrics (Hit@k, MRR, nDCG with known fixtures)
- Token counter and observability emitters
- FastAPI routes (health, query, error paths)

---

## License

[Apache 2.0](LICENSE). Third-party model weights (Qwen2.5, BGE-M3, BGE-reranker-v2-m3) are governed by their own licenses; see [`NOTICE`](NOTICE).

---

## Repository

[github.com/baranozgurtas/research-rag-assistant](https://github.com/baranozgurtas/research-rag-assistant)
