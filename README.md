# Scholar RAG

**Citation-enforced question answering over ML research papers, with reproducible retrieval evaluation.**

Scholar RAG answers questions about a corpus of 15 machine-learning papers. It retrieves with BGE-M3 dense and sparse vectors fused by Reciprocal Rank Fusion, reranks with a BGE cross-encoder, and generates with Qwen2.5:7b. Every answer must cite the passages it was given, or it is not released. A FastAPI service serves the answers and a web UI that shows the evidence behind each one. An evaluation harness records every run, from models and index to commit and question-file hash, and CI checks it deterministically.

[![CI](https://github.com/baranozgurtas/scholar-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/baranozgurtas/scholar-rag/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

- **Hybrid retrieval**: BGE-M3 dense + sparse vectors from one forward pass, fused with RRF (k=60), stored as named vectors in Qdrant.
- **Cross-encoder reranking**: `bge-reranker-v2-m3` reorders the top 20 candidates, and the top 5 go to the generator.
- **Enforced citation policy**: an answer is released only if it carries at least one citation tag and every tag matches a supplied passage. Otherwise the user gets an explicit refusal with a recorded reason.
- **Evidence-first UI**: citation pills link to the cited chunk, alongside dense, sparse and rerank scores.
- **Reproducible evaluation**: retrieval-only ablations, a resumable generation evaluation, per-run manifests, and a model-free CI suite.

---

## Demo

**A cited answer with per-chunk dense, sparse and rerank scores.**

<img width="1440" height="850" alt="Cited answer with retrieved sources and score breakdown" src="https://github.com/user-attachments/assets/1f717c77-fc2f-4f65-bc00-e3d6e8b91ecc" />

**A comparative question answered from two papers (NCF and BPR).**

<img width="1439" height="816" alt="Answer citing both the NCF and BPR papers" src="https://github.com/user-attachments/assets/caaf3857-2325-456b-8336-4fc0da8576eb" />

---

## Architecture

```
INGESTION
  15 PDFs → section-aware chunker (drops figure debris) → 1,278 chunks
          → BGE-M3 (dense + sparse, one forward pass) → Qdrant (named vectors)

QUERY
  question → dense top-30 + sparse top-30 → RRF (k=60) → top-20
           → cross-encoder rerank (bge-reranker-v2-m3) → top-5
           → [optional gate: top-1 rerank score < threshold → refuse]
           → Qwen2.5:7b
           → answer policy
               abstention sentence only       → model_abstained
               abstention + other content     → withheld
               no citation tag                → withheld
               tag not in supplied passages   → withheld
               otherwise                      → released
```

Every chunk carries `paper_title`, `page` and `section`. The prompt asks for a `[Paper: TITLE | p.N | §SECTION]` tag after each claim. The same fields in parentheses are accepted too, under identical rules. The API response includes an `outcome` field explaining any refusal, and with `debug=true` it also returns the raw model output.

---

## Key engineering decisions

1. **Citations are a release condition, not decoration.** [`citation_checker.py`](src/rag/guards/citation_checker.py) accepts a tag only if it matches a supplied passage: exactly, with a different section on the same page, or with a whole-word shortened title of at least 6 characters. Uncited answers, out-of-context tags and refusal-plus-answer outputs are withheld. This is a structural check: a valid tag shows that the cited page was in context, not that it supports the claim.
2. **Retrieval is evaluated separately from generation.** The four-config ablation never loads the generator. The dense-only and dense+rerank configs share one candidate set, and the two hybrid configs share another, so the embedder and the reranker run in separate phases.
3. **Coverage is measured for multi-paper questions.** Hit@5 is satisfied by any one expected paper. All-sources@5 requires every expected paper in the five chunks the generator sees.
4. **Refusal is gated before generation only when evidence supports it.** The top-1 rerank-score gate is off by default. Its threshold can only be selected on the dev split, and recorded outputs let it be replayed exactly.
5. **Runs are reproducible and resumable.** Each run writes a manifest with the commit SHA, model names, generator digest, settings, PDF hashes, index size and question-file hash, and saves results one question at a time. It refuses to resume if the model or data changed.
6. **Index hygiene is enforced at ingestion.** A lexical-diversity filter drops PDF figure debris, such as scatter-plot markers extracted as text, which surfaced as the top dense results in one anomalous run ([incident write-up](docs/evaluation.md#index-hygiene-the-adam-question-returned-causal-forest-chunks-incident)).
7. **CI needs no models.** The suite runs a fake retriever, reranker and LLM, checks every evidence quote in the question set against the PDFs, and regenerates the historical results, all without torch, Ollama or a Qdrant server.

More rationale: [`docs/design_decisions.md`](docs/design_decisions.md).

---

## Evaluation results

### Exploratory retrieval run (35 questions)

Run [`retrieval_v2draft35_20260925T222828Z_f416193`](eval/results/runs/retrieval_v2draft35_20260925T222828Z_f416193/) evaluated four retrieval configurations on [`eval/questions_v2_draft.jsonl`](eval/questions_v2_draft.jsonl): 25 answerable and 10 unanswerable questions, with fixed dev and held-out splits. The results are **exploratory**: the questions were drafted by an LLM, have not been human-reviewed, and their outcomes have been inspected.

- **All four configurations reached Hit@5 = 1.000** on answerable questions in both splits.
- **All-sources@5 separated them on held-out.** Dense-only (A) scored 14/14; the API default, hybrid + rerank (D), scored 13/14.
- **Two multi-paper questions exposed incomplete coverage:**
  - **D05** (BPR vs NCF): no configuration placed NCF in the top 5.
  - **H13** (DeepAR and conformalized quantile regression): only dense-only placed DeepAR in the top 5.

These are retrieval measurements. They do not rank the configurations overall, and they say nothing about answer quality. Per-split tables, latency and per-question failures are in [docs/evaluation.md](docs/evaluation.md#exploratory-35-question-retrieval-run). The status of the generation evaluation is there too.

### Historical results (legacy 25-question set)

An earlier end-to-end run on 20 LLM-generated answerable questions and 5 off-topic questions, before the citation policy existed. Re-derived from the committed JSONs by `python -m eval.reanalyze_legacy` and checked in CI. Only five ranks were stored, so ranking metrics are @5. See [`legacy_reanalysis.md`](eval/results/legacy_reanalysis.md) for caveats.

<!-- METRICS:ABLATION:START -->
| Config | Hit@5 | MRR@5 | nDCG@5 | False abstention (answerable) | Answered unanswerable | p50 / p95 latency (s) |
|---|---|---|---|---|---|---|
| `A_dense_only` Dense only | 0.950 | 0.925 | 0.932 | 3/20 | 0/5 | 5.7 / 8.9 |
| `B_dense_plus_rerank` Dense + rerank | 1.000 | **0.967** | 0.975 | 1/20 | 0/5 | 6.6 / 9.8 |
| `C_hybrid_no_rerank` Hybrid (RRF), no rerank | 0.950 | 0.950 | 0.950 | 3/20 | 0/5 | 5.6 / 9.3 |
| `D_hybrid_plus_rerank` Hybrid + rerank (API default) | 0.950 | 0.950 | 0.950 | 0/20 | 0/5 | 6.8 / 10.3 |
<!-- METRICS:ABLATION:END -->

---

## Stack

| Layer | Choice |
|---|---|
| Generator | Qwen2.5:7b, self-hosted via Ollama |
| Embeddings | BAAI/bge-m3 (1024-dim dense + sparse) |
| Reranker | BAAI/bge-reranker-v2-m3 (cross-encoder) |
| Vector store | Qdrant, named dense + sparse vectors (server, or embedded via `QDRANT_PATH`) |
| Fusion | Reciprocal Rank Fusion, k=60 |
| Orchestration | LangChain core (LCEL) |
| API and UI | FastAPI + Uvicorn; vanilla HTML/CSS/JS served as static files |
| Observability | Prometheus metrics, structlog JSON logs, optional Langfuse tracing |
| Quality | pytest (deterministic), ruff, GitHub Actions |
| Packaging | Docker Compose (dev and prod) |

---

## Quickstart

Requires Python 3.11, [Ollama](https://ollama.com/), and Docker (or `QDRANT_PATH` for embedded Qdrant).

```bash
git clone https://github.com/baranozgurtas/scholar-rag.git
cd scholar-rag
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e .

ollama pull qwen2.5:7b
cp .env.example .env              # models, devices, QDRANT_URL or QDRANT_PATH
make services-up                  # Qdrant + Langfuse (skip when using QDRANT_PATH)
python -m rag.ingestion.pipeline --pdf-dir ./data/pdfs --recreate

make api                          # FastAPI + UI on http://localhost:8000
```

Evaluation:

```bash
make test                         # deterministic suite, no models needed
make eval-retrieval-draft         # retrieval-only ablation on the 35-question draft set
make eval-generation RUN=<run>    # generation for one config, resumable
```

All commands are listed in [docs/evaluation.md](docs/evaluation.md#commands).

---

## Question sets

- [`eval/questions_v2_draft.jsonl`](eval/questions_v2_draft.jsonl) has 35 questions on fixed dev and held-out splits: 25 answerable (4 need two papers) and 10 unanswerable, one with a false premise. Thirteen are harder items that do not name their target paper and have close distractors. CI verifies every evidence quote against its PDF page and every "absent" term against the scoped papers. A human review packet is in [`eval/review/`](eval/review/questions_v2_draft_review.md).
- [`eval/questions.jsonl`](eval/questions.jsonl) is the historical 25-question set, annotated with known label issues.

---

## Project layout

```
scholar-rag/
├── src/rag/
│   ├── ingestion/        # PDF loader, section-aware chunker, debris filter, ingest pipeline
│   ├── embeddings/       # BGE-M3 wrapper (dense + sparse in one call)
│   ├── vectorstore/      # Qdrant (server or embedded)
│   ├── retrieval/        # dense, sparse, hybrid (RRF), reranker
│   ├── generation/       # prompts, AnswerGenerator, RAGChain
│   ├── guards/           # citation extraction, validation, answer policy
│   ├── observability/    # Prometheus metrics, token ledger, Langfuse tracer
│   └── api/              # FastAPI routes, dependencies, schemas
├── static/               # web UI
├── eval/                 # question sets, harness, runners, committed runs
├── tests/                # deterministic suite
├── docs/                 # architecture, design decisions, evaluation
└── data/pdfs/            # the 15 source papers
```

---

## License

[Apache 2.0](LICENSE). Model weights (Qwen2.5, BGE-M3, BGE-reranker-v2-m3) are covered by their own licenses; see [`NOTICE`](NOTICE).
