# Scholar RAG
> Retrieval-augmented question answering over 15 ML papers, built to be measured.
> Hybrid retrieval (BGE-M3 dense + sparse, RRF) with a BGE cross-encoder reranker, answers from Qwen2.5:7b via Ollama, an enforced citation policy, served by FastAPI with a vanilla-JS UI.

[![CI](https://github.com/baranozgurtas/scholar-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/baranozgurtas/scholar-rag/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

---

## What this project is for

The question is whether a small, fully local RAG system can be evaluated honestly: what it retrieves, when it refuses, and what its citations do and do not prove.

- **Citations are enforced, within limits.** An answer is released only if it has at least one citation tag and every tag refers to a passage the model was given. Otherwise the user sees the abstention sentence, and the reason is recorded. A matching tag shows that the cited page was in the context, **not** that it supports the claim, and not that every claim is cited.
- **Refusal is measured, not assumed.** The eval reports the false-answer rate on unanswerable questions and the false-abstention rate on answerable ones separately, with confidence intervals.
- **Retrieval is ablated.** Four configurations, scored on retrieval alone (no LLM in the loop).
- **Everything runs locally.** No hosted APIs.

---

## Results

### Committed results (legacy 25-question set)

Four retrieval configurations, run end-to-end on a Colab T4 before this revision: 20 answerable questions (LLM-generated from single paper sections, **not human-reviewed**) and 5 out-of-domain questions. Numbers are recomputed from the committed JSONs by `python -m eval.reanalyze_legacy` (checked in CI); see [`eval/results/legacy_reanalysis.md`](eval/results/legacy_reanalysis.md).

<!-- METRICS:ABLATION:START -->
| Config | Hit@5 | MRR@5 | nDCG@5 | False abstention (answerable) | Answered unanswerable | p50 / p95 latency (s) |
|---|---|---|---|---|---|---|
| `A_dense_only` Dense only | 0.950 | 0.925 | 0.932 | 3/20 | 0/5 | 5.7 / 8.9 |
| `B_dense_plus_rerank` Dense + rerank | 1.000 | **0.967** | 0.975 | 1/20 | 0/5 | 6.6 / 9.8 |
| `C_hybrid_no_rerank` Hybrid (RRF), no rerank | 0.950 | 0.950 | 0.950 | 3/20 | 0/5 | 5.6 / 9.3 |
| `D_hybrid_plus_rerank` Hybrid + rerank (API default) | 0.950 | 0.950 | 0.950 | 0/20 | 0/5 | 6.8 / 10.3 |
<!-- METRICS:ABLATION:END -->

How to read this table:

- **Dense + reranker (B) ranks best**: MRR 0.967 and Hit@5 1.000, against 0.950 / 0.950 for hybrid + reranker (D). The committed JSON labels this MRR@10, but only the 5 chunks sent to the LLM were stored, so it is effectively MRR@5.
- **D abstained least on answerable questions**: 0/20 false abstentions against B's 1/20. The difference is one question (B refused L18, the Jeopardy metric question). Retrieval found the right paper for L18 under both configs, so this is a generation-side difference.
- **The trade-off is small and not established.** B retrieves slightly better and D abstained once less. With n=20 both gaps are single questions, and the 95% intervals overlap (D 0–16%, B 1–24%). D's one retrieval "miss" (L04) is a question that doesn't name its paper and fits NCF as well as the labelled BPR (see [evaluation notes](docs/evaluation.md#question-sets)). The API default is D; the data do not show it is better than B.
- **Reranking helped dense retrieval** (A → B: MRR 0.925 → 0.967). Hybrid without reranking (C) matched D on retrieval.
- **5/5 abstention on unanswerable questions is a small smoke test, not a guarantee.** The five are obviously off-topic (restaurants, Claude 3's parameter count, code generation). A 0/5 false-answer rate has a 95% upper bound of 43%. None is a near-miss question about an indexed paper.
- **What produced those refusals**: the rerank threshold was 0.0 (disabled), so every question reached the generator, and the model wrote the abstention sentence itself. At the time, "abstained" was a substring test. In configs A, B and D, 2–3 of the 5 refusals still contained citation tags, which suggests the outputs mixed a refusal with an answer. Answer text wasn't saved, so this can't be checked.
- **The citation guard did not exist for these runs.** One to three released answers per config had no citation tags (A 1, B 3, C 2, D 2). Under the current policy those would be withheld, so the legacy false-abstention numbers are probably optimistic for the current code (not re-measured).
- The generator model was not recorded for these runs.

### Current code

| Evaluation | Status |
|---|---|
| Retrieval-only ablation (A–D), current code and index | **NOT RUN** |
| Generation eval with the enforced answer policy (config D) | **NOT RUN** |
| LLM-judged grounding (RAGAS) | **NOT RUN** |
| Rerank abstention threshold (dev → held-out) | **NOT RUN**; gate disabled |

Commands to produce these, and the procedure for a full GPU run, are in [docs/evaluation.md](docs/evaluation.md#commands). A first local attempt on an 8 GB laptop was stopped: generation took about 5 minutes per question, and it surfaced a retrieval anomaly. The investigation (debris chunks from a PDF figure; the cause of the one degenerate query embedding is not established) is written up in [docs/evaluation.md](docs/evaluation.md#index-hygiene-the-adam-question-returned-causal-forest-chunks-incident).

---

## Demo

Three screenshots of the UI on different question types. They were taken before the changes in this revision; the badge in screenshot 3 now reads "model declined: context judged insufficient", and tag badges now say "citation tags match retrieved passages".

### 1. Grounded answer with citation breakdown

[Grounded answer with score breakdown] <img width="1440" height="850" alt="Screenshot 2026-05-26 at 8 17 04 PM" src="https://github.com/user-attachments/assets/1f717c77-fc2f-4f65-bc00-e3d6e8b91ecc" />


A simple architectural question about M3-Embedding. The model produces a short, citation-attached answer; the right panel shows the top retrieved chunks with their dense / sparse / rerank scores. The inline `[1]` pill is clickable and scrolls the sources panel to the cited chunk. A pill means the tag matches a supplied passage; whether that passage supports the sentence is for the reader to check.

### 2. Cross-paper synthesis via hybrid retrieval

[Cross-paper retrieval across BPR and NCF papers] <img width="1439" height="816" alt="Screenshot 2026-05-26 at 8 21 21 PM" src="https://github.com/user-attachments/assets/caaf3857-2325-456b-8336-4fc0da8576eb" />


A comparative question needs evidence from two papers (`ncf-he-2017` and `bpr-rendle-2009`). In this example hybrid retrieval plus the reranker put both in the top 5 and the answer cites both. This is one example, not a measured comparison against dense-only retrieval on multi-paper questions.

### 3. Out-of-corpus query: the generator declines

[Adversarial out-of-corpus question correctly abstained] <img width="1436" height="845" alt="Screenshot 2026-05-26 at 10 18 45 PM 1" src="https://github.com/user-attachments/assets/09003461-1d26-4609-a7e6-196fb361ba2e" />


An off-topic query (*"What is the best ramen restaurant in Zurich?"*). The retriever still returns five chunks (N-BEATS, Adam, BPR, NCF) with a top rerank score of `0.671`. The rerank threshold was 0.0 (disabled), so those chunks went to the generator, and **the generator** wrote the abstention sentence. No score-based abstain branch was involved. The same mechanism produced the 5/5 abstentions in the table above.

---

## Architecture
```
+-- INGESTION ---------------------------------------------------+
|   15 arXiv PDFs → section-aware chunker (drops figure debris)   |
|   → BGE-M3 (dense + sparse, one forward pass) → Qdrant          |
+----------------------------------------------------------------+
                            |
+-- QUERY -------------------------------------------------------+
|   question → dense + sparse → RRF (k=60) → top-20              |
|   → cross-encoder rerank (bge-reranker-v2-m3) → top-5          |
|   → [optional gate: top-1 rerank score < threshold → abstain]  |
|   → Qwen2.5:7b via Ollama                                      |
|   → answer policy:                                             |
|        abstention sentence only        → model_abstained       |
|        abstention + other text         → withheld              |
|        no citation tag                 → withheld              |
|        tag not in supplied passages    → withheld              |
|        otherwise                       → released              |
+----------------------------------------------------------------+
```

Every chunk carries `paper_title`, `page` and `section`. The prompt asks for a `[Paper: TITLE | p.N | §SECTION]` tag after each claim. After generation, the policy in [`citation_checker.py`](src/rag/guards/citation_checker.py) parses the tags and compares them with the tags of the supplied passages. Exact matches pass, as do matches that differ only in section or use a whole-word shortened title of at least 6 characters on the same page. A withheld answer is replaced by

> *"I could not find sufficient information in the indexed papers to answer this question."*

and the response's `outcome` field says why. With `debug=true` the raw generator output is returned too.

The pre-generation rerank gate (`RETRIEVAL_RERANK_SCORE_THRESHOLD`) is **off by default**. If it's enabled, choose the value with `eval.select_threshold` on the dev split, and report it on held-out questions.

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| Generator (and RAGAS judge) | **Qwen2.5:7b** via Ollama | Open-weight, runs locally |
| Embeddings | **BAAI/bge-m3** (1024-dim, dense + sparse) | One model emits both signals |
| Reranker | **BAAI/bge-reranker-v2-m3** | Raised dense-only MRR 0.925 → 0.967 on the legacy set (n=20) |
| Vector DB | **Qdrant** (named vectors; server or embedded via `QDRANT_PATH`) | Dense and sparse in one collection |
| Fusion | **Reciprocal Rank Fusion**, k=60 | No score normalisation needed |
| Orchestration | **LangChain core** (LCEL) | Prompt → LLM → parser |
| API | **FastAPI** + Uvicorn | Async, OpenAPI docs, serves the static UI |
| Frontend | **Vanilla HTML + CSS + JS** (~750 lines) | Citation pills, score breakdowns |
| Observability | **Langfuse** (self-hosted, optional) + **Prometheus** + **structlog** | Traces, metrics, JSON logs |
| Tests | **pytest**, deterministic | No models, no Ollama, no Qdrant server |
| CI | **GitHub Actions** | ruff + pytest + legacy re-analysis check |
| Packaging | **Docker Compose** (dev + prod) | Qdrant + Langfuse + API |

---

## Quickstart

### Requirements

- macOS or Linux, Python 3.11
- [Ollama](https://ollama.com/) with `qwen2.5:7b`
- Docker for Qdrant, or set `QDRANT_PATH` to use embedded Qdrant
- ~10 GB free disk (models + index)

### Setup

```bash
git clone https://github.com/baranozgurtas/scholar-rag.git
cd scholar-rag
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e .

ollama pull qwen2.5:7b
cp .env.example .env            # device, models, QDRANT_URL or QDRANT_PATH
make services-up                # Qdrant + Langfuse (skip if using QDRANT_PATH)
python -m rag.ingestion.pipeline --pdf-dir ./data/pdfs --recreate

make test                       # seconds; no model downloads
```

### Run

```bash
make api      # FastAPI on :8000 (loads BGE-M3 + reranker, calls Ollama)
make ui       # opens http://localhost:8000
```

---

## The UI

A single-page vanilla-JS app served by FastAPI's `StaticFiles` mount.

| Panel | Contents |
|---|---|
| **Left** (collapsible) | Service status (Qdrant, Ollama), chunk count, pipeline summary, recent queries in `localStorage` |
| **Center** | Answer with inline citation pills, per-stage latency, and a badge: `N/N citation tags match retrieved passages`, or the reason for not answering (`model declined…`, `withheld: answer had no citation tags`, `withheld: cited a passage outside the supplied context`, …) |
| **Right** (collapsible) | Top-k chunks with dense / sparse / rerank scores and paper, page, section |

---

## Evaluation

Full details: [docs/evaluation.md](docs/evaluation.md).

- **Retrieval-only ablation** (`make eval-retrieval`, `make eval-retrieval-draft`): Hit@5, MRR@10, nDCG@10 on answerable questions for A–D, with no generator call. The embedder and reranker run in separate phases. Resumable.
- **Generation for one config** (`make eval-generation RUN=<dir>`): runs one question at a time and appends each record to disk as it goes; restarting resumes from where it stopped. It reports false abstention on answerable questions, false answers on unanswerable ones, citation tag validity (structural), what the guard withheld and why, p50/p95 latency, and per-question failures.
- **Grounding** (`make eval-ragas RECORDS=…`): LLM-judged RAGAS on the saved answers. The judge is the same model family as the generator, and failed judgments are reported as missing, never scored.
- Every run writes a `manifest.json`: commit SHA and dirty flag, model names and the generator's Ollama digest, settings, PDF hashes, index size, and question-file hash with review-status counts.

**Question sets.** [`eval/questions.jsonl`](eval/questions.jsonl) is the legacy 25, now annotated per question with known label problems. [`eval/questions_v2_draft.jsonl`](eval/questions_v2_draft.jsonl) has 22 dev/held-out questions: 15 answerable, whose evidence quotes CI checks against the PDFs, and 7 near-miss unanswerable, whose absent terms CI checks. They were **drafted by an LLM and are not human-reviewed**. The 15 old `[FILL IN]` templates were never evaluated and have been removed. The plan for a larger reviewed set is in [docs/evaluation.md](docs/evaluation.md#proposed-reviewed-held-out-set).

**Corpus label fixes.** Two PDFs were mislabelled. `ml-tips-domingos-2012.pdf` was actually Paullada et al. 2020, "Data and its (dis)contents", and `dropout-srivastava-2014.pdf` was actually Hinton et al. 2012. They are now `data-discontents-paullada-2020` and `dropout-hinton-2012`. Re-ingest with `--recreate` after pulling.

---

## Defensive design choices

1. **Citation tags are validated and enforced.** Answers with no tag, or with a tag outside the supplied passages, are withheld ([tests](tests/test_citation_checker.py)). This is a structural check only.
2. **Mixed outputs are not released.** If the model writes the abstention sentence *and* an answer, the answer is withheld.
3. **Generation errors are not shown as answers.** The user gets the abstention sentence, and `outcome=generation_error` records what happened.
4. **Optional rerank gate, off by default.** It acts on the top-1 score before generation, so its effect can be replayed exactly on recorded runs, and it is only tuned on dev.
5. **Section-aware chunking with a debris filter.** Chunks never span sections, and figure-marker debris is dropped at ingestion.
6. **Sparse + dense from one model.** BGE-M3 emits both in one forward pass.
7. **RRF instead of weighted fusion.** No weights to tune.
8. **Self-hosted observability.** Langfuse runs in Docker, and it is optional.
9. **No streaming answer path.** The full answer passes the policy before it is returned.

More rationale: [`docs/design_decisions.md`](docs/design_decisions.md).

---

## Project layout

```
scholar-rag/
├── src/rag/
│   ├── ingestion/        # PDF loader, section-aware chunker (+ debris filter), ingest pipeline
│   ├── embeddings/       # BGE-M3 wrapper (dense + sparse in one call)
│   ├── vectorstore/      # Qdrant (server or embedded)
│   ├── retrieval/        # dense, sparse, hybrid (RRF), reranker
│   ├── generation/       # prompts, AnswerGenerator, RAGChain, LLM factories
│   ├── guards/           # citation extraction, validation, answer policy
│   ├── observability/    # Prometheus metrics, token ledger, Langfuse tracer
│   └── api/              # FastAPI routes, dependencies, schemas
├── static/               # vanilla HTML + CSS + JS frontend
├── tests/                # deterministic suite (no models / services)
├── eval/
│   ├── questions.jsonl           # legacy 25 (annotated)
│   ├── questions_v2_draft.jsonl  # 22 unreviewed dev/held-out drafts
│   ├── questions.py              # schema, loader, PDF evidence checks
│   ├── retrieval_ablation.py     # retrieval-only A–D, resumable
│   ├── generation_eval.py        # one config, one question at a time, resumable
│   ├── ragas_eval.py             # LLM-judged grounding on saved answers
│   ├── select_threshold.py       # dev → held-out threshold selection
│   ├── reanalyze_legacy.py       # re-derives committed numbers (CI)
│   ├── harness.py, metrics.py    # summaries, Hit/MRR/nDCG, Wilson CIs, p50/p95
│   └── results/                  # committed legacy results + re-analysis
├── docs/                 # architecture, design decisions, evaluation
├── data/pdfs/            # the 15 source papers
└── scripts/update_readme_metrics.py
```

---

## Testing

```bash
make test       # full deterministic suite, a few seconds
make test-ci    # exactly what CI runs (ruff + pytest + legacy re-analysis)
```

CI installs only `requirements-ci.txt`: no torch, no model downloads, no Ollama or Qdrant server. The suite covers the chunker and debris filter, RRF, the citation checker and answer policy (uncited, out-of-context, spoofed-title, mixed and error outputs), RAGChain with fake retriever, reranker and LLM (gate, withholding, ranking depth), the eval harness (metrics, CIs, error handling, resume, model-mismatch refusal), threshold selection (dev only), question-file evidence checks against the PDFs, a regression that re-derives the committed legacy numbers, and the API.

---

## License

[Apache 2.0](LICENSE). Model weights (Qwen2.5, BGE-M3, BGE-reranker-v2-m3) have their own licenses; see [`NOTICE`](NOTICE).
