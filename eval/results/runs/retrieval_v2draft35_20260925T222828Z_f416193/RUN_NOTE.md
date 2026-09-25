# Run note: retrieval_v2draft35_20260925T222828Z_f416193

**INDICATIVE.** All 35 questions are LLM-drafted and not human-reviewed. This is not a benchmark result.

## What was run
- Retrieval-only 4-config ablation (A dense, B dense+rerank, C hybrid, D hybrid+rerank). No generator (Qwen) was loaded.
- Code: commit `f416193` (clean tree, per `manifest.json`).
- Index: embedded Qdrant `./qdrant_local`, 1,278 chunks from the 15 correctly named PDFs, figure-debris filter applied (0 low-content chunks).
- Models: BAAI/bge-m3 (embedder, MPS), BAAI/bge-reranker-v2-m3 (reranker, MPS). Embedder and reranker ran in separate phases.
- Completion: 35/35 questions for every config, 0 errors. Wall time 996 s, peak RSS 2.2 GB (8 GB Mac).
- No abstention threshold was set or tuned.

## Exact question set
- `questions_v2_draft.snapshot.jsonl` is a byte-exact copy of `eval/questions_v2_draft.jsonl` as evaluated.
- SHA-256 `b587122590998b9cb70aea3ada49294be53cb530adfea6a72034f9e14446df07`, identical to `manifest.json → questions.sha256`.

## Held-out exposure (recorded here, not in the question file)
Per-question outcomes of the hard items **H13–H20** (held-out) and **D11–D15** (dev) were inspected when these results were reported. Items D01–D10 / H01–H12 were already marked `inspected_before_freeze` in the question file. **After this run, no held-out item in this file is pristine.** Machine-readable record: `inspection.json`.

## Outputs
- `candidates_{dense,hybrid}.jsonl`: top-20 candidates per question, with scores and text.
- `ranked_<config>.jsonl`: top-10 ranking per question and config (text kept for the top 5).
- `summary.json`, `report.md`: all 35 questions.
- `summary_by_split.json`, `report_by_split.md`: dev and held-out separately, labelled INDICATIVE.
