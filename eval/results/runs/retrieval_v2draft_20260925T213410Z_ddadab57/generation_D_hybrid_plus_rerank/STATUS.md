# Generation run status: STOPPED (incomplete), 2/22 questions

- Config: `D_hybrid_plus_rerank`, from retrieval run `retrieval_v2draft_20260925T213410Z_ddadab57`
- Generator: `qwen2.5:7b` (Ollama digest `845dbda0ea48…`, 7.6B, Q4_K_M), prompt `rag_answer@1.2`, commit `9713192` (clean)
- Machine: Apple Silicon Mac, 8 GB RAM; only Ollama loaded (no embedder / reranker).
- Completed: D01 (generation 81.9 s, outcome `uncited_answer`), D02 (70.2 s, `answered`).
- Stopped deliberately during D03: kernel memory-pressure level 2 (WARN), swap 6.43 / 7.17 GB used,
  ~200k pages swapped in and out per 20 s (thrashing). The in-flight question was not written.
- **No summary is computed**: 2/22 records cannot support rates. False abstention, false answer,
  tag validity, threshold selection and RAGAS for this run are **NOT RUN**.

Observed failure (D01): the answer is correct and cites RAG p.2, but the tag is written in parentheses
`(Paper: … | p.2 | §methods)` instead of `[Paper: …]`; the parser does not match it, so the policy
withheld the answer as `uncited_answer` (a format-strictness false abstention).

Resume on a machine with more memory (same model and prompt, or the run refuses to resume):

    GENERATOR_MODEL=qwen2.5:7b python -m eval.generation_eval eval/results/runs/retrieval_v2draft_20260925T213410Z_ddadab57 --config D_hybrid_plus_rerank
