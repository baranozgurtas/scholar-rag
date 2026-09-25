# Retrieval ablation

Retrieval-only run `retrieval_v2draft_20260925T213410Z_ddadab57` · no generator called · embedding `BAAI/bge-m3` · reranker `BAAI/bge-reranker-v2-m3` · questions `eval/questions_v2_draft.jsonl` (n=22, review status {'unreviewed': 22}) · commit `ddadab57e346ee3f1bcbd77697d6204ba71bfd12` · 15 PDFs / 1278 chunks.

## Retrieval (answerable questions)

| Config | n | Hit@5 | MRR@10 | nDCG@10 | Retrieval+rerank p50 / p95 (ms) | Errors |
|---|---|---|---|---|---|---|
| `A_dense_only` | 15 | 1.000 | 1.000 | 0.974 | 268 / 674 | 0 |
| `B_dense_plus_rerank` | 15 | 1.000 | 1.000 | 0.989 | 8611 / 14688 | 0 |
| `C_hybrid_no_rerank` | 15 | 1.000 | 1.000 | 0.989 | 644 / 843 | 0 |
| `D_hybrid_plus_rerank` | 15 | 1.000 | 1.000 | 0.987 | 7677 / 9508 | 0 |

## Top-1 rerank score by answerability (input for a dev-split threshold; not a result)

| Config | answerable min / p50 / max | unanswerable min / p50 / max |
|---|---|---|
| `B_dense_plus_rerank` | 0.411 / 0.884 / 0.996 | 0.125 / 0.628 / 0.725 |
| `D_hybrid_plus_rerank` | 0.482 / 0.884 / 0.996 | 0.125 / 0.636 / 0.725 |

## Per-question failures

### `A_dense_only`

None.

### `B_dense_plus_rerank`

None.

### `C_hybrid_no_rerank`

None.

### `D_hybrid_plus_rerank`

None.

