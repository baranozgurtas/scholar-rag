# dev split

**INDICATIVE** — 10/10 questions are not human-reviewed (LLM-drafted; evidence machine-checked only). Do not cite as a benchmark result.

## Retrieval (answerable questions)

| Config | n | Hit@5 | MRR@10 | nDCG@10 | Retrieval+rerank p50 / p95 (ms) | Errors |
|---|---|---|---|---|---|---|
| `A_dense_only` | 7 | 1.000 | 1.000 | 0.945 | 266 / 2877 | 0 |
| `B_dense_plus_rerank` | 7 | 1.000 | 1.000 | 0.976 | 8651 / 15795 | 0 |
| `C_hybrid_no_rerank` | 7 | 1.000 | 1.000 | 0.976 | 620 / 767 | 0 |
| `D_hybrid_plus_rerank` | 7 | 1.000 | 1.000 | 0.971 | 7661 / 9637 | 0 |

## Top-1 rerank score by answerability (input for a dev-split threshold; not a result)

| Config | answerable min / p50 / max | unanswerable min / p50 / max |
|---|---|---|
| `B_dense_plus_rerank` | 0.411 / 0.871 / 0.992 | 0.525 / 0.645 / 0.725 |
| `D_hybrid_plus_rerank` | 0.641 / 0.871 / 0.992 | 0.645 / 0.652 / 0.725 |

## Per-question failures

### `A_dense_only`

None.

### `B_dense_plus_rerank`

None.

### `C_hybrid_no_rerank`

None.

### `D_hybrid_plus_rerank`

None.


# heldout split

**INDICATIVE** — 12/12 questions are not human-reviewed (LLM-drafted; evidence machine-checked only). Do not cite as a benchmark result.

## Retrieval (answerable questions)

| Config | n | Hit@5 | MRR@10 | nDCG@10 | Retrieval+rerank p50 / p95 (ms) | Errors |
|---|---|---|---|---|---|---|
| `A_dense_only` | 8 | 1.000 | 1.000 | 1.000 | 291 / 574 | 0 |
| `B_dense_plus_rerank` | 8 | 1.000 | 1.000 | 1.000 | 8611 / 11228 | 0 |
| `C_hybrid_no_rerank` | 8 | 1.000 | 1.000 | 1.000 | 665 / 852 | 0 |
| `D_hybrid_plus_rerank` | 8 | 1.000 | 1.000 | 1.000 | 8046 / 9486 | 0 |

## Top-1 rerank score by answerability (input for a dev-split threshold; not a result)

| Config | answerable min / p50 / max | unanswerable min / p50 / max |
|---|---|---|
| `B_dense_plus_rerank` | 0.482 / 0.885 / 0.996 | 0.125 / 0.514 / 0.636 |
| `D_hybrid_plus_rerank` | 0.482 / 0.885 / 0.996 | 0.125 / 0.514 / 0.636 |

## Per-question failures

### `A_dense_only`

None.

### `B_dense_plus_rerank`

None.

### `C_hybrid_no_rerank`

None.

### `D_hybrid_plus_rerank`

None.

