# dev split

**INDICATIVE** — 15/15 questions are not human-reviewed (LLM-drafted; evidence machine-checked only). Do not cite as a benchmark result.

## Retrieval (answerable questions)

| Config | n | Hit@5 | All-sources@5 | Multi-paper n / Hit@5 / All-sources@5 | MRR@10 | nDCG@10 | Retrieval+rerank p50 / p95 (ms) | Errors |
|---|---|---|---|---|---|---|---|---|
| `A_dense_only` | 11 | 1.000 | 0.909 | 2 / 1.000 / 0.500 | 1.000 | 0.965 | 348 / 2065 | 0 |
| `B_dense_plus_rerank` | 11 | 1.000 | 0.909 | 2 / 1.000 / 0.500 | 1.000 | 0.985 | 12602 / 21273 | 0 |
| `C_hybrid_no_rerank` | 11 | 1.000 | 0.909 | 2 / 1.000 / 0.500 | 1.000 | 0.985 | 899 / 1085 | 0 |
| `D_hybrid_plus_rerank` | 11 | 1.000 | 0.909 | 2 / 1.000 / 0.500 | 1.000 | 0.982 | 15371 / 17771 | 0 |

## Top-1 rerank score by answerability (input for a dev-split threshold; not a result)

| Config | answerable min / p50 / max | unanswerable min / p50 / max |
|---|---|---|
| `B_dense_plus_rerank` | 0.411 / 0.990 / 0.995 | 0.525 / 0.691 / 0.917 |
| `D_hybrid_plus_rerank` | 0.641 / 0.990 / 0.995 | 0.652 / 0.691 / 0.917 |

## Per-question failures

### `A_dense_only`

| id | failures | expected | top-5 sources | question |
|---|---|---|---|---|
| D05 | missing_required_source_at_5 | bpr-rendle-2009, ncf-he-2017 | bpr-rendle-2009, bpr-rendle-2009, bpr-rendle-2009, bpr-rendle-2009, bpr-rendle-2009 | How does the training objective of BPR differ from the objective used by Neural Collaborative Filtering? |

### `B_dense_plus_rerank`

| id | failures | expected | top-5 sources | question |
|---|---|---|---|---|
| D05 | missing_required_source_at_5 | bpr-rendle-2009, ncf-he-2017 | bpr-rendle-2009, bpr-rendle-2009, bpr-rendle-2009, bpr-rendle-2009, bpr-rendle-2009 | How does the training objective of BPR differ from the objective used by Neural Collaborative Filtering? |

### `C_hybrid_no_rerank`

| id | failures | expected | top-5 sources | question |
|---|---|---|---|---|
| D05 | missing_required_source_at_5 | bpr-rendle-2009, ncf-he-2017 | bpr-rendle-2009, bpr-rendle-2009, bpr-rendle-2009, bpr-rendle-2009, bpr-rendle-2009 | How does the training objective of BPR differ from the objective used by Neural Collaborative Filtering? |

### `D_hybrid_plus_rerank`

| id | failures | expected | top-5 sources | question |
|---|---|---|---|---|
| D05 | missing_required_source_at_5 | bpr-rendle-2009, ncf-he-2017 | bpr-rendle-2009, bpr-rendle-2009, bpr-rendle-2009, bpr-rendle-2009, bpr-rendle-2009 | How does the training objective of BPR differ from the objective used by Neural Collaborative Filtering? |


# heldout split

**INDICATIVE** — 20/20 questions are not human-reviewed (LLM-drafted; evidence machine-checked only). Do not cite as a benchmark result.

## Retrieval (answerable questions)

| Config | n | Hit@5 | All-sources@5 | Multi-paper n / Hit@5 / All-sources@5 | MRR@10 | nDCG@10 | Retrieval+rerank p50 / p95 (ms) | Errors |
|---|---|---|---|---|---|---|---|---|
| `A_dense_only` | 14 | 1.000 | 1.000 | 2 / 1.000 / 1.000 | 1.000 | 0.991 | 304 / 406 | 0 |
| `B_dense_plus_rerank` | 14 | 1.000 | 0.929 | 2 / 1.000 / 0.500 | 1.000 | 0.974 | 10777 / 13490 | 0 |
| `C_hybrid_no_rerank` | 14 | 1.000 | 0.929 | 2 / 1.000 / 0.500 | 0.952 | 0.949 | 751 / 1001 | 0 |
| `D_hybrid_plus_rerank` | 14 | 1.000 | 0.929 | 2 / 1.000 / 0.500 | 1.000 | 0.962 | 14821 / 23767 | 0 |

## Top-1 rerank score by answerability (input for a dev-split threshold; not a result)

| Config | answerable min / p50 / max | unanswerable min / p50 / max |
|---|---|---|
| `B_dense_plus_rerank` | 0.416 / 0.887 / 0.997 | 0.125 / 0.537 / 0.803 |
| `D_hybrid_plus_rerank` | 0.416 / 0.887 / 0.997 | 0.125 / 0.537 / 0.803 |

## Per-question failures

### `A_dense_only`

None.

### `B_dense_plus_rerank`

| id | failures | expected | top-5 sources | question |
|---|---|---|---|---|
| H13 | missing_required_source_at_5 | deepar-salinas-2017, conformal-qr-romano-2019 | conformal-qr-romano-2019, conformal-qr-romano-2019, conformal-qr-romano-2019, conformal-qr-romano-2019, conformal-qr-romano-2019 | Which two works in the corpus produce probabilistic outputs, one through a parametric likelihood for count data and one  |

### `C_hybrid_no_rerank`

| id | failures | expected | top-5 sources | question |
|---|---|---|---|---|
| H13 | missing_required_source_at_5 | deepar-salinas-2017, conformal-qr-romano-2019 | conformal-qr-romano-2019, conformal-qr-romano-2019, conformal-qr-romano-2019, conformal-qr-romano-2019, conformal-qr-romano-2019 | Which two works in the corpus produce probabilistic outputs, one through a parametric likelihood for count data and one  |

### `D_hybrid_plus_rerank`

| id | failures | expected | top-5 sources | question |
|---|---|---|---|---|
| H13 | missing_required_source_at_5 | deepar-salinas-2017, conformal-qr-romano-2019 | conformal-qr-romano-2019, conformal-qr-romano-2019, conformal-qr-romano-2019, conformal-qr-romano-2019, conformal-qr-romano-2019 | Which two works in the corpus produce probabilistic outputs, one through a parametric likelihood for count data and one  |

