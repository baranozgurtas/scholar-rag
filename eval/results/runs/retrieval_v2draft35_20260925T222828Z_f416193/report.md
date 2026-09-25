# Retrieval ablation

Retrieval-only run `retrieval_v2draft35_20260925T222828Z_f416193` · no generator called · embedding `BAAI/bge-m3` · reranker `BAAI/bge-reranker-v2-m3` · questions `eval/questions_v2_draft.jsonl` (n=35, review status {'unreviewed': 35}) · commit `f416193d82ef2195a2e8fb80461ff4c8a23c5b69` · 15 PDFs / 1278 chunks.

## Retrieval (answerable questions)

| Config | n | Hit@5 | All-sources@5 | Multi-paper n / Hit@5 / All-sources@5 | MRR@10 | nDCG@10 | Retrieval+rerank p50 / p95 (ms) | Errors |
|---|---|---|---|---|---|---|---|---|
| `A_dense_only` | 25 | 1.000 | 0.960 | 4 / 1.000 / 0.750 | 1.000 | 0.980 | 333 / 533 | 0 |
| `B_dense_plus_rerank` | 25 | 1.000 | 0.920 | 4 / 1.000 / 0.500 | 1.000 | 0.979 | 11632 / 18359 | 0 |
| `C_hybrid_no_rerank` | 25 | 1.000 | 0.920 | 4 / 1.000 / 0.500 | 0.973 | 0.965 | 818 / 1063 | 0 |
| `D_hybrid_plus_rerank` | 25 | 1.000 | 0.920 | 4 / 1.000 / 0.500 | 1.000 | 0.970 | 14979 / 22531 | 0 |

## Top-1 rerank score by answerability (input for a dev-split threshold; not a result)

| Config | answerable min / p50 / max | unanswerable min / p50 / max |
|---|---|---|
| `B_dense_plus_rerank` | 0.411 / 0.899 / 0.997 | 0.125 / 0.632 / 0.917 |
| `D_hybrid_plus_rerank` | 0.416 / 0.899 / 0.997 | 0.125 / 0.644 / 0.917 |

## Per-question failures

### `A_dense_only`

| id | failures | expected | top-5 sources | question |
|---|---|---|---|---|
| D05 | missing_required_source_at_5 | bpr-rendle-2009, ncf-he-2017 | bpr-rendle-2009, bpr-rendle-2009, bpr-rendle-2009, bpr-rendle-2009, bpr-rendle-2009 | How does the training objective of BPR differ from the objective used by Neural Collaborative Filtering? |

### `B_dense_plus_rerank`

| id | failures | expected | top-5 sources | question |
|---|---|---|---|---|
| D05 | missing_required_source_at_5 | bpr-rendle-2009, ncf-he-2017 | bpr-rendle-2009, bpr-rendle-2009, bpr-rendle-2009, bpr-rendle-2009, bpr-rendle-2009 | How does the training objective of BPR differ from the objective used by Neural Collaborative Filtering? |
| H13 | missing_required_source_at_5 | deepar-salinas-2017, conformal-qr-romano-2019 | conformal-qr-romano-2019, conformal-qr-romano-2019, conformal-qr-romano-2019, conformal-qr-romano-2019, conformal-qr-romano-2019 | Which two works in the corpus produce probabilistic outputs, one through a parametric likelihood for count data and one  |

### `C_hybrid_no_rerank`

| id | failures | expected | top-5 sources | question |
|---|---|---|---|---|
| D05 | missing_required_source_at_5 | bpr-rendle-2009, ncf-he-2017 | bpr-rendle-2009, bpr-rendle-2009, bpr-rendle-2009, bpr-rendle-2009, bpr-rendle-2009 | How does the training objective of BPR differ from the objective used by Neural Collaborative Filtering? |
| H13 | missing_required_source_at_5 | deepar-salinas-2017, conformal-qr-romano-2019 | conformal-qr-romano-2019, conformal-qr-romano-2019, conformal-qr-romano-2019, conformal-qr-romano-2019, conformal-qr-romano-2019 | Which two works in the corpus produce probabilistic outputs, one through a parametric likelihood for count data and one  |

### `D_hybrid_plus_rerank`

| id | failures | expected | top-5 sources | question |
|---|---|---|---|---|
| D05 | missing_required_source_at_5 | bpr-rendle-2009, ncf-he-2017 | bpr-rendle-2009, bpr-rendle-2009, bpr-rendle-2009, bpr-rendle-2009, bpr-rendle-2009 | How does the training objective of BPR differ from the objective used by Neural Collaborative Filtering? |
| H13 | missing_required_source_at_5 | deepar-salinas-2017, conformal-qr-romano-2019 | conformal-qr-romano-2019, conformal-qr-romano-2019, conformal-qr-romano-2019, conformal-qr-romano-2019, conformal-qr-romano-2019 | Which two works in the corpus produce probabilistic outputs, one through a parametric likelihood for count data and one  |

