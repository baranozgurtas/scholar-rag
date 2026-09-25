# Legacy results, re-analysed

Re-analysis of the committed 25-question results (`eval/results/ablation_*.json`,
generated on a Colab T4 before this harness existed) with the current metric
definitions. No model was re-run. Limitations:

- Only the 5 chunks sent to the LLM were stored, so MRR@10 / nDCG@10 are really @5
  and Hit@10 = Hit@5.
- `abstained` was a substring test; outputs containing the abstention sentence plus
  other text counted as abstentions. Answer text was not stored.
- Citation tag validity cannot be recomputed (allowed tags not stored).
- The 20 answerable questions are LLM-generated and not human-reviewed; the 5
  unanswerable ones are obvious out-of-domain questions. Small n: see the 95% CIs.
- The pre-generation rerank threshold was 0.0 (disabled). Every abstention in these
  runs came from the generator emitting the abstention sentence.

## Retrieval (answerable questions)

| Config | n | Hit@5 | MRR@10 | nDCG@10 | Note |
|---|---|---|---|---|---|
| `A_dense_only` | 20 | 0.950 | 0.925 | 0.932 | Only 5 ranked sources were recorded per question, so the @10 metrics are effectively @5. |
| `B_dense_plus_rerank` | 20 | 1.000 | 0.967 | 0.975 | Only 5 ranked sources were recorded per question, so the @10 metrics are effectively @5. |
| `C_hybrid_no_rerank` | 20 | 0.950 | 0.950 | 0.950 | Only 5 ranked sources were recorded per question, so the @10 metrics are effectively @5. |
| `D_hybrid_plus_rerank` | 20 | 0.950 | 0.950 | 0.950 | Only 5 ranked sources were recorded per question, so the @10 metrics are effectively @5. |

## Abstention

| Config | False abstention (answerable) | False answer (unanswerable) | Errors |
|---|---|---|---|
| `A_dense_only` | 3/20 (15.0%; 95% CI 5%-36%) | 0/5 (0.0%; 95% CI 0%-43%) | 0 |
| `B_dense_plus_rerank` | 1/20 (5.0%; 95% CI 1%-24%) | 0/5 (0.0%; 95% CI 0%-43%) | 0 |
| `C_hybrid_no_rerank` | 3/20 (15.0%; 95% CI 5%-36%) | 0/5 (0.0%; 95% CI 0%-43%) | 0 |
| `D_hybrid_plus_rerank` | 0/20 (0.0%; 95% CI 0%-16%) | 0/5 (0.0%; 95% CI 0%-43%) | 0 |

## Citations (tag validity is structural, not factual grounding)

| Config | Released answers | Released with all tags valid | Tag validity (all generated tags) | Withheld by guard |
|---|---|---|---|---|
| `A_dense_only` | 17 | n/a (n=0) | n/a (n=0) | not recorded (no guard) |
| `B_dense_plus_rerank` | 19 | n/a (n=0) | n/a (n=0) | not recorded (no guard) |
| `C_hybrid_no_rerank` | 17 | n/a (n=0) | n/a (n=0) | not recorded (no guard) |
| `D_hybrid_plus_rerank` | 20 | n/a (n=0) | n/a (n=0) | not recorded (no guard) |

Factual grounding: Not measured by this harness. Citation tag validity only shows that a cited (title, page) was among the supplied passages, not that the passage supports the claim. Use `python -m eval.ragas_eval` (LLM judge) or human review for grounding.

## Latency

| Config | n | p50 total (s) | p95 total (s) |
|---|---|---|---|
| `A_dense_only` | 25 | 5.7 | 8.9 |
| `B_dense_plus_rerank` | 25 | 6.6 | 9.8 |
| `C_hybrid_no_rerank` | 25 | 5.6 | 9.3 |
| `D_hybrid_plus_rerank` | 25 | 6.8 | 10.3 |

## Per-question failures

### `A_dense_only`

| id | failures | expected | top-5 sources | question |
|---|---|---|---|---|
| L04 | retrieval_miss_at_5, false_abstention:legacy_abstained_substring | bpr-rendle-2009 | n-beats-oreshkin-2019, xgboost-chen-2016, xgboost-chen-2016, xgboost-chen-2016, n-beats-oreshkin-2019 | What are the two datasets used in the evaluation, and what do they contain? |
| L07 | false_abstention:legacy_abstained_substring | colbertv2-santhanam-2022 | lost-in-the-middle-2023, colbertv2-santhanam-2022, bge-m3-chen-2024, bge-m3-chen-2024, bge-m3-chen-2024 | What metric does the paper use to evaluate retrieval quality, and how is it defined? |
| L15 | false_abstention:legacy_abstained_substring | n-beats-oreshkin-2019 | n-beats-oreshkin-2019, n-beats-oreshkin-2019, n-beats-oreshkin-2019, n-beats-oreshkin-2019, n-beats-oreshkin-2019 | What is the performance improvement of N-BEATS over a statistical benchmark according to the paper's abstract? |

### `B_dense_plus_rerank`

| id | failures | expected | top-5 sources | question |
|---|---|---|---|---|
| L18 | false_abstention:legacy_abstained_substring | rag-lewis-2020 | rag-lewis-2020, rag-lewis-2020, rag-lewis-2020, rag-lewis-2020, rag-lewis-2020 | What metric is used to evaluate the Jeopardy question generation task, and what makes it suitable for this task compared |

### `C_hybrid_no_rerank`

| id | failures | expected | top-5 sources | question |
|---|---|---|---|---|
| L04 | retrieval_miss_at_5, false_abstention:legacy_abstained_substring | bpr-rendle-2009 | xgboost-chen-2016, xgboost-chen-2016, data-discontents-paullada-2020, deepar-salinas-2017, ncf-he-2017 | What are the two datasets used in the evaluation, and what do they contain? |
| L15 | false_abstention:legacy_abstained_substring | n-beats-oreshkin-2019 | n-beats-oreshkin-2019, n-beats-oreshkin-2019, n-beats-oreshkin-2019, n-beats-oreshkin-2019, n-beats-oreshkin-2019 | What is the performance improvement of N-BEATS over a statistical benchmark according to the paper's abstract? |
| L18 | false_abstention:legacy_abstained_substring | rag-lewis-2020 | rag-lewis-2020, rag-lewis-2020, rag-lewis-2020, rag-lewis-2020, rag-lewis-2020 | What metric is used to evaluate the Jeopardy question generation task, and what makes it suitable for this task compared |

### `D_hybrid_plus_rerank`

| id | failures | expected | top-5 sources | question |
|---|---|---|---|---|
| L04 | retrieval_miss_at_5 | bpr-rendle-2009 | xgboost-chen-2016, deepar-salinas-2017, ncf-he-2017, colbertv2-santhanam-2022, deepar-salinas-2017 | What are the two datasets used in the evaluation, and what do they contain? |

## Legacy-only observations

| Config | Unanswerable 'abstained' but emitted citation tags | Answerable released with 0 tags |
|---|---|---|
| `A_dense_only` | 3/5 | 1 |
| `B_dense_plus_rerank` | 2/5 | 3 |
| `C_hybrid_no_rerank` | 0/5 | 2 |
| `D_hybrid_plus_rerank` | 3/5 | 2 |
