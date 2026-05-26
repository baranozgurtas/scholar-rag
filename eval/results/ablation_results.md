# Retrieval Ablation Results

| Config | Description | Hit@5 | Hit@10 | MRR@10 | nDCG@10 | Abstain% | Adv.Abstain% |
|---|---|---|---|---|---|---|---|
| `A_dense_only` | Dense-only baseline | 0.950 | 0.950 | 0.925 | 0.932 | 32.0% | 1.000 |
| `B_dense_plus_rerank` | Dense + reranker | 1.000 | 1.000 | 0.967 | 0.975 | 24.0% | 1.000 |
| `C_hybrid_no_rerank` | Hybrid (dense+sparse, RRF) | 0.950 | 0.950 | 0.950 | 0.950 | 32.0% | 1.000 |
| `D_hybrid_plus_rerank` | Hybrid + reranker (production) | 0.950 | 0.950 | 0.950 | 0.950 | 20.0% | 1.000 |

**Production (D) vs Dense-only baseline (A):**
- **Hit@5**: 0.950 → 0.950  (**+0.0%**)
- **Hit@10**: 0.950 → 0.950  (**+0.0%**)
- **MRR@10**: 0.925 → 0.950  (**+2.7%**)
- **nDCG@10**: 0.932 → 0.950  (**+2.0%**)