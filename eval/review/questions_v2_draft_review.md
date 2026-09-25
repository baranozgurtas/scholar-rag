# Review packet: `questions_v2_draft.jsonl`

SHA-256 `b051f2b8d73f7b72` · 35 questions (25 answerable, 10 unanswerable) · review status: unreviewed.

All questions were drafted by an LLM. Quotes and absent terms are machine-checked against the PDF text; that does not establish that a question is well-posed, unambiguous, or that the reference answer is complete. Nothing here is human-reviewed. After review, set `review_status` to `human_reviewed` in the JSONL for the questions you accept (and update `tests/test_eval_harness.py::TestQuestionFiles::test_no_question_claims_human_review`).

Held-out caveat: items marked *outcomes already inspected* (D01-D10 and H01-H12, including revised ones) had their retrieval results looked at before this revision, so the held-out split is not pristine for them. Only the new hard items (D11-D15, H13-H20) have not been run or inspected.

| id | split | type | rev | expected source(s) | inspected | kind |
|---|---|---|---|---|---|---|
| D01 | dev | methodology | 1 | rag-lewis-2020 | yes | answerable |
| D02 | dev | comparison | 2 | colbertv2-santhanam-2022 | yes | answerable |
| D03 | dev | factoid | 1 | lost-in-the-middle-2023 | yes | answerable |
| D04 | dev | methodology | 1 | conformal-qr-romano-2019 | yes | answerable |
| D05 | dev | comparison | 2 | bpr-rendle-2009, ncf-he-2017 | yes | answerable |
| D06 | dev | methodology | 1 | dropout-hinton-2012 | yes | answerable |
| D07 | dev | methodology | 1 | causal-forest-wager-2018 | yes | answerable |
| D08 | dev | near_miss_unanswerable | 1 | — | yes | unanswerable |
| D09 | dev | near_miss_unanswerable | 1 | — | yes | unanswerable |
| D10 | dev | comparison | 2 | ncf-he-2017 | yes | answerable |
| D11 | dev | multi_paper | 1 | rag-lewis-2020, lost-in-the-middle-2023 | no | answerable |
| D12 | dev | factoid | 1 | bge-m3-chen-2024 | no | answerable |
| D13 | dev | methodology | 1 | dropout-hinton-2012 | no | answerable |
| D14 | dev | near_miss_unanswerable | 1 | — | no | unanswerable |
| D15 | dev | near_miss_unanswerable | 1 | — | no | unanswerable |
| H01 | heldout | methodology | 1 | bge-m3-chen-2024 | yes | answerable |
| H02 | heldout | methodology | 1 | ragas-es-2023 | yes | answerable |
| H03 | heldout | methodology | 1 | n-beats-oreshkin-2019 | yes | answerable |
| H04 | heldout | factoid | 1 | deepar-salinas-2017 | yes | answerable |
| H05 | heldout | factoid | 1 | xgboost-chen-2016 | yes | answerable |
| H06 | heldout | methodology | 2 | adam-kingma-2014 | yes | answerable |
| H07 | heldout | reasoning | 2 | data-discontents-paullada-2020 | yes | answerable |
| H08 | heldout | factoid | 1 | ragas-es-2023 | yes | answerable |
| H09 | heldout | near_miss_unanswerable | 1 | — | yes | unanswerable |
| H10 | heldout | near_miss_unanswerable | 1 | — | yes | unanswerable |
| H11 | heldout | near_miss_unanswerable | 1 | — | yes | unanswerable |
| H12 | heldout | false_premise | 2 | — | yes | false premise |
| H13 | heldout | multi_paper | 1 | deepar-salinas-2017, conformal-qr-romano-2019 | no | answerable |
| H14 | heldout | multi_paper | 1 | bpr-rendle-2009, ncf-he-2017 | no | answerable |
| H15 | heldout | methodology | 1 | causal-forest-wager-2018 | no | answerable |
| H16 | heldout | factoid | 1 | colbertv2-santhanam-2022 | no | answerable |
| H17 | heldout | factoid | 1 | n-beats-oreshkin-2019 | no | answerable |
| H18 | heldout | factoid | 1 | xgboost-chen-2016 | no | answerable |
| H19 | heldout | near_miss_unanswerable | 1 | — | no | unanswerable |
| H20 | heldout | near_miss_unanswerable | 1 | — | no | unanswerable |

### D01 · dev · answerable · rev 1 · outcomes already inspected

**Question:** How does the RAG model of Lewis et al. combine parametric and non-parametric memory?

**Expected source(s):** `rag-lewis-2020`

**Reference answer:** The parametric memory is a pre-trained seq2seq transformer and the non-parametric memory is a dense vector index of Wikipedia accessed with a pre-trained neural retriever (DPR); the two are combined in a probabilistic model trained end-to-end.

**Supporting evidence (machine-checked verbatim on the stated page):**
- `rag-lewis-2020` p.2: “the parametric memory is a pre-trained seq2seq transformer, and the non-parametric memory is a dense vector index of Wikipedia, accessed with a pre-trained neural retriever”

**Mechanical checks:** pass

**Notes for the reviewer:**
- Retrieval outcomes for this item were inspected (run retrieval_v2draft_20260925T213410Z_ddadab57) before this revision; not a pristine held-out item.

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### D02 · dev · answerable · rev 2 · outcomes already inspected

**Question:** How does ColBERTv2's late interaction differ from single-vector retrieval models?

**Expected source(s):** `colbertv2-santhanam-2022`

**Reference answer:** Single-vector models encode each query and document into one vector and score relevance with a dot product; late interaction produces multi-vector, token-level representations and decomposes relevance into token-level computations.

**Supporting evidence (machine-checked verbatim on the stated page):**
- `colbertv2-santhanam-2022` p.1: “encode each query and each document into a single high-dimensional vector”
- `colbertv2-santhanam-2022` p.1: “at the granularity of each token and decompose relevance modeling into scalable token-level computations”

**Mechanical checks:** pass

**Notes for the reviewer:**
- Retrieval outcomes for this item were inspected (run retrieval_v2draft_20260925T213410Z_ddadab57) before this revision; not a pristine held-out item.
- Rev 2: added a quote for the late-interaction half (multi-vector, token-level); previously only the single-vector half was quoted.

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### D03 · dev · answerable · rev 1 · outcomes already inspected

**Question:** According to Lost in the Middle, where in the input context is relevant information used worst?

**Expected source(s):** `lost-in-the-middle-2023`

**Reference answer:** In the middle: performance follows a U-shaped curve, highest when relevant information is at the very beginning or end of the context and degrading significantly in the middle.

**Supporting evidence (machine-checked verbatim on the stated page):**
- `lost-in-the-middle-2023` p.2: “we observe a distinctive U-shaped performance curve”

**Mechanical checks:** pass

**Notes for the reviewer:**
- Retrieval outcomes for this item were inspected (run retrieval_v2draft_20260925T213410Z_ddadab57) before this revision; not a pristine held-out item.

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### D04 · dev · answerable · rev 1 · outcomes already inspected

**Question:** How does conformalized quantile regression calibrate its prediction intervals?

**Expected source(s):** `conformal-qr-romano-2019`

**Reference answer:** The training data are split into a proper training set and a calibration set; two quantile regressors are fit on the proper training set, and the calibration set is used to conformalize (and if necessary correct) the interval.

**Supporting evidence (machine-checked verbatim on the stated page):**
- `conformal-qr-romano-2019` p.2: “a proper training set and a calibration set”

**Mechanical checks:** pass

**Notes for the reviewer:**
- Retrieval outcomes for this item were inspected (run retrieval_v2draft_20260925T213410Z_ddadab57) before this revision; not a pristine held-out item.

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### D05 · dev · answerable · rev 2 · outcomes already inspected

**Question:** How does the training objective of BPR differ from the objective used by Neural Collaborative Filtering?

**Expected source(s):** `bpr-rendle-2009`, `ncf-he-2017`

**Reference answer:** BPR optimizes BPR-Opt, a pairwise personalized-ranking criterion derived as a maximum posterior estimator; NCF treats implicit feedback as binary classification and minimizes the pointwise binary cross-entropy (log loss).

**Supporting evidence (machine-checked verbatim on the stated page):**
- `bpr-rendle-2009` p.1: “a generic optimization criterion BPR-Opt for personalized ranking”
- `bpr-rendle-2009` p.1: “that is the maximum posterior estimator”
- `bpr-rendle-2009` p.2: “criterion for personalized ranking that is based on pairs of items”
- `ncf-he-2017` p.4: “it is the same as the binary cross-entropy loss, also known as log loss”
- `ncf-he-2017` p.4: “as a binary classification problem”

**Mechanical checks:** pass

**Notes for the reviewer:**
- Retrieval outcomes for this item were inspected (run retrieval_v2draft_20260925T213410Z_ddadab57) before this revision; not a pristine held-out item.
- Rev 2: evidence now covers every part of the reference: BPR-Opt, maximum posterior estimator, pairs of items (BPR); log loss and binary classification (NCF).

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### D06 · dev · answerable · rev 1 · outcomes already inspected

**Question:** In Hinton et al.'s dropout paper, how is the network used at test time?

**Expected source(s):** `dropout-hinton-2012`

**Reference answer:** They use the 'mean network' containing all hidden units with outgoing weights halved, to compensate for twice as many units being active.

**Supporting evidence (machine-checked verbatim on the stated page):**
- `dropout-hinton-2012` p.2: “with their outgoing weights halved to compensate for the fact that twice as many of them are active”

**Mechanical checks:** pass

**Notes for the reviewer:**
- Retrieval outcomes for this item were inspected (run retrieval_v2draft_20260925T213410Z_ddadab57) before this revision; not a pristine held-out item.

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### D07 · dev · answerable · rev 1 · outcomes already inspected

**Question:** What condition do Wager and Athey call honesty for a tree?

**Expected source(s):** `causal-forest-wager-2018`

**Reference answer:** A tree is honest if, for each training example, it uses the response either to estimate the within-leaf treatment effect or to decide where to place splits, but not both.

**Supporting evidence (machine-checked verbatim on the stated page):**
- `causal-forest-wager-2018` p.8: “it only uses the response Yi to estimate the within-leaf treatment effect”

**Mechanical checks:** pass

**Notes for the reviewer:**
- Retrieval outcomes for this item were inspected (run retrieval_v2draft_20260925T213410Z_ddadab57) before this revision; not a pristine held-out item.

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### D08 · dev · UNANSWERABLE · rev 1 · outcomes already inspected

**Question:** What nDCG@10 does the M3-Embedding paper report on the TREC-COVID subset of BEIR?

**Why unanswerable:** none of “trec-covid” occurs anywhere in `bge-m3-chen-2024` (machine-checked).
**Distractor / note:** TREC-COVID appears in the ColBERTv2 paper, so retrieval will surface a plausible distractor.

**Mechanical checks:** pass

**Notes for the reviewer:**
- Retrieval outcomes for this item were inspected (run retrieval_v2draft_20260925T213410Z_ddadab57) before this revision; not a pristine held-out item.

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### D09 · dev · UNANSWERABLE · rev 1 · outcomes already inspected

**Question:** How much faster does XGBoost train on a GPU than on a CPU, according to the XGBoost paper?

**Why unanswerable:** none of “gpu”, “cuda” occurs anywhere in `xgboost-chen-2016` (machine-checked).
**Distractor / note:** XGBoost paper discusses system speedups (out-of-core, parallel), inviting a wrong transfer.

**Mechanical checks:** pass

**Notes for the reviewer:**
- Retrieval outcomes for this item were inspected (run retrieval_v2draft_20260925T213410Z_ddadab57) before this revision; not a pristine held-out item.

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### D10 · dev · answerable · rev 2 · outcomes already inspected

**Question:** How does the pairwise-ranking matrix factorization baseline BPR compare with NeuMF on the MovieLens and Pinterest recommendation experiments?

**Expected source(s):** `ncf-he-2017`

**Reference answer:** NCF reports that NeuMF significantly outperforms BPR on both datasets, with an average relative improvement of 4.9% over BPR (4.5% over eALS). BPR's per-dataset HR@10 and NDCG@10 values are shown only in NCF's Figure 4, not in the text.

**Supporting evidence (machine-checked verbatim on the stated page):**
- `ncf-he-2017` p.6: “the relative improvement over eALS and BPR is 4.5% and 4.9%, respectively”
- `ncf-he-2017` p.6: “Figure 4 shows the performance of HR@10 and NDCG@10 with respect to the number of predictive factors”

**Mechanical checks:** pass

**Notes for the reviewer:**
- Retrieval outcomes for this item were inspected (run retrieval_v2draft_20260925T213410Z_ddadab57) before this revision; not a pristine held-out item.
- Rev 2: was 'What hit ratio does BPR achieve on MovieLens in the original BPR paper?' labelled unanswerable. NCF reports BPR as a MovieLens baseline, so that label was contestable; replaced by an answerable comparison grounded in NCF's text. The exact BPR hit ratio is only in a figure, so the question no longer asks for it.

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### D11 · dev · answerable · rev 1 · not yet inspected

**Question:** In the corpus's studies of retrieval-augmented question answering, how does retrieving more documents affect open-domain QA accuracy?

**Expected source(s):** `rag-lewis-2020`, `lost-in-the-middle-2023`

**Reference answer:** Two papers address it. In RAG, retrieving more documents at test time monotonically improves open-domain QA for RAG-Sequence, while RAG-Token peaks at 10 documents. Lost in the Middle finds reader performance saturates long before retriever recall: using 50 instead of 20 documents helps only marginally.

**Supporting evidence (machine-checked verbatim on the stated page):**
- `rag-lewis-2020` p.8: “retrieving more documents at test time monotonically improves Open-domain QA results for RAG-Sequence, but performance peaks for RAG-Token at 10 retrieved documents”
- `lost-in-the-middle-2023` p.3: “saturates long before retriever recall”
- `lost-in-the-middle-2023` p.3: “using 50 documents instead of 20 retrieved documents only marginally”

**Mechanical checks:** pass

**Notes for the reviewer:**
- Close distractor(s): ColBERTv2 and BGE-M3 (retrieval quality, not reader accuracy).

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### D12 · dev · answerable · rev 1 · not yet inspected

**Question:** Which embedding model in the corpus uses the [CLS] embedding for dense retrieval and the other tokens' embeddings for sparse and multi-vector retrieval?

**Expected source(s):** `bge-m3-chen-2024`

**Reference answer:** M3-Embedding (BGE-M3).

**Supporting evidence (machine-checked verbatim on the stated page):**
- `bge-m3-chen-2024` p.2: “In M3-Embedding, the [CLS] embedding is used for dense retrieval”

**Mechanical checks:** pass

**Notes for the reviewer:**
- Close distractor(s): ColBERTv2 (multi-vector late interaction).

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### D13 · dev · answerable · rev 1 · not yet inspected

**Question:** Which regularization method randomly omits each hidden unit with probability 0.5 on each training case, and why?

**Expected source(s):** `dropout-hinton-2012`

**Reference answer:** Dropout (Hinton et al. 2012): each hidden unit is omitted with probability 0.5 on each presentation of each training case, so a unit cannot rely on other hidden units being present, preventing complex co-adaptations.

**Supporting evidence (machine-checked verbatim on the stated page):**
- `dropout-hinton-2012` p.1: “to prevent complex co-adaptations on the training data”
- `dropout-hinton-2012` p.1: “each hidden unit is randomly omitted from the network with a probability of 0.5, so a hidden unit cannot rely on other hidden units being present”

**Mechanical checks:** pass

**Notes for the reviewer:**
- Close distractor(s): Adam (uses dropout noise in its experiments).

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### D14 · dev · UNANSWERABLE · rev 1 · not yet inspected

**Question:** By how much does the autoregressive recurrent probabilistic forecaster improve on the winner of the M4 competition?

**Why unanswerable:** none of “m4” occurs anywhere in `deepar-salinas-2017` (machine-checked).

**Mechanical checks:** pass

**Notes for the reviewer:**
- Why unanswerable: DeepAR never mentions M4; N-BEATS (a different forecaster) reports improving on the M4 winner, a close distractor.

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### D15 · dev · UNANSWERABLE · rev 1 · not yet inspected

**Question:** What coverage does conformalized quantile regression achieve on the M4 forecasting competition data?

**Why unanswerable:** none of “m4” occurs anywhere in `conformal-qr-romano-2019` (machine-checked).

**Mechanical checks:** pass

**Notes for the reviewer:**
- Why unanswerable: The CQR paper does not use M4; only N-BEATS does.

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### H01 · heldout · answerable · rev 1 · outcomes already inspected

**Question:** In M3-Embedding, what is used as the teacher signal for self-knowledge distillation?

**Expected source(s):** `bge-m3-chen-2024`

**Reference answer:** The relevance scores from the different retrieval functions (dense, sparse/lexical, multi-vector) are integrated and used as the teacher signal.

**Supporting evidence (machine-checked verbatim on the stated page):**
- `bge-m3-chen-2024` p.2: “we integrate the relevance scores from different retrieval functions as the teacher signal”

**Mechanical checks:** pass

**Notes for the reviewer:**
- Retrieval outcomes for this item were inspected (run retrieval_v2draft_20260925T213410Z_ddadab57) before this revision; not a pristine held-out item.

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### H02 · heldout · answerable · rev 1 · outcomes already inspected

**Question:** How does Ragas estimate faithfulness before verifying the answer against the context?

**Expected source(s):** `ragas-es-2023`

**Reference answer:** It first uses an LLM to extract a set of statements from the answer, decomposing longer sentences into shorter, more focused assertions, which are then checked against the context.

**Supporting evidence (machine-checked verbatim on the stated page):**
- `ragas-es-2023` p.3: “we first use an LLM to extract a set of statements”

**Mechanical checks:** pass

**Notes for the reviewer:**
- Retrieval outcomes for this item were inspected (run retrieval_v2draft_20260925T213410Z_ddadab57) before this revision; not a pristine held-out item.

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### H03 · heldout · answerable · rev 1 · outcomes already inspected

**Question:** What does each N-BEATS block predict?

**Expected source(s):** `n-beats-oreshkin-2019`

**Reference answer:** Each block is a multi-layer fully connected network with ReLU nonlinearities that predicts basis expansion coefficients both forward (forecast) and backward (backcast).

**Supporting evidence (machine-checked verbatim on the stated page):**
- `n-beats-oreshkin-2019` p.3: “It predicts basis expansion coefficients both forward”

**Mechanical checks:** pass

**Notes for the reviewer:**
- Retrieval outcomes for this item were inspected (run retrieval_v2draft_20260925T213410Z_ddadab57) before this revision; not a pristine held-out item.

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### H04 · heldout · answerable · rev 1 · outcomes already inspected

**Question:** Which likelihood does DeepAR propose for count data?

**Expected source(s):** `deepar-salinas-2017`

**Reference answer:** A negative Binomial likelihood.

**Supporting evidence (machine-checked verbatim on the stated page):**
- `deepar-salinas-2017` p.2: “incorporating a negative Binomial likelihood for count data”

**Mechanical checks:** pass

**Notes for the reviewer:**
- Retrieval outcomes for this item were inspected (run retrieval_v2draft_20260925T213410Z_ddadab57) before this revision; not a pristine held-out item.

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### H05 · heldout · answerable · rev 1 · outcomes already inspected

**Question:** What regularization term does XGBoost add to its objective?

**Expected source(s):** `xgboost-chen-2016`

**Reference answer:** Omega(f) = gamma*T + (1/2)*lambda*||w||^2, penalizing the number of leaves T and the squared leaf weights, which smooths the learnt weights to avoid over-fitting.

**Supporting evidence (machine-checked verbatim on the stated page):**
- `xgboost-chen-2016` p.2: “penalizes the complexity of the model”

**Mechanical checks:** pass

**Notes for the reviewer:**
- Retrieval outcomes for this item were inspected (run retrieval_v2draft_20260925T213410Z_ddadab57) before this revision; not a pristine held-out item.

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### H06 · heldout · answerable · rev 2 · outcomes already inspected

**Question:** How does Adam correct the bias of its moment estimates?

**Expected source(s):** `adam-kingma-2014`

**Reference answer:** It divides the first moment estimate by (1 - beta1^t) and the second raw moment estimate by (1 - beta2^t) to obtain bias-corrected estimates used in the update.

**Supporting evidence (machine-checked verbatim on the stated page):**
- `adam-kingma-2014` p.2: “Compute bias-corrected first moment estimate”
- `adam-kingma-2014` p.2: “Compute bias-corrected second raw moment estimate”

**Mechanical checks:** pass

**Notes for the reviewer:**
- Retrieval outcomes for this item were inspected (run retrieval_v2draft_20260925T213410Z_ddadab57) before this revision; not a pristine held-out item.
- Rev 2: added the second-moment correction line. The divisors (1 - beta^t) are in the algorithm box on p.2 but extracted as garbled math, so they are not quoted.

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### H07 · heldout · answerable · rev 2 · outcomes already inspected

**Question:** According to the Paullada et al. survey, what happened when ImageNet was re-created following its original construction process?

**Expected source(s):** `data-discontents-paullada-2020`

**Reference answer:** The newly constructed dataset had different distributional properties; the differences were largely due to how ground-truth labels were built from multiple annotations, and different inter-annotator agreement thresholds produced vastly different datasets.

**Supporting evidence (machine-checked verbatim on the stated page):**
- `data-discontents-paullada-2020` p.5: “the newly constructed dataset was found to have different distributional properties”
- `data-discontents-paullada-2020` p.5: “The differences were largely localized to variations in constructing ground truth labels from multiple annotations”
- `data-discontents-paullada-2020` p.5: “different thresholds for inter-annotator agreement were found to produce vastly different datasets”

**Mechanical checks:** pass

**Notes for the reviewer:**
- Retrieval outcomes for this item were inspected (run retrieval_v2draft_20260925T213410Z_ddadab57) before this revision; not a pristine held-out item.
- Rev 2: question now names ImageNet (the passage is about Recht et al. 2019 re-creating ImageNet); evidence covers every clause of the reference.

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### H08 · heldout · answerable · rev 1 · outcomes already inspected

**Question:** How often did the two WikiEval annotators in the Ragas paper agree?

**Expected source(s):** `ragas-es-2023`

**Reference answer:** About 95% of cases for faithfulness and context relevance, and about 90% for answer relevance.

**Supporting evidence (machine-checked verbatim on the stated page):**
- `ragas-es-2023` p.4: “the two annotators agreed in around 95% of cases”

**Mechanical checks:** pass

**Notes for the reviewer:**
- Retrieval outcomes for this item were inspected (run retrieval_v2draft_20260925T213410Z_ddadab57) before this revision; not a pristine held-out item.

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### H09 · heldout · UNANSWERABLE · rev 1 · outcomes already inspected

**Question:** What coverage does conformalized quantile regression achieve on ImageNet?

**Why unanswerable:** none of “imagenet” occurs anywhere in `conformal-qr-romano-2019` (machine-checked).
**Distractor / note:** ImageNet occurs in other corpus papers (Adam, dropout).

**Mechanical checks:** pass

**Notes for the reviewer:**
- Retrieval outcomes for this item were inspected (run retrieval_v2draft_20260925T213410Z_ddadab57) before this revision; not a pristine held-out item.

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### H10 · heldout · UNANSWERABLE · rev 1 · outcomes already inspected

**Question:** How much does GPU acceleration speed up causal forest training in Wager and Athey's experiments?

**Why unanswerable:** none of “gpu” occurs anywhere in `causal-forest-wager-2018` (machine-checked).
**Distractor / note:** On-topic paper, absent detail.

**Mechanical checks:** pass

**Notes for the reviewer:**
- Retrieval outcomes for this item were inspected (run retrieval_v2draft_20260925T213410Z_ddadab57) before this revision; not a pristine held-out item.

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### H11 · heldout · UNANSWERABLE · rev 1 · outcomes already inspected

**Question:** How does dropout compare with batch normalization on CIFAR-10 in Hinton et al.'s dropout paper?

**Why unanswerable:** none of “batch normalization”, “batchnorm” occurs anywhere in `dropout-hinton-2012` (machine-checked).
**Distractor / note:** Batch normalization is mentioned in DeepAR, a distractor.

**Mechanical checks:** pass

**Notes for the reviewer:**
- Retrieval outcomes for this item were inspected (run retrieval_v2draft_20260925T213410Z_ddadab57) before this revision; not a pristine held-out item.

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### H12 · heldout · FALSE PREMISE · rev 2 · outcomes already inspected

**Question:** Which Claude 3 models are evaluated in the Lost in the Middle paper, and how do they perform?

**Why unanswerable:** none of “claude 3”, “claude-3” occurs anywhere in the whole corpus (machine-checked).
**Distractor / note:** The paper evaluates Claude-1.3; a released answer that says so is scored as premise_corrected (see premise_correction), not as a false answer.

**Premise correction (scored separately):** acceptable answer: The paper does not evaluate Claude 3; the Anthropic model it evaluates is Claude-1.3. Counted as corrected if released and it mentions any of “claude-1.3”, “claude 1.3”.
- `lost-in-the-middle-2023` p.2: “Anthropic’s Claude-1.3”

**Mechanical checks:** pass

**Notes for the reviewer:**
- Retrieval outcomes for this item were inspected (run retrieval_v2draft_20260925T213410Z_ddadab57) before this revision; not a pristine held-out item.
- Rev 2: false-premise item. Abstaining is correct; a released answer that corrects the premise (mentions Claude-1.3) is scored as premise_corrected, not as a false answer; any other released answer is an unsupported (false) answer.

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### H13 · heldout · answerable · rev 1 · not yet inspected

**Question:** Which two works in the corpus produce probabilistic outputs, one through a parametric likelihood for count data and one through prediction intervals, and what guarantee does the interval method give?

**Expected source(s):** `deepar-salinas-2017`, `conformal-qr-romano-2019`

**Reference answer:** DeepAR uses a negative binomial likelihood for count data; conformalized quantile regression builds prediction intervals whose conformalized version is guaranteed to satisfy the coverage requirement regardless of the quantile regression estimator.

**Supporting evidence (machine-checked verbatim on the stated page):**
- `deepar-salinas-2017` p.2: “incorporating a negative Binomial likelihood for count data”
- `conformal-qr-romano-2019` p.2: “the conformalized prediction interval is guaranteed to satisfy the coverage requirement”

**Mechanical checks:** pass

**Notes for the reviewer:**
- Close distractor(s): N-BEATS (point forecasts), causal forest (confidence intervals for treatment effects).

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### H14 · heldout · answerable · rev 1 · not yet inspected

**Question:** Among the corpus's recommendation papers, which trains on item pairs and which treats implicit feedback as binary classification, and how does each obtain its negative examples?

**Expected source(s):** `bpr-rendle-2009`, `ncf-he-2017`

**Reference answer:** BPR optimizes a pairwise criterion using item pairs, trained by LearnBPR with bootstrap sampling of training triples. NCF casts implicit feedback as binary classification with log loss and uniformly samples negatives from unobserved interactions in each iteration.

**Supporting evidence (machine-checked verbatim on the stated page):**
- `bpr-rendle-2009` p.3: “using item pairs as training data”
- `bpr-rendle-2009` p.5: “we propose LearnBPR, a stochastic gradient-descent algorithm based on bootstrap sampling of training triples”
- `ncf-he-2017` p.4: “as a binary classification problem”
- `ncf-he-2017` p.4: “also known as log loss”
- `ncf-he-2017` p.4: “we uniformly sample them from unobserved”

**Mechanical checks:** pass

**Notes for the reviewer:**
- Close distractor(s): Each paper describes the other's family of methods (NCF lists BPR as a baseline).

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### H15 · heldout · answerable · rev 1 · not yet inspected

**Question:** Which method grows each tree on one subsample while estimating the leaf predictions on a different subsample, and what is the purpose?

**Expected source(s):** `causal-forest-wager-2018`

**Reference answer:** Causal forests (Wager & Athey): trees grown this way are called 'honest', a condition used to reduce bias.

**Supporting evidence (machine-checked verbatim on the stated page):**
- `causal-forest-wager-2018` p.2: “the tree is grown using one subsample, while the predictions at the leaves of the tree are estimated using a”
- `causal-forest-wager-2018` p.2: “a condition we call “honesty” to reduce bias”

**Mechanical checks:** pass

**Notes for the reviewer:**
- Close distractor(s): XGBoost (tree boosting with subsampling).

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### H16 · heldout · answerable · rev 1 · not yet inspected

**Question:** What mechanism reduces the space footprint of token-level late-interaction indexes by 6 to 10 times?

**Expected source(s):** `colbertv2-santhanam-2022`

**Reference answer:** ColBERTv2's residual compression mechanism.

**Supporting evidence (machine-checked verbatim on the stated page):**
- `colbertv2-santhanam-2022` p.2: “uses a residual compression mechanism (§3.3) to reduce the space footprint of late interaction by 6–10× while preserving quality”

**Mechanical checks:** pass

**Notes for the reviewer:**
- Close distractor(s): BGE-M3 (also produces multi-vector representations).

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### H17 · heldout · answerable · rev 1 · not yet inspected

**Question:** Which forecasting model reports improving forecast accuracy by 11% over a statistical benchmark and by 3% over the previous M4 winner?

**Expected source(s):** `n-beats-oreshkin-2019`

**Reference answer:** N-BEATS.

**Supporting evidence (machine-checked verbatim on the stated page):**
- `n-beats-oreshkin-2019` p.1: “improving forecast accuracy by 11% over a statistical benchmark and by 3% over last year’s winner of the M4 competition”

**Mechanical checks:** pass

**Notes for the reviewer:**
- Close distractor(s): DeepAR (probabilistic forecasting, reports accuracy improvements on other data).

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### H18 · heldout · answerable · rev 1 · not yet inspected

**Question:** Which system proposes a sparsity-aware split-finding algorithm and a weighted quantile sketch for approximate tree learning?

**Expected source(s):** `xgboost-chen-2016`

**Reference answer:** XGBoost.

**Supporting evidence (machine-checked verbatim on the stated page):**
- `xgboost-chen-2016` p.1: “We propose a novel sparsity-aware algorithm for sparse data”
- `xgboost-chen-2016` p.4: “Weighted Quantile Sketch”

**Mechanical checks:** pass

**Notes for the reviewer:**
- Close distractor(s): Causal forests (tree ensembles).

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### H19 · heldout · UNANSWERABLE · rev 1 · not yet inspected

**Question:** Which negative binomial likelihood does the doubly residual basis-expansion forecaster use for count data?

**Why unanswerable:** none of “negative binomial”, “binomial” occurs anywhere in `n-beats-oreshkin-2019` (machine-checked).

**Mechanical checks:** pass

**Notes for the reviewer:**
- Why unanswerable: N-BEATS makes point forecasts and never mentions a binomial likelihood; DeepAR does, a close distractor.

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### H20 · heldout · UNANSWERABLE · rev 1 · not yet inspected

**Question:** What error does the honest causal forest achieve on the IHDP benchmark?

**Why unanswerable:** none of “ihdp” occurs anywhere in the whole corpus (machine-checked).

**Mechanical checks:** pass

**Notes for the reviewer:**
- Why unanswerable: No corpus paper mentions IHDP; the causal-forest paper evaluates on its own simulations.

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______
