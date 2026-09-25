# Review packet: `questions_v2_draft.jsonl`

SHA-256 `ac4853c50364ecd4` · 22 questions (15 answerable, 7 unanswerable) · review status: unreviewed.

All questions were drafted by an LLM. Quotes and absent terms are machine-checked against the PDF text; that does not establish that a question is well-posed, unambiguous, or that the reference answer is complete. Nothing here is human-reviewed. After review, set `review_status` to `human_reviewed` in the JSONL for the questions you accept (and update `tests/test_eval_harness.py::TestQuestionFiles::test_no_question_claims_human_review`).

| id | split | answerable | expected source(s) | open concerns |
|---|---|---|---|---|
| D01 | dev | yes | rag-lewis-2020 |  |
| D02 | dev | yes | colbertv2-santhanam-2022 | 1 |
| D03 | dev | yes | lost-in-the-middle-2023 |  |
| D04 | dev | yes | conformal-qr-romano-2019 |  |
| D05 | dev | yes | bpr-rendle-2009, ncf-he-2017 | 1 |
| D06 | dev | yes | dropout-hinton-2012 |  |
| D07 | dev | yes | causal-forest-wager-2018 |  |
| D08 | dev | no | — |  |
| D09 | dev | no | — |  |
| D10 | dev | no | — | 1 |
| H01 | heldout | yes | bge-m3-chen-2024 |  |
| H02 | heldout | yes | ragas-es-2023 |  |
| H03 | heldout | yes | n-beats-oreshkin-2019 |  |
| H04 | heldout | yes | deepar-salinas-2017 |  |
| H05 | heldout | yes | xgboost-chen-2016 |  |
| H06 | heldout | yes | adam-kingma-2014 | 1 |
| H07 | heldout | yes | data-discontents-paullada-2020 | 1 |
| H08 | heldout | yes | ragas-es-2023 |  |
| H09 | heldout | no | — |  |
| H10 | heldout | no | — |  |
| H11 | heldout | no | — |  |
| H12 | heldout | no | — | 1 |

### D01 · dev · answerable

**Question:** How does the RAG model of Lewis et al. combine parametric and non-parametric memory?

**Expected source(s):** `rag-lewis-2020`

**Reference answer:** The parametric memory is a pre-trained seq2seq transformer and the non-parametric memory is a dense vector index of Wikipedia accessed with a pre-trained neural retriever (DPR); the two are combined in a probabilistic model trained end-to-end.

**Supporting evidence (machine-checked verbatim on the stated page):**
- `rag-lewis-2020` p.2: “the parametric memory is a pre-trained seq2seq transformer, and the non-parametric memory is a dense vector index of Wikipedia, accessed with a pre-trained neural retriever”

**Mechanical checks:** pass

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### D02 · dev · answerable

**Question:** How does ColBERTv2's late interaction differ from single-vector retrieval models?

**Expected source(s):** `colbertv2-santhanam-2022`

**Reference answer:** Single-vector models encode each query and document into one vector and score relevance with a dot product; late interaction produces multi-vector, token-level representations and decomposes relevance into token-level computations.

**Supporting evidence (machine-checked verbatim on the stated page):**
- `colbertv2-santhanam-2022` p.1: “encode each query and each document into a single high-dimensional vector”

**Mechanical checks:** pass

**Open concerns:**
- Evidence quote covers only the single-vector half of the comparison; the late-interaction half is on the same page but split by a line-break hyphen ('multi-vector repre- sentations'), so it was not quoted verbatim.

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### D03 · dev · answerable

**Question:** According to Lost in the Middle, where in the input context is relevant information used worst?

**Expected source(s):** `lost-in-the-middle-2023`

**Reference answer:** In the middle: performance follows a U-shaped curve, highest when relevant information is at the very beginning or end of the context and degrading significantly in the middle.

**Supporting evidence (machine-checked verbatim on the stated page):**
- `lost-in-the-middle-2023` p.2: “we observe a distinctive U-shaped performance curve”

**Mechanical checks:** pass

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### D04 · dev · answerable

**Question:** How does conformalized quantile regression calibrate its prediction intervals?

**Expected source(s):** `conformal-qr-romano-2019`

**Reference answer:** The training data are split into a proper training set and a calibration set; two quantile regressors are fit on the proper training set, and the calibration set is used to conformalize (and if necessary correct) the interval.

**Supporting evidence (machine-checked verbatim on the stated page):**
- `conformal-qr-romano-2019` p.2: “a proper training set and a calibration set”

**Mechanical checks:** pass

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### D05 · dev · answerable

**Question:** How does the training objective of BPR differ from the objective used by Neural Collaborative Filtering?

**Expected source(s):** `bpr-rendle-2009`, `ncf-he-2017`

**Reference answer:** BPR optimizes BPR-Opt, a pairwise personalized-ranking criterion derived as a maximum posterior estimator; NCF treats implicit feedback as binary classification and minimizes the pointwise binary cross-entropy (log loss).

**Supporting evidence (machine-checked verbatim on the stated page):**
- `bpr-rendle-2009` p.1: “a generic optimization criterion BPR-Opt for personalized ranking”
- `ncf-he-2017` p.4: “it is the same as the binary cross-entropy loss, also known as log loss”

**Mechanical checks:** pass

**Open concerns:**
- Needs two papers; check both quotes support the stated contrast.

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### D06 · dev · answerable

**Question:** In Hinton et al.'s dropout paper, how is the network used at test time?

**Expected source(s):** `dropout-hinton-2012`

**Reference answer:** They use the 'mean network' containing all hidden units with outgoing weights halved, to compensate for twice as many units being active.

**Supporting evidence (machine-checked verbatim on the stated page):**
- `dropout-hinton-2012` p.2: “with their outgoing weights halved to compensate for the fact that twice as many of them are active”

**Mechanical checks:** pass

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### D07 · dev · answerable

**Question:** What condition do Wager and Athey call honesty for a tree?

**Expected source(s):** `causal-forest-wager-2018`

**Reference answer:** A tree is honest if, for each training example, it uses the response either to estimate the within-leaf treatment effect or to decide where to place splits, but not both.

**Supporting evidence (machine-checked verbatim on the stated page):**
- `causal-forest-wager-2018` p.8: “it only uses the response Yi to estimate the within-leaf treatment effect”

**Mechanical checks:** pass

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### D08 · dev · UNANSWERABLE

**Question:** What nDCG@10 does the M3-Embedding paper report on the TREC-COVID subset of BEIR?

**Why unanswerable:** none of “trec-covid” occurs anywhere in `bge-m3-chen-2024` (machine-checked).
**Distractor / note:** TREC-COVID appears in the ColBERTv2 paper, so retrieval will surface a plausible distractor.

**Mechanical checks:** pass

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### D09 · dev · UNANSWERABLE

**Question:** How much faster does XGBoost train on a GPU than on a CPU, according to the XGBoost paper?

**Why unanswerable:** none of “gpu”, “cuda” occurs anywhere in `xgboost-chen-2016` (machine-checked).
**Distractor / note:** XGBoost paper discusses system speedups (out-of-core, parallel), inviting a wrong transfer.

**Mechanical checks:** pass

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### D10 · dev · UNANSWERABLE

**Question:** What hit ratio does BPR achieve on the MovieLens dataset in the original BPR paper?

**Why unanswerable:** none of “movielens” occurs anywhere in `bpr-rendle-2009` (machine-checked).
**Distractor / note:** NCF reports hit ratio on MovieLens and compares to BPR, a strong distractor.

**Mechanical checks:** pass

**Open concerns:**
- Ambiguous: NCF (ncf-he-2017) reports BPR's hit ratio on MovieLens as a baseline. The question restricts to 'the original BPR paper', so it is labelled unanswerable, but an answer citing NCF's BPR baseline is defensible. Decide: keep as unanswerable, relabel as answerable from NCF, or rephrase.

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### H01 · heldout · answerable

**Question:** In M3-Embedding, what is used as the teacher signal for self-knowledge distillation?

**Expected source(s):** `bge-m3-chen-2024`

**Reference answer:** The relevance scores from the different retrieval functions (dense, sparse/lexical, multi-vector) are integrated and used as the teacher signal.

**Supporting evidence (machine-checked verbatim on the stated page):**
- `bge-m3-chen-2024` p.2: “we integrate the relevance scores from different retrieval functions as the teacher signal”

**Mechanical checks:** pass

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### H02 · heldout · answerable

**Question:** How does Ragas estimate faithfulness before verifying the answer against the context?

**Expected source(s):** `ragas-es-2023`

**Reference answer:** It first uses an LLM to extract a set of statements from the answer, decomposing longer sentences into shorter, more focused assertions, which are then checked against the context.

**Supporting evidence (machine-checked verbatim on the stated page):**
- `ragas-es-2023` p.3: “we first use an LLM to extract a set of statements”

**Mechanical checks:** pass

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### H03 · heldout · answerable

**Question:** What does each N-BEATS block predict?

**Expected source(s):** `n-beats-oreshkin-2019`

**Reference answer:** Each block is a multi-layer fully connected network with ReLU nonlinearities that predicts basis expansion coefficients both forward (forecast) and backward (backcast).

**Supporting evidence (machine-checked verbatim on the stated page):**
- `n-beats-oreshkin-2019` p.3: “It predicts basis expansion coefficients both forward”

**Mechanical checks:** pass

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### H04 · heldout · answerable

**Question:** Which likelihood does DeepAR propose for count data?

**Expected source(s):** `deepar-salinas-2017`

**Reference answer:** A negative Binomial likelihood.

**Supporting evidence (machine-checked verbatim on the stated page):**
- `deepar-salinas-2017` p.2: “incorporating a negative Binomial likelihood for count data”

**Mechanical checks:** pass

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### H05 · heldout · answerable

**Question:** What regularization term does XGBoost add to its objective?

**Expected source(s):** `xgboost-chen-2016`

**Reference answer:** Omega(f) = gamma*T + (1/2)*lambda*||w||^2, penalizing the number of leaves T and the squared leaf weights, which smooths the learnt weights to avoid over-fitting.

**Supporting evidence (machine-checked verbatim on the stated page):**
- `xgboost-chen-2016` p.2: “penalizes the complexity of the model”

**Mechanical checks:** pass

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### H06 · heldout · answerable

**Question:** How does Adam correct the bias of its moment estimates?

**Expected source(s):** `adam-kingma-2014`

**Reference answer:** It divides the first moment estimate by (1 - beta1^t) and the second raw moment estimate by (1 - beta2^t) to obtain bias-corrected estimates used in the update.

**Supporting evidence (machine-checked verbatim on the stated page):**
- `adam-kingma-2014` p.2: “Compute bias-corrected first moment estimate”

**Mechanical checks:** pass

**Open concerns:**
- Evidence quote shows the first-moment correction only; the second-moment line (same algorithm box, p.2) should also be checked against the reference answer.

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### H07 · heldout · answerable

**Question:** What did the Paullada et al. survey report about inter-annotator agreement thresholds when a dataset was reconstructed?

**Expected source(s):** `data-discontents-paullada-2020`

**Reference answer:** Different thresholds for inter-annotator agreement produced vastly different datasets, indicating that ground-truth labels do not correspond to truth.

**Supporting evidence (machine-checked verbatim on the stated page):**
- `data-discontents-paullada-2020` p.5: “different thresholds for inter-annotator agreement were found to produce vastly different datasets”

**Mechanical checks:** pass

**Open concerns:**
- Passage (p.5) is about the ImageNet re-creation by Recht et al. 2019; the question does not name ImageNet. Confirm it is unambiguous or add the dataset name.

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### H08 · heldout · answerable

**Question:** How often did the two WikiEval annotators in the Ragas paper agree?

**Expected source(s):** `ragas-es-2023`

**Reference answer:** About 95% of cases for faithfulness and context relevance, and about 90% for answer relevance.

**Supporting evidence (machine-checked verbatim on the stated page):**
- `ragas-es-2023` p.4: “the two annotators agreed in around 95% of cases”

**Mechanical checks:** pass

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### H09 · heldout · UNANSWERABLE

**Question:** What coverage does conformalized quantile regression achieve on ImageNet?

**Why unanswerable:** none of “imagenet” occurs anywhere in `conformal-qr-romano-2019` (machine-checked).
**Distractor / note:** ImageNet occurs in other corpus papers (Adam, dropout).

**Mechanical checks:** pass

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### H10 · heldout · UNANSWERABLE

**Question:** How much does GPU acceleration speed up causal forest training in Wager and Athey's experiments?

**Why unanswerable:** none of “gpu” occurs anywhere in `causal-forest-wager-2018` (machine-checked).
**Distractor / note:** On-topic paper, absent detail.

**Mechanical checks:** pass

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### H11 · heldout · UNANSWERABLE

**Question:** How does dropout compare with batch normalization on CIFAR-10 in Hinton et al.'s dropout paper?

**Why unanswerable:** none of “batch normalization”, “batchnorm” occurs anywhere in `dropout-hinton-2012` (machine-checked).
**Distractor / note:** Batch normalization is mentioned in DeepAR, a distractor.

**Mechanical checks:** pass

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______

### H12 · heldout · UNANSWERABLE

**Question:** Which Claude 3 models are evaluated in the Lost in the Middle paper, and how do they perform?

**Why unanswerable:** none of “claude 3”, “claude-3” occurs anywhere in the whole corpus (machine-checked).
**Distractor / note:** The paper evaluates an earlier Claude model; an answer that corrects the premise is arguably acceptable but is counted as a non-abstention by the current metric.

**Mechanical checks:** pass

**Open concerns:**
- False-premise style: the paper evaluates Claude-1.3, not Claude 3. An answer that corrects the premise is arguably correct but is scored as a false answer by the current metric. Decide whether to keep, rephrase, or score separately.

Reviewer: ☐ keep as is ☐ edit (describe) ☐ drop  — correct label? ☐ yes ☐ no — notes: ______
