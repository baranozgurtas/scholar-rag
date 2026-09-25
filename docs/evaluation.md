# Evaluation

How the system is evaluated, what the current question sets can and cannot
support, and the plan for a larger reviewed held-out set.

## Status of results in this repository

| Evaluation | Status |
|---|---|
| Legacy 25-question results (`eval/results/ablation_*.json`, Colab T4, generator not recorded) | Committed; re-analysed deterministically in `eval/results/legacy_reanalysis.md` |
| Retrieval-only ablation with the current code and index | **NOT RUN** |
| Generation eval (answer policy, false abstention / false answer, tag validity) with the current code | **NOT RUN** |
| RAGAS (LLM-judged grounding) | **NOT RUN** |
| Rerank threshold selection on dev / held-out | **NOT RUN** (gate stays disabled) |

## Commands

The pipeline is split so that models are never loaded together and every
step can be interrupted and resumed with the same command:

| Step | Command | Loads | Writes |
|---|---|---|---|
| Tests (CI) | `make test` or `make test-ci` | nothing | — |
| Legacy re-analysis | `make eval-legacy` | nothing | `eval/results/legacy_reanalysis.{json,md}` |
| 1. Retrieval-only ablation, A–D | `make eval-retrieval` (legacy 25) / `make eval-retrieval-draft` (v2 draft) | embedder, released; then reranker | `eval/results/runs/retrieval_<ts>_<sha>/` |
| 2. Generation, one config | `make eval-generation RUN=<run dir> [CONFIG=D_hybrid_plus_rerank]` | Ollama generator only | `<run dir>/generation_<config>/` |
| 3. LLM-judged grounding | `make eval-ragas RECORDS=<run dir>/generation_<config>/records.jsonl` | Ollama judge + `nomic-embed-text` | `ragas_eval.json`, `ragas_summary.md` next to the records |
| Threshold (optional) | `python -m eval.select_threshold <run dir>/generation_D_hybrid_plus_rerank/records.jsonl` | nothing | `..._threshold.json` (needs dev + held-out records) |
| All of 1–3 | `make eval` (`python -m eval.run_full_eval --with-ragas [--run-dir <dir>]`) | one step per subprocess | as above |

Resume: `python -m eval.retrieval_ablation --run-dir <dir>` and
`python -m eval.generation_eval <dir>` skip question IDs already written.
Both refuse to resume if the question file, models (including the Ollama
digest of the generator), prompt version, index size or settings changed.
A generator failure stops the run without writing that question.

The retrieval step never calls the generator. B and D rerank the saved
top-20 candidates of A and C respectively, so each question is embedded once
per candidate kind.

Without Docker, set `QDRANT_PATH=./qdrant_local` to use embedded Qdrant for
both `make ingest` and the eval (exact search, single process).

### Later full GPU evaluation

Run on a machine with a GPU and enough memory for the generator
(the published numbers must name the model that produced them):

```bash
export GENERATOR_MODEL=qwen2.5:7b JUDGE_MODEL=qwen2.5:7b \
       EMBEDDING_DEVICE=cuda RERANKER_DEVICE=cuda LANGFUSE_ENABLED=false
ollama pull qwen2.5:7b && ollama pull nomic-embed-text
make services-up                                   # or QDRANT_PATH=...
python -m rag.ingestion.pipeline --pdf-dir ./data/pdfs --recreate
python -m eval.run_full_eval --questions eval/questions_v2_draft.jsonl --with-ragas
python -m eval.select_threshold eval/results/runs/<dir>/generation_D_hybrid_plus_rerank/records.jsonl
python scripts/update_readme_metrics.py eval/results/runs/<dir>/generation_D_hybrid_plus_rerank/summary.json
```

Results from a different generator (for example a smaller model chosen to fit
a laptop) are a different system. Report them in their own table with the
model name from the run manifest; do not put them next to, or in place of,
the Qwen2.5 numbers.

Each run directory contains `manifest.json` (git commit and dirty flag,
model names, generator Ollama digest, retrieval and chunking settings, PDF
hashes, Qdrant point count, question-file hash and review-status counts,
package versions) and the raw per-question JSONL files.

## Metrics, reported separately

| Metric | Population | Definition |
|---|---|---|
| Hit@5, MRR@10, nDCG@10 | answerable | Paper-level: a ranked chunk is relevant if its PDF is in `expected_sources`; each paper counts once (nDCG). Ranking is the top-10 after reranking (or fusion), not just the 5 chunks sent to the LLM. |
| False abstention | answerable | Share answered with the abstention text, broken down by `outcome` (model abstained, withheld by guard, rerank gate). |
| False answer | unanswerable | Share where an answer was released. |
| Citation tag validity | all generated tags | Share of tags whose (title, page) was among the supplied passages. **Structural only.** |
| Factual grounding | — | Not measured by the harness. `eval.ragas_eval` gives an LLM-judged estimate (same model family as the generator); human review is the reference. |
| Latency | all non-error | Retrieval-only runs: p50 / p95 of retrieval + rerank. Generation runs: retrieval + rerank + generation, measured in separate processes and summed. |

All rates carry 95% Wilson intervals because n is small.

## Index hygiene: the "Adam question returned causal-forest chunks" incident

During an aborted local end-to-end run (8 GB machine, embedded Qdrant,
generator running concurrently), config A answered L01 (Adam vs SGD/Adagrad
on MNIST) from ten causal-forest chunks. Findings:

- **Index and ingestion are complete**: 15 sources, 1,429 points, one file
  hash per source, no NaN vectors, all unit-norm.
- **The retrieved chunks were extraction debris**: `1a12476c_0131 … _0223`
  are runs of `G\n` (scatter-plot markers) from a figure on p.21 of the
  causal-forest PDF. 151 such chunks were indexed; 90 of them share one
  identical dense vector.
- **Retrieval is correct in a fresh process**: re-embedding L01 on MPS
  (fp16) and CPU (fp32) gives the same query vector (cos ≈ 1.0), and dense
  search returns ten Adam chunks (top cosine 0.73; debris chunks average
  0.34). Replaying the exact config-A code path three times, with the
  reranker also loaded, reproduced the correct ranking every time.
- **A degenerate query vector reproduces the symptom**: an all-ones query
  vector ranks the `G\n` debris first; a NaN vector raises an error in
  Qdrant, and a zero vector returns other chunks.
- **Not established**: why the query embedding in that one process was
  degenerate. The run did not save query vectors or dense scores. Its
  retrieval step took 14.7 s against ~1 s in replays. That the generator
  was sharing the GPU is a candidate cause, but it has not been tested.

Changes: the chunker now drops such debris (`is_low_content`: fewer than
10% distinct tokens *and* under 20% of characters in real words; on this
corpus it flags exactly the 151 marker chunks and keeps every numeric
table). Retrieval records store `top_dense_score`; the harness flags any
question whose best dense score is below 0.4 as `low_top_dense_score`.
Apply it by re-ingesting into a recreated collection:
`python -m rag.ingestion.pipeline --pdf-dir ./data/pdfs --recreate`. The
committed legacy results were produced by the same chunker before this
filter, so their index most likely contained the same debris; that index
was not saved, so this cannot be checked.

## Answer policy (what the citation guard enforces)

`rag.guards.citation_checker.apply_answer_policy`, in order:

1. generation raised → withheld (`generation_error`)
2. output is exactly the abstention sentence → `model_abstained`
3. abstention sentence **and** other content → withheld (`mixed_abstention`)
4. no parseable citation tag → withheld (`uncited_answer`)
5. any tag not matching a supplied passage → withheld (`invalid_citation`)
6. otherwise → released (`answered`)

"Withheld" means the user sees the abstention sentence; the raw output is
kept in the response (`debug=true`) and in eval records. A released answer
has at least one tag and no out-of-context tags. That does **not** establish
that every claim is cited, or that a cited passage supports its claim.

The optional pre-generation gate (`RETRIEVAL_RERANK_SCORE_THRESHOLD`) acts on
the top-1 reranker score only and is off by default (0.0). Because it acts
before generation on a recorded score, `eval.select_threshold` can replay any
threshold exactly on recorded outputs. It selects on `split=dev` and reports
`split=heldout` once; it refuses the `legacy_test` split.

## Question sets

### `eval/questions.jsonl` — legacy test set (25)

The set behind the committed results. 20 answerable questions generated by
Qwen2.5 from single PDF sections (`review_status: unreviewed`) and 5
hand-written out-of-corpus questions. Audit findings, recorded per question
in `label_issues`:

- **Two corpus files were mislabeled.** `ml-tips-domingos-2012.pdf` is arXiv
  2012.05345, Paullada et al. 2020, "Data and its (dis)contents"; the two
  questions labeled with it (Levendowski copyright argument; reasons for
  dataset review) are about that paper. `dropout-srivastava-2014.pdf` is
  arXiv 1207.0580, Hinton et al. 2012. Files and labels are now
  `data-discontents-paullada-2020` and `dropout-hinton-2012`.
- **Paper not named.** L04 ("What are the two datasets used in the
  evaluation…") fits NCF as well as BPR (both use exactly two datasets). It
  is the only question config D "misses", and D cited NCF p.5. L05, L07, L11,
  L14 also say "the paper"/"the work".
- **Reference incomplete.** L06 asks for k-NN MSE values at d=30; the
  reference answer only says "an order of magnitude worse".
- **Easy negatives only.** The 5 unanswerable questions are off-topic
  (restaurants, Claude 3 parameters, code generation). None tests a
  plausible question about an indexed paper whose answer is absent.

The 15 `[FILL IN]` manual templates were never completed and were never part
of the 25 evaluated questions (the old loader skipped them). They have been
removed from this file; the loader now rejects any `[FILL IN]` record.

### `eval/questions_v2_draft.jsonl` — proposed dev / held-out draft (22)

- 15 answerable questions drafted from the former template hints (the
  "ML Tips bias-variance" hint was replaced because that paper is not in the
  corpus; the cross-cluster "conformal prediction vs RAG abstention" hint was
  dropped because no passage answers it). Each carries `evidence` quotes
  that CI checks verbatim against the stated PDF page.
- 7 near-miss unanswerable questions about indexed papers asking for a fact
  those papers do not contain (e.g. BPR hit ratio on MovieLens, which NCF
  reports but BPR does not). Each lists `absent_terms` that CI checks never
  occur in the scoped PDFs.
- Split fixed before any run: dev 7 + 3, held-out 8 + 4.
- **Drafted by an LLM (Claude Code), not human-reviewed.** Machine checks
  confirm quotes exist and terms are absent; they do not confirm the
  question is well-posed or the reference answer is complete.

## Proposed reviewed held-out set

Target for claims beyond "indicative": at 20 answerable questions a 0/20
false-abstention result still has a 95% upper bound of 16%. For ±5-point
intervals near 10% error rates, ~150 questions per class are needed.

| Block | n | Construction |
|---|---|---|
| Answerable, single paper | 90 | 6 per paper; mix of factoid / method / result-number questions; each names its paper or is unambiguous corpus-wide |
| Answerable, multi-paper | 30 | Comparisons that need two papers (e.g. BPR vs NCF objectives) |
| Unanswerable, near-miss | 45 | On-topic, fact absent from the named paper, ideally present in a *different* corpus paper (distractor) |
| Unanswerable, false premise | 15 | Presupposes a claim the paper does not make; correct behaviour is refusal or premise correction (scored separately) |
| Unanswerable, off-topic | 10 | Current adversarial style; a smoke test |

Protocol:

1. Two annotators draft independently from the PDFs, not from system output.
   Every answerable item cites page + quote; every unanswerable item lists
   search terms checked (`absent_terms`), so `verify_question` covers it.
2. Cross-review: the other annotator answers each question blind from the
   PDFs; disagreements are resolved or the item is dropped. Record agreement.
3. Only then set `review_status: "human_reviewed"` and assign splits
   (stratified by paper and block, ~30% dev / 70% held-out), freeze the file
   and record its SHA-256.
4. Tune anything (threshold, prompt, top-k) on dev only. Report held-out once
   per frozen system version, with the run manifest.
5. For grounding, have a person label a sample of released answers per claim
   (supported / unsupported / uncited) and compare with the RAGAS judge.

`tests/test_eval_harness.py::TestQuestionFiles::test_no_question_claims_human_review`
fails if any record claims human review; update it when step 3 is done.
