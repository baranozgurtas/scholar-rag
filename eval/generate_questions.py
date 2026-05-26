"""Generate the eval question set.

Three sources combined:

1. **Synthetic** (30 questions): Qwen2.5:14b reads section excerpts and
   produces (question, reference_answer, type) triples in strict JSON.
   Distribution: 10 single-paper, 10 cross-paper (forces multi-doc retrieval),
   10 multi-cluster (forces semantic generalization across domains).

2. **Manual** (15 questions, template): a JSONL template the user can fill
   in 30 minutes. Skeletons cover factoid / comparison / methodology /
   reasoning question types — explicit slots for question + answer +
   expected source papers.

3. **Adversarial** (5 questions, hard-coded): designed to be unanswerable
   from the indexed corpus, to measure abstention rate (should be 5/5).

Output: `eval/questions.jsonl` — every line a record with fields:
    question, reference_answer, expected_sources (list of paper aliases),
    expected_clusters, question_type, source_kind (synthetic|manual|adversarial)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rag.config import get_settings
from rag.generation.llm import build_generator_llm
from rag.generation.prompts import QGEN_SYSTEM, QGEN_USER
from rag.ingestion.pdf_loader import load_pdfs
from rag.logging_config import configure_logging, get_logger

logger = get_logger(__name__)

QUESTIONS_OUT = Path(__file__).resolve().parent / "questions.jsonl"


@dataclass
class EvalQuestion:
    """One eval question with ground truth."""

    question: str
    reference_answer: str
    expected_sources: list[str] = field(default_factory=list)
    expected_clusters: list[str] = field(default_factory=list)
    question_type: str = "factoid"
    source_kind: str = "synthetic"  # synthetic | manual | adversarial
    notes: str = ""


# ─── Adversarial set (hard-coded, defensive) ──────────────────────
ADVERSARIAL_QUESTIONS: list[EvalQuestion] = [
    EvalQuestion(
        question="What is the inference latency of Qwen2.5:14b on Apple M4 Pro for batched RAG queries?",
        reference_answer="I could not find sufficient information in the indexed papers to answer this question.",
        expected_sources=[],
        expected_clusters=[],
        question_type="out_of_corpus_factoid",
        source_kind="adversarial",
        notes="Specific runtime benchmark not in any indexed paper; should abstain.",
    ),
    EvalQuestion(
        question="How many parameters does Claude 3 Opus have, according to Anthropic's technical report?",
        reference_answer="I could not find sufficient information in the indexed papers to answer this question.",
        expected_sources=[],
        expected_clusters=[],
        question_type="out_of_corpus_factoid",
        source_kind="adversarial",
        notes="Not in any indexed paper; should abstain (and not hallucinate a number).",
    ),
    EvalQuestion(
        question="What was the F1 score of the SCIBERT model on the BLIRT-2024 benchmark?",
        reference_answer="I could not find sufficient information in the indexed papers to answer this question.",
        expected_sources=[],
        expected_clusters=[],
        question_type="hallucination_trap",
        source_kind="adversarial",
        notes="BLIRT-2024 is a fabricated benchmark name. Model must not invent a number.",
    ),
    EvalQuestion(
        question="Write Python code that implements FlashAttention-2 from scratch.",
        reference_answer="I could not find sufficient information in the indexed papers to answer this question.",
        expected_sources=[],
        expected_clusters=[],
        question_type="out_of_scope_codegen",
        source_kind="adversarial",
        notes="Out-of-scope task (code generation); also not in indexed papers.",
    ),
    EvalQuestion(
        question="Which restaurant in Zurich serves the best ramen?",
        reference_answer="I could not find sufficient information in the indexed papers to answer this question.",
        expected_sources=[],
        expected_clusters=[],
        question_type="out_of_domain",
        source_kind="adversarial",
        notes="Completely off-topic; should abstain rather than fall back to general knowledge.",
    ),
]


# ─── Manual template (15 placeholders for the user to fill) ───────
def _manual_template() -> list[EvalQuestion]:
    """Return 15 EvalQuestion stubs the user can fill in by hand.

    Each carries a hint in `notes` describing the question type to write.
    """
    stubs: list[tuple[str, str, list[str]]] = [
        ("methodology", "RAG paper: describe the retriever+generator setup", ["rag-lewis-2020"]),
        ("methodology", "BGE-M3: how are dense+sparse jointly learned?", ["bge-m3-chen-2024"]),
        ("comparison", "ColBERTv2 vs DPR on retrieval architecture", ["colbertv2-santhanam-2022"]),
        ("reasoning", "RAGAS: why is faithfulness decomposed into claims?", ["ragas-es-2023"]),
        ("factoid", "Lost in the Middle: which position has the lowest recall?", ["lost-in-the-middle-2023"]),
        ("methodology", "N-BEATS: how do the backcast/forecast blocks work?", ["n-beats-oreshkin-2019"]),
        ("methodology", "Conformal QR: how are intervals calibrated?", ["conformal-qr-romano-2019"]),
        ("methodology", "DeepAR: what likelihood family is used?", ["deepar-salinas-2017"]),
        ("comparison", "BPR vs NCF: what's the loss formulation difference?", ["bpr-rendle-2009", "ncf-he-2017"]),
        ("factoid", "XGBoost: what is the regularization term?", ["xgboost-chen-2016"]),
        ("methodology", "Dropout: how does it act at training vs inference?", ["dropout-srivastava-2014"]),
        ("methodology", "Adam: how are first/second moments combined?", ["adam-kingma-2014"]),
        ("methodology", "Causal Forest: how is honest splitting defined?", ["causal-forest-wager-2018"]),
        ("reasoning", "ML Tips: pick one bias-variance insight", ["ml-tips-domingos-2012"]),
        ("cross_cluster", "How does conformal prediction (forecasting) relate to RAG abstention?",
            ["conformal-qr-romano-2019", "rag-lewis-2020"]),
    ]
    out: list[EvalQuestion] = []
    for qtype, hint, sources in stubs:
        out.append(
            EvalQuestion(
                question=f"[FILL IN] {hint}",
                reference_answer="[FILL IN — 1 to 3 sentences drawn from the paper text]",
                expected_sources=sources,
                expected_clusters=[],  # auto-filled below from papers.txt
                question_type=qtype,
                source_kind="manual",
                notes=f"Template — hint: {hint}",
            )
        )
    return out


# ─── Synthetic generation via Qwen ────────────────────────────────
def _build_qgen_chain():
    """Build a JSON-output chain for question generation."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [("system", QGEN_SYSTEM), ("human", QGEN_USER)]
    )
    llm = build_generator_llm()
    return prompt | llm | StrOutputParser()


def _safe_parse_json_list(raw: str) -> list[dict[str, Any]]:
    """Best-effort JSON list extraction (handles fenced code blocks)."""
    text = raw.strip()
    if text.startswith("```"):
        # Strip leading/trailing fence
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Try to locate the JSON object inside any preamble
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return []
        else:
            return []
    if isinstance(parsed, dict) and "questions" in parsed:
        return parsed["questions"]
    if isinstance(parsed, list):
        return parsed
    return []


def _generate_for_paper(
    paper_title: str,
    section_label: str,
    text: str,
    paper_alias: str,
    cluster: str,
    n: int,
    chain: Any,
) -> list[EvalQuestion]:
    """Generate n questions grounded in a single paper section."""
    try:
        raw = chain.invoke(
            {
                "paper_title": paper_title,
                "section": section_label,
                "text": text[:4000],  # cap to avoid huge prompts
                "n": n,
            }
        )
    except Exception as e:
        logger.warning("qgen_invoke_failed", error=str(e), paper=paper_alias)
        return []

    items = _safe_parse_json_list(raw)
    questions: list[EvalQuestion] = []
    for it in items[:n]:
        q = (it.get("question") or "").strip()
        a = (it.get("reference_answer") or "").strip()
        qt = (it.get("question_type") or "factoid").strip()
        if not q or not a:
            continue
        questions.append(
            EvalQuestion(
                question=q,
                reference_answer=a,
                expected_sources=[paper_alias],
                expected_clusters=[cluster],
                question_type=qt,
                source_kind="synthetic",
                notes=f"Generated from §{section_label}",
            )
        )
    return questions


def generate_synthetic_questions(
    n_total: int = 30,
    seed: int = 42,
) -> list[EvalQuestion]:
    """Generate `n_total` synthetic questions grounded in indexed PDFs.

    Strategy:
    - Load PDFs from data/pdfs/
    - For each paper, pick 2 distinct sections (preferring methods/results/discussion)
    - Generate 1 question per (paper, section) until n_total is reached
    """
    settings = get_settings()
    pdf_dir = settings.ingestion.pdf_dir
    if not pdf_dir.exists() or not any(pdf_dir.glob("*.pdf")):
        logger.warning("no_pdfs_for_qgen", dir=str(pdf_dir))
        return []

    # Map alias → cluster from papers.txt
    alias_to_cluster: dict[str, str] = {}
    papers_file = Path(__file__).resolve().parent / "papers.txt"
    if papers_file.exists():
        for line in papers_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                alias_to_cluster[parts[1]] = parts[2]

    docs = load_pdfs(pdf_dir)
    if not docs:
        return []

    chain = _build_qgen_chain()
    rng = random.Random(seed)

    # Group pages by section, prefer methods/results/discussion
    preferred = {"methods", "results", "experiments", "discussion", "evaluation"}
    questions: list[EvalQuestion] = []
    n_per_paper = max(1, n_total // len(docs))

    for doc in docs:
        alias = doc.source_path.stem
        cluster = alias_to_cluster.get(alias, "unknown")
        by_section: dict[str, list[str]] = defaultdict(list)
        for page in doc.pages:
            by_section[page.section].append(page.text)
        # Pick sections: preferred first, then any
        chosen_sections = [s for s in by_section if s in preferred] or list(by_section.keys())
        rng.shuffle(chosen_sections)
        chosen_sections = chosen_sections[: max(1, n_per_paper)]

        for sec in chosen_sections:
            if len(questions) >= n_total:
                break
            section_text = "\n\n".join(by_section[sec])[:6000]
            if len(section_text) < 400:
                continue
            new_qs = _generate_for_paper(
                paper_title=doc.title,
                section_label=sec,
                text=section_text,
                paper_alias=alias,
                cluster=cluster,
                n=1,
                chain=chain,
            )
            questions.extend(new_qs)
        if len(questions) >= n_total:
            break

    logger.info("synthetic_questions_generated", n=len(questions), target=n_total)
    return questions[:n_total]


# ─── Orchestrator ────────────────────────────────────────────────
def build_question_set(
    n_synthetic: int = 30,
    include_manual_template: bool = True,
    include_adversarial: bool = True,
    out_path: Path = QUESTIONS_OUT,
    skip_synthetic: bool = False,
) -> Path:
    """Build the full question set and write it to JSONL."""
    questions: list[EvalQuestion] = []
    if not skip_synthetic:
        questions.extend(generate_synthetic_questions(n_total=n_synthetic))
    if include_manual_template:
        questions.extend(_manual_template())
    if include_adversarial:
        questions.extend(ADVERSARIAL_QUESTIONS)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")

    logger.info(
        "question_set_written",
        path=str(out_path),
        total=len(questions),
        synthetic=sum(1 for q in questions if q.source_kind == "synthetic"),
        manual=sum(1 for q in questions if q.source_kind == "manual"),
        adversarial=sum(1 for q in questions if q.source_kind == "adversarial"),
    )
    return out_path


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    p = argparse.ArgumentParser(description="Generate eval question set.")
    p.add_argument("--n-synthetic", type=int, default=30)
    p.add_argument("--no-manual", action="store_true")
    p.add_argument("--no-adversarial", action="store_true")
    p.add_argument("--skip-synthetic", action="store_true", help="Skip Qwen-based generation.")
    p.add_argument("--out", type=Path, default=QUESTIONS_OUT)
    args = p.parse_args(argv)

    out = build_question_set(
        n_synthetic=args.n_synthetic,
        include_manual_template=not args.no_manual,
        include_adversarial=not args.no_adversarial,
        out_path=args.out,
        skip_synthetic=args.skip_synthetic,
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "ADVERSARIAL_QUESTIONS",
    "EvalQuestion",
    "build_question_set",
    "generate_synthetic_questions",
    "main",
]
