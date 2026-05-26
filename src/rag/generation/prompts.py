"""Versioned prompt templates.

Every prompt has a stable VERSION string. Changing a prompt means bumping
the version, which:
- shows up in Langfuse traces (so you can compare metric deltas across versions)
- is logged with every /query response (so eval runs are reproducible)
- is referenced from `eval/results/ragas_summary.md` so a metric jump can
  be traced back to a specific prompt change

The RAG answer prompt is engineered for:
- **Strict grounding**: only answer from context; abstain otherwise.
- **Citation discipline**: every claim → at least one citation tag.
- **Hedging**: when context is partial, say so explicitly.
- **No hallucination of citations**: if a citation tag isn't in the
  provided context, the model must not invent one.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate


@dataclass(frozen=True)
class PromptVersion:
    name: str
    version: str
    description: str


# ─── RAG answer prompt (v1.2) ─────────────────────────────────────
RAG_ANSWER_PROMPT_VERSION = PromptVersion(
    name="rag_answer",
    version="1.2",
    description="Strict-grounding RAG prompt with abstention + citation discipline.",
)

RAG_ANSWER_SYSTEM = """You are a helpful research assistant answering questions about academic papers.

You will be given excerpts from one or more papers as context. Use the context to answer the question.

CITATION FORMAT
Cite each factual claim with a tag in this format right after the claim:
    [Paper: <title> | p.<page> | §<section>]
You can have multiple citations per claim, separated by spaces.

RULES
1. Base your answer on the provided context. The context contains the relevant information.
2. Cite every factual claim using the format above. Use only tags whose title/page/section appear in the context.
3. Be concise: 2-5 sentences. Get straight to the answer.
4. Only respond with "I could not find sufficient information in the indexed papers to answer this question." if the context is genuinely unrelated to the question. If the context discusses related concepts, synthesize an answer from what is available.
5. Do not invent citation tags or facts not present in the context.
"""

RAG_ANSWER_USER = """CONTEXT EXCERPTS
---
{context}
---

QUESTION
{question}

ANSWER (with citations):"""


def build_rag_answer_prompt() -> ChatPromptTemplate:
    """Build the LangChain ChatPromptTemplate for the RAG answer step."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", RAG_ANSWER_SYSTEM),
            ("human", RAG_ANSWER_USER),
        ]
    )


# ─── Synthetic question generation (v1.0) ─────────────────────────
QGEN_PROMPT_VERSION = PromptVersion(
    name="question_generation",
    version="1.0",
    description="Generate diverse, grounded eval questions from a paper section.",
)

QGEN_SYSTEM = """You are an expert ML/IR research evaluator. You write evaluation questions \
about academic papers that test whether a RAG system can retrieve and faithfully use \
specific information from those papers.

REQUIREMENTS for every question you generate:
1. The question must be answerable ONLY from the specific paper text you are given.
2. Avoid yes/no questions. Prefer "what", "how", "why", "compare", "describe" forms.
3. The question must be specific — name a method, dataset, metric, or claim from the text.
4. Provide a concise reference answer (1-3 sentences) drawn strictly from the text.
5. Output strictly valid JSON, no preamble, no markdown fences.
"""

QGEN_USER = """PAPER TITLE: {paper_title}
SECTION: {section}
TEXT EXCERPT:
---
{text}
---

Generate exactly {n} evaluation questions about this excerpt. Return JSON only:
{{
  "questions": [
    {{"question": "...", "reference_answer": "...", "question_type": "factoid|methodology|comparison|reasoning"}},
    ...
  ]
}}"""


def build_qgen_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [("system", QGEN_SYSTEM), ("human", QGEN_USER)],
    )


# ─── Registry (for logging / Langfuse) ────────────────────────────
PROMPT_REGISTRY: dict[str, PromptVersion] = {
    p.name: p for p in [RAG_ANSWER_PROMPT_VERSION, QGEN_PROMPT_VERSION]
}


def format_context(chunks_with_tags: list[tuple[str, str]]) -> str:
    """Format a list of (citation_tag, chunk_text) pairs into the prompt context block.

    Each excerpt is prefixed with its citation tag, separated by a divider.
    The LLM is instructed to reuse these tags verbatim in citations.
    """
    if not chunks_with_tags:
        return "(No context available.)"
    blocks: list[str] = []
    for tag, text in chunks_with_tags:
        blocks.append(f"{tag}\n{text}")
    return "\n\n---\n\n".join(blocks)


__all__ = [
    "PROMPT_REGISTRY",
    "QGEN_PROMPT_VERSION",
    "RAG_ANSWER_PROMPT_VERSION",
    "PromptVersion",
    "build_qgen_prompt",
    "build_rag_answer_prompt",
    "format_context",
]
