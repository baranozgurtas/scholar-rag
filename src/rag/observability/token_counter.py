"""Token counting + cost ledger using tiktoken.

We use `cl100k_base` (GPT-4 tokenizer) as a proxy. Qwen2.5 uses a
different tokenizer (Qwen2Tokenizer / tiktoken-like ~150k vocab), but
cl100k counts are within ~10% of true Qwen counts in practice and don't
require downloading a separate tokenizer at runtime. This is acceptable
because the value of this number is *trend tracking* and *cost-awareness
signal*, not invoice-grade billing.

For absolute precision on a production deployment, swap in
`transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

import tiktoken

from rag.config import get_settings
from rag.logging_config import get_logger

logger = get_logger(__name__)

_ENCODING_NAME = "cl100k_base"
_encoding: tiktoken.Encoding | None = None
_encoding_lock = Lock()


def _get_encoding() -> tiktoken.Encoding:
    """Lazy-load the tokenizer (thread-safe)."""
    global _encoding
    if _encoding is None:
        with _encoding_lock:
            if _encoding is None:
                _encoding = tiktoken.get_encoding(_ENCODING_NAME)
    return _encoding


def count_tokens(text: str) -> int:
    """Approximate token count for a single string."""
    if not text:
        return 0
    return len(_get_encoding().encode(text))


def count_messages_tokens(messages: list[dict[str, str]]) -> int:
    """Approximate token count for a chat messages list (role + content)."""
    return sum(count_tokens(m.get("content", "")) + 4 for m in messages)


class TokenLedger:
    """JSONL append-only ledger of (query, prompt_tokens, completion_tokens, ts).

    Lightweight, no external DB. Useful for "avg tokens/query" reporting and
    early cost alerts in production.
    """

    def __init__(self, path: Path | None = None) -> None:
        cfg = get_settings()
        self.path = path or (cfg.logs_dir / "token_ledger.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def record(
        self,
        question: str,
        prompt_tokens: int,
        completion_tokens: int,
        extra: dict | None = None,
    ) -> None:
        entry = {
            "ts": datetime.now(UTC).isoformat(),
            "question_preview": question[:120],
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(prompt_tokens + completion_tokens),
        }
        if extra:
            entry.update(extra)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def aggregate(self) -> dict:
        """Compute summary statistics from the ledger."""
        if not self.path.exists():
            return {"queries": 0}
        n = 0
        sum_prompt = 0
        sum_completion = 0
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n += 1
                sum_prompt += e.get("prompt_tokens", 0)
                sum_completion += e.get("completion_tokens", 0)
        if n == 0:
            return {"queries": 0}
        return {
            "queries": n,
            "avg_prompt_tokens": round(sum_prompt / n, 1),
            "avg_completion_tokens": round(sum_completion / n, 1),
            "total_prompt_tokens": sum_prompt,
            "total_completion_tokens": sum_completion,
        }


__all__ = ["TokenLedger", "count_messages_tokens", "count_tokens"]
