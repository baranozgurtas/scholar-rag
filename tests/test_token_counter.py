"""Tests for token counting and the ledger."""

from __future__ import annotations

import json
from pathlib import Path

from rag.observability.token_counter import TokenLedger, count_tokens


class TestCounting:
    def test_count_tokens_empty(self) -> None:
        assert count_tokens("") == 0

    def test_count_tokens_short(self) -> None:
        n = count_tokens("hello world")
        assert n > 0
        assert n < 10  # "hello world" → 2 tokens in cl100k_base

    def test_count_tokens_grows_with_length(self) -> None:
        short = count_tokens("hello")
        long = count_tokens("hello " * 100)
        assert long > short * 10


class TestTokenLedger:
    def test_record_and_aggregate(self, tmp_path: Path) -> None:
        ledger = TokenLedger(path=tmp_path / "ledger.jsonl")
        ledger.record("q1", prompt_tokens=100, completion_tokens=20)
        ledger.record("q2", prompt_tokens=200, completion_tokens=40)
        agg = ledger.aggregate()
        assert agg["queries"] == 2
        assert agg["avg_prompt_tokens"] == 150.0
        assert agg["avg_completion_tokens"] == 30.0
        assert agg["total_prompt_tokens"] == 300

    def test_aggregate_empty_file(self, tmp_path: Path) -> None:
        ledger = TokenLedger(path=tmp_path / "ledger.jsonl")
        agg = ledger.aggregate()
        assert agg == {"queries": 0}

    def test_record_preserves_extra(self, tmp_path: Path) -> None:
        path = tmp_path / "ledger.jsonl"
        ledger = TokenLedger(path=path)
        ledger.record(
            "q1", 50, 10, extra={"prompt_version": "rag_answer@1.2", "abstained": False}
        )
        line = path.read_text().strip()
        record = json.loads(line)
        assert record["prompt_version"] == "rag_answer@1.2"
        assert record["abstained"] is False

    def test_jsonl_append(self, tmp_path: Path) -> None:
        path = tmp_path / "ledger.jsonl"
        ledger = TokenLedger(path=path)
        for i in range(5):
            ledger.record(f"q{i}", 10 + i, 5 + i)
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 5
