"""Full local evaluation as separate processes (models never co-resident).

    1. retrieval-only 4-config ablation  (embedder, then reranker)
    2. generation for ONE config          (Ollama generator only)
    3. optional RAGAS on step 2's records (Ollama judge only)

Each step is resumable; re-running this command with the same `--run-dir`
continues where it stopped.

    python -m eval.run_full_eval --questions eval/questions_v2_draft.jsonl
    python -m eval.run_full_eval --run-dir eval/results/runs/<dir> --with-ragas
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from eval.generation_eval import DEFAULT_CONFIG
from eval.questions import QUESTIONS_PATH
from eval.retrieval_ablation import RUNS_DIR


def _step(args: list[str]) -> int:
    print("+ python -m " + " ".join(args), flush=True)
    return subprocess.call([sys.executable, "-m", *args])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Retrieval ablation → generation (one config) → optional RAGAS.")
    p.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--with-ragas", action="store_true")
    args = p.parse_args(argv)

    run_dir = args.run_dir or RUNS_DIR / f"full_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    if rc := _step(["eval.retrieval_ablation", "--questions", str(args.questions), "--run-dir", str(run_dir)]):
        return rc
    if rc := _step(["eval.generation_eval", str(run_dir), "--config", args.config]):
        return rc
    if args.with_ragas:
        records = run_dir / f"generation_{args.config}" / "records.jsonl"
        if rc := _step(["eval.ragas_eval", str(records)]):
            return rc
    print(f"Done: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main"]
