.PHONY: help install install-dev clean lint format test test-ci test-fast test-integration \
        services-up services-down qdrant-up langfuse-up ollama-pull \
        download-papers ingest generate-questions eval eval-retrieval eval-retrieval-draft \
        eval-generation eval-ragas eval-legacy verify-questions \
        api ui docker-build docker-up docker-down ci

# Default target shows help
.DEFAULT_GOAL := help

PYTHON ?= python3.11
VENV   ?= .venv
PIP    := $(VENV)/bin/pip
PY     := $(VENV)/bin/python
PYTEST := $(VENV)/bin/pytest
RUFF   := $(VENV)/bin/ruff
UVICORN := $(VENV)/bin/uvicorn
STREAMLIT := $(VENV)/bin/streamlit

# ─── Help ─────────────────────────────────────────────────────────
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'

# ─── Setup ────────────────────────────────────────────────────────
$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel

install: $(VENV)/bin/activate  ## Install runtime dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev: $(VENV)/bin/activate  ## Install dev dependencies (tests, lint, eval)
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .
	$(VENV)/bin/pre-commit install || true

clean:  ## Remove build artifacts and caches
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ─── Lint & format ────────────────────────────────────────────────
lint:  ## Run ruff lint check
	$(RUFF) check src tests eval

format:  ## Auto-format with ruff
	$(RUFF) format src tests eval
	$(RUFF) check --fix src tests eval

# ─── Tests ────────────────────────────────────────────────────────
test:  ## Run all tests (no models / services needed)
	$(PYTEST) tests/

test-ci:  ## Run exactly what CI runs (lint + tests + legacy re-analysis check)
	$(RUFF) check src tests eval
	$(PYTEST) tests/ -m "not integration"
	$(PY) -m eval.reanalyze_legacy

test-fast:  ## Run tests excluding slow + integration
	$(PYTEST) tests/ -m "not slow and not integration"

test-integration:  ## Run integration tests (requires Qdrant + Ollama)
	$(PYTEST) tests/ -m integration

# ─── Services (Docker) ────────────────────────────────────────────
qdrant-up:  ## Start Qdrant only
	docker compose -f docker-compose.dev.yml up -d qdrant

langfuse-up:  ## Start Langfuse only
	docker compose -f docker-compose.dev.yml up -d langfuse-db langfuse

services-up:  ## Start Qdrant + Langfuse (dev: API runs on host)
	docker compose -f docker-compose.dev.yml up -d

services-down:  ## Stop all dev services
	docker compose -f docker-compose.dev.yml down

# ─── Ollama (host) ────────────────────────────────────────────────
ollama-pull:  ## Pull the generator model from .env.example (qwen2.5:7b)
	ollama pull qwen2.5:7b

# ─── Data pipeline ────────────────────────────────────────────────
download-papers:  ## Download 15 eval corpus papers from arXiv
	$(PY) -m eval.download_papers

ingest:  ## Ingest PDFs from data/pdfs into Qdrant
	$(PY) -m rag.ingestion.pipeline --pdf-dir ./data/pdfs

generate-questions:  ## Generate new LLM-written questions (writes eval/questions_synthetic_new.jsonl)
	$(PY) -m eval.generate_questions

verify-questions:  ## Check question files against PDF text (evidence quotes, absent terms)
	$(PYTEST) tests/test_eval_harness.py -k "QuestionFiles"

# ─── Evaluation ───────────────────────────────────────────────────
eval-retrieval:  ## Retrieval-only 4-config ablation, legacy 25 (embedder, then reranker; no LLM)
	$(PY) -m eval.retrieval_ablation

eval-retrieval-draft:  ## Retrieval-only ablation on the unreviewed v2 dev/held-out draft
	$(PY) -m eval.retrieval_ablation --questions eval/questions_v2_draft.jsonl

eval-generation:  ## Generation for one config from a retrieval run: make eval-generation RUN=eval/results/runs/<dir>
	$(PY) -m eval.generation_eval $(RUN) --config $(or $(CONFIG),D_hybrid_plus_rerank)

eval-ragas:  ## LLM-judged RAGAS on generation records: make eval-ragas RECORDS=<run>/generation_<cfg>/records.jsonl
	$(PY) -m eval.ragas_eval $(RECORDS)

eval-legacy:  ## Re-analyse the committed results (no models)
	$(PY) -m eval.reanalyze_legacy

eval:  ## Full local eval as separate steps: retrieval → generation (D) → RAGAS
	$(PY) -m eval.run_full_eval --with-ragas

# ─── Serving ──────────────────────────────────────────────────────
api:  ## Start FastAPI server (port 8000)
	$(UVICORN) rag.api.main:app --host 0.0.0.0 --port 8000 --reload

api-prod:  ## Start FastAPI server in production mode
	$(UVICORN) rag.api.main:app --host 0.0.0.0 --port 8000 --workers 2

ui:  ## Open the web UI in browser (FastAPI must be running via 'make api')
	@echo "Opening http://localhost:8000 ..."
	@echo "Make sure 'make api' is running in another terminal."
	@if command -v open >/dev/null 2>&1; then \
		open http://localhost:8000; \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open http://localhost:8000; \
	else \
		echo "Open http://localhost:8000 manually"; \
	fi

# ─── Docker (production) ──────────────────────────────────────────
docker-build:  ## Build production Docker image
	docker compose build

docker-up:  ## Start full stack via Docker (API + Qdrant + Langfuse)
	docker compose up -d

docker-down:  ## Stop full stack
	docker compose down

# ─── CI surrogate (local) ─────────────────────────────────────────
ci: test-ci  ## Run what CI runs
	@echo "✅ CI checks passed"
