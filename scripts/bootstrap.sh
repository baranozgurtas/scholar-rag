#!/usr/bin/env bash
# Bootstrap script: first-time setup on Apple Silicon Mac.
# Idempotent — safe to re-run.
set -euo pipefail

echo "==> Research RAG Assistant bootstrap"
echo

# ─── 1. Check prerequisites ──────────────────────────────────────
echo "[1/5] Checking prerequisites..."
command -v python3.11 >/dev/null 2>&1 || command -v python3.12 >/dev/null 2>&1 || {
    echo "ERROR: Python 3.11 or 3.12 not found. Install via 'brew install python@3.11'" >&2
    exit 1
}
command -v docker >/dev/null 2>&1 || {
    echo "ERROR: Docker not found. Install Docker Desktop for Mac." >&2
    exit 1
}
command -v ollama >/dev/null 2>&1 || {
    echo "ERROR: Ollama not found. Install via 'brew install ollama' then 'brew services start ollama'" >&2
    exit 1
}
echo "  ✓ Python, Docker, Ollama present"

# ─── 2. Install Python dependencies ──────────────────────────────
echo
echo "[2/5] Installing Python deps (this may take a few minutes)..."
make install-dev
echo "  ✓ venv + deps installed"

# ─── 3. Pull Qwen2.5:14b via Ollama ──────────────────────────────
echo
echo "[3/5] Pulling qwen2.5:7b via Ollama (~9 GB)..."
if ollama list | grep -q "qwen2.5:7b"; then
    echo "  ✓ qwen2.5:7b already present"
else
    ollama pull qwen2.5:7b
    echo "  ✓ qwen2.5:7b ready"
fi

# ─── 4. Start Qdrant + Langfuse ──────────────────────────────────
echo
echo "[4/5] Starting Qdrant + Langfuse via Docker compose..."
make services-up
sleep 3
echo "  ✓ Services started (qdrant:6333, langfuse:3001)"

# ─── 5. Copy .env.example to .env if missing ─────────────────────
echo
echo "[5/5] Setting up .env..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "  ✓ Created .env from .env.example"
    echo "  ⚠️  Edit .env to add LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY"
    echo "     (sign up at http://localhost:3001 first)"
else
    echo "  ✓ .env already exists, not overwriting"
fi

echo
echo "==> Bootstrap complete!"
echo
echo "Next steps:"
echo "  make download-papers      # download 15 eval corpus papers from arXiv"
echo "  make ingest               # ingest into Qdrant"
echo "  make generate-questions   # generate 30 synthetic + 5 adversarial questions"
echo "  make eval                 # run ablation + RAGAS (≈ 30 min on M-series)"
echo "  make api                  # start FastAPI on :8000"
echo "  make ui                   # start Streamlit on :8501"
