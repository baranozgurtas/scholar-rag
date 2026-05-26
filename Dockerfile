# ──────────────────────────────────────────────────────────────────
# Research RAG Assistant — production image
#
# Multi-stage build:
#   1. `builder`  installs deps into a venv
#   2. `runtime`  copies the venv + source into a slim final image
#
# Runs on linux/arm64 (Apple Silicon native) and linux/amd64 (CI / Linux).
# Buildx multi-arch builds: `docker buildx build --platform linux/arm64,linux/amd64 .`
# ──────────────────────────────────────────────────────────────────

# ─── Stage 1: builder ─────────────────────────────────────────────
FROM --platform=$BUILDPLATFORM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build deps for sentence-transformers / qdrant-client wheels on ARM
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Create venv to keep the runtime image clean
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python deps. Copy only requirements first for layer caching.
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt

# Copy source and install the package
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-deps -e .

# ─── Stage 2: runtime ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    APP_HOME=/app

# Minimal runtime deps; tini for proper PID 1 signal handling
RUN apt-get update && apt-get install -y --no-install-recommends \
        tini \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 app \
    && useradd  --uid 1000 --gid app --shell /bin/bash --create-home app

# Copy the venv from the builder
COPY --from=builder /opt/venv /opt/venv

WORKDIR $APP_HOME

# Copy source (keep ownership for non-root user)
COPY --chown=app:app pyproject.toml ./
COPY --chown=app:app src ./src
COPY --chown=app:app app.py ./
COPY --chown=app:app eval ./eval
COPY --chown=app:app scripts ./scripts

# Pre-create data/log directories the app writes to
RUN mkdir -p data/pdfs logs .cache eval/results \
    && chown -R app:app /app

USER app

# Docker healthcheck hits the FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:${API_PORT:-8000}/health || exit 1

EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
