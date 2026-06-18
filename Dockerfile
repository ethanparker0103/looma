# Looma — Hugging Face Spaces Docker image
# ============================================
# Deploy:   Push repo → HF Spaces (Docker runtime) → Done
# Build locally (optional):
#   docker build -t looma .
#   docker run -p 7860:7860 -e ANTHROPIC_API_KEY=sk-... looma

FROM python:3.12-slim

# ── System dependencies ──────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ──────────────────────────────────────────
WORKDIR /app
COPY backend/requirements.txt .
# Upgrade pip/setuptools first; use --no-build-isolation so
# openai-whisper can use the host's setuptools (pkg_resources).
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --no-build-isolation -r requirements.txt

# ── Application code ─────────────────────────────────────────────
COPY backend/ backend/
COPY frontend/ frontend/

# ── Runtime defaults (override via Space Secrets) ────────────────
# HF Spaces passes PORT env var automatically (default 7860)
ENV WHISPER_MODEL=medium
ENV MAX_CONCURRENT_JOBS=2
ENV MAX_VIDEO_SECONDS=1800
ENV JOB_TIMEOUT_SECONDS=3600
ENV DATA_DIR=/app/data

# ── Health check for HF Spaces ───────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:$PORT/healthz')"

EXPOSE 7860

CMD cd backend && uvicorn app.main:create_app --factory \
    --host 0.0.0.0 \
    --port ${PORT:-7860}
