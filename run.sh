#!/usr/bin/env bash
# Launch the Looma backend in dev mode.
# Usage: bash run.sh [--reload]
set -euo pipefail

cd "$(dirname "$0")"

# Load .env if present
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Activate the backend virtualenv if it exists
if [[ -f backend/.venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source backend/.venv/bin/activate
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
RELOAD_FLAG=""
if [[ "${1:-}" == "--reload" ]]; then
  RELOAD_FLAG="--reload"
fi

exec uvicorn backend.app.main:app --host "$HOST" --port "$PORT" $RELOAD_FLAG
