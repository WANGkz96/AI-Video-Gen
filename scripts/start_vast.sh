#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8080}"

cd "${ROOT_DIR}"
source .venv/bin/activate
export PORT="${PORT}"
export CORS_ORIGINS="${CORS_ORIGINS:-http://127.0.0.1:${PORT},http://localhost:${PORT}}"

exec uvicorn backend.app.main:app --host 0.0.0.0 --port "${PORT}"
