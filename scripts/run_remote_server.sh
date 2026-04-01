#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8090}"
BACKEND="${GENERATOR_BACKEND:-cogvideox-5b}"
CORS="${CORS_ORIGINS:-http://127.0.0.1:8080,http://localhost:8080}"

mkdir -p "${ROOT_DIR}/.run"
pkill -f "uvicorn backend.app.main:app .* --port ${PORT}" || true

nohup env PORT="${PORT}" GENERATOR_BACKEND="${BACKEND}" CORS_ORIGINS="${CORS}" \
  "${ROOT_DIR}/scripts/start_vast.sh" \
  > "${ROOT_DIR}/.run/backend.out.log" \
  2> "${ROOT_DIR}/.run/backend.err.log" \
  < /dev/null &

echo $!
