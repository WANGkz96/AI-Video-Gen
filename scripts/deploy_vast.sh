#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8090}"
MODELS="${MODELS:-ltx-2.3-distilled}"
BACKEND="${GENERATOR_BACKEND:-ltx-2.3-distilled}"
CORS="${CORS_ORIGINS:-http://127.0.0.1:8080,http://localhost:8080}"

cd "${ROOT_DIR}"

PORT="${PORT}" MODELS="${MODELS}" GENERATOR_BACKEND="${BACKEND}" CORS_ORIGINS="${CORS}" bash "${ROOT_DIR}/scripts/bootstrap_vast.sh"
PORT="${PORT}" GENERATOR_BACKEND="${BACKEND}" CORS_ORIGINS="${CORS}" bash "${ROOT_DIR}/scripts/run_remote_server.sh"

cat <<EOF
Deploy complete.
Backend is starting on port ${PORT}.

Recommended SSH tunnel:
  ssh -L 8080:localhost:${PORT} <user>@<host>

Then open:
  http://127.0.0.1:8080/
EOF
