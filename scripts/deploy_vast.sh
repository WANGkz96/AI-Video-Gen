#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8090}"
MODELS="${MODELS:-}"
BACKEND="${GENERATOR_BACKEND:-comfyui-ltx23}"
GENERATOR_API_URL="${GENERATOR_API_URL:-http://127.0.0.1:18188}"
AI_VIDEO_GEN_DOWNLOAD_COMFY_MODELS="${AI_VIDEO_GEN_DOWNLOAD_COMFY_MODELS:-0}"
CORS="${CORS_ORIGINS:-http://127.0.0.1:8080,http://localhost:8080}"

cd "${ROOT_DIR}"

PORT="${PORT}" MODELS="${MODELS}" GENERATOR_BACKEND="${BACKEND}" GENERATOR_API_URL="${GENERATOR_API_URL}" AI_VIDEO_GEN_DOWNLOAD_COMFY_MODELS="${AI_VIDEO_GEN_DOWNLOAD_COMFY_MODELS}" CORS_ORIGINS="${CORS}" bash "${ROOT_DIR}/scripts/bootstrap_vast.sh"

if [ "${BACKEND}" = "comfyui-ltx23" ]; then
  "${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/scripts/download_comfy_ltx23_models.py" --comfy-root "${COMFYUI_ROOT:-/workspace/ComfyUI}" --verify-only
  "${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/scripts/wait_for_comfyui_ready.py"
fi

PORT="${PORT}" GENERATOR_BACKEND="${BACKEND}" GENERATOR_API_URL="${GENERATOR_API_URL}" CORS_ORIGINS="${CORS}" bash "${ROOT_DIR}/scripts/run_remote_server.sh"

cat <<EOF
Deploy complete.
Backend is starting on port ${PORT}.

Recommended SSH tunnel:
  ssh -L 8080:localhost:${PORT} <user>@<host>

Then open:
  http://127.0.0.1:8080/
EOF
