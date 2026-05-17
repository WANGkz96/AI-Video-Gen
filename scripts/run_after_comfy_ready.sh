#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8090}"
BACKEND="${GENERATOR_BACKEND:-comfyui-ltx23}"
GENERATOR_API_URL="${GENERATOR_API_URL:-http://127.0.0.1:18188}"
CORS="${CORS_ORIGINS:-http://127.0.0.1:${PORT},http://localhost:${PORT}}"
COMFY_ROOT="${COMFYUI_ROOT:-/workspace/ComfyUI}"

cd "${ROOT_DIR}"

if [ "${BACKEND}" = "comfyui-ltx23" ]; then
  "${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/scripts/download_comfy_ltx23_models.py" --comfy-root "${COMFY_ROOT}" --verify-only
  "${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/scripts/wait_for_comfyui_ready.py"
fi

PORT="${PORT}" GENERATOR_BACKEND="${BACKEND}" GENERATOR_API_URL="${GENERATOR_API_URL}" CORS_ORIGINS="${CORS}" bash "${ROOT_DIR}/scripts/run_remote_server.sh"
