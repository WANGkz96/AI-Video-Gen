#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8090}"
BACKEND="${GENERATOR_BACKEND:-comfyui-ltx23}"
GENERATOR_API_URL="${GENERATOR_API_URL:-http://127.0.0.1:18188}"
CORS="${CORS_ORIGINS:-http://127.0.0.1:8080,http://localhost:8080}"
COMFYUI_ROOT="${COMFYUI_ROOT:-/workspace/ComfyUI}"
AI_VIDEO_GEN_PROVISIONING_STATUS="${AI_VIDEO_GEN_PROVISIONING_STATUS:-${ROOT_DIR}/data/provisioning-status.json}"

mkdir -p "${ROOT_DIR}/.run"
pkill -f "uvicorn backend.app.main:app .* --port ${PORT}" || true

nohup env PORT="${PORT}" GENERATOR_BACKEND="${BACKEND}" GENERATOR_API_URL="${GENERATOR_API_URL}" CORS_ORIGINS="${CORS}" COMFYUI_ROOT="${COMFYUI_ROOT}" AI_VIDEO_GEN_PROVISIONING_STATUS="${AI_VIDEO_GEN_PROVISIONING_STATUS}" AI_VIDEO_GEN_AUTH_REQUIRED="${AI_VIDEO_GEN_AUTH_REQUIRED:-1}" AI_VIDEO_GEN_API_TOKEN="${AI_VIDEO_GEN_API_TOKEN:-}" \
  bash "${ROOT_DIR}/scripts/start_vast.sh" \
  > "${ROOT_DIR}/.run/backend.out.log" \
  2> "${ROOT_DIR}/.run/backend.err.log" \
  < /dev/null &

echo $!
