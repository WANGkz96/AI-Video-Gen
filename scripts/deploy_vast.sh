#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8090}"
MODELS="${MODELS:-}"
BACKEND="${GENERATOR_BACKEND:-comfyui-ltx23}"
GENERATOR_API_URL="${GENERATOR_API_URL:-http://127.0.0.1:18188}"
AI_VIDEO_GEN_DOWNLOAD_COMFY_MODELS="${AI_VIDEO_GEN_DOWNLOAD_COMFY_MODELS:-0}"
CORS="${CORS_ORIGINS:-http://127.0.0.1:8080,http://localhost:8080}"

case "${BACKEND}" in
  comfyui-ltx23)
    ;;
  ltx-2.3-distilled|ltx23-distilled|ltx-native|ltx-2.3|wan2.2-ti2v-5b)
    echo "Legacy GENERATOR_BACKEND='${BACKEND}' detected; using comfyui-ltx23."
    BACKEND="comfyui-ltx23"
    ;;
esac

if [ "${AI_VIDEO_GEN_FORCE_LTX23_DISTILLED:-0}" = "1" ]; then
  echo "Ignoring legacy AI_VIDEO_GEN_FORCE_LTX23_DISTILLED=1; generation is handled by ComfyUI."
  BACKEND="comfyui-ltx23"
fi

cd "${ROOT_DIR}"

PORT="${PORT}" MODELS="${MODELS}" GENERATOR_BACKEND="${BACKEND}" GENERATOR_API_URL="${GENERATOR_API_URL}" AI_VIDEO_GEN_DOWNLOAD_COMFY_MODELS="${AI_VIDEO_GEN_DOWNLOAD_COMFY_MODELS}" CORS_ORIGINS="${CORS}" bash "${ROOT_DIR}/scripts/bootstrap_vast.sh"

mkdir -p "${ROOT_DIR}/.run"
nohup env PORT="${PORT}" GENERATOR_BACKEND="${BACKEND}" GENERATOR_API_URL="${GENERATOR_API_URL}" CORS_ORIGINS="${CORS}" COMFYUI_ROOT="${COMFYUI_ROOT:-/workspace/ComfyUI}" \
  bash "${ROOT_DIR}/scripts/run_after_comfy_ready.sh" \
  > "${ROOT_DIR}/.run/after-comfy-ready.out.log" \
  2> "${ROOT_DIR}/.run/after-comfy-ready.err.log" \
  < /dev/null &

cat <<EOF
Deploy complete.
AI-Video-Gen will start on port ${PORT} after ComfyUI and the LTX 2.3 workflows are ready.

Recommended SSH tunnel:
  ssh -L 8080:localhost:${PORT} <user>@<host>

Then open:
  http://127.0.0.1:8080/
EOF
