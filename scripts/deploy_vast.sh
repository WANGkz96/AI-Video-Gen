#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8090}"
MODELS="${MODELS:-}"
BACKEND="${GENERATOR_BACKEND:-comfyui-ltx23}"
GENERATOR_API_URL="${GENERATOR_API_URL:-http://127.0.0.1:18188}"
AI_VIDEO_GEN_DOWNLOAD_COMFY_MODELS="${AI_VIDEO_GEN_DOWNLOAD_COMFY_MODELS:-0}"
AI_VIDEO_GEN_PROVISIONING_STATUS="${AI_VIDEO_GEN_PROVISIONING_STATUS:-${ROOT_DIR}/data/provisioning-status.json}"
AI_VIDEO_GEN_MODEL_DOWNLOAD_MAX_ATTEMPTS="${AI_VIDEO_GEN_MODEL_DOWNLOAD_MAX_ATTEMPTS:-60}"
AI_VIDEO_GEN_MODEL_DOWNLOAD_RETRY_DELAY="${AI_VIDEO_GEN_MODEL_DOWNLOAD_RETRY_DELAY:-20}"
CORS="${CORS_ORIGINS:-http://127.0.0.1:8080,http://localhost:8080}"
COMFY_ROOT="${COMFYUI_ROOT:-/workspace/ComfyUI}"

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

PORT="${PORT}" MODELS="${MODELS}" GENERATOR_BACKEND="${BACKEND}" GENERATOR_API_URL="${GENERATOR_API_URL}" AI_VIDEO_GEN_DOWNLOAD_COMFY_MODELS="${AI_VIDEO_GEN_DOWNLOAD_COMFY_MODELS}" AI_VIDEO_GEN_PROVISIONING_STATUS="${AI_VIDEO_GEN_PROVISIONING_STATUS}" CORS_ORIGINS="${CORS}" bash "${ROOT_DIR}/scripts/bootstrap_vast.sh"

mkdir -p "${ROOT_DIR}/.run"
if [ "${BACKEND}" = "comfyui-ltx23" ]; then
  if [ "${AI_VIDEO_GEN_DOWNLOAD_COMFY_MODELS}" = "1" ]; then
    nohup env HF_TOKEN="${HF_TOKEN:-}" HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}" \
      "${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/scripts/download_comfy_ltx23_models.py" \
      --comfy-root "${COMFY_ROOT}" \
      --status-file "${AI_VIDEO_GEN_PROVISIONING_STATUS}" \
      --max-attempts "${AI_VIDEO_GEN_MODEL_DOWNLOAD_MAX_ATTEMPTS}" \
      --retry-delay "${AI_VIDEO_GEN_MODEL_DOWNLOAD_RETRY_DELAY}" \
      > "${ROOT_DIR}/.run/model-download.out.log" \
      2> "${ROOT_DIR}/.run/model-download.err.log" \
      < /dev/null &
    echo $! > "${ROOT_DIR}/.run/model-download.pid"
  else
    "${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/scripts/download_comfy_ltx23_models.py" \
      --comfy-root "${COMFY_ROOT}" \
      --status-file "${AI_VIDEO_GEN_PROVISIONING_STATUS}" \
      --verify-only || true
  fi
fi

PORT="${PORT}" GENERATOR_BACKEND="${BACKEND}" GENERATOR_API_URL="${GENERATOR_API_URL}" CORS_ORIGINS="${CORS}" COMFYUI_ROOT="${COMFY_ROOT}" AI_VIDEO_GEN_PROVISIONING_STATUS="${AI_VIDEO_GEN_PROVISIONING_STATUS}" \
  bash "${ROOT_DIR}/scripts/run_remote_server.sh" >/dev/null

cat <<EOF
Deploy complete.
AI-Video-Gen starts immediately on port ${PORT}.
The frontend shows ComfyUI/model download readiness from:
  ${AI_VIDEO_GEN_PROVISIONING_STATUS}

Recommended SSH tunnel:
  ssh -L 8080:localhost:${PORT} <user>@<host>

Then open:
  http://127.0.0.1:8080/
EOF
