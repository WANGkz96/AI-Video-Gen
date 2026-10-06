#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8090}"
# A Vast base image may contain its own ComfyUI checkout. Keep the pinned
# runtime isolated so its Git checkout cannot overwrite provider image files.
COMFY_ROOT="${AI_VIDEO_GEN_COMFYUI_ROOT:-/workspace/AI-Video-Gen-ComfyUI}"
GENERATOR_API_URL="http://127.0.0.1:18188"
# Existing Vast templates export LTX 2.3 workflow paths. Override them in the
# parent shell too: bootstrap_vast.sh cannot change inherited API variables.
export COMFYUI_ROOT="${COMFY_ROOT}"
export COMFYUI_T2V_WORKFLOW="${COMFY_ROOT}/blueprints/video_ltx2_5_t2v.json"
export COMFYUI_I2V_WORKFLOW="${COMFY_ROOT}/blueprints/video_ltx2_5_i2v.json"
export AI_VIDEO_GEN_LTX_MODEL_ROOT="${COMFY_ROOT}"
export LTX25_MEGAPIXELS="${LTX25_MEGAPIXELS:-0.9}"
STATUS_FILE="${AI_VIDEO_GEN_PROVISIONING_STATUS:-${ROOT_DIR}/data/provisioning-status.json}"
LONGCAT_STATUS_FILE="${LONGCAT_PROVISIONING_STATUS:-${ROOT_DIR}/data/longcat-provisioning-status.json}"
LONGCAT_ENV_DIR="${LONGCAT_CONDA_ENV_DIR:-/workspace/.venvs/longcat-video}"
ENABLE_LTX="${AI_VIDEO_GEN_ENABLE_LTX:-1}"
ENABLE_LONGCAT="${AI_VIDEO_GEN_ENABLE_LONGCAT:-0}"
DOWNLOAD_WORKERS="${AI_VIDEO_GEN_MODEL_DOWNLOAD_CONCURRENCY:-3}"
CORS="${CORS_ORIGINS:-http://127.0.0.1:${PORT},http://localhost:${PORT}}"

# Vast RTX PRO 6000 instances have 96 GB VRAM. The two generators use
# separate Python runtimes and run their generation branches sequentially.
PORT="${PORT}" \
GENERATOR_BACKEND="comfyui-ltx25" \
GENERATOR_API_URL="${GENERATOR_API_URL}" \
COMFYUI_ROOT="${COMFY_ROOT}" \
AI_VIDEO_GEN_LTX_MODEL_ROOT="${COMFY_ROOT}" \
COMFYUI_T2V_WORKFLOW="${COMFY_ROOT}/blueprints/video_ltx2_5_t2v.json" \
COMFYUI_I2V_WORKFLOW="${COMFY_ROOT}/blueprints/video_ltx2_5_i2v.json" \
LTX25_MEGAPIXELS="${LTX25_MEGAPIXELS}" \
AI_VIDEO_GEN_ENABLE_LTX="${ENABLE_LTX}" \
AI_VIDEO_GEN_ENABLE_LONGCAT="${ENABLE_LONGCAT}" \
AI_VIDEO_GEN_RELEASE_LONGCAT_WEIGHTS_AFTER_BRANCH="0" \
AI_VIDEO_GEN_EXECUTION_PROFILE="standard" \
AI_VIDEO_GEN_PROVISIONING_STATUS="${STATUS_FILE}" \
LONGCAT_PROVISIONING_STATUS="${LONGCAT_STATUS_FILE}" \
LONGCAT_CONDA_ENV_DIR="${LONGCAT_ENV_DIR}" \
AI_VIDEO_GEN_MODEL_DOWNLOAD_CONCURRENCY="${DOWNLOAD_WORKERS}" \
CORS_ORIGINS="${CORS}" \
bash "${ROOT_DIR}/scripts/bootstrap_vast.sh"

mkdir -p "${ROOT_DIR}/.run"
export PATH="${ROOT_DIR}/.venv/bin:${PATH}"

if [ "${ENABLE_LTX}" = "1" ]; then
  COMFYUI_ROOT="${COMFY_ROOT}" \
  AI_VIDEO_GEN_LTX_MODEL_ROOT="${COMFY_ROOT}" \
  COMFYUI_PORT="18188" \
  COMFY_PYTHON="${ROOT_DIR}/.venv/bin/python" \
  bash "${ROOT_DIR}/scripts/provision_packet_comfyui.sh"

  nohup env HF_TOKEN="${HF_TOKEN:-}" HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}" \
    "${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/scripts/download_comfy_ltx25_models.py" \
    --comfy-root "${COMFY_ROOT}" \
    --status-file "${STATUS_FILE}" \
    --max-workers "${DOWNLOAD_WORKERS}" \
    --max-attempts "${AI_VIDEO_GEN_MODEL_DOWNLOAD_MAX_ATTEMPTS:-60}" \
    --retry-delay "${AI_VIDEO_GEN_MODEL_DOWNLOAD_RETRY_DELAY:-20}" \
    > "${ROOT_DIR}/.run/ltx25-download.out.log" \
    2> "${ROOT_DIR}/.run/ltx25-download.err.log" < /dev/null &
  echo $! > "${ROOT_DIR}/.run/ltx25-download.pid"
fi

if [ "${ENABLE_LONGCAT}" = "1" ]; then
  if [ ! -x "${LONGCAT_CONDA_BIN:-/opt/conda/bin/conda}" ] \
    && ! command -v uv >/dev/null 2>&1; then
    "${ROOT_DIR}/.venv/bin/python" -m pip install uv
  fi
  nohup env HF_TOKEN="${HF_TOKEN:-}" \
    PATH="${PATH}" \
    LONGCAT_REPO_DIR="${LONGCAT_REPO_DIR:-/workspace/LongCat-Video}" \
    LONGCAT_CONDA_ENV_DIR="${LONGCAT_ENV_DIR}" \
    LONGCAT_PROVISIONING_STATUS="${LONGCAT_STATUS_FILE}" \
    AI_VIDEO_GEN_MODEL_DOWNLOAD_CONCURRENCY="${DOWNLOAD_WORKERS}" \
    bash "${ROOT_DIR}/scripts/provision_longcat_avatar.sh" \
    > "${ROOT_DIR}/.run/longcat-provision.out.log" \
    2> "${ROOT_DIR}/.run/longcat-provision.err.log" < /dev/null &
  echo $! > "${ROOT_DIR}/.run/longcat-provision.pid"
fi

if [ "${ENABLE_LTX}" = "1" ]; then
  GENERATOR_API_URL="${GENERATOR_API_URL}" \
  COMFYUI_T2V_WORKFLOW="${COMFYUI_T2V_WORKFLOW}" \
  COMFYUI_I2V_WORKFLOW="${COMFYUI_I2V_WORKFLOW}" \
  "${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/scripts/wait_for_comfyui_ready.py"
fi

PORT="${PORT}" \
GENERATOR_BACKEND="comfyui-ltx25" \
GENERATOR_API_URL="${GENERATOR_API_URL}" \
COMFYUI_ROOT="${COMFY_ROOT}" \
COMFYUI_T2V_WORKFLOW="${COMFYUI_T2V_WORKFLOW}" \
COMFYUI_I2V_WORKFLOW="${COMFYUI_I2V_WORKFLOW}" \
AI_VIDEO_GEN_PROVISIONING_STATUS="${STATUS_FILE}" \
LONGCAT_PROVISIONING_STATUS="${LONGCAT_STATUS_FILE}" \
AI_VIDEO_GEN_EXECUTION_PROFILE="standard" \
CORS_ORIGINS="${CORS}" \
bash "${ROOT_DIR}/scripts/run_remote_server.sh" >/dev/null

echo "Vast deploy started: LTX 2.5=${ENABLE_LTX}, LongCat=${ENABLE_LONGCAT}, API port ${PORT}."
