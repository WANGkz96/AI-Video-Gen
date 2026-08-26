#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8090}"
COMFY_ROOT="${COMFYUI_ROOT:-/workspace/ComfyUI}"
GENERATOR_API_URL="${GENERATOR_API_URL:-http://127.0.0.1:18188}"
AI_VIDEO_GEN_ENABLE_LTX="${AI_VIDEO_GEN_ENABLE_LTX:-1}"
AI_VIDEO_GEN_ENABLE_LONGCAT="${AI_VIDEO_GEN_ENABLE_LONGCAT:-0}"
# Packet's 150 GB ephemeral root cannot retain both LongCat's runtime weights
# and LTX 2.5 while a mixed batch is running.  The worker releases LongCat
# only after its branch has finished and LTX is still pending.
AI_VIDEO_GEN_RELEASE_LONGCAT_WEIGHTS_AFTER_BRANCH="${AI_VIDEO_GEN_RELEASE_LONGCAT_WEIGHTS_AFTER_BRANCH:-1}"
STATUS_FILE="${AI_VIDEO_GEN_PROVISIONING_STATUS:-${ROOT_DIR}/data/provisioning-status.json}"

cd "${ROOT_DIR}"

# Python 3.12 powers the service/ComfyUI. LongCat remains isolated in a Python
# 3.10 uv environment, so its requirements cannot disturb ComfyUI.
if ! command -v uv >/dev/null 2>&1; then
  # Packet's Ubuntu 24.04 image marks the system Python as externally
  # managed (PEP 668). uv is only a bootstrap utility here; the project
  # itself is installed into its own virtual environments below.
  python3 -m pip install --break-system-packages uv
fi
export PATH="${HOME}/.local/bin:${PATH}"

PORT="${PORT}" \
PYTHON_BIN="${PYTHON_BIN:-python3}" \
GENERATOR_BACKEND="comfyui-ltx25" \
GENERATOR_API_URL="${GENERATOR_API_URL}" \
COMFYUI_ROOT="${COMFY_ROOT}" \
COMFYUI_T2V_WORKFLOW="${COMFY_ROOT}/blueprints/LTX-2.5_T2V_I2V_Single_Stage_Distilled.json" \
COMFYUI_I2V_WORKFLOW="${COMFY_ROOT}/blueprints/LTX-2.5_T2V_I2V_Single_Stage_Distilled.json" \
AI_VIDEO_GEN_ENABLE_LTX="${AI_VIDEO_GEN_ENABLE_LTX}" \
AI_VIDEO_GEN_ENABLE_LONGCAT="${AI_VIDEO_GEN_ENABLE_LONGCAT}" \
AI_VIDEO_GEN_RELEASE_LONGCAT_WEIGHTS_AFTER_BRANCH="${AI_VIDEO_GEN_RELEASE_LONGCAT_WEIGHTS_AFTER_BRANCH}" \
AI_VIDEO_GEN_PROVISIONING_STATUS="${STATUS_FILE}" \
LONGCAT_CONDA_ENV_DIR="${LONGCAT_CONDA_ENV_DIR:-/workspace/.venvs/longcat-video}" \
bash "${ROOT_DIR}/scripts/bootstrap_vast.sh"

# ComfyUI shares the application's Python 3.12 virtual environment.  Packet's
# base Ubuntu image marks its system interpreter as externally managed, while
# the venv is exactly where its runtime dependencies belong.
COMFYUI_ROOT="${COMFY_ROOT}" \
COMFYUI_PORT="18188" \
COMFY_PYTHON="${COMFY_PYTHON:-${ROOT_DIR}/.venv/bin/python}" \
bash "${ROOT_DIR}/scripts/provision_packet_comfyui.sh"

mkdir -p "${ROOT_DIR}/.run"
if [ "${AI_VIDEO_GEN_ENABLE_LTX}" = "1" ]; then
  nohup env HF_TOKEN="${HF_TOKEN:-}" HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}" \
    "${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/scripts/download_comfy_ltx25_models.py" \
    --comfy-root "${COMFY_ROOT}" \
    --status-file "${STATUS_FILE}" \
    --max-attempts "${AI_VIDEO_GEN_MODEL_DOWNLOAD_MAX_ATTEMPTS:-60}" \
    --retry-delay "${AI_VIDEO_GEN_MODEL_DOWNLOAD_RETRY_DELAY:-20}" \
    > "${ROOT_DIR}/.run/ltx25-download.out.log" \
    2> "${ROOT_DIR}/.run/ltx25-download.err.log" < /dev/null &
  echo $! > "${ROOT_DIR}/.run/ltx25-download.pid"
fi

if [ "${AI_VIDEO_GEN_ENABLE_LONGCAT}" = "1" ]; then
  nohup env HF_TOKEN="${HF_TOKEN:-}" \
    LONGCAT_REPO_DIR="${LONGCAT_REPO_DIR:-/workspace/LongCat-Video}" \
    LONGCAT_AVATAR_CHECKPOINT_DIR="${LONGCAT_AVATAR_CHECKPOINT_DIR:-/workspace/LongCat-Video/weights/LongCat-Video-Avatar-1.5}" \
    LONGCAT_CONDA_ENV_DIR="${LONGCAT_CONDA_ENV_DIR:-/workspace/.venvs/longcat-video}" \
    LONGCAT_PROVISIONING_STATUS="${LONGCAT_PROVISIONING_STATUS:-${ROOT_DIR}/data/longcat-provisioning-status.json}" \
    bash "${ROOT_DIR}/scripts/provision_longcat_avatar.sh" \
    > "${ROOT_DIR}/.run/longcat-provision.out.log" \
    2> "${ROOT_DIR}/.run/longcat-provision.err.log" < /dev/null &
  echo $! > "${ROOT_DIR}/.run/longcat-provision.pid"
fi

if [ "${AI_VIDEO_GEN_ENABLE_LTX}" = "1" ] && [ "${AI_VIDEO_GEN_ENABLE_LONGCAT}" = "1" ]; then
  nohup env \
    LTX_PID_FILE="${ROOT_DIR}/.run/ltx25-download.pid" \
    LONGCAT_WEIGHTS_DIR="${LONGCAT_REPO_DIR:-/workspace/LongCat-Video}/weights" \
    PACKET_DISK_GUARD_PATH="/workspace" \
    AI_VIDEO_GEN_PACKET_LTX_MIN_FREE_GB="${AI_VIDEO_GEN_PACKET_LTX_MIN_FREE_GB:-70}" \
    AI_VIDEO_GEN_PACKET_LTX_GUARD_POLL_SEC="${AI_VIDEO_GEN_PACKET_LTX_GUARD_POLL_SEC:-5}" \
    bash "${ROOT_DIR}/scripts/guard_packet_ltx_disk.sh" \
    > "${ROOT_DIR}/.run/ltx25-disk-guard.out.log" \
    2> "${ROOT_DIR}/.run/ltx25-disk-guard.err.log" < /dev/null &
  echo $! > "${ROOT_DIR}/.run/ltx25-disk-guard.pid"
fi

# The API starts before model downloads complete. JobService independently
# schedules each branch as soon as its own provisioning becomes ready.
PORT="${PORT}" \
GENERATOR_BACKEND="comfyui-ltx25" \
GENERATOR_API_URL="${GENERATOR_API_URL}" \
COMFYUI_ROOT="${COMFY_ROOT}" \
AI_VIDEO_GEN_PROVISIONING_STATUS="${STATUS_FILE}" \
bash "${ROOT_DIR}/scripts/run_remote_server.sh" >/dev/null

echo "Packet deploy started: API port ${PORT}; ComfyUI stays private on 18188."
