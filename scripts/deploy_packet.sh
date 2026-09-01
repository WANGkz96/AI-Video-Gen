#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8090}"
COMFY_ROOT="${COMFYUI_ROOT:-/workspace/ComfyUI}"
GENERATOR_API_URL="${GENERATOR_API_URL:-http://127.0.0.1:18188}"
AI_VIDEO_GEN_ENABLE_LTX="${AI_VIDEO_GEN_ENABLE_LTX:-1}"
AI_VIDEO_GEN_ENABLE_LONGCAT="${AI_VIDEO_GEN_ENABLE_LONGCAT:-0}"
# A mixed Packet job is deliberately serialized: LongCat downloads and renders
# first, then releases its weights before the LTX model pack starts.  This
# avoids filling Packet's 150 GB ephemeral root and needs no process pausing.
AI_VIDEO_GEN_RELEASE_LONGCAT_WEIGHTS_AFTER_BRANCH="${AI_VIDEO_GEN_RELEASE_LONGCAT_WEIGHTS_AFTER_BRANCH:-1}"
STATUS_FILE="${AI_VIDEO_GEN_PROVISIONING_STATUS:-${ROOT_DIR}/data/provisioning-status.json}"
LONGCAT_STATUS_FILE="${LONGCAT_PROVISIONING_STATUS:-${ROOT_DIR}/data/longcat-provisioning-status.json}"
LONGCAT_RELEASE_FILE="${AI_VIDEO_GEN_LONGCAT_BRANCH_RELEASE_FILE:-${ROOT_DIR}/.run/longcat-branch-released.json}"
MODEL_DOWNLOAD_CONCURRENCY="${AI_VIDEO_GEN_MODEL_DOWNLOAD_CONCURRENCY:-3}"

if ! [[ "${MODEL_DOWNLOAD_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]]; then
  MODEL_DOWNLOAD_CONCURRENCY=3
elif [ "${MODEL_DOWNLOAD_CONCURRENCY}" -gt 3 ]; then
  MODEL_DOWNLOAD_CONCURRENCY=3
fi
export AI_VIDEO_GEN_MODEL_DOWNLOAD_CONCURRENCY="${MODEL_DOWNLOAD_CONCURRENCY}"

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
AI_VIDEO_GEN_LONGCAT_BRANCH_RELEASE_FILE="${LONGCAT_RELEASE_FILE}" \
AI_VIDEO_GEN_MODEL_DOWNLOAD_CONCURRENCY="${MODEL_DOWNLOAD_CONCURRENCY}" \
AI_VIDEO_GEN_PROVISIONING_STATUS="${STATUS_FILE}" \
LONGCAT_PROVISIONING_STATUS="${LONGCAT_STATUS_FILE}" \
LONGCAT_CONDA_ENV_DIR="${LONGCAT_CONDA_ENV_DIR:-/workspace/.venvs/longcat-video}" \
bash "${ROOT_DIR}/scripts/bootstrap_vast.sh"

# ComfyUI shares the application's Python 3.12 virtual environment.  Packet's
# base Ubuntu image marks its system interpreter as externally managed, while
# the venv is exactly where its runtime dependencies belong.
if [ "${AI_VIDEO_GEN_ENABLE_LTX}" = "1" ]; then
  COMFYUI_ROOT="${COMFY_ROOT}" \
  COMFYUI_PORT="18188" \
  COMFY_PYTHON="${COMFY_PYTHON:-${ROOT_DIR}/.venv/bin/python}" \
  bash "${ROOT_DIR}/scripts/provision_packet_comfyui.sh"
else
  echo "Skipping ComfyUI provisioning: this Packet job has no LTX branch."
fi

mkdir -p "${ROOT_DIR}/.run"
rm -f "${LONGCAT_RELEASE_FILE}"

start_ltx_download() {
  nohup env HF_TOKEN="${HF_TOKEN:-}" HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}" \
    "${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/scripts/download_comfy_ltx25_models.py" \
    --comfy-root "${COMFY_ROOT}" \
    --status-file "${STATUS_FILE}" \
    --max-workers "${MODEL_DOWNLOAD_CONCURRENCY}" \
    --max-attempts "${AI_VIDEO_GEN_MODEL_DOWNLOAD_MAX_ATTEMPTS:-60}" \
    --retry-delay "${AI_VIDEO_GEN_MODEL_DOWNLOAD_RETRY_DELAY:-20}" \
    > "${ROOT_DIR}/.run/ltx25-download.out.log" \
    2> "${ROOT_DIR}/.run/ltx25-download.err.log" < /dev/null &
  echo $! > "${ROOT_DIR}/.run/ltx25-download.pid"
}

if [ "${AI_VIDEO_GEN_ENABLE_LONGCAT}" = "1" ]; then
  nohup env HF_TOKEN="${HF_TOKEN:-}" \
    LONGCAT_REPO_DIR="${LONGCAT_REPO_DIR:-/workspace/LongCat-Video}" \
    LONGCAT_AVATAR_CHECKPOINT_DIR="${LONGCAT_AVATAR_CHECKPOINT_DIR:-/workspace/LongCat-Video/weights/LongCat-Video-Avatar-1.5}" \
    LONGCAT_CONDA_ENV_DIR="${LONGCAT_CONDA_ENV_DIR:-/workspace/.venvs/longcat-video}" \
    LONGCAT_PROVISIONING_STATUS="${LONGCAT_STATUS_FILE}" \
    AI_VIDEO_GEN_MODEL_DOWNLOAD_CONCURRENCY="${MODEL_DOWNLOAD_CONCURRENCY}" \
    bash "${ROOT_DIR}/scripts/provision_longcat_avatar.sh" \
    > "${ROOT_DIR}/.run/longcat-provision.out.log" \
    2> "${ROOT_DIR}/.run/longcat-provision.err.log" < /dev/null &
  echo $! > "${ROOT_DIR}/.run/longcat-provision.pid"
fi

if [ "${AI_VIDEO_GEN_ENABLE_LTX}" = "1" ] && [ "${AI_VIDEO_GEN_ENABLE_LONGCAT}" = "1" ]; then
  nohup env \
    "${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/scripts/coordinate_packet_model_branches.py" \
    --longcat-status-file "${LONGCAT_STATUS_FILE}" \
    --ltx-status-file "${STATUS_FILE}" \
    --release-file "${LONGCAT_RELEASE_FILE}" \
    --poll-sec "${AI_VIDEO_GEN_PACKET_BRANCH_SEQUENCE_POLL_SEC:-2}" \
    -- \
    "${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/scripts/download_comfy_ltx25_models.py" \
    --comfy-root "${COMFY_ROOT}" \
    --status-file "${STATUS_FILE}" \
    --max-workers "${MODEL_DOWNLOAD_CONCURRENCY}" \
    --max-attempts "${AI_VIDEO_GEN_MODEL_DOWNLOAD_MAX_ATTEMPTS:-60}" \
    --retry-delay "${AI_VIDEO_GEN_MODEL_DOWNLOAD_RETRY_DELAY:-20}" \
    > "${ROOT_DIR}/.run/ltx25-sequence.out.log" \
    2> "${ROOT_DIR}/.run/ltx25-sequence.err.log" < /dev/null &
  echo $! > "${ROOT_DIR}/.run/ltx25-sequence.pid"
elif [ "${AI_VIDEO_GEN_ENABLE_LTX}" = "1" ]; then
  start_ltx_download
fi

# The API starts before model downloads complete.  For a mixed batch the
# coordinator keeps LTX unavailable until LongCat's completed branch releases
# its model directory; single-backend batches retain immediate startup.
PORT="${PORT}" \
GENERATOR_BACKEND="comfyui-ltx25" \
GENERATOR_API_URL="${GENERATOR_API_URL}" \
COMFYUI_ROOT="${COMFY_ROOT}" \
AI_VIDEO_GEN_PROVISIONING_STATUS="${STATUS_FILE}" \
LONGCAT_PROVISIONING_STATUS="${LONGCAT_STATUS_FILE}" \
AI_VIDEO_GEN_LONGCAT_BRANCH_RELEASE_FILE="${LONGCAT_RELEASE_FILE}" \
bash "${ROOT_DIR}/scripts/run_remote_server.sh" >/dev/null

echo "Packet deploy started: API port ${PORT}; ComfyUI stays private on 18188."
