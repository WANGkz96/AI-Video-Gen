#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LONGCAT_REPO_DIR="${LONGCAT_REPO_DIR:-/workspace/LongCat-Video}"
LONGCAT_REPO_URL="${LONGCAT_REPO_URL:-https://github.com/meituan-longcat/LongCat-Video.git}"
LONGCAT_REPO_REF="${LONGCAT_REPO_REF:-6b3f4b8582a8bc3f20f795735f5383716c4ba794}"
LONGCAT_CONDA_ENV_DIR="${LONGCAT_CONDA_ENV_DIR:-/opt/conda/envs/longcat-video}"
LONGCAT_MODEL_ROOT="${LONGCAT_MODEL_ROOT:-${LONGCAT_REPO_DIR}/weights}"
LONGCAT_PROVISIONING_STATUS="${LONGCAT_PROVISIONING_STATUS:-${ROOT_DIR}/data/longcat-provisioning-status.json}"
CONDA_BIN="${LONGCAT_CONDA_BIN:-/opt/conda/bin/conda}"
MODEL_DOWNLOAD_CONCURRENCY="${AI_VIDEO_GEN_MODEL_DOWNLOAD_CONCURRENCY:-3}"

# Hugging Face defaults to eight workers, which competes aggressively with the
# rest of a Packet bootstrap.  Keep the per-branch fan-out bounded while still
# allowing several independent checkpoint files to fill the available link.
if ! [[ "${MODEL_DOWNLOAD_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]]; then
  MODEL_DOWNLOAD_CONCURRENCY=3
elif [ "${MODEL_DOWNLOAD_CONCURRENCY}" -gt 3 ]; then
  MODEL_DOWNLOAD_CONCURRENCY=3
fi

write_status() {
  local status="$1"
  local progress="$2"
  local message="$3"
  local error="${4:-}"
  mkdir -p "$(dirname "${LONGCAT_PROVISIONING_STATUS}")"
  "${ROOT_DIR}/.venv/bin/python" - "${LONGCAT_PROVISIONING_STATUS}" "${status}" "${progress}" "${message}" "${error}" <<'PY'
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

target, status, progress, message, error = sys.argv[1:]
Path(target).write_text(json.dumps({
    "schemaVersion": "ai-video-gen.longcat-provisioning.v1",
    "updatedAt": datetime.now(UTC).isoformat(),
    "status": status,
    "progressPercent": float(progress),
    "message": message,
    "error": error or None,
}, indent=2), encoding="utf-8")
PY
}

fail() {
  local code=$?
  write_status "error" "0" "LongCat provisioning failed." "exit ${code}"
  exit "${code}"
}
trap fail ERR

mkdir -p "${ROOT_DIR}/data/tmp"
# The API can receive a retry while the instance bootstrap is still running.
# A second installer would race the first one over git/conda/pip/HF cache and
# can leave an otherwise valid Avatar setup only partially installed.  Keep a
# process-held advisory lock for this instance; a later retry may safely run
# once the first provisioning process has exited.
if command -v flock >/dev/null 2>&1; then
  exec 9>"${ROOT_DIR}/data/tmp/longcat-provision.lock"
  if ! flock -n 9; then
    echo "LongCat provisioning is already running; leaving the active installer in control." >&2
    exit 0
  fi
else
  echo "flock is unavailable; continuing without a LongCat provisioning lock." >&2
fi

write_status "provisioning" "2" "Preparing LongCat runtime."
if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  if [ "$(id -u)" -eq 0 ]; then
    apt-get update
    apt-get install -y --no-install-recommends ffmpeg git libsndfile1
    rm -rf /var/lib/apt/lists/*
  elif sudo -n true >/dev/null 2>&1; then
    sudo apt-get update
    sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg git libsndfile1
    sudo rm -rf /var/lib/apt/lists/*
  else
    echo "LongCat provisioning requires root or passwordless sudo for system packages." >&2
    exit 1
  fi
fi
if [ ! -d "${LONGCAT_REPO_DIR}/.git" ]; then
  # The runtime only needs the pinned checkout. Avoid downloading the full
  # LongCat history/tags on the small ephemeral Packet root disk.
  git clone --depth 1 --no-tags "${LONGCAT_REPO_URL}" "${LONGCAT_REPO_DIR}"
fi
git -C "${LONGCAT_REPO_DIR}" fetch --depth 1 origin "${LONGCAT_REPO_REF}"
git -C "${LONGCAT_REPO_DIR}" checkout "${LONGCAT_REPO_REF}"
"${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/scripts/patch_longcat_runtime.py" --repo "${LONGCAT_REPO_DIR}"
install -m 0755 "${ROOT_DIR}/scripts/run_longcat_avatar_batch.py" "${LONGCAT_REPO_DIR}/run_longcat_avatar_batch.py"

write_status "provisioning" "8" "Preparing Python 3.10 environment."
if [ ! -x "${LONGCAT_CONDA_ENV_DIR}/bin/python" ]; then
  if [ -x "${CONDA_BIN}" ]; then
    "${CONDA_BIN}" create -y -p "${LONGCAT_CONDA_ENV_DIR}" python=3.10
  elif command -v uv >/dev/null 2>&1; then
    # Newer Vast templates ship uv and a system venv instead of /opt/conda.
    # Keep the existing env path contract used by the backend adapter while
    # letting uv install a managed Python 3.10 runtime into that directory.
    uv venv --python 3.10 --seed "${LONGCAT_CONDA_ENV_DIR}"
  else
    echo "Neither conda (${CONDA_BIN}) nor uv is available to create the LongCat Python 3.10 environment." >&2
    exit 127
  fi
fi
PYTHON_BIN="${LONGCAT_CONDA_ENV_DIR}/bin/python"
HF_BIN="${LONGCAT_CONDA_ENV_DIR}/bin/hf"
if ! "${PYTHON_BIN}" -m pip --version >/dev/null 2>&1; then
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "${PYTHON_BIN}" pip "setuptools<82" wheel
  else
    "${PYTHON_BIN}" -m ensurepip --upgrade
  fi
fi
"${PYTHON_BIN}" -m pip install --upgrade pip "setuptools<82" wheel
"${PYTHON_BIN}" -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
"${PYTHON_BIN}" -m pip install numpy==1.26.4 ninja psutil packaging huggingface_hub

REQ_BASE="${ROOT_DIR}/data/tmp/longcat-requirements.txt"
REQ_AVATAR="${ROOT_DIR}/data/tmp/longcat-requirements-avatar.txt"
mkdir -p "$(dirname "${REQ_BASE}")"
grep -Ev '^(torch|torchvision|torchaudio|numpy|flash-attn)([<=>].*)?$' "${LONGCAT_REPO_DIR}/requirements.txt" > "${REQ_BASE}"
grep -Ev '^(torch|torchvision|torchaudio|numpy|sympy|libsndfile1|tritonserverclient)([<=>].*)?$' "${LONGCAT_REPO_DIR}/requirements_avatar.txt" > "${REQ_AVATAR}"
"${PYTHON_BIN}" -m pip install -r "${REQ_BASE}" -r "${REQ_AVATAR}"
"${PYTHON_BIN}" -m pip install flash-attn==2.7.4.post1 --no-build-isolation

write_status "downloading" "20" "Downloading only LongCat files used by Avatar 1.5 INT8 + distilled runtime."
HF_ARGS=()
if [ -n "${HF_TOKEN:-}" ]; then
  HF_ARGS+=(--token "${HF_TOKEN}")
fi
"${HF_BIN}" download meituan-longcat/LongCat-Video \
  --local-dir "${LONGCAT_MODEL_ROOT}/LongCat-Video" \
  --max-workers "${MODEL_DOWNLOAD_CONCURRENCY}" \
  --include "tokenizer/**" "text_encoder/**" "vae/**" \
  "${HF_ARGS[@]}"
write_status "downloading" "58" "LongCat runtime weights ready; downloading Avatar 1.5 INT8 components."
"${HF_BIN}" download meituan-longcat/LongCat-Video-Avatar-1.5 \
  --local-dir "${LONGCAT_MODEL_ROOT}/LongCat-Video-Avatar-1.5" \
  --max-workers "${MODEL_DOWNLOAD_CONCURRENCY}" \
  --include \
  "base_model_int8/**" \
  "lora/dmd_lora.safetensors" \
  "whisper-large-v3/added_tokens.json" \
  "whisper-large-v3/config.json" \
  "whisper-large-v3/generation_config.json" \
  "whisper-large-v3/merges.txt" \
  "whisper-large-v3/model.safetensors" \
  "whisper-large-v3/normalizer.json" \
  "whisper-large-v3/preprocessor_config.json" \
  "whisper-large-v3/special_tokens_map.json" \
  "whisper-large-v3/tokenizer.json" \
  "whisper-large-v3/tokenizer_config.json" \
  "whisper-large-v3/vocab.json" \
  "vocal_separator/Kim_Vocal_2.onnx" \
  "scheduler/**" \
  "${HF_ARGS[@]}"

write_status "ready" "100" "LongCat Video Avatar 1.5 is ready."
trap - ERR
