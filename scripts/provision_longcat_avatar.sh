#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LONGCAT_REPO_DIR="${LONGCAT_REPO_DIR:-/workspace/LongCat-Video}"
LONGCAT_REPO_URL="${LONGCAT_REPO_URL:-https://github.com/meituan-longcat/LongCat-Video.git}"
LONGCAT_REPO_REF="${LONGCAT_REPO_REF:-6b3f4b8582a8bc3f20f795735f5383716c4ba794}"
LONGCAT_CONDA_ENV_DIR="${LONGCAT_CONDA_ENV_DIR:-/opt/conda/envs/longcat-video}"
LONGCAT_PROVISIONING_STATUS="${LONGCAT_PROVISIONING_STATUS:-${ROOT_DIR}/data/longcat-provisioning-status.json}"
CONDA_BIN="${LONGCAT_CONDA_BIN:-/opt/conda/bin/conda}"

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

write_status "provisioning" "2" "Preparing LongCat runtime."
if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends ffmpeg git libsndfile1
  rm -rf /var/lib/apt/lists/*
fi
if [ ! -d "${LONGCAT_REPO_DIR}/.git" ]; then
  git clone "${LONGCAT_REPO_URL}" "${LONGCAT_REPO_DIR}"
fi
git -C "${LONGCAT_REPO_DIR}" fetch --all --tags --prune
git -C "${LONGCAT_REPO_DIR}" checkout "${LONGCAT_REPO_REF}"
"${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/scripts/patch_longcat_runtime.py" --repo "${LONGCAT_REPO_DIR}"

write_status "provisioning" "8" "Preparing Python 3.10 environment."
if [ ! -x "${LONGCAT_CONDA_ENV_DIR}/bin/python" ]; then
  "${CONDA_BIN}" create -y -p "${LONGCAT_CONDA_ENV_DIR}" python=3.10
fi
PYTHON_BIN="${LONGCAT_CONDA_ENV_DIR}/bin/python"
HF_BIN="${LONGCAT_CONDA_ENV_DIR}/bin/hf"
"${PYTHON_BIN}" -m pip install --upgrade pip "setuptools<82" wheel
"${PYTHON_BIN}" -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
"${PYTHON_BIN}" -m pip install numpy==1.26.4 ninja psutil packaging huggingface_hub

REQ_BASE="${ROOT_DIR}/data/tmp/longcat-requirements.txt"
REQ_AVATAR="${ROOT_DIR}/data/tmp/longcat-requirements-avatar.txt"
mkdir -p "$(dirname "${REQ_BASE}")"
grep -Ev '^(torch|torchvision|torchaudio|numpy|flash-attn)([<=>].*)?$' "${LONGCAT_REPO_DIR}/requirements.txt" > "${REQ_BASE}"
grep -Ev '^(torch|torchvision|torchaudio|numpy|libsndfile1|tritonserverclient)([<=>].*)?$' "${LONGCAT_REPO_DIR}/requirements_avatar.txt" > "${REQ_AVATAR}"
"${PYTHON_BIN}" -m pip install -r "${REQ_BASE}" -r "${REQ_AVATAR}"
"${PYTHON_BIN}" -m pip install flash-attn==2.7.4.post1 --no-build-isolation

write_status "downloading" "20" "Downloading LongCat base and Avatar 1.5 weights."
HF_ARGS=()
if [ -n "${HF_TOKEN:-}" ]; then
  HF_ARGS+=(--token "${HF_TOKEN}")
fi
"${HF_BIN}" download meituan-longcat/LongCat-Video \
  --local-dir "${LONGCAT_REPO_DIR}/weights/LongCat-Video" "${HF_ARGS[@]}"
write_status "downloading" "58" "LongCat base weights ready; downloading Avatar 1.5."
"${HF_BIN}" download meituan-longcat/LongCat-Video-Avatar-1.5 \
  --local-dir "${LONGCAT_REPO_DIR}/weights/LongCat-Video-Avatar-1.5" "${HF_ARGS[@]}"

write_status "ready" "100" "LongCat Video Avatar 1.5 is ready."
trap - ERR
