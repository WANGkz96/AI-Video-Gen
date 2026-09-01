#!/usr/bin/env bash
set -euo pipefail

# This file is uploaded as Packet's startup_script. It is deliberately a small
# immutable launcher; all real setup stays versioned in the checked-out app.
REPO_URL="${REPO_URL:-https://github.com/WANGkz96/AI-Video-Gen.git}"
REPO_REF="${REPO_REF:-master}"
WORK_ROOT="${WORK_ROOT:-/workspace}"
APP_DIR="${APP_DIR:-${WORK_ROOT}/AI-Video-Gen}"

mkdir -p "${WORK_ROOT}"
if [ ! -d "${APP_DIR}/.git" ]; then
  git clone "${REPO_URL}" "${APP_DIR}"
fi
git -C "${APP_DIR}" fetch --all --tags --prune
git -C "${APP_DIR}" checkout "${REPO_REF}"
if git -C "${APP_DIR}" show-ref --verify --quiet "refs/remotes/origin/${REPO_REF}"; then
  git -C "${APP_DIR}" pull --ff-only origin "${REPO_REF}"
fi

exec env \
  PORT="${PORT:-8090}" \
  GENERATOR_BACKEND="comfyui-ltx25" \
  GENERATOR_API_URL="${GENERATOR_API_URL:-http://127.0.0.1:18188}" \
  AI_VIDEO_GEN_ENABLE_LTX="${AI_VIDEO_GEN_ENABLE_LTX:-1}" \
  AI_VIDEO_GEN_ENABLE_LONGCAT="${AI_VIDEO_GEN_ENABLE_LONGCAT:-0}" \
  AI_VIDEO_GEN_RELEASE_LONGCAT_WEIGHTS_AFTER_BRANCH="${AI_VIDEO_GEN_RELEASE_LONGCAT_WEIGHTS_AFTER_BRANCH:-1}" \
  AI_VIDEO_GEN_PERSISTENT_MODEL_CACHE_DIR="${AI_VIDEO_GEN_PERSISTENT_MODEL_CACHE_DIR:-}" \
  AI_VIDEO_GEN_AUTH_REQUIRED="${AI_VIDEO_GEN_AUTH_REQUIRED:-1}" \
  AI_VIDEO_GEN_API_TOKEN="${AI_VIDEO_GEN_API_TOKEN:-}" \
  HF_TOKEN="${HF_TOKEN:-}" \
  REPO_URL="${REPO_URL}" \
  REPO_REF="${REPO_REF}" \
  bash "${APP_DIR}/scripts/deploy_packet.sh"
