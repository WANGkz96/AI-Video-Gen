#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/WANGkz96/AI-Video-Gen.git}"
REPO_REF="${REPO_REF:-master}"
WORK_ROOT="${WORK_ROOT:-/root/work}"
APP_DIR="${APP_DIR:-${WORK_ROOT}/AI-Video-Gen}"
PORT="${PORT:-8090}"
MODELS="${MODELS:-all}"
GENERATOR_BACKEND="${GENERATOR_BACKEND:-ltx-2.3}"
CORS_ORIGINS="${CORS_ORIGINS:-http://127.0.0.1:${PORT},http://localhost:${PORT}}"

mkdir -p "${WORK_ROOT}"

if [ ! -d "${APP_DIR}/.git" ]; then
  git clone "${REPO_URL}" "${APP_DIR}"
fi

cd "${APP_DIR}"
git fetch --all --tags --prune
git checkout "${REPO_REF}"

if git show-ref --verify --quiet "refs/remotes/origin/${REPO_REF}"; then
  git pull --ff-only origin "${REPO_REF}"
fi

PORT="${PORT}" \
MODELS="${MODELS}" \
GENERATOR_BACKEND="${GENERATOR_BACKEND}" \
CORS_ORIGINS="${CORS_ORIGINS}" \
bash ./scripts/deploy_vast.sh
