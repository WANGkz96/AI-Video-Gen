#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
MODELS="${MODELS:-all}"
PORT="${PORT:-8080}"

ensure_apt_packages() {
  if ! command -v apt-get >/dev/null 2>&1; then
    return 0
  fi
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y "$@"
}

ensure_nodejs() {
  if command -v npm >/dev/null 2>&1; then
    return 0
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "npm is not installed and apt-get is unavailable; frontend build will be skipped." >&2
    return 0
  fi

  ensure_apt_packages curl ca-certificates gnupg
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
  ensure_apt_packages nodejs
}

cd "${ROOT_DIR}"
mkdir -p models data/jobs data/archives data/tmp

if ! python3 -m venv --help >/dev/null 2>&1; then
  ensure_apt_packages python3 python3-venv python3-pip
fi

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip "setuptools<82" wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
python -m pip install -e ".[models]"

ensure_nodejs
if command -v npm >/dev/null 2>&1; then
  (cd frontend && npm install && npm run build)
fi

if [ ! -f .env ]; then
  cp .env.example .env
fi
python -m backend.app.cli download-models --models "${MODELS}"

echo "Bootstrap complete."
echo "Run with:"
echo "  source .venv/bin/activate"
echo "  PORT=${PORT} uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT}"
