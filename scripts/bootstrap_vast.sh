#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
MODELS="${MODELS:-all}"
PORT="${PORT:-8080}"

cd "${ROOT_DIR}"
mkdir -p models data/jobs data/archives data/tmp

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
python -m pip install -e ".[models]"

if command -v npm >/dev/null 2>&1; then
  (cd frontend && npm install && npm run build)
fi

cp -n .env.example .env || true
python -m backend.app.cli download-models --models "${MODELS}"

echo "Bootstrap complete."
echo "Run with:"
echo "  source .venv/bin/activate"
echo "  PORT=${PORT} uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT}"

