#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
MODELS="${MODELS:-all}"
PORT="${PORT:-8080}"
PYTHON_BIN="${PYTHON_BIN:-}"

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

python_supports_project() {
  local candidate="$1"
  "${candidate}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)'
}

resolve_python() {
  if [ -n "${PYTHON_BIN}" ] && command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    if python_supports_project "${PYTHON_BIN}"; then
      echo "${PYTHON_BIN}"
      return 0
    fi
    echo "Configured PYTHON_BIN='${PYTHON_BIN}' is below Python 3.12." >&2
    return 1
  fi

  for candidate in /usr/bin/python3.12 python3.12 python3; do
    if ! command -v "${candidate}" >/dev/null 2>&1; then
      continue
    fi
    if python_supports_project "${candidate}"; then
      echo "${candidate}"
      return 0
    fi
  done

  if command -v apt-get >/dev/null 2>&1; then
    ensure_apt_packages python3.12 python3.12-venv python3-pip git
    if command -v /usr/bin/python3.12 >/dev/null 2>&1 && python_supports_project /usr/bin/python3.12; then
      echo "/usr/bin/python3.12"
      return 0
    fi
  fi

  echo "A Python 3.12+ interpreter is required, but none was found." >&2
  return 1
}

cd "${ROOT_DIR}"
mkdir -p models data/jobs data/archives data/tmp

ensure_apt_packages git

PYTHON_BIN="$(resolve_python)"
echo "Using Python interpreter: ${PYTHON_BIN} ($("${PYTHON_BIN}" -V 2>&1))"

if ! "${PYTHON_BIN}" -m venv --help >/dev/null 2>&1; then
  if [ "${PYTHON_BIN}" = "/usr/bin/python3.12" ] || [ "${PYTHON_BIN}" = "python3.12" ]; then
    ensure_apt_packages python3.12-venv python3-pip
  else
    ensure_apt_packages python3-venv python3-pip
  fi
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
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
