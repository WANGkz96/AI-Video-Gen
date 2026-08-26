#!/usr/bin/env bash
set -euo pipefail

# Build the ComfyUI side explicitly. Vast supplied this through an opaque image;
# Packet instances are intentionally self-contained and reproducible instead.

COMFY_ROOT="${COMFYUI_ROOT:-/workspace/ComfyUI}"
COMFY_PYTHON="${COMFY_PYTHON:-python3}"
COMFY_PORT="${COMFYUI_PORT:-18188}"
COMFY_REF="${COMFYUI_REF:-12d5279438bfefc058a269eae805ceab6047777f}"
LTX_NODE_REF="${COMFYUI_LTXVIDEO_REF:-15d09abb5a187a8dcaea2fc31fe51ee96e6c9d0d}"
CONVERTER_REF="${COMFYUI_CONVERTER_REF:-bc8538278f82053b3ca10a44d62d02596f8e1a37}"
LTX_NODE_DIR="${COMFY_ROOT}/custom_nodes/ComfyUI-LTXVideo"
CONVERTER_DIR="${COMFY_ROOT}/custom_nodes/comfyui-workflow-to-api-converter-endpoint"
BLUEPRINT_DIR="${COMFY_ROOT}/blueprints"
WORKFLOW_NAME="LTX-2.5_T2V_I2V_Single_Stage_Distilled.json"

sync_repo() {
  local repo_url="$1"
  local target="$2"
  local ref="$3"
  if [ ! -d "${target}/.git" ]; then
    # The Packet bootstrap creates ComfyUI's persistent model/custom-node
    # directories before this script runs.  `git clone <url> <target>` rejects
    # that otherwise harmless non-empty directory, so initialise the checkout
    # in place.  This also keeps future model caches outside Git's control.
    mkdir -p "${target}"
    git -C "${target}" init -q
    git -C "${target}" remote add origin "${repo_url}"
  else
    git -C "${target}" remote set-url origin "${repo_url}"
  fi
  git -C "${target}" fetch --depth=1 origin "${ref}"
  git -C "${target}" checkout --detach --force FETCH_HEAD
}

if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends git ffmpeg libgl1 libglib2.0-0 libsndfile1
  rm -rf /var/lib/apt/lists/*
fi

mkdir -p "${COMFY_ROOT}/custom_nodes" "${BLUEPRINT_DIR}"
sync_repo "https://github.com/comfyanonymous/ComfyUI.git" "${COMFY_ROOT}" "${COMFY_REF}"
sync_repo "https://github.com/Lightricks/ComfyUI-LTXVideo.git" "${LTX_NODE_DIR}" "${LTX_NODE_REF}"
sync_repo "https://github.com/SethRobinson/comfyui-workflow-to-api-converter-endpoint.git" "${CONVERTER_DIR}" "${CONVERTER_REF}"

"${COMFY_PYTHON}" -m pip install --upgrade pip "setuptools<82" wheel
"${COMFY_PYTHON}" -m pip install -r "${COMFY_ROOT}/requirements.txt"
"${COMFY_PYTHON}" -m pip install -r "${LTX_NODE_DIR}/requirements.txt"
# The pinned LTX node revision still imports ``pad`` from Kornia's pyramid
# module. Kornia 0.8.3 removed that re-export, which makes the whole custom
# node pack fail to load (including the otherwise present LTXFloatToInt node).
# Keep the reproducible Packet bootstrap on the last compatible release.
"${COMFY_PYTHON}" -m pip install "kornia<0.8.3"

install -m 0644 "${LTX_NODE_DIR}/example_workflows/2.5/${WORKFLOW_NAME}" "${BLUEPRINT_DIR}/${WORKFLOW_NAME}"

mkdir -p "${COMFY_ROOT}/.run"
pkill -f "${COMFY_ROOT}/main.py.*--port ${COMFY_PORT}" || true
nohup "${COMFY_PYTHON}" "${COMFY_ROOT}/main.py" \
  --listen 127.0.0.1 \
  --port "${COMFY_PORT}" \
  --disable-auto-launch \
  > "${COMFY_ROOT}/.run/comfyui.out.log" \
  2> "${COMFY_ROOT}/.run/comfyui.err.log" \
  < /dev/null &
echo $! > "${COMFY_ROOT}/.run/comfyui.pid"

echo "ComfyUI LTX 2.5 bootstrap started on 127.0.0.1:${COMFY_PORT}."
