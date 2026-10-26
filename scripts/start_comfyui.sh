#!/usr/bin/env bash
set -euo pipefail

COMFY_ROOT="${COMFYUI_ROOT:-/workspace/ComfyUI}"
COMFY_PYTHON="${COMFY_PYTHON:-python3}"
COMFY_PORT="${COMFYUI_PORT:-18188}"
LTX_MODEL_ROOT="${AI_VIDEO_GEN_LTX_MODEL_ROOT:-${COMFY_ROOT}}"
if [ ! -f "${COMFY_ROOT}/main.py" ] || ! command -v "${COMFY_PYTHON}" >/dev/null 2>&1; then
  echo "Prepared ComfyUI checkout or Python runtime is missing." >&2
  exit 1
fi
COMFY_MODEL_PATH_ARGS=()
if [ "${LTX_MODEL_ROOT}" != "${COMFY_ROOT}" ]; then
  mkdir -p "${LTX_MODEL_ROOT}/models/diffusion_models" "${LTX_MODEL_ROOT}/models/text_encoders" \
    "${LTX_MODEL_ROOT}/models/vae" "${LTX_MODEL_ROOT}/models/latent_upscale_models"
  EXTRA_MODEL_PATHS_FILE="${COMFY_ROOT}/extra_model_paths.ai-video-gen.yaml"
  printf '%s\n' 'ai_video_gen_persistent_ltx:' "  base_path: ${LTX_MODEL_ROOT}" \
    '  diffusion_models: models/diffusion_models' '  text_encoders: models/text_encoders' \
    '  vae: models/vae' '  latent_upscale_models: models/latent_upscale_models' > "${EXTRA_MODEL_PATHS_FILE}"
  COMFY_MODEL_PATH_ARGS=(--extra-model-paths-config "${EXTRA_MODEL_PATHS_FILE}")
fi

mkdir -p "${COMFY_ROOT}/.run"
COMFY_MEMORY_ARGS=()
if [ "${COMFYUI_SAFE_CUDA_MEMORY:-0}" = "1" ]; then
  COMFY_MEMORY_ARGS=(--disable-cuda-malloc --disable-dynamic-vram --disable-async-offload --disable-pinned-memory)
fi
pkill -f "${COMFY_ROOT}/main.py.*--port ${COMFY_PORT}" || true
nohup "${COMFY_PYTHON}" "${COMFY_ROOT}/main.py" \
  --listen 127.0.0.1 --port "${COMFY_PORT}" --disable-auto-launch \
  "${COMFY_MEMORY_ARGS[@]}" "${COMFY_MODEL_PATH_ARGS[@]}" \
  > "${COMFY_ROOT}/.run/comfyui.out.log" 2> "${COMFY_ROOT}/.run/comfyui.err.log" < /dev/null &
echo $! > "${COMFY_ROOT}/.run/comfyui.pid"
echo "ComfyUI started on 127.0.0.1:${COMFY_PORT}."
