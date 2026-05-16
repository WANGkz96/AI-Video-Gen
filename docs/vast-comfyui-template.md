# Vast AI ComfyUI Template

AI-Video-Gen production generation runs through the ComfyUI LTX 2.3 workflows.
The service no longer downloads or executes native LTX weights directly.

## Base Image

Use the Vast AI ComfyUI image:

```text
vastai/comfy
```

The template must expose the AI-Video-Gen web app port and keep ComfyUI running
inside the container.

Recommended extra portal entry:

```text
localhost:8090:8090:/:AI Video Gen
```

## Environment

Set these variables in the Vast template:

```text
PORT=8090
REPO_URL=https://github.com/WANGkz96/AI-Video-Gen.git
REPO_REF=master
WORK_ROOT=/workspace
APP_DIR=/workspace/AI-Video-Gen
PROVISIONING_SCRIPT=https://raw.githubusercontent.com/WANGkz96/AI-Video-Gen/master/scripts/onstart_vast_instance.sh
GENERATOR_BACKEND=comfyui-ltx23
GENERATOR_API_URL=http://127.0.0.1:18188
COMFYUI_ROOT=/workspace/ComfyUI
COMFYUI_T2V_WORKFLOW=/workspace/ComfyUI/blueprints/Text to Video (LTX-2.3).json
COMFYUI_I2V_WORKFLOW=/workspace/ComfyUI/blueprints/Image to Video (LTX-2.3).json
COMFYUI_OUTPUT_PREFIX=video/AI_Video_Gen
COMFYUI_STRIP_AUDIO=1
AI_VIDEO_GEN_DOWNLOAD_COMFY_MODELS=1
ENABLE_LEGACY_BACKENDS=0
ENABLE_MOCK_BACKEND=0
SEGMENT_VARIANTS=2
VIDEO_DURATION_SEC=8
PORTRAIT_RESOLUTION=720x1280
LANDSCAPE_RESOLUTION=1280x720
OUTPUT_UPSCALE=off
```

`HF_TOKEN` is optional unless Hugging Face requires authentication for the model
files or rate limits the machine.

## Required Workflows

The ComfyUI template must contain these blueprint files:

```text
/workspace/ComfyUI/blueprints/Text to Video (LTX-2.3).json
/workspace/ComfyUI/blueprints/Image to Video (LTX-2.3).json
```

AI-Video-Gen calls:

```text
video_ltx2_3_t2v
video_ltx2_3_i2v
```

through ComfyUI's HTTP API. No other production generation backend is selected
by default.

The image-to-video path uploads a first-frame image to ComfyUI and wires it into
the converted I2V workflow before queueing the prompt.

## Required ComfyUI Extension

The template must include:

```text
/workspace/ComfyUI/custom_nodes/comfyui-workflow-to-api-converter-endpoint
```

AI-Video-Gen uses its endpoint:

```text
POST http://127.0.0.1:18188/workflow/convert
```

to convert the ComfyUI blueprint JSON into API prompt JSON.

## Required Model Files

`scripts/download_comfy_ltx23_models.py` downloads or verifies these files:

```text
Comfy-Org/ltx-2:
  split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors
  -> /workspace/ComfyUI/models/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors

Lightricks/LTX-2.3-fp8:
  ltx-2.3-22b-dev-fp8.safetensors
  -> /workspace/ComfyUI/models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors

Lightricks/LTX-2.3:
  ltx-2.3-22b-distilled-lora-384.safetensors
  -> /workspace/ComfyUI/models/loras/ltx-2.3-22b-distilled-lora-384.safetensors

Lightricks/LTX-2.3:
  ltx-2.3-spatial-upscaler-x2-1.1.safetensors
  -> /workspace/ComfyUI/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors
```

The startup script skips existing non-empty files, so repeated instance starts
do not re-download the weights.

## Startup Flow

Use this on-start command in the Vast template:

```text
entrypoint.sh
```

and keep `PROVISIONING_SCRIPT` pointed at:

```text
https://raw.githubusercontent.com/WANGkz96/AI-Video-Gen/master/scripts/onstart_vast_instance.sh
```

The script:

1. clones or updates `/workspace/AI-Video-Gen`;
2. installs the backend into a local venv;
3. builds the Vue frontend;
4. downloads missing ComfyUI model files when `AI_VIDEO_GEN_DOWNLOAD_COMFY_MODELS=1`;
5. starts AI-Video-Gen on `PORT`.

## Health Checks

Inside the instance:

```bash
curl http://127.0.0.1:18188/system_stats
curl http://127.0.0.1:8090/api/health
curl http://127.0.0.1:8090/api/backends
```

`/api/backends` should list only the ready production backend:

```text
comfyui-ltx23
```
