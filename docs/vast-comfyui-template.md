# Vast AI LTX 2.5 + LongCat template

Vast uses the same pinned ComfyUI checkout, LTX 2.5 stock I2V/T2V workflows,
model pack, and LongCat Avatar provisioner as Packet. The AI-Video-Gen startup
script installs these components on the instance; the image's bundled ComfyUI
version and any legacy LTX 2.3 template variables are not used.

## Vast template

Use an Ubuntu GPU image with NVIDIA drivers, `apt-get`, root or passwordless
`sudo`, and an SSH/direct port. Expose port `8090` for AI-Video-Gen. The pinned
ComfyUI API listens only on `127.0.0.1:18188` inside the instance.

```text
PORT=8090
WORK_ROOT=/workspace
APP_DIR=/workspace/AI-Video-Gen
REPO_URL=https://github.com/WANGkz96/AI-Video-Gen.git
REPO_REF=master
PROVISIONING_SCRIPT=https://raw.githubusercontent.com/WANGkz96/AI-Video-Gen/master/scripts/onstart_vast_instance.sh
AI_VIDEO_GEN_ENABLE_LTX=1
AI_VIDEO_GEN_ENABLE_LONGCAT=1
LTX25_MEGAPIXELS=0.9
```

Set `PROVISIONING_SCRIPT` in the Vast template and retain its normal
`entrypoint.sh` on-start command. The Video-pipeline launch payload sets the
two `AI_VIDEO_GEN_ENABLE_*` flags for each batch. It requests 400 GB of disk
and selects one RTX PRO 6000 WS, S, or Max-Q under the configured hourly cap.
`HF_TOKEN` is optional unless Hugging Face requires authentication.

The startup script checks out `REPO_REF`, installs Python 3.12 for the API and
ComfyUI, installs Python 3.10 in a separate LongCat environment, and downloads
the model files required by the selected branches. Both model packs remain on
the 400 GB instance disk. Generation is sequential (`standard` profile); the
LongCat weights are not deleted between branches. The backend may accept a
batch while downloads finish, then waits for each required branch to be ready.
Before exposing the API, startup verifies that ComfyUI converts both stock
workflows and exposes their required nodes.
Its pinned ComfyUI checkout lives at `/workspace/AI-Video-Gen-ComfyUI`, separate
from any ComfyUI copy bundled in the Vast image.

The LTX controls are the pipeline image, prompt, duration, nearest supported
aspect ratio, and `LTX25_MEGAPIXELS` (default `0.9`). The stock template keeps
its `multiple`, frame rate, and sampler settings. See
[`packet-bootstrap.md`](packet-bootstrap.md) for the pinned ComfyUI workflow
and six model files used by both providers.

## Readiness checks

From SSH on the instance:

```bash
git -C /workspace/AI-Video-Gen rev-parse HEAD
curl --fail http://127.0.0.1:18188/system_stats
curl --fail -H "Authorization: Bearer $AI_VIDEO_GEN_API_TOKEN" http://127.0.0.1:8090/api/health
```

The health response reports separate LTX and LongCat provisioning states.
Inspect `.run/ltx25-download.*.log`, `.run/longcat-provision.*.log`,
`.run/backend.*.log`, and `/workspace/AI-Video-Gen-ComfyUI/.run/comfyui.*.log` when a
branch is not ready. No Vast instance is started by these instructions;
launch is an explicit action in Video-pipeline's `live_vast` mode.
