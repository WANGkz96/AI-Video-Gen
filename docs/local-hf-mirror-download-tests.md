# Local HF Mirror Download Tests

This test contour is local-only. It lets us exercise LTX 2.3 model download
logic against a HuggingFace-compatible localhost server with bandwidth limits,
jitter, stalls, and forced disconnects.

Production defaults are unchanged. Vast continues to use:

- `https://huggingface.co`
- `https://hf-mirror.com`

unless `AI_VIDEO_GEN_HF_BASE_URLS` or `--base-url` is explicitly set.

## 1. Prepare Mirror Files

From existing ComfyUI model files:

```powershell
python scripts/prepare_local_hf_mirror.py `
  --mirror-root C:\ai-video-gen-hf-mirror `
  --source-comfy-root C:\ComfyUI `
  --copy-mode hardlink
```

If local files do not exist yet, download missing files from HuggingFace:

```powershell
python scripts/prepare_local_hf_mirror.py `
  --mirror-root C:\ai-video-gen-hf-mirror `
  --download-missing
```

The mirror layout matches HuggingFace resolver paths:

```text
C:\ai-video-gen-hf-mirror\Comfy-Org\ltx-2\split_files\...
C:\ai-video-gen-hf-mirror\Lightricks\LTX-2.3-fp8\...
```

## 2. Start Slow/Turbulent Mirror

```powershell
python scripts/local_hf_mirror_server.py `
  --mirror-root C:\ai-video-gen-hf-mirror `
  --host 127.0.0.1 `
  --port 8765 `
  --rate-mbps 200 `
  --jitter 0.35 `
  --stall-every-mb 2048 `
  --stall-duration-sec 20
```

The downloader base URL is:

```text
http://127.0.0.1:8765
```

## 3. Run Downloader Against Local Mirror

Use a separate local ComfyUI root so production files are not touched:

```powershell
python scripts/download_comfy_ltx23_models.py `
  --comfy-root C:\ai-video-gen-download-test\ComfyUI `
  --status-file C:\ai-video-gen-download-test\provisioning-status.json `
  --base-url http://127.0.0.1:8765 `
  --max-attempts 3
```

Equivalent env var:

```powershell
$env:AI_VIDEO_GEN_HF_BASE_URLS = "http://127.0.0.1:8765"
```

Do not put localhost mirror values into Vast template env.

## 4. Watch Rolling ETA / Recycle Decision

```powershell
python scripts/monitor_provisioning_eta.py `
  --status-file C:\ai-video-gen-download-test\provisioning-status.json `
  --window-sec 600 `
  --warmup-sec 180 `
  --bad-eta-min 60
```

The monitor reports:

- rolling bytes downloaded over the window;
- stable effective speed;
- ETA across current and pending sequential files when sizes are known;
- `warming_up`, `keep`, `keep_grace`, `recycle`, or `ready`.

It is a test/diagnostic tool. Video-Pipeline can later port the same decision
logic into the Vast attempt supervisor.
