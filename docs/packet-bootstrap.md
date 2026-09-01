# Packet.ai bootstrap and LTX 2.5

`scripts/onstart_packet_instance.sh` is the Packet startup entrypoint.  It
checks out the ref supplied by Video-pipeline and delegates setup to
`scripts/deploy_packet.sh`.

## What the bootstrap creates

- Python 3.12 AI-Video-Gen API on public port `8090` (authenticated by the
  per-run API token);
- pinned ComfyUI, `ComfyUI-LTXVideo`, and workflow-to-API converter;
- private ComfyUI on `127.0.0.1:18188`;
- the official LTX 2.5 single-stage T2V/I2V workflow;
- a separate Python 3.10 LongCat Avatar runtime, only if the batch contains
  dialogue scenes.

The API begins accepting jobs while background provisioning continues.  A
single-backend batch starts its model download immediately.  A mixed LongCat +
LTX batch is intentionally serialized: LongCat downloads and renders first;
after its results are written, the worker removes its weights and opens the LTX
download gate.  The two large model packs therefore never compete for the
same ephemeral disk.  If LongCat provisioning itself fails, its partial model
directory is released and LTX is still allowed to finish its independent
branch instead of being blocked by the failed first branch.

Model downloads within the active branch use at most three workers.  Override
the lower value with `AI_VIDEO_GEN_MODEL_DOWNLOAD_CONCURRENCY=1` or `2`; values
above `3` are capped to keep Packet bootstrap bandwidth and disk I/O stable.

## Storage contract

Packet's 150 GB ephemeral disk is shared by both branches.  The bootstrap
therefore downloads only the five LTX 2.5 ComfyUI files used by the selected
INT8/convrot workflow pack and only Avatar's actual INT8 + distilled LongCat
subtrees:

- LTX transformer, text encoder, prompt enhancer, video VAE, audio VAE;
- LongCat `tokenizer`, `text_encoder`, `vae`;
- Avatar `base_model_int8`, `lora`, `whisper-large-v3`, `vocal_separator`, and
  `scheduler`.

The adapter rewrites the official workflow's BF16 default names to those exact
local model files before submitting a ComfyUI prompt.  This is required: the
BF16 defaults are intentionally not downloaded on the 150 GB profile.

## Persistent model cache

Set `AI_VIDEO_GEN_PERSISTENT_MODEL_CACHE_DIR` to the mount path of a durable
Packet volume (for example `/data/share123/ai-video-gen-model-cache`). The
bootstrap keeps application code and runtimes on the instance, but downloads
the LTX and LongCat model payloads into that directory. ComfyUI is given that
directory as an extra model search path and LongCat loads its checkpoint from
there directly, so a later instance reuses verified files instead of fetching
them again.

The cache is intentionally optional: an empty value keeps the original
ephemeral-only behaviour. A durable cache is never removed when the LongCat
branch opens the LTX branch. On the 100 GB Packet volume, the exact INT8 LTX
pack and the trimmed LongCat Avatar payload fit together; the Avatar download
keeps only the `safetensors` Whisper weights used by Transformers and omits
redundant TensorFlow, Flax, and PyTorch checkpoint formats.

`HF_TOKEN` must have accepted access to the gated `Lightricks/LTX-2.5` model.
If it has not, the LTX branch remains unavailable with a readable provisioning
error; Avatar jobs do not depend on that acceptance.
