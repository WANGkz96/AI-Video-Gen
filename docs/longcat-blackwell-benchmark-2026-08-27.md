# LongCat Avatar Blackwell benchmark — 2026-08-27

## Scope

This is an isolated Avatar-only benchmark.  LTX/ComfyUI were disabled in the
Packet provisioning request, and no LTX files or inference code were changed.

Both measurements use the exact same production scene input, prompt, seed,
audio and starting image:

- LongCat-Video-Avatar-1.5;
- 480p (`832x480`), 25 fps;
- one 93-frame AI2V segment (3.72 seconds);
- distilled profile: 8 inference steps, text/audio CFG 1.0;
- PyTorch 2.7.1 + CUDA 12.8 on an RTX PRO 6000 Blackwell Dynamic instance.

Audio preparation is measured separately from diffusion and is not included in
the speed comparison.

## Results

| DiT weights | AI2V inference | Audio preparation | Total scene time |
| --- | ---: | ---: | ---: |
| Production `base_model_int8` | 254.504 s | 103.555 s | 363.749 s |
| BF16 `base_model` | 276.717 s | 103.527 s | 382.051 s |

BF16 was **8.7% slower** for this comparable run and consumes an additional
~30 GB on disk.  It is not a production candidate on this Packet Dynamic
profile.

Every diffusion step in both runs emitted a scheduler warning that it was
waiting for the GPU lock.  Therefore the timings include Dynamic-pool
scheduling latency and cannot be treated as bare-GPU throughput.

The generated raw clips were downloaded for visual comparison; both are 3.72 s
at 832x480 with the original audio.

## Follow-up candidates

1. Repeat the exact benchmark on a Packet Dedicated RTX PRO 6000 or other
   exclusive GPU before changing inference code.
2. Only then evaluate a separate vLLM-Omni/cuDNN-attention environment.  Native
   LongCat currently disables cuDNN on `sm_120` to avoid its existing
   Conv-initialisation failure, so enabling it in place would not be safe.
3. Keep production on INT8 until an exclusive-GPU measurement demonstrates a
   material improvement.
