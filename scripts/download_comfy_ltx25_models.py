"""Download the exact ComfyUI LTX 2.5 runtime pack used by AI-Video-Gen.

The resilient range-download implementation is shared with the earlier LTX
downloader. Keeping the model manifest here makes a release change explicit
and keeps status-file semantics identical for the remote automation.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LEGACY_PATH = ROOT / "download_comfy_ltx23_models.py"
SPEC = importlib.util.spec_from_file_location("aivg_comfy_downloader", LEGACY_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - filesystem invariant
    raise RuntimeError(f"Cannot load shared downloader: {LEGACY_PATH}")
downloader = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = downloader
SPEC.loader.exec_module(downloader)


MODEL_FILES = [
    downloader.ComfyModelFile(
        id="ltx_2_5_22b_distilled_transformer_comfy_int8",
        label="LTX 2.5 distilled transformer (ComfyUI INT8)",
        repo_id="Lightricks/LTX-2.5",
        repo_filename="diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
        target_subdir="models/diffusion_models",
        target_filename="ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
    ),
    downloader.ComfyModelFile(
        id="gemma4_12b_ltx_2_5_comfy_int8",
        label="Gemma 4 LTX 2.5 text encoder (ComfyUI INT8)",
        repo_id="Lightricks/LTX-2.5",
        repo_filename="text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
        target_subdir="models/text_encoders",
        target_filename="gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    ),
    downloader.ComfyModelFile(
        id="gemma4_e2b_it_bf16",
        label="Gemma 4 prompt enhancer",
        repo_id="Comfy-Org/gemma-4",
        repo_filename="text_encoders/gemma4_e2b_it_bf16.safetensors",
        target_subdir="models/text_encoders",
        target_filename="gemma4_e2b_it_bf16.safetensors",
    ),
    downloader.ComfyModelFile(
        id="ltx_2_5_video_vae_conv",
        label="LTX 2.5 convolutional video VAE",
        repo_id="Lightricks/LTX-2.5",
        repo_filename="vae/ltx-2.5-video-vae-conv-bf16.safetensors",
        target_subdir="models/vae",
        target_filename="ltx-2.5-video-vae-conv-bf16.safetensors",
    ),
    downloader.ComfyModelFile(
        id="ltx_2_5_audio_vae",
        label="LTX 2.5 audio VAE",
        repo_id="Lightricks/LTX-2.5",
        repo_filename="vae/ltx-2.5-audio-vae-bf16.safetensors",
        target_subdir="models/vae",
        target_filename="ltx-2.5-audio-vae-bf16.safetensors",
    ),
]


downloader.MODEL_FILES = MODEL_FILES


if __name__ == "__main__":
    downloader.main()
