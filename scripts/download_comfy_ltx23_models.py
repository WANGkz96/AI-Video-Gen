from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download


@dataclass(frozen=True)
class ComfyModelFile:
    repo_id: str
    repo_filename: str
    target_subdir: str
    target_filename: str


MODEL_FILES = [
    ComfyModelFile(
        repo_id="Comfy-Org/ltx-2",
        repo_filename="split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
        target_subdir="models/text_encoders",
        target_filename="gemma_3_12B_it_fp4_mixed.safetensors",
    ),
    ComfyModelFile(
        repo_id="Lightricks/LTX-2.3-fp8",
        repo_filename="ltx-2.3-22b-dev-fp8.safetensors",
        target_subdir="models/checkpoints",
        target_filename="ltx-2.3-22b-dev-fp8.safetensors",
    ),
    ComfyModelFile(
        repo_id="Lightricks/LTX-2.3",
        repo_filename="ltx-2.3-22b-distilled-lora-384.safetensors",
        target_subdir="models/loras",
        target_filename="ltx-2.3-22b-distilled-lora-384.safetensors",
    ),
    ComfyModelFile(
        repo_id="Lightricks/LTX-2.3",
        repo_filename="ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        target_subdir="models/latent_upscale_models",
        target_filename="ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    ),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the ComfyUI LTX 2.3 workflow model files into a ComfyUI tree."
    )
    parser.add_argument(
        "--comfy-root",
        default=os.getenv("COMFYUI_ROOT", "/workspace/ComfyUI"),
        help="ComfyUI root directory. Defaults to COMFYUI_ROOT or /workspace/ComfyUI.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite files that already exist.",
    )
    return parser


def copy_hf_file(model: ComfyModelFile, comfy_root: Path, *, force: bool) -> None:
    target_dir = comfy_root / model.target_subdir
    target_path = target_dir / model.target_filename
    target_dir.mkdir(parents=True, exist_ok=True)

    if target_path.is_file() and target_path.stat().st_size > 0 and not force:
        print(f"[skip] {target_path}")
        return

    print(f"[download] {model.repo_id}:{model.repo_filename} -> {target_path}")
    downloaded_path = Path(
        hf_hub_download(
            repo_id=model.repo_id,
            filename=model.repo_filename,
            token=os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"),
            force_download=force,
        )
    )
    tmp_path = target_path.with_suffix(f"{target_path.suffix}.tmp")
    shutil.copyfile(downloaded_path, tmp_path)
    tmp_path.replace(target_path)
    print(f"[ready] {target_path}")


def main() -> None:
    args = build_parser().parse_args()
    comfy_root = Path(args.comfy_root).expanduser().resolve()
    comfy_root.mkdir(parents=True, exist_ok=True)
    for model in MODEL_FILES:
        copy_hf_file(model, comfy_root, force=args.force)


if __name__ == "__main__":
    main()
