from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import httpx


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
        repo_id="Comfy-Org/ltx-2",
        repo_filename="split_files/loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors",
        target_subdir="models/loras",
        target_filename="gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors",
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
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify that all files exist and are non-empty. Do not download anything.",
    )
    return parser


def model_target_path(model: ComfyModelFile, comfy_root: Path) -> Path:
    return comfy_root / model.target_subdir / model.target_filename


def verify_hf_file(model: ComfyModelFile, comfy_root: Path) -> bool:
    target_path = model_target_path(model, comfy_root)
    ok = target_path.is_file() and target_path.stat().st_size > 0
    status = "ready" if ok else "missing"
    print(f"[{status}] {target_path}")
    return ok


def copy_hf_file(model: ComfyModelFile, comfy_root: Path, *, force: bool) -> None:
    target_dir = comfy_root / model.target_subdir
    target_path = model_target_path(model, comfy_root)
    target_dir.mkdir(parents=True, exist_ok=True)

    if target_path.is_file() and target_path.stat().st_size > 0 and not force:
        print(f"[skip] {target_path}")
        return

    encoded_filename = quote(model.repo_filename, safe="/")
    url = f"https://huggingface.co/{model.repo_id}/resolve/main/{encoded_filename}"
    headers = {}
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    print(f"[download] {url} -> {target_path}")
    tmp_path = target_path.with_suffix(f"{target_path.suffix}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=None) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as output:
            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
    tmp_path.replace(target_path)
    print(f"[ready] {target_path}")


def main() -> None:
    args = build_parser().parse_args()
    comfy_root = Path(args.comfy_root).expanduser().resolve()
    comfy_root.mkdir(parents=True, exist_ok=True)
    if args.verify_only:
        missing = [
            model_target_path(model, comfy_root)
            for model in MODEL_FILES
            if not verify_hf_file(model, comfy_root)
        ]
        if missing:
            raise SystemExit(
                "Missing required ComfyUI model files:\n"
                + "\n".join(f"  - {path}" for path in missing)
            )
        return

    for model in MODEL_FILES:
        copy_hf_file(model, comfy_root, force=args.force)


if __name__ == "__main__":
    main()
