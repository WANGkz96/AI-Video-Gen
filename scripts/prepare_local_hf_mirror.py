from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.download_comfy_ltx23_models import MODEL_FILES, model_target_path, model_url  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a local HuggingFace-style mirror for the LTX 2.3 model files."
    )
    parser.add_argument("--mirror-root", required=True)
    parser.add_argument(
        "--source-comfy-root",
        default=os.getenv("COMFYUI_ROOT", ""),
        help="Copy/link existing downloaded target files from this ComfyUI root when available.",
    )
    parser.add_argument("--download-missing", action="store_true", help="Download missing mirror files from HF.")
    parser.add_argument("--copy-mode", choices=["copy", "hardlink"], default="copy")
    parser.add_argument("--base-url", default=os.getenv("AI_VIDEO_GEN_SOURCE_HF_BASE_URL", "https://huggingface.co"))
    parser.add_argument("--connect-timeout", type=float, default=30.0)
    parser.add_argument("--read-timeout", type=float, default=120.0)
    return parser


def mirror_path(mirror_root: Path, repo_id: str, repo_filename: str) -> Path:
    return mirror_root / repo_id / repo_filename


def copy_or_link(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == source.stat().st_size:
        print(f"[ready] {target}")
        return
    target.unlink(missing_ok=True)
    if mode == "hardlink":
        try:
            os.link(source, target)
            print(f"[hardlink] {source} -> {target}")
            return
        except OSError as exc:
            print(f"[hardlink-fallback] {exc}; copying instead.")
    shutil.copy2(source, target)
    print(f"[copy] {source} -> {target}")


def download_file(url: str, target: Path, *, connect_timeout: float, read_timeout: float) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(f"{target.suffix}.tmp")
    headers: dict[str, str] = {}
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    downloaded = tmp_path.stat().st_size if tmp_path.is_file() else 0
    if downloaded > 0:
        headers["Range"] = f"bytes={downloaded}-"

    timeout = httpx.Timeout(timeout=read_timeout, connect=connect_timeout)
    print(f"[download] {url} -> {target}")
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        with client.stream("GET", url, headers=headers) as response:
            if downloaded > 0 and response.status_code != 206:
                tmp_path.unlink(missing_ok=True)
                downloaded = 0
                headers.pop("Range", None)
                response.close()
                with client.stream("GET", url, headers=headers) as restarted:
                    restarted.raise_for_status()
                    with tmp_path.open("wb") as output:
                        for chunk in restarted.iter_bytes(chunk_size=1024 * 1024):
                            if chunk:
                                output.write(chunk)
            else:
                response.raise_for_status()
                mode = "ab" if downloaded > 0 and response.status_code == 206 else "wb"
                with tmp_path.open(mode) as output:
                    for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                        if chunk:
                            output.write(chunk)
    tmp_path.replace(target)
    print(f"[ready] {target}")


def main() -> None:
    args = build_parser().parse_args()
    mirror_root = Path(args.mirror_root).expanduser().resolve()
    source_comfy_root = Path(args.source_comfy_root).expanduser().resolve() if args.source_comfy_root else None
    missing: list[Path] = []
    for model in MODEL_FILES:
        target = mirror_path(mirror_root, model.repo_id, model.repo_filename)
        if target.is_file() and target.stat().st_size > 0:
            print(f"[ready] {target}")
            continue

        source = model_target_path(model, source_comfy_root) if source_comfy_root else None
        if source and source.is_file() and source.stat().st_size > 0:
            copy_or_link(source, target, args.copy_mode)
            continue

        if args.download_missing:
            download_file(
                model_url(model, base_url=args.base_url),
                target,
                connect_timeout=max(1.0, args.connect_timeout),
                read_timeout=max(1.0, args.read_timeout),
            )
            continue

        missing.append(target)

    if missing:
        print("\nMissing mirror files:")
        for path in missing:
            print(f"  - {path}")
        raise SystemExit("Run again with --download-missing or provide --source-comfy-root.")

    print(f"\nMirror ready: {mirror_root}")


if __name__ == "__main__":
    main()
