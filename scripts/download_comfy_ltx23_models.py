from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

import httpx


@dataclass(frozen=True)
class ComfyModelFile:
    id: str
    label: str
    repo_id: str
    repo_filename: str
    target_subdir: str
    target_filename: str


MODEL_FILES = [
    ComfyModelFile(
        id="gemma_3_12B_it_fp4_mixed",
        label="Gemma 3 text encoder",
        repo_id="Comfy-Org/ltx-2",
        repo_filename="split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
        target_subdir="models/text_encoders",
        target_filename="gemma_3_12B_it_fp4_mixed.safetensors",
    ),
    ComfyModelFile(
        id="ltx_2_3_22b_dev_fp8",
        label="LTX 2.3 FP8 checkpoint",
        repo_id="Lightricks/LTX-2.3-fp8",
        repo_filename="ltx-2.3-22b-dev-fp8.safetensors",
        target_subdir="models/checkpoints",
        target_filename="ltx-2.3-22b-dev-fp8.safetensors",
    ),
    ComfyModelFile(
        id="ltx_2_3_22b_distilled_lora_384",
        label="LTX 2.3 distilled LoRA",
        repo_id="Lightricks/LTX-2.3",
        repo_filename="ltx-2.3-22b-distilled-lora-384.safetensors",
        target_subdir="models/loras",
        target_filename="ltx-2.3-22b-distilled-lora-384.safetensors",
    ),
    ComfyModelFile(
        id="gemma_3_12b_abliterated_lora_rank64_bf16",
        label="Gemma 3 LoRA",
        repo_id="Comfy-Org/ltx-2",
        repo_filename="split_files/loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors",
        target_subdir="models/loras",
        target_filename="gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors",
    ),
    ComfyModelFile(
        id="ltx_2_3_spatial_upscaler_x2_1_1",
        label="LTX 2.3 spatial upscaler",
        repo_id="Lightricks/LTX-2.3",
        repo_filename="ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        target_subdir="models/latent_upscale_models",
        target_filename="ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    ),
]


def _env_int(name: str, fallback: int) -> int:
    try:
        return int(os.getenv(name, str(fallback)))
    except ValueError:
        return fallback


def _env_float(name: str, fallback: float) -> float:
    try:
        return float(os.getenv(name, str(fallback)))
    except ValueError:
        return fallback


DOWNLOAD_CHUNK_SIZE = max(16 * 1024, _env_int("AI_VIDEO_GEN_MODEL_DOWNLOAD_CHUNK_BYTES", 256 * 1024))


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
    parser.add_argument(
        "--status-file",
        default=os.getenv("AI_VIDEO_GEN_PROVISIONING_STATUS"),
        help="JSON status file updated during verification and download.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=_env_int("AI_VIDEO_GEN_MODEL_DOWNLOAD_MAX_ATTEMPTS", 3),
        help="Per-file download attempts. Use 0 for unlimited retries.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=_env_float("AI_VIDEO_GEN_MODEL_DOWNLOAD_RETRY_DELAY", 20.0),
        help="Seconds to wait before retrying a failed file download.",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=_env_float("AI_VIDEO_GEN_MODEL_DOWNLOAD_CONNECT_TIMEOUT", 30.0),
        help="HTTP connect timeout in seconds.",
    )
    parser.add_argument(
        "--read-timeout",
        type=float,
        default=_env_float("AI_VIDEO_GEN_MODEL_DOWNLOAD_READ_TIMEOUT", 120.0),
        help="HTTP read/write/pool timeout in seconds.",
    )
    parser.add_argument(
        "--base-url",
        action="append",
        default=None,
        help=(
            "HuggingFace-compatible resolver base URL. Can be passed more than once. "
            "Defaults to AI_VIDEO_GEN_HF_BASE_URLS or huggingface.co plus hf-mirror.com."
        ),
    )
    return parser


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def model_target_path(model: ComfyModelFile, comfy_root: Path) -> Path:
    return comfy_root / model.target_subdir / model.target_filename


def parse_base_urls(values: list[str] | None = None) -> list[str]:
    raw_values = values or []
    env_value = os.getenv("AI_VIDEO_GEN_HF_BASE_URLS") or os.getenv("AI_VIDEO_GEN_HF_BASE_URL")
    if env_value:
        raw_values.extend(env_value.split(","))
    if not raw_values:
        raw_values = ["https://huggingface.co", "https://hf-mirror.com"]
    urls: list[str] = []
    for value in raw_values:
        url = str(value or "").strip().rstrip("/")
        if url and url not in urls:
            urls.append(url)
    return urls or ["https://huggingface.co"]


def model_url(model: ComfyModelFile, base_url: str = "https://huggingface.co") -> str:
    encoded_filename = quote(model.repo_filename, safe="/")
    return f"{base_url.rstrip('/')}/{model.repo_id}/resolve/main/{encoded_filename}"


def is_ready(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def file_size(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0


def parse_total_size(response: httpx.Response, starting_bytes: int) -> int | None:
    content_range = response.headers.get("content-range", "")
    if "/" in content_range:
        total_raw = content_range.rsplit("/", 1)[-1].strip()
        if total_raw.isdigit():
            return int(total_raw)
    content_length = response.headers.get("content-length")
    if content_length and content_length.isdigit():
        return starting_bytes + int(content_length)
    return None


def read_auth_headers() -> dict[str, str]:
    headers = {}
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def probe_remote_size(
    client: httpx.Client,
    model: ComfyModelFile,
    base_url: str,
) -> int | None:
    url = model_url(model, base_url=base_url)
    headers = read_auth_headers()
    try:
        response = client.head(url, headers=headers)
        if response.status_code < 400:
            content_length = response.headers.get("content-length")
            if content_length and content_length.isdigit():
                return int(content_length)
    except Exception:
        pass

    headers = {**headers, "Range": "bytes=0-0"}
    try:
        with client.stream("GET", url, headers=headers) as response:
            if response.status_code < 400:
                return parse_total_size(response, 0)
    except Exception:
        return None
    return None


def discover_model_sizes(
    comfy_root: Path,
    base_urls: list[str],
    *,
    connect_timeout: float,
    read_timeout: float,
) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for model in MODEL_FILES:
        target = model_target_path(model, comfy_root)
        if is_ready(target):
            sizes[model.id] = file_size(target)
            continue
        tmp_size = file_size(target.with_suffix(f"{target.suffix}.tmp"))
        timeout = httpx.Timeout(timeout=read_timeout, connect=connect_timeout)
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            for base_url in base_urls:
                size = probe_remote_size(client, model, base_url)
                if size:
                    sizes[model.id] = max(size, tmp_size)
                    break
    return sizes


def write_json_atomic(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def build_model_states(
    comfy_root: Path,
    *,
    current_model: ComfyModelFile | None = None,
    current_status: str | None = None,
    current_bytes: int | None = None,
    current_total: int | None = None,
    expected_total_by_model: dict[str, int] | None = None,
    error_by_model: dict[str, str] | None = None,
) -> list[dict]:
    errors = error_by_model or {}
    expected_sizes = expected_total_by_model or {}
    states: list[dict] = []
    for model in MODEL_FILES:
        target_path = model_target_path(model, comfy_root)
        downloaded = file_size(target_path)
        tmp_downloaded = file_size(target_path.with_suffix(f"{target_path.suffix}.tmp"))
        expected_total = expected_sizes.get(model.id)
        if current_model and current_model.id == model.id:
            downloaded = current_bytes if current_bytes is not None else max(downloaded, tmp_downloaded)
            total = current_total or expected_total
            status = current_status or "downloading"
        else:
            total = expected_total
            status = "ready" if is_ready(target_path) else "missing"
            if model.id in errors:
                status = "error"
            if status == "ready" and not total:
                total = downloaded
        states.append(
            {
                "id": model.id,
                "label": model.label,
                "repoId": model.repo_id,
                "repoFilename": model.repo_filename,
                "targetPath": target_path.as_posix(),
                "status": status,
                "bytesDownloaded": downloaded,
                "totalBytes": total,
                "progressPercent": (
                    round((downloaded / total) * 100, 2)
                    if total and downloaded <= total
                    else (100.0 if status == "ready" else None)
                ),
                "error": errors.get(model.id),
            }
        )
    return states


def write_status(
    status_file: Path | None,
    comfy_root: Path,
    *,
    status: str,
    message: str,
    current_model: ComfyModelFile | None = None,
    current_index: int | None = None,
    current_bytes: int | None = None,
    current_total: int | None = None,
    attempt: int | None = None,
    max_attempts: int | None = None,
    error: str | None = None,
    error_by_model: dict[str, str] | None = None,
    current_url: str | None = None,
    expected_total_by_model: dict[str, int] | None = None,
) -> None:
    models = build_model_states(
        comfy_root,
        current_model=current_model,
        current_status=status if current_model else None,
        current_bytes=current_bytes,
        current_total=current_total,
        expected_total_by_model=expected_total_by_model,
        error_by_model=error_by_model,
    )
    ready_count = sum(1 for model in models if model["status"] == "ready")
    total_count = len(models)
    byte_total = 0
    byte_downloaded = 0
    byte_totals_complete = True
    for model_state in models:
        model_total = model_state.get("totalBytes")
        if model_total:
            model_downloaded = int(model_state.get("bytesDownloaded") or 0)
            byte_downloaded += min(model_downloaded, int(model_total))
            byte_total += int(model_total)
        else:
            byte_totals_complete = False
            break
    if byte_totals_complete and byte_total > 0:
        progress = (byte_downloaded / byte_total) * 100
    elif current_index:
        current_ratio = 0.0
        if current_model and current_total:
            current_ratio = max(0.0, min(1.0, float(current_bytes or 0) / float(current_total)))
        progress = ((current_index - 1 + current_ratio) / total_count) * 100
    else:
        progress = (ready_count / total_count) * 100 if total_count else 100.0

    payload = {
        "schemaVersion": "ai-video-gen.provisioning.v1",
        "updatedAt": utc_now(),
        "status": status,
        "message": message,
        "progressPercent": round(max(0.0, min(100.0, progress)), 2),
        "modelFilesReady": ready_count,
        "modelFilesTotal": total_count,
        "models": models,
        "current": (
            {
                "id": current_model.id,
                "label": current_model.label,
                "index": current_index,
                "total": total_count,
                "bytesDownloaded": current_bytes,
                "totalBytes": current_total,
                "attempt": attempt,
                "maxAttempts": max_attempts,
                "url": current_url or model_url(current_model),
                "targetPath": model_target_path(current_model, comfy_root).as_posix(),
            }
            if current_model
            else None
        ),
        "error": error,
    }
    write_json_atomic(status_file, payload)


def verify_hf_file(model: ComfyModelFile, comfy_root: Path) -> bool:
    target_path = model_target_path(model, comfy_root)
    ok = is_ready(target_path)
    status = "ready" if ok else "missing"
    print(f"[{status}] {target_path}", flush=True)
    return ok


def copy_hf_file_once(
    model: ComfyModelFile,
    comfy_root: Path,
    *,
    force: bool,
    status_file: Path | None,
    current_index: int,
    attempt: int,
    max_attempts: int,
    connect_timeout: float,
    read_timeout: float,
    base_url: str,
    expected_total_by_model: dict[str, int],
) -> None:
    target_dir = comfy_root / model.target_subdir
    target_path = model_target_path(model, comfy_root)
    target_dir.mkdir(parents=True, exist_ok=True)

    if force:
        target_path.unlink(missing_ok=True)
        target_path.with_suffix(f"{target_path.suffix}.tmp").unlink(missing_ok=True)

    if is_ready(target_path):
        print(f"[skip] {target_path}", flush=True)
        return

    url = model_url(model, base_url=base_url)
    headers = read_auth_headers()

    tmp_path = target_path.with_suffix(f"{target_path.suffix}.tmp")
    starting_bytes = file_size(tmp_path)
    if starting_bytes > 0:
        headers["Range"] = f"bytes={starting_bytes}-"

    print(f"[download] {url} -> {target_path}", flush=True)
    write_status(
        status_file,
        comfy_root,
        status="downloading",
        message=f"Downloading {model.label}.",
        current_model=model,
        current_index=current_index,
        current_bytes=starting_bytes,
        current_total=expected_total_by_model.get(model.id),
        attempt=attempt,
        max_attempts=max_attempts,
        current_url=url,
        expected_total_by_model=expected_total_by_model,
    )

    timeout = httpx.Timeout(timeout=read_timeout, connect=connect_timeout)
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        with client.stream("GET", url, headers=headers) as response:
            if starting_bytes > 0 and response.status_code != 206:
                tmp_path.unlink(missing_ok=True)
                starting_bytes = 0
                headers.pop("Range", None)
                response.close()
                with client.stream("GET", url, headers=headers) as restarted:
                    write_response_to_file(
                        restarted,
                        tmp_path,
                        target_path,
                        model,
                        comfy_root,
                        status_file=status_file,
                        current_index=current_index,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        starting_bytes=0,
                        url=url,
                        expected_total_by_model=expected_total_by_model,
                    )
                    return
            write_response_to_file(
                response,
                tmp_path,
                target_path,
                model,
                comfy_root,
                status_file=status_file,
                current_index=current_index,
                attempt=attempt,
                max_attempts=max_attempts,
                starting_bytes=starting_bytes,
                url=url,
                expected_total_by_model=expected_total_by_model,
            )


def write_response_to_file(
    response: httpx.Response,
    tmp_path: Path,
    target_path: Path,
    model: ComfyModelFile,
    comfy_root: Path,
    *,
    status_file: Path | None,
    current_index: int,
    attempt: int,
    max_attempts: int,
    starting_bytes: int,
    url: str,
    expected_total_by_model: dict[str, int],
) -> None:
    response.raise_for_status()
    total_bytes = parse_total_size(response, starting_bytes) or expected_total_by_model.get(model.id)
    mode = "ab" if response.status_code == 206 and starting_bytes > 0 else "wb"
    downloaded = starting_bytes if mode == "ab" else 0
    last_status_update = 0.0
    with tmp_path.open(mode) as output:
        for chunk in response.iter_bytes(chunk_size=DOWNLOAD_CHUNK_SIZE):
            if not chunk:
                continue
            output.write(chunk)
            downloaded += len(chunk)
            now = time.monotonic()
            if now - last_status_update >= 1.0:
                last_status_update = now
                write_status(
                    status_file,
                    comfy_root,
                    status="downloading",
                    message=f"Downloading {model.label}.",
                    current_model=model,
                    current_index=current_index,
                    current_bytes=downloaded,
                    current_total=total_bytes,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    current_url=url,
                    expected_total_by_model=expected_total_by_model,
                )
    tmp_path.replace(target_path)
    print(f"[ready] {target_path}", flush=True)


def copy_hf_file(
    model: ComfyModelFile,
    comfy_root: Path,
    *,
    force: bool,
    status_file: Path | None,
    current_index: int,
    max_attempts: int,
    retry_delay: float,
    connect_timeout: float,
    read_timeout: float,
    error_by_model: dict[str, str],
    base_urls: list[str],
    expected_total_by_model: dict[str, int],
) -> None:
    if is_ready(model_target_path(model, comfy_root)) and not force:
        print(f"[skip] {model_target_path(model, comfy_root)}", flush=True)
        return

    attempt = 1
    while True:
        base_url = base_urls[(attempt - 1) % len(base_urls)]
        try:
            copy_hf_file_once(
                model,
                comfy_root,
                force=force and attempt == 1,
                status_file=status_file,
                current_index=current_index,
                attempt=attempt,
                max_attempts=max_attempts,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
                base_url=base_url,
                expected_total_by_model=expected_total_by_model,
            )
            error_by_model.pop(model.id, None)
            return
        except Exception as exc:
            error = str(exc)
            error_by_model[model.id] = error
            should_retry = max_attempts == 0 or attempt < max_attempts
            status = "retrying" if should_retry else "error"
            print(f"[{status}] {model.label}: {error}", flush=True)
            target_path = model_target_path(model, comfy_root)
            write_status(
                status_file,
                comfy_root,
                status=status,
                message=(
                    f"Retrying {model.label} in {retry_delay:g}s."
                    if should_retry
                    else f"Failed to download {model.label}."
                ),
                current_model=model,
                current_index=current_index,
                current_bytes=file_size(target_path.with_suffix(f"{target_path.suffix}.tmp")),
                current_total=expected_total_by_model.get(model.id),
                attempt=attempt,
                max_attempts=max_attempts,
                error=error,
                error_by_model=error_by_model,
                current_url=model_url(model, base_url=base_url),
                expected_total_by_model=expected_total_by_model,
            )
            if not should_retry:
                raise
            attempt += 1
            time.sleep(retry_delay)


def main() -> None:
    args = build_parser().parse_args()
    comfy_root = Path(args.comfy_root).expanduser().resolve()
    status_file = Path(args.status_file).expanduser().resolve() if args.status_file else None
    comfy_root.mkdir(parents=True, exist_ok=True)
    error_by_model: dict[str, str] = {}
    base_urls = parse_base_urls(args.base_url)
    expected_total_by_model = discover_model_sizes(
        comfy_root,
        base_urls,
        connect_timeout=max(1.0, args.connect_timeout),
        read_timeout=max(1.0, args.read_timeout),
    )

    write_status(
        status_file,
        comfy_root,
        status="verifying",
        message="Checking required LTX 2.3 files.",
        expected_total_by_model=expected_total_by_model,
    )

    if args.verify_only:
        missing = [
            model_target_path(model, comfy_root)
            for model in MODEL_FILES
            if not verify_hf_file(model, comfy_root)
        ]
        if missing:
            write_status(
                status_file,
                comfy_root,
                status="error",
                message="Required ComfyUI model files are missing.",
                error="\n".join(path.as_posix() for path in missing),
                expected_total_by_model=expected_total_by_model,
            )
            raise SystemExit(
                "Missing required ComfyUI model files:\n"
                + "\n".join(f"  - {path}" for path in missing)
            )
        write_status(
            status_file,
            comfy_root,
            status="ready",
            message="All required LTX 2.3 files are ready.",
            expected_total_by_model=expected_total_by_model,
        )
        return

    try:
        for index, model in enumerate(MODEL_FILES, start=1):
            copy_hf_file(
                model,
                comfy_root,
                force=args.force,
                status_file=status_file,
                current_index=index,
                max_attempts=max(0, args.max_attempts),
                retry_delay=max(1.0, args.retry_delay),
                connect_timeout=max(1.0, args.connect_timeout),
                read_timeout=max(1.0, args.read_timeout),
                error_by_model=error_by_model,
                base_urls=base_urls,
                expected_total_by_model=expected_total_by_model,
            )
        write_status(
            status_file,
            comfy_root,
            status="ready",
            message="All required LTX 2.3 files are ready.",
            expected_total_by_model=expected_total_by_model,
        )
    except Exception as exc:
        write_status(
            status_file,
            comfy_root,
            status="error",
            message="Model download failed.",
            error=str(exc),
            error_by_model=error_by_model,
            expected_total_by_model=expected_total_by_model,
        )
        raise


if __name__ == "__main__":
    main()
