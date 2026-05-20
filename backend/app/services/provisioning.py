from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from backend.app.config import Settings


COMFY_LTX23_MODELS = [
    {
        "id": "gemma_3_12B_it_fp4_mixed",
        "label": "Gemma 3 text encoder",
        "targetSubdir": "models/text_encoders",
        "targetFilename": "gemma_3_12B_it_fp4_mixed.safetensors",
    },
    {
        "id": "ltx_2_3_22b_dev_fp8",
        "label": "LTX 2.3 FP8 checkpoint",
        "targetSubdir": "models/checkpoints",
        "targetFilename": "ltx-2.3-22b-dev-fp8.safetensors",
    },
    {
        "id": "ltx_2_3_22b_distilled_lora_384",
        "label": "LTX 2.3 distilled LoRA",
        "targetSubdir": "models/loras",
        "targetFilename": "ltx-2.3-22b-distilled-lora-384.safetensors",
    },
    {
        "id": "gemma_3_12b_abliterated_lora_rank64_bf16",
        "label": "Gemma 3 LoRA",
        "targetSubdir": "models/loras",
        "targetFilename": "gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors",
    },
    {
        "id": "ltx_2_3_spatial_upscaler_x2_1_1",
        "label": "LTX 2.3 spatial upscaler",
        "targetSubdir": "models/latent_upscale_models",
        "targetFilename": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    },
]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _target_path(settings: Settings, model: dict[str, str]) -> Path:
    return settings.comfyui_root / model["targetSubdir"] / model["targetFilename"]


def list_comfy_ltx23_model_files(settings: Settings) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for model in COMFY_LTX23_MODELS:
        path = _target_path(settings, model)
        size = path.stat().st_size if path.is_file() else 0
        models.append(
            {
                **model,
                "targetPath": path.as_posix(),
                "status": "ready" if size > 0 else "missing",
                "bytesDownloaded": size,
                "ready": size > 0,
            }
        )
    return models


def missing_comfy_ltx23_model_files(settings: Settings) -> list[Path]:
    return [
        Path(model["targetPath"])
        for model in list_comfy_ltx23_model_files(settings)
        if not model["ready"]
    ]


def _read_downloader_status(settings: Settings) -> dict[str, Any] | None:
    path = settings.provisioning_status_file
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _check_comfy(settings: Settings) -> tuple[bool, str | None]:
    try:
        response = httpx.get(f"{settings.generator_api_url.rstrip('/')}/system_stats", timeout=2.0)
        response.raise_for_status()
    except Exception as exc:
        return False, f"ComfyUI API is unavailable: {exc}"
    return True, None


def get_provisioning_status(settings: Settings) -> dict[str, Any]:
    model_files = list_comfy_ltx23_model_files(settings)
    missing = [model for model in model_files if not model["ready"]]
    workflows = [
        settings.comfyui_t2v_workflow,
        settings.comfyui_i2v_workflow,
    ]
    missing_workflows = [path.as_posix() for path in workflows if not path.is_file()]
    comfy_ready, comfy_error = _check_comfy(settings)
    downloader = _read_downloader_status(settings)
    downloader_status = downloader.get("status") if downloader else None
    downloader_error = downloader.get("error") if downloader else None
    downloader_message = downloader.get("message") if downloader else None

    model_total = len(model_files)
    model_ready = model_total - len(missing)
    model_percent = round((model_ready / model_total) * 100, 2) if model_total else 100.0
    progress_percent = float((downloader or {}).get("progressPercent") or model_percent)

    ready = not missing and not missing_workflows and comfy_ready
    if ready:
        status = "ready"
        message = "ComfyUI and all required LTX 2.3 files are ready."
        progress_percent = 100.0
    elif downloader_status in {"downloading", "retrying", "verifying"}:
        status = downloader_status
        message = str(downloader_message or "Downloading required LTX 2.3 files.")
    elif downloader_status == "error":
        status = "error"
        message = str(downloader_error or downloader_message or "Model download failed.")
    elif missing:
        status = "missing"
        message = "Required LTX 2.3 model files are missing."
    elif missing_workflows:
        status = "missing"
        message = "Required ComfyUI workflow files are missing."
    else:
        status = "waiting_comfy"
        message = comfy_error or "Waiting for ComfyUI API."

    return {
        "schemaVersion": "ai-video-gen.provisioning.v1",
        "updatedAt": _utc_now(),
        "ready": ready,
        "status": status,
        "message": message,
        "progressPercent": max(0.0, min(100.0, progress_percent)),
        "modelFilesReady": model_ready,
        "modelFilesTotal": model_total,
        "models": (downloader or {}).get("models") or model_files,
        "current": (downloader or {}).get("current"),
        "error": downloader_error,
        "comfyui": {
            "ready": comfy_ready,
            "apiUrl": settings.generator_api_url.rstrip("/"),
            "error": comfy_error,
        },
        "workflows": {
            "ready": not missing_workflows,
            "missing": missing_workflows,
        },
        "statusFile": settings.provisioning_status_file.as_posix(),
    }
