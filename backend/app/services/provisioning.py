from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from backend.app.config import Settings


COMFY_LTX25_MODELS = [
    {
        "id": "ltx_2_5_22b_distilled_transformer_comfy_int8",
        "label": "LTX 2.5 distilled transformer (ComfyUI INT8)",
        "targetSubdir": "models/diffusion_models",
        "targetFilename": "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
    },
    {
        "id": "gemma4_12b_ltx_2_5_comfy_int8",
        "label": "Gemma 4 LTX 2.5 text encoder (ComfyUI INT8)",
        "targetSubdir": "models/text_encoders",
        "targetFilename": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    },
    {
        "id": "gemma4_e2b_it_bf16",
        "label": "Gemma 4 prompt enhancer",
        "targetSubdir": "models/text_encoders",
        "targetFilename": "gemma4_e2b_it_bf16.safetensors",
    },
    {
        "id": "ltx_2_5_video_vae_conv",
        "label": "LTX 2.5 convolutional video VAE",
        "targetSubdir": "models/vae",
        "targetFilename": "ltx-2.5-video-vae-conv-bf16.safetensors",
    },
    {
        "id": "ltx_2_5_audio_vae",
        "label": "LTX 2.5 audio VAE",
        "targetSubdir": "models/vae",
        "targetFilename": "ltx-2.5-audio-vae-bf16.safetensors",
    },
]

# Keep these names in one place: the Packet bootstrap deliberately uses the
# ComfyUI INT8/convrot pack so LTX and LongCat Avatar can coexist on Packet's
# 150 GB ephemeral disk.  The official workflow defaults to BF16 names, so the
# adapter must override those defaults in the converted API graph.
COMFY_LTX25_MODEL_NAMES = {
    "transformer": "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
    "text_encoder": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    "text_enhancer": "gemma4_e2b_it_bf16.safetensors",
    "video_vae": "ltx-2.5-video-vae-conv-bf16.safetensors",
    "audio_vae": "ltx-2.5-audio-vae-bf16.safetensors",
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _target_path(settings: Settings, model: dict[str, str]) -> Path:
    return settings.comfyui_root / model["targetSubdir"] / model["targetFilename"]


def list_comfy_ltx25_model_files(settings: Settings) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for model in COMFY_LTX25_MODELS:
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


def missing_comfy_ltx25_model_files(settings: Settings) -> list[Path]:
    return [
        Path(model["targetPath"])
        for model in list_comfy_ltx25_model_files(settings)
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


def _read_status_file(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _check_longcat(settings: Settings) -> dict[str, Any]:
    required_files = [
        settings.longcat_repo_dir / "run_demo_avatar_multi_audio_to_video.py",
        settings.longcat_repo_dir / "weights" / "LongCat-Video" / "tokenizer" / "tokenizer_config.json",
        settings.longcat_checkpoint_dir / "base_model_int8" / "quantized_model.safetensors.index.json",
        settings.longcat_checkpoint_dir / "lora" / "dmd_lora.safetensors",
        settings.longcat_checkpoint_dir / "whisper-large-v3" / "config.json",
        settings.longcat_conda_env_dir / "bin" / "torchrun",
    ]
    missing = [item.as_posix() for item in required_files if not item.is_file()]
    status_payload = _read_status_file(settings.longcat_provisioning_status_file) or {}
    ready = not missing
    return {
        "required": settings.enable_longcat,
        "ready": ready,
        "status": "ready" if ready else str(status_payload.get("status") or "missing"),
        "message": (
            "LongCat Video Avatar is ready."
            if ready
            else str(status_payload.get("message") or "LongCat Video Avatar provisioning is incomplete.")
        ),
        "progressPercent": 100.0 if ready else float(status_payload.get("progressPercent") or 0),
        "missing": missing,
        "statusFile": settings.longcat_provisioning_status_file.as_posix(),
        "error": status_payload.get("error"),
    }


def _check_comfy(settings: Settings) -> tuple[bool, str | None]:
    try:
        response = httpx.get(f"{settings.generator_api_url.rstrip('/')}/system_stats", timeout=2.0)
        response.raise_for_status()
    except Exception as exc:
        return False, f"ComfyUI API is unavailable: {exc}"
    return True, None


def get_provisioning_status(settings: Settings) -> dict[str, Any]:
    model_files = list_comfy_ltx25_model_files(settings) if settings.enable_ltx else []
    missing = [model for model in model_files if not model["ready"]]
    workflows = [
        settings.comfyui_t2v_workflow,
        settings.comfyui_i2v_workflow,
    ] if settings.enable_ltx else []
    missing_workflows = [path.as_posix() for path in workflows if not path.is_file()]
    comfy_ready, comfy_error = _check_comfy(settings) if settings.enable_ltx else (True, None)
    downloader = _read_downloader_status(settings) if settings.enable_ltx else None
    downloader_status = downloader.get("status") if downloader else None
    downloader_error = downloader.get("error") if downloader else None
    downloader_message = downloader.get("message") if downloader else None
    sequence_waiting_for_longcat = downloader_status == "waiting_for_longcat"

    model_total = len(model_files)
    model_ready = model_total - len(missing)
    model_percent = round((model_ready / model_total) * 100, 2) if model_total else 100.0
    progress_percent = float((downloader or {}).get("progressPercent") or model_percent)

    ltx_ready = not sequence_waiting_for_longcat and not missing and not missing_workflows and comfy_ready
    longcat = _check_longcat(settings)
    longcat_ready = not settings.enable_longcat or longcat["ready"]
    if not settings.enable_ltx:
        ltx_status = "disabled"
        ltx_message = "LTX 2.5 is not required for this instance."
        ltx_progress_percent = 100.0
        ltx_error = None
    elif ltx_ready:
        ltx_status = "ready"
        ltx_message = "LTX 2.5 and ComfyUI are ready."
        ltx_progress_percent = 100.0
        ltx_error = None
    elif downloader_status in {"waiting_for_longcat", "starting"}:
        ltx_status = str(downloader_status)
        ltx_message = str(downloader_message or "Waiting to start the LTX 2.5 model download.")
        ltx_progress_percent = progress_percent
        ltx_error = None
    elif downloader_status in {"downloading", "retrying", "verifying"}:
        ltx_status = downloader_status
        ltx_message = str(downloader_message or "Downloading required LTX 2.5 files.")
        ltx_progress_percent = progress_percent
        ltx_error = None
    elif downloader_status == "error":
        ltx_status = "error"
        ltx_message = str(downloader_error or downloader_message or "Model download failed.")
        ltx_progress_percent = progress_percent
        ltx_error = downloader_error or downloader_message
    elif missing:
        ltx_status = "missing"
        ltx_message = "Required LTX 2.5 model files are missing."
        ltx_progress_percent = progress_percent
        ltx_error = None
    elif missing_workflows:
        ltx_status = "missing"
        ltx_message = "Required ComfyUI workflow files are missing."
        ltx_progress_percent = progress_percent
        ltx_error = None
    else:
        ltx_status = "waiting_comfy"
        ltx_message = comfy_error or "Waiting for ComfyUI API."
        ltx_progress_percent = progress_percent
        ltx_error = None

    ltx = {
        "required": settings.enable_ltx,
        "ready": ltx_ready,
        "status": ltx_status,
        "message": ltx_message,
        "progressPercent": max(0.0, min(100.0, ltx_progress_percent)),
        "error": ltx_error,
    }
    ready = ltx_ready and longcat_ready
    if ready:
        status = "ready"
        enabled = [name for name, flag in [("LTX 2.5", settings.enable_ltx), ("LongCat Avatar", settings.enable_longcat)] if flag]
        message = f"All required generators are ready: {', '.join(enabled) or 'none'}."
        progress_percent = 100.0
    elif settings.enable_longcat and not longcat_ready:
        status = longcat["status"]
        message = longcat["message"]
        progress_percent = longcat["progressPercent"]
    elif downloader_status in {"waiting_for_longcat", "starting"}:
        status = str(downloader_status)
        message = str(downloader_message or "Waiting to start the LTX 2.5 model download.")
    elif downloader_status in {"downloading", "retrying", "verifying"}:
        status = downloader_status
        message = str(downloader_message or "Downloading required LTX 2.5 files.")
    elif downloader_status == "error":
        status = "error"
        message = str(downloader_error or downloader_message or "Model download failed.")
    elif missing:
        status = "missing"
        message = "Required LTX 2.5 model files are missing."
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
            "required": settings.enable_ltx,
            "ready": comfy_ready,
            "apiUrl": settings.generator_api_url.rstrip("/"),
            "error": comfy_error,
        },
        "workflows": {
            "ready": not missing_workflows,
            "missing": missing_workflows,
        },
        "statusFile": settings.provisioning_status_file.as_posix(),
        "requirements": {
            "ltx": settings.enable_ltx,
            "longcatVideoAvatar": settings.enable_longcat,
        },
        "branches": {
            "comfyui-ltx25": ltx,
            "longcat-video-avatar": {
                **longcat,
                "ready": longcat_ready,
            },
        },
        "longcat": longcat,
    }
