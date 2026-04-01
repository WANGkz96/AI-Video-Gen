from __future__ import annotations

import asyncio
import gc
import inspect
import os
from pathlib import Path
from typing import Any

from backend.app.adapters.base import AdapterUnavailableError, BaseGeneratorAdapter
from backend.app.adapters.catalog import DiffusersBackendSpec
from backend.app.config import Settings
from backend.app.models import AdapterInfo, GenerationArtifact, SegmentGenerationRequest


class DiffusersVideoAdapter(BaseGeneratorAdapter):
    def __init__(self, settings: Settings, spec: DiffusersBackendSpec) -> None:
        self._settings = settings
        self._spec = spec
        self._pipeline: Any | None = None
        self._load_lock = asyncio.Lock()

    def info(self) -> AdapterInfo:
        local_path = self._local_model_dir()
        deps_ok, dep_error = self._check_dependencies()
        available = deps_ok and self._spec.status != "planned"
        notes = self._spec.notes

        if dep_error:
            available = False
            notes = dep_error
        elif self._spec.status == "planned":
            notes = self._spec.notes or "Backend scaffold exists, but runtime integration is not wired yet."
        elif not local_path.exists():
            notes = (
                f"Model is not downloaded yet. Expected path: {local_path.as_posix()}. "
                "Use the bootstrap/download script or first-run lazy download."
            )

        return AdapterInfo(
            key=self._spec.key,
            label=self._spec.label,
            description=self._spec.description,
            status=self._spec.status,
            available=available,
            supportsBatch=True,
            supportsDirect=True,
            requiresRemote=True,
            requiresDownload=not local_path.exists(),
            modelId=self._spec.model_id,
            localPath=local_path.as_posix(),
            minimumVramGb=self._spec.minimum_vram_gb,
            notes=notes,
        )

    async def generate_segment(
        self, request: SegmentGenerationRequest
    ) -> GenerationArtifact:
        pipe = await self._get_pipeline()
        debug, output_path = await asyncio.to_thread(self._run_generation, pipe, request)
        return GenerationArtifact(
            modelName=self._spec.key,
            modelVersion=self._spec.model_id,
            outputPath=output_path,
            debug=debug,
        )

    def release(self) -> None:
        pipe = self._pipeline
        self._pipeline = None
        if pipe is None:
            return

        try:
            if hasattr(pipe, "remove_all_hooks"):
                pipe.remove_all_hooks()
        except Exception:
            pass

        del pipe
        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def _check_dependencies(self) -> tuple[bool, str | None]:
        try:
            import diffusers  # noqa: F401
            import huggingface_hub  # noqa: F401
            import imageio  # noqa: F401
            import torch  # noqa: F401
        except Exception as exc:
            return False, f"Runtime dependencies are missing for '{self._spec.key}': {exc}"
        return True, None

    def _local_model_dir(self) -> Path:
        return self._settings.models_dir / self._spec.storage_dir_name

    async def _get_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        async with self._load_lock:
            if self._pipeline is not None:
                return self._pipeline
            self._pipeline = await asyncio.to_thread(self._load_pipeline)
            return self._pipeline

    async def download_assets(self) -> Path:
        return await asyncio.to_thread(self._download_assets)

    def _load_pipeline(self) -> Any:
        try:
            import diffusers
            import torch
        except Exception as exc:
            raise AdapterUnavailableError(f"Cannot load diffusers runtime: {exc}") from exc

        if self._spec.status == "planned":
            raise AdapterUnavailableError(self._spec.notes or f"{self._spec.key} is not wired yet.")

        local_dir = self._download_assets()

        pipeline_class = None
        for candidate in self._spec.pipeline_candidates:
            pipeline_class = getattr(diffusers, candidate, None)
            if pipeline_class is not None:
                break
        if pipeline_class is None:
            joined = ", ".join(self._spec.pipeline_candidates)
            raise AdapterUnavailableError(
                f"No supported diffusers pipeline class was found for {self._spec.key}: {joined}"
            )

        dtype = self._resolve_dtype(torch)
        pipe = pipeline_class.from_pretrained(
            local_dir,
            torch_dtype=dtype,
            token=self._settings.hf_token,
            local_files_only=True,
            **self._spec.load_kwargs,
        )

        if self._spec.uses_float32_vae and hasattr(pipe, "vae") and pipe.vae is not None:
            pipe.vae.to(torch.float32)
        if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None and dtype != torch.float32:
            try:
                pipe.text_encoder.to(dtype)
            except Exception:
                pass
        if self._spec.enable_vae_tiling and hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        if self._spec.enable_vae_slicing and hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if self._spec.enable_attention_slicing and hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        device_map = getattr(pipe, "hf_device_map", None)
        if device_map:
            return pipe
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        elif hasattr(pipe, "enable_sequential_cpu_offload"):
            pipe.enable_sequential_cpu_offload()
        elif hasattr(pipe, "to"):
            pipe.to("cuda")

        return pipe

    def _download_assets(self) -> Path:
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
        except Exception as exc:
            raise AdapterUnavailableError(f"Cannot download model assets: {exc}") from exc

        if self._spec.status == "planned" and not self._spec.download_files:
            raise AdapterUnavailableError(self._spec.notes or f"{self._spec.key} is not wired yet.")

        local_dir = self._local_model_dir()
        if local_dir.exists():
            return local_dir

        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        local_dir.mkdir(parents=True, exist_ok=True)
        if self._spec.download_files:
            for filename in self._spec.download_files:
                hf_hub_download(
                    repo_id=self._spec.model_id,
                    filename=filename,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False,
                    token=self._settings.hf_token,
                )
            return local_dir

        snapshot_download(
            repo_id=self._spec.model_id,
            local_dir=local_dir,
            token=self._settings.hf_token,
            allow_patterns=list(self._spec.allow_patterns) or None,
        )
        return local_dir

    def _resolve_dtype(self, torch: Any):
        if self._spec.recommended_dtype == "float16":
            return torch.float16
        if self._spec.recommended_dtype == "bfloat16":
            return torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _run_generation(
        self,
        pipe: Any,
        request: SegmentGenerationRequest,
    ) -> tuple[dict[str, Any], Path]:
        import imageio.v3 as iio
        import numpy as np
        import torch

        kwargs = self._build_call_kwargs(pipe, request, torch)
        result = pipe(**kwargs)
        frames = self._extract_frames(result)
        if not isinstance(frames, list) or not frames:
            raise RuntimeError(f"{self._spec.key} returned no frames.")

        request.outputPath.parent.mkdir(parents=True, exist_ok=True)
        fps = max(1, int(round(kwargs.get("frame_rate", kwargs.get("fps", request.fps)))))
        np_frames = [self._frame_to_ndarray(frame, np) for frame in frames]
        iio.imwrite(request.outputPath, np_frames, fps=fps)

        debug = {
            "modelId": self._spec.model_id,
            "resolvedPrompt": request.resolvedPrompt,
            "resolvedNegativePrompt": request.resolvedNegativePrompt,
            "generationArgs": {
                key: self._stringify_debug_value(value)
                for key, value in kwargs.items()
                if key != "generator"
            },
            "frameCount": len(np_frames),
        }
        return debug, request.outputPath

    def _extract_frames(self, result: Any) -> Any:
        candidates = (
            getattr(result, "frames", None),
            getattr(result, "videos", None),
            result[0] if isinstance(result, tuple) and result else None,
            result,
        )
        for candidate in candidates:
            if candidate is None:
                continue
            frames = candidate
            if isinstance(frames, list) and frames and isinstance(frames[0], list):
                frames = frames[0]
            if isinstance(frames, tuple):
                frames = list(frames)
            if hasattr(frames, "shape"):
                shape = tuple(int(item) for item in frames.shape)
                if len(shape) == 5:
                    frames = frames[0]
                    shape = tuple(int(item) for item in frames.shape)
                if len(shape) == 4:
                    return [frames[index] for index in range(shape[0])]
            if isinstance(frames, list) and frames:
                return frames
        return result

    def _build_call_kwargs(
        self,
        pipe: Any,
        request: SegmentGenerationRequest,
        torch: Any,
    ) -> dict[str, Any]:
        params = inspect.signature(pipe.__call__).parameters
        backend_params = request.backendParams or {}
        width = self._normalize_dimension(
            int(backend_params.get("width", request.width or self._spec.default_width)),
            self._spec.resolution_multiple,
            self._spec.default_width,
        )
        height = self._normalize_dimension(
            int(backend_params.get("height", request.height or self._spec.default_height)),
            self._spec.resolution_multiple,
            self._spec.default_height,
        )
        fps = float(backend_params.get("fps", request.fps or self._spec.default_fps))
        num_frames = self._resolve_num_frames(request, fps, backend_params)
        steps = int(backend_params.get("num_inference_steps", self._spec.default_steps))
        guidance = float(backend_params.get("guidance_scale", self._spec.default_guidance_scale))
        seed = int(backend_params.get("seed", 42))
        prompt = request.resolvedPrompt or request.prompt
        negative_prompt = request.resolvedNegativePrompt or request.negativePrompt

        kwargs: dict[str, Any] = {}
        if "prompt" in params:
            kwargs["prompt"] = prompt
        if "negative_prompt" in params and negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        if "height" in params:
            kwargs["height"] = height
        if "width" in params:
            kwargs["width"] = width
        if "frames" in params:
            kwargs["frames"] = num_frames
        if "num_frames" in params:
            kwargs["num_frames"] = num_frames
        if "num_inference_steps" in params:
            kwargs["num_inference_steps"] = steps
        if "guidance_scale" in params:
            kwargs["guidance_scale"] = guidance
        if "frame_rate" in params:
            kwargs["frame_rate"] = fps
        if "fps" in params:
            kwargs["fps"] = fps
        if "generator" in params:
            kwargs["generator"] = torch.Generator(device="cuda").manual_seed(seed)
        if "motion_score" in params and "motion_score" in backend_params:
            kwargs["motion_score"] = int(backend_params["motion_score"])
        if "max_sequence_length" in params and "max_sequence_length" in backend_params:
            kwargs["max_sequence_length"] = int(backend_params["max_sequence_length"])
        if "image" in params and backend_params.get("image") is not None:
            kwargs["image"] = backend_params["image"]
        if "prompt_embeds" in params and backend_params.get("prompt_embeds") is not None:
            kwargs["prompt_embeds"] = backend_params["prompt_embeds"]
        for key, value in self._spec.static_call_kwargs.items():
            if key in params:
                kwargs[key] = value
        return kwargs

    def _resolve_num_frames(
        self,
        request: SegmentGenerationRequest,
        fps: float,
        backend_params: dict[str, Any],
    ) -> int:
        if "num_frames" in backend_params:
            frames = int(backend_params["num_frames"])
        else:
            frames = int(round(max(1.0, request.durationSec) * max(1.0, fps)))

        multiple = max(1, self._spec.frame_multiple)
        offset = self._spec.frame_offset
        if frames <= offset:
            frames = offset + multiple
        remainder = (frames - offset) % multiple
        if remainder != 0:
            frames += multiple - remainder
        return frames

    def _normalize_dimension(self, value: int, multiple: int, fallback: int) -> int:
        value = max(multiple, value)
        value = value - (value % multiple)
        return value or fallback

    def _frame_to_ndarray(self, frame: Any, np: Any):
        if hasattr(frame, "convert"):
            return np.asarray(frame.convert("RGB"))
        if hasattr(frame, "shape"):
            array = np.asarray(frame)
        else:
            raise RuntimeError(f"Unsupported frame type: {type(frame)!r}")
        if array.ndim == 4:
            array = array[0]
        if array.dtype != np.uint8:
            if array.max() <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        return array

    def _stringify_debug_value(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [self._stringify_debug_value(item) for item in value]
        return repr(value)
