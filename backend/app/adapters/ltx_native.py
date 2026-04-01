from __future__ import annotations

import asyncio
import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from backend.app.adapters.base import AdapterUnavailableError, BaseGeneratorAdapter
from backend.app.config import REPO_ROOT, Settings
from backend.app.models import AdapterInfo, GenerationArtifact, SegmentGenerationRequest


@dataclass(slots=True)
class LtxNativeBackendSpec:
    key: str
    label: str
    description: str
    model_id: str
    checkpoint_filename: str
    gemma_model_id: str
    gemma_dir_name: str
    runtime_repo_url: str
    runtime_repo_ref: str
    pipeline_module: str = "ltx_pipelines.ti2vid_one_stage"
    status: Literal["ready", "planned", "experimental"] = "experimental"
    minimum_vram_gb: int = 32
    default_width: int = 768
    default_height: int = 512
    default_fps: float = 16.0
    default_steps: int = 30
    resolution_multiple: int = 32
    frame_multiple: int = 8
    frame_offset: int = 1
    notes: str | None = None

    @property
    def storage_dir_name(self) -> str:
        return self.key


def get_ltx_native_backend_specs() -> dict[str, LtxNativeBackendSpec]:
    return {
        "ltx-2.3": LtxNativeBackendSpec(
            key="ltx-2.3",
            label="LTX 2.3",
            description="Official Lightricks LTX 2.3 one-stage runtime with native PyTorch pipelines.",
            model_id="Lightricks/LTX-2.3",
            checkpoint_filename="ltx-2.3-22b-dev.safetensors",
            gemma_model_id="google/gemma-3-12b-it-qat-q4_0-unquantized",
            gemma_dir_name="gemma-3-12b-it-qat-q4_0-unquantized",
            runtime_repo_url="https://github.com/Lightricks/LTX-2.git",
            runtime_repo_ref="59ca828d5ae24358832ffd7003c2306fbceeba3a",
            status="experimental",
            minimum_vram_gb=32,
            default_width=768,
            default_height=512,
            default_fps=16.0,
            default_steps=20,
            notes=(
                "Uses the official native Lightricks pipeline and requires the Gemma 3 text encoder assets "
                "plus the external LTX runtime packages."
            ),
        )
    }


class LtxNativeAdapter(BaseGeneratorAdapter):
    def __init__(self, settings: Settings, spec: LtxNativeBackendSpec) -> None:
        self._settings = settings
        self._spec = spec

    def info(self) -> AdapterInfo:
        checkpoint = self._checkpoint_path()
        gemma_root = self._gemma_root()
        deps_ok, dep_error = self._check_dependencies()
        runtime_ready = self._runtime_installed()
        assets_ready = checkpoint.is_file() and self._gemma_ready()
        available = deps_ok and runtime_ready and assets_ready and self._spec.status != "planned"
        notes = self._spec.notes

        if dep_error:
            available = False
            notes = dep_error
        elif self._spec.status == "planned":
            available = False
            notes = self._spec.notes or "Native runtime scaffold exists, but generation is not wired yet."
        elif not runtime_ready:
            available = False
            notes = (
                "Official LTX runtime packages are not installed yet. "
                "Run the bootstrap/download script to clone and install the native Lightricks runtime."
            )
        elif not checkpoint.is_file():
            available = False
            notes = (
                f"LTX checkpoint is not downloaded yet. Expected file: {checkpoint.as_posix()}. "
                "Run the bootstrap/download script or the model download CLI."
            )
        elif not self._gemma_ready():
            available = False
            if not self._settings.hf_token:
                notes = (
                    "Gemma 3 assets are not downloaded yet and the official repository is gated on Hugging Face. "
                    "Set HF_TOKEN with approved access to "
                    f"'{self._spec.gemma_model_id}', then rerun bootstrap/download. "
                    f"Expected path: {gemma_root.as_posix()}."
                )
            else:
                notes = (
                    f"Gemma assets are not downloaded yet. Expected path: {gemma_root.as_posix()}. "
                    "Run the bootstrap/download script or the model download CLI."
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
            requiresDownload=not assets_ready,
            modelId=self._spec.model_id,
            localPath=self._checkpoint_dir().as_posix(),
            minimumVramGb=self._spec.minimum_vram_gb,
            notes=notes,
        )

    async def generate_segment(self, request: SegmentGenerationRequest) -> GenerationArtifact:
        await asyncio.to_thread(self._prepare_runtime)
        command, env, debug = self._build_command(request)
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=REPO_ROOT.as_posix(),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        stdout_text = stdout.decode("utf-8", errors="ignore").strip()
        stderr_text = stderr.decode("utf-8", errors="ignore").strip()

        if process.returncode != 0:
            raise RuntimeError(
                "\n".join(
                    item
                    for item in [
                        f"Official LTX runtime failed with exit code {process.returncode}.",
                        stdout_text,
                        stderr_text,
                    ]
                    if item
                )
            )

        if not request.outputPath.is_file():
            raise FileNotFoundError(
                f"LTX runtime exited successfully but did not create {request.outputPath.name}."
            )

        debug["stdout"] = stdout_text
        debug["stderr"] = stderr_text
        return GenerationArtifact(
            modelName=self._spec.key,
            modelVersion=self._spec.model_id,
            outputPath=request.outputPath,
            debug=debug,
        )

    async def download_assets(self) -> Path:
        return await asyncio.to_thread(self._download_assets)

    def _check_dependencies(self) -> tuple[bool, str | None]:
        try:
            import huggingface_hub  # noqa: F401
        except Exception as exc:
            return False, f"Runtime dependencies are missing for '{self._spec.key}': {exc}"
        return True, None

    def _runtime_installed(self) -> bool:
        return importlib.util.find_spec("ltx_pipelines") is not None

    def _checkpoint_dir(self) -> Path:
        return self._settings.models_dir / self._spec.storage_dir_name

    def _checkpoint_path(self) -> Path:
        return self._checkpoint_dir() / self._spec.checkpoint_filename

    def _gemma_root(self) -> Path:
        return self._settings.models_dir / self._spec.gemma_dir_name

    def _gemma_ready(self) -> bool:
        gemma_root = self._gemma_root()
        if not gemma_root.is_dir():
            return False
        tokenizer = next(gemma_root.rglob("tokenizer.model"), None)
        preprocessor = next(gemma_root.rglob("preprocessor_config.json"), None)
        weights = next(gemma_root.rglob("model*.safetensors"), None)
        return tokenizer is not None and preprocessor is not None and weights is not None

    def _runtime_repo_dir(self) -> Path:
        return self._settings.ltx_runtime_repo_dir

    def _download_assets(self) -> Path:
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
            from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
        except Exception as exc:
            raise AdapterUnavailableError(f"Cannot download model assets: {exc}") from exc

        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

        checkpoint_dir = self._checkpoint_dir()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if not self._checkpoint_path().is_file():
            hf_hub_download(
                repo_id=self._spec.model_id,
                filename=self._spec.checkpoint_filename,
                local_dir=checkpoint_dir,
                local_dir_use_symlinks=False,
                token=self._settings.hf_token,
            )
            hf_hub_download(
                repo_id=self._spec.model_id,
                filename="README.md",
                local_dir=checkpoint_dir,
                local_dir_use_symlinks=False,
                token=self._settings.hf_token,
            )

        gemma_root = self._gemma_root()
        if not self._gemma_ready():
            try:
                snapshot_download(
                    repo_id=self._spec.gemma_model_id,
                    local_dir=gemma_root,
                    token=self._settings.hf_token,
                )
            except GatedRepoError as exc:
                raise AdapterUnavailableError(
                    "Gemma 3 access is gated on Hugging Face. "
                    f"Approve access to '{self._spec.gemma_model_id}' and set HF_TOKEN before downloading LTX 2.3 assets."
                ) from exc
            except HfHubHTTPError as exc:
                status = getattr(exc.response, "status_code", None)
                if status in {401, 403}:
                    raise AdapterUnavailableError(
                        "Gemma 3 download was denied by Hugging Face. "
                        f"Approve access to '{self._spec.gemma_model_id}' and set HF_TOKEN before downloading LTX 2.3 assets."
                    ) from exc
                raise

        self._prepare_runtime()
        return checkpoint_dir

    def _prepare_runtime(self) -> None:
        if not self._checkpoint_path().is_file() or not self._gemma_ready():
            self._download_assets()
            return
        if not self._runtime_installed():
            self._install_runtime()

    def _install_runtime(self) -> None:
        git_executable = shutil.which("git")
        if git_executable is None:
            raise AdapterUnavailableError(
                "git is required to install the official LTX runtime, but it was not found in PATH."
            )

        repo_dir = self._runtime_repo_dir()
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        if not (repo_dir / ".git").is_dir():
            self._run_command(
                [git_executable, "clone", self._spec.runtime_repo_url, repo_dir.as_posix()],
                "Failed to clone the official LTX runtime repository.",
            )
        self._run_command(
            [git_executable, "-C", repo_dir.as_posix(), "fetch", "--tags", "--force", "origin"],
            "Failed to refresh the official LTX runtime repository.",
        )
        self._run_command(
            [git_executable, "-C", repo_dir.as_posix(), "checkout", "--force", self._spec.runtime_repo_ref],
            "Failed to checkout the pinned official LTX runtime revision.",
        )

        runtime_env = os.environ.copy()
        runtime_env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
        shared_deps = ["scipy>=1.14", "av>=16.0.1", "tqdm>=4.67.1", "pillow>=11.3.0"]
        self._run_command(
            [sys.executable, "-m", "pip", "install", *shared_deps],
            "Failed to install shared native LTX runtime dependencies.",
            env=runtime_env,
        )
        self._run_command(
            [sys.executable, "-m", "pip", "install", "--no-deps", "-e", (repo_dir / "packages" / "ltx-core").as_posix()],
            "Failed to install the official ltx-core package.",
            env=runtime_env,
        )
        self._run_command(
            [sys.executable, "-m", "pip", "install", "--no-deps", "-e", (repo_dir / "packages" / "ltx-pipelines").as_posix()],
            "Failed to install the official ltx-pipelines package.",
            env=runtime_env,
        )

        if not self._runtime_installed():
            raise AdapterUnavailableError("The official LTX runtime was installed, but ltx_pipelines is still unavailable.")

    def _build_command(
        self, request: SegmentGenerationRequest
    ) -> tuple[list[str], dict[str, str], dict[str, object]]:
        backend_params = dict(request.backendParams or {})
        width = self._normalize_dimension(
            int(backend_params.get("width", request.width or self._spec.default_width))
        )
        height = self._normalize_dimension(
            int(backend_params.get("height", request.height or self._spec.default_height))
        )
        fps = float(backend_params.get("fps", request.fps or self._spec.default_fps))
        num_frames = self._resolve_num_frames(request, fps, backend_params)
        steps = int(backend_params.get("num_inference_steps", self._spec.default_steps))
        seed = int(backend_params.get("seed", 42))
        prompt = request.resolvedPrompt or request.prompt
        negative_prompt = request.resolvedNegativePrompt or request.negativePrompt
        command = [
            sys.executable,
            "-m",
            self._spec.pipeline_module,
            "--checkpoint-path",
            self._checkpoint_path().as_posix(),
            "--gemma-root",
            self._gemma_root().as_posix(),
            "--prompt",
            prompt,
            "--output-path",
            request.outputPath.as_posix(),
            "--seed",
            str(seed),
            "--height",
            str(height),
            "--width",
            str(width),
            "--num-frames",
            str(num_frames),
            "--frame-rate",
            str(fps),
            "--num-inference-steps",
            str(steps),
        ]

        if negative_prompt:
            command.extend(["--negative-prompt", negative_prompt])
        if "streaming_prefetch_count" in backend_params:
            command.extend(
                ["--streaming-prefetch-count", str(int(backend_params["streaming_prefetch_count"]))]
            )
        if "max_batch_size" in backend_params:
            command.extend(["--max-batch-size", str(int(backend_params["max_batch_size"]))])
        if "video_cfg_guidance_scale" in backend_params:
            command.extend(
                ["--video-cfg-guidance-scale", str(float(backend_params["video_cfg_guidance_scale"]))]
            )
        if "audio_cfg_guidance_scale" in backend_params:
            command.extend(
                ["--audio-cfg-guidance-scale", str(float(backend_params["audio_cfg_guidance_scale"]))]
            )
        if backend_params.get("enhance_prompt"):
            command.append("--enhance-prompt")

        env = os.environ.copy()
        env.setdefault("HF_HUB_DISABLE_XET", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if self._settings.hf_token:
            env.setdefault("HF_TOKEN", self._settings.hf_token)
            env.setdefault("HUGGING_FACE_HUB_TOKEN", self._settings.hf_token)

        debug = {
            "modelId": self._spec.model_id,
            "pipelineModule": self._spec.pipeline_module,
            "checkpointPath": self._checkpoint_path().as_posix(),
            "gemmaRoot": self._gemma_root().as_posix(),
            "generationArgs": {
                "prompt": prompt,
                "negative_prompt": negative_prompt or "<official-default>",
                "seed": seed,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "frame_rate": fps,
                "num_inference_steps": steps,
                "streaming_prefetch_count": backend_params.get("streaming_prefetch_count"),
                "max_batch_size": backend_params.get("max_batch_size"),
            },
            "command": command,
        }
        return command, env, debug

    def _resolve_num_frames(
        self,
        request: SegmentGenerationRequest,
        fps: float,
        backend_params: dict[str, object],
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

    def _normalize_dimension(self, value: int) -> int:
        value = max(self._spec.resolution_multiple, value)
        value = value - (value % self._spec.resolution_multiple)
        return value or self._spec.resolution_multiple

    def _run_command(
        self,
        command: list[str],
        failure_message: str,
        *,
        env: dict[str, str] | None = None,
    ) -> None:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            return
        error_blob = "\n".join(
            item for item in [failure_message, completed.stdout.strip(), completed.stderr.strip()] if item
        )
        raise AdapterUnavailableError(error_blob)
