from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path

from backend.app.adapters.base import AdapterUnavailableError, BaseGeneratorAdapter
from backend.app.config import Settings
from backend.app.models import (
    AdapterInfo,
    DialogueSceneGenerationRequest,
    GenerationArtifact,
    SegmentGenerationRequest,
)


class LongCatAvatarAdapter(BaseGeneratorAdapter):
    key = "longcat-video-avatar"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._root = Path(os.getenv("LONGCAT_REPO_DIR", "/workspace/LongCat-Video")).resolve()
        self._checkpoint = Path(
            os.getenv(
                "LONGCAT_AVATAR_CHECKPOINT_DIR",
                self._root / "weights" / "LongCat-Video-Avatar-1.5",
            )
        ).resolve()
        self._python_env = Path(os.getenv("LONGCAT_CONDA_ENV_DIR", "/opt/conda/envs/longcat-video")).resolve()
        self._torchrun = Path(os.getenv("LONGCAT_TORCHRUN", self._python_env / "bin" / "torchrun")).resolve()

    def info(self) -> AdapterInfo:
        runner = self._root / "run_demo_avatar_multi_audio_to_video.py"
        available = runner.is_file() and self._checkpoint.is_dir() and self._torchrun.is_file()
        missing: list[str] = []
        if not runner.is_file():
            missing.append(f"runner: {runner.as_posix()}")
        if not self._checkpoint.is_dir():
            missing.append(f"checkpoint: {self._checkpoint.as_posix()}")
        if not self._torchrun.is_file():
            missing.append(f"environment: {self._torchrun.as_posix()}")
        return AdapterInfo(
            key=self.key,
            label="LongCat Video Avatar 1.5",
            description="Two-person image and separate-audio talking-avatar generation.",
            status="ready" if available else "experimental",
            available=available,
            supportsBatch=True,
            supportsDirect=False,
            requiresRemote=True,
            requiresDownload=not available,
            modelId="meituan-longcat/LongCat-Video-Avatar-1.5",
            localPath=self._checkpoint.as_posix(),
            minimumVramGb=90,
            notes=None if available else "LongCat provisioning is incomplete: " + "; ".join(missing),
        )

    async def generate_segment(self, request: SegmentGenerationRequest) -> GenerationArtifact:
        raise AdapterUnavailableError("LongCat Video Avatar only accepts dialogue scene requests.")

    async def generate_scene(self, request: DialogueSceneGenerationRequest) -> GenerationArtifact:
        info = self.info()
        if not info.available:
            raise AdapterUnavailableError(info.notes or "LongCat Video Avatar is unavailable.")

        request.outputPath.parent.mkdir(parents=True, exist_ok=True)
        run_dir = request.outputPath.parent / "longcat-runtime"
        generated_dir = run_dir / "generated"
        run_dir.mkdir(parents=True, exist_ok=True)
        generated_dir.mkdir(parents=True, exist_ok=True)
        num_segments = max(1, 1 + math.ceil(max(0.0, request.durationSec - 3.72) / 3.2))
        stable_prompt = " ".join(
            item
            for item in [
                request.prompt.strip(),
                "Preserve the exact two people, location and framing from the conditioning image.",
                "Natural multi-person lip sync: each mouth moves only for its own audio track.",
                "Use restrained physically coherent conversational gestures and reactions.",
                "Keep both faces visible. No cuts, subtitles, visible text, speech bubbles, zoom, push-in, dolly, reframing, duplicated people, or additional characters.",
            ]
            if item
        )
        input_doc = {
            "prompt": stable_prompt,
            "prompt_schedule": [
                f"Continuation segment {index + 1}. {stable_prompt} Do not repeat a completed gesture as a loop."
                for index in range(num_segments)
            ],
            "cond_image": request.imagePath.as_posix(),
            "cond_audio": {
                "person1": request.speaker1Path.as_posix(),
                "person2": request.speaker2Path.as_posix(),
            },
            "audio_type": "para",
            "target_orientation": "portrait" if request.height > request.width else "landscape",
        }
        input_path = run_dir / "avatar_input.json"
        input_path.write_text(json.dumps(input_doc, ensure_ascii=False, indent=2), encoding="utf-8")
        command = [
            self._torchrun.as_posix(),
            "--master_port",
            str(29600 + (request.sceneIndex % 300)),
            "--nproc_per_node=1",
            "run_demo_avatar_multi_audio_to_video.py",
            "--input_json",
            input_path.as_posix(),
            "--output_dir",
            generated_dir.as_posix(),
            "--resolution",
            "480p",
            "--num_segments",
            str(num_segments),
            "--ref_img_index",
            "30",
            "--mask_frame_range",
            "5",
            "--checkpoint_dir",
            self._checkpoint.as_posix(),
            "--model_type",
            "avatar-v1.5",
            "--use_distill",
            "--use_int8",
        ]
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=self._root,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await process.communicate()
        log_text = stdout.decode("utf-8", errors="replace")
        (run_dir / "longcat.log").write_text(log_text, encoding="utf-8")
        if process.returncode != 0:
            raise RuntimeError(
                f"LongCat exited with code {process.returncode}: {log_text[-6000:]}"
            )

        generated_files = sorted(
            generated_dir.rglob("*.mp4"),
            key=lambda item: item.stat().st_mtime,
        )
        if not generated_files:
            raise FileNotFoundError(f"LongCat created no mp4 in {generated_dir.as_posix()}")
        preferred_name = f"video_continue_{num_segments}.mp4"
        source_video = next(
            (item for item in reversed(generated_files) if item.name == preferred_name),
            generated_files[-1],
        )
        await self._normalize_video(source_video, request)
        return GenerationArtifact(
            modelName=self.key,
            modelVersion="LongCat-Video-Avatar-1.5@6b3f4b8-int8-distill",
            outputPath=request.outputPath,
            debug={
                "input": input_path.as_posix(),
                "rawOutput": source_video.as_posix(),
                "log": (run_dir / "longcat.log").as_posix(),
                "numSegments": num_segments,
                "durationSec": request.durationSec,
                "fps": 25,
            },
        )

    async def _normalize_video(
        self,
        source_video: Path,
        request: DialogueSceneGenerationRequest,
    ) -> None:
        video_filter = (
            f"scale={request.width}:{request.height}:force_original_aspect_ratio=increase,"
            f"crop={request.width}:{request.height},fps=25,"
            f"trim=duration={request.durationSec},setpts=PTS-STARTPTS"
        )
        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            source_video.as_posix(),
            "-vf",
            video_filter,
            "-t",
            str(request.durationSec),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            request.outputPath.as_posix(),
        ]
        process = await asyncio.create_subprocess_exec(*command)
        code = await process.wait()
        if code != 0 or not request.outputPath.is_file():
            raise RuntimeError(f"ffmpeg failed to normalize LongCat output (exit {code}).")
