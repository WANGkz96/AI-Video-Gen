from __future__ import annotations

import asyncio
import json
import math
import os
import wave
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
        artifacts, errors = await self.generate_scenes([request])
        if request.sceneId in errors:
            raise RuntimeError(errors[request.sceneId])
        return artifacts[request.sceneId]

    async def generate_scenes(
        self,
        requests: list[DialogueSceneGenerationRequest],
    ) -> tuple[dict[str, GenerationArtifact], dict[str, str]]:
        """Generate one dialogue batch while retaining LongCat weights in VRAM.

        The upstream demo reloads every model inside its one-scene entrypoint.
        A job normally contains several independent dialogue scenes, so the
        maintained batch runner loads the INT8/distilled runtime once, renders
        those scenes serially, and returns an artifact per scene.
        """
        info = self.info()
        if not info.available:
            raise AdapterUnavailableError(info.notes or "LongCat Video Avatar is unavailable.")
        if not requests:
            return {}, {}

        prepared = [self._prepare_scene(request) for request in requests]
        batch_root = requests[0].outputPath.parent.parent / "longcat-batches"
        batch_root.mkdir(parents=True, exist_ok=True)
        batch_key = "-".join(f"{request.sceneIndex:03d}" for request in requests)
        batch_dir = batch_root / f"batch-{batch_key}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        batch_manifest = batch_dir / "batch-input.json"
        result_manifest = batch_dir / "batch-result.json"
        batch_manifest.write_text(
            json.dumps(
                {
                    "scenes": [
                        {
                            "sceneId": item["request"].sceneId,
                            "inputJson": item["inputPath"].as_posix(),
                            "outputDir": item["generatedDir"].as_posix(),
                            "numSegments": item["numSegments"],
                        }
                        for item in prepared
                    ]
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        runner = self._root / "run_longcat_avatar_batch.py"
        if not runner.is_file():
            raise AdapterUnavailableError(
                "LongCat batch runner is unavailable. Re-run LongCat provisioning for this instance."
            )
        command = [
            self._torchrun.as_posix(),
            "--master_port",
            str(29600 + (requests[0].sceneIndex % 300)),
            "--nproc_per_node=1",
            runner.as_posix(),
            "--batch_manifest",
            batch_manifest.as_posix(),
            "--result_manifest",
            result_manifest.as_posix(),
            "--resolution",
            "480p",
            "--checkpoint_dir",
            self._checkpoint.as_posix(),
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
        log_path = batch_dir / "longcat-batch.log"
        log_path.write_text(log_text, encoding="utf-8")
        if not result_manifest.is_file():
            raise RuntimeError(
                f"LongCat batch exited with code {process.returncode} without a result manifest: {log_text[-6000:]}"
            )
        results = json.loads(result_manifest.read_text(encoding="utf-8"))
        by_scene = {str(item.get("sceneId")): item for item in results.get("scenes", [])}
        artifacts: dict[str, GenerationArtifact] = {}
        errors: dict[str, str] = {}
        for item in prepared:
            request = item["request"]
            result = by_scene.get(request.sceneId)
            if not result or result.get("status") != "completed":
                errors[request.sceneId] = str((result or {}).get("error") or "LongCat did not return a scene result.")
                continue
            source_video = Path(str(result.get("outputPath") or ""))
            if not source_video.is_file():
                errors[request.sceneId] = f"LongCat returned a missing output: {source_video.as_posix()}"
                continue
            await self._normalize_video(source_video, request)
            artifacts[request.sceneId] = GenerationArtifact(
                modelName=self.key,
                modelVersion="LongCat-Video-Avatar-1.5@6b3f4b8-int8-distill-batched",
                outputPath=request.outputPath,
                debug={
                    "input": item["inputPath"].as_posix(),
                    "rawOutput": source_video.as_posix(),
                    "log": log_path.as_posix(),
                    "batchManifest": batch_manifest.as_posix(),
                    "batchResult": result_manifest.as_posix(),
                    "batchSceneCount": len(requests),
                    "numSegments": item["numSegments"],
                    "durationSec": request.durationSec,
                    "fps": 25,
                    "substitutedSilentTracks": item["substitutedSilentTracks"],
                    "processReturnCode": process.returncode,
                    "processReturnedNonZeroAfterResults": process.returncode != 0,
                },
            )
        return artifacts, errors

    def _prepare_scene(self, request: DialogueSceneGenerationRequest) -> dict[str, object]:

        request.outputPath.parent.mkdir(parents=True, exist_ok=True)
        run_dir = request.outputPath.parent / "longcat-runtime"
        generated_dir = run_dir / "generated"
        run_dir.mkdir(parents=True, exist_ok=True)
        generated_dir.mkdir(parents=True, exist_ok=True)
        speaker_paths = {
            "person1": request.speaker1Path,
            "person2": request.speaker2Path,
        }
        substituted_silent_tracks: list[str] = []
        for person, source_path in tuple(speaker_paths.items()):
            if not self._pcm_wav_is_fully_silent(source_path):
                continue
            prepared_path = run_dir / f"{person}_technical_silence.wav"
            self._write_longcat_compatible_silence(source_path, prepared_path, request.durationSec)
            speaker_paths[person] = prepared_path
            substituted_silent_tracks.append(person)
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
                "person1": speaker_paths["person1"].as_posix(),
                "person2": speaker_paths["person2"].as_posix(),
            },
            "audio_type": "para",
            "target_orientation": "portrait" if request.height > request.width else "landscape",
        }
        input_path = run_dir / "avatar_input.json"
        input_path.write_text(json.dumps(input_doc, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "request": request,
            "runDir": run_dir,
            "generatedDir": generated_dir,
            "inputPath": input_path,
            "numSegments": num_segments,
            "substitutedSilentTracks": substituted_silent_tracks,
        }

    @staticmethod
    def _pcm_wav_is_fully_silent(path: Path) -> bool:
        """Return True only for a valid PCM WAV whose samples are all digital silence."""
        try:
            with wave.open(path.as_posix(), "rb") as source:
                sample_width = source.getsampwidth()
                frames = source.readframes(source.getnframes())
        except (OSError, EOFError, wave.Error):
            return False

        if not frames:
            return True
        if sample_width == 1:
            # Eight-bit PCM WAV uses unsigned 128 as its zero level.
            return all(sample == 128 for sample in frames)
        return not any(frames)

    @staticmethod
    def _write_longcat_compatible_silence(source_path: Path, output_path: Path, duration_sec: float) -> None:
        """Preserve silence while adding one inaudible LSB sample for LongCat's separator."""
        try:
            with wave.open(source_path.as_posix(), "rb") as source:
                channels = max(1, source.getnchannels())
                sample_width = max(1, source.getsampwidth())
                sample_rate = max(1, source.getframerate())
                frames = bytearray(source.readframes(source.getnframes()))
        except (OSError, EOFError, wave.Error):
            channels = 1
            sample_width = 2
            sample_rate = 24_000
            frames = bytearray()

        frame_width = channels * sample_width
        if not frames:
            frame_count = max(1, round(max(0.01, duration_sec) * sample_rate))
            if sample_width == 1:
                frames = bytearray([128]) * (frame_count * frame_width)
            else:
                frames = bytearray(frame_count * frame_width)

        # audio-separator rejects mathematically all-zero input. A single
        # least-significant-bit sample is inaudible but keeps the silent actor valid.
        frames[0] = 129 if sample_width == 1 else 1
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(output_path.as_posix(), "wb") as target:
            target.setnchannels(channels)
            target.setsampwidth(sample_width)
            target.setframerate(sample_rate)
            target.writeframes(frames)

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
