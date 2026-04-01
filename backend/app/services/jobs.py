from __future__ import annotations

import asyncio
import json
import sys
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.app.adapters.base import AdapterUnavailableError, BaseGeneratorAdapter
from backend.app.adapters.catalog import get_diffusers_backend_specs
from backend.app.adapters.comfyui import ComfyUiWorkflowAdapter
from backend.app.adapters.diffusers_video import DiffusersVideoAdapter
from backend.app.adapters.mock_gen import MockGenAdapter
from backend.app.adapters.planned import PlannedAdapter
from backend.app.config import REPO_ROOT, Settings
from backend.app.models import (
    AdapterInfo,
    BatchExport,
    DirectGenerationRequest,
    GenerationArtifact,
    JobQueuedResponse,
    JobSnapshot,
    LogEntry,
    ManifestSegment,
    SegmentGenerationRequest,
    VariantInfo,
    VariantManifest,
    VideoInfo,
)
from backend.app.services.media import probe_video


def utc_now() -> datetime:
    return datetime.now(UTC)


def compact_timestamp() -> str:
    return utc_now().strftime("%Y%m%d_%H%M%S_%f")


@dataclass(slots=True)
class JobRuntime:
    snapshot: JobSnapshot
    batch: BatchExport
    workspace_dir: Path
    input_path: Path
    result_path: Path
    archive_path: Path
    snapshot_path: Path
    logs_path: Path
    logs: list[LogEntry] = field(default_factory=list)
    subscribers: list[asyncio.Queue[dict[str, Any]]] = field(default_factory=list)


class JobService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._settings.ensure_dirs()
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._jobs: dict[str, JobRuntime] = {}
        self._worker_task: asyncio.Task[None] | None = None
        self._job_counter = 0
        self._adapters = self._build_registry()

    async def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker_loop(), name="video-job-worker")

    async def stop(self) -> None:
        if self._worker_task is None:
            return
        self._worker_task.cancel()
        try:
            await self._worker_task
        except asyncio.CancelledError:
            pass
        self._worker_task = None

    def health(self) -> dict[str, Any]:
        active_jobs = sum(
            1
            for runtime in self._jobs.values()
            if runtime.snapshot.status in {"queued", "running"}
        )
        return {
            "status": "ok",
            "defaultBackend": self._settings.generator_backend,
            "queuedJobs": self._queue.qsize(),
            "activeJobs": active_jobs,
            "frontendBuilt": self._settings.frontend_dist_dir.exists(),
        }

    def list_backends(self) -> list[AdapterInfo]:
        return [adapter.info() for adapter in self._adapters.values()]

    async def create_job(
        self, batch_payload: dict[str, Any], backend: str | None = None
    ) -> JobQueuedResponse:
        batch = BatchExport.model_validate(batch_payload)
        selected_backend = backend or self._settings.generator_backend
        self._ensure_backend_can_run(selected_backend)
        total_videos = len(batch.videos)
        total_variants = sum(len(video.variants) for video in batch.videos)
        total_segments = sum(
            len(variant.manifest.segments)
            for video in batch.videos
            for variant in video.variants
            if variant.manifest is not None
        )

        job_id = self._next_job_id()
        workspace_dir = self._settings.jobs_dir / job_id
        input_dir = workspace_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        input_path = input_dir / "batch.json"
        result_path = workspace_dir / "result.json"
        archive_path = self._settings.archive_dir / f"{job_id}.zip"
        snapshot_path = workspace_dir / "job.json"
        logs_path = workspace_dir / "logs.jsonl"

        snapshot = JobSnapshot(
            jobId=job_id,
            status="queued",
            backend=selected_backend,
            totalVideos=total_videos,
            totalVariants=total_variants,
            totalSegments=total_segments,
            updatedAt=utc_now(),
            inputFile=input_path.relative_to(workspace_dir).as_posix(),
        )
        runtime = JobRuntime(
            snapshot=snapshot,
            batch=batch,
            workspace_dir=workspace_dir,
            input_path=input_path,
            result_path=result_path,
            archive_path=archive_path,
            snapshot_path=snapshot_path,
            logs_path=logs_path,
        )
        self._jobs[job_id] = runtime
        self._write_json(input_path, batch.model_dump(mode="json"))
        self._write_snapshot(runtime)
        await self._log(runtime, "info", f"Job {job_id} queued with backend '{selected_backend}'.")
        await self._queue.put(job_id)
        return JobQueuedResponse(jobId=job_id, status=snapshot.status)

    async def create_direct_job(self, request: DirectGenerationRequest) -> JobQueuedResponse:
        payload = self._build_direct_batch_payload(request)
        return await self.create_job(payload, backend=request.backend)

    def get_job(self, job_id: str) -> JobSnapshot:
        return self._runtime(job_id).snapshot

    def get_logs(self, job_id: str) -> list[LogEntry]:
        return list(self._runtime(job_id).logs)

    def get_result(self, job_id: str) -> dict[str, Any]:
        runtime = self._runtime(job_id)
        if not runtime.result_path.is_file():
            raise FileNotFoundError("Job result.json is not ready yet.")
        return json.loads(runtime.result_path.read_text(encoding="utf-8"))

    def get_archive_path(self, job_id: str) -> Path:
        runtime = self._runtime(job_id)
        if not runtime.archive_path.is_file():
            raise FileNotFoundError("Job archive is not ready yet.")
        return runtime.archive_path

    def get_job_file(self, job_id: str, relative_path: str) -> Path:
        runtime = self._runtime(job_id)
        target = (runtime.workspace_dir / relative_path).resolve()
        workspace = runtime.workspace_dir.resolve()
        if workspace not in target.parents and target != workspace:
            raise PermissionError("Requested path is outside the job workspace.")
        if not target.is_file():
            raise FileNotFoundError(relative_path)
        return target

    async def stream_events(self, job_id: str):
        runtime = self._runtime(job_id)
        yield {"type": "snapshot", "data": runtime.snapshot.model_dump(mode="json")}
        for log in runtime.logs:
            yield {"type": "log", "data": log.model_dump(mode="json")}
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=200)
        runtime.subscribers.append(queue)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    event = {"type": "heartbeat", "data": {"ts": utc_now().isoformat()}}
                yield event
                if (
                    event["type"] == "snapshot"
                    and event["data"]["status"] in {"completed", "completed_with_errors", "failed"}
                ):
                    break
        finally:
            if queue in runtime.subscribers:
                runtime.subscribers.remove(queue)

    def _build_registry(self) -> dict[str, BaseGeneratorAdapter]:
        registry: dict[str, BaseGeneratorAdapter] = {
            "mock-gen": MockGenAdapter(self._settings.mock_media_dir),
            "comfyui-workflow": ComfyUiWorkflowAdapter(self._settings.generator_api_url),
        }
        for spec in get_diffusers_backend_specs().values():
            registry[spec.key] = DiffusersVideoAdapter(self._settings, spec)
        registry["ltx-video-2"] = PlannedAdapter(
            "ltx-video-2",
            "LTX Video 2",
            "Use 'ltx-2-distilled' instead.",
        )
        registry["ltx-video-2-distilled"] = PlannedAdapter(
            "ltx-video-2-distilled",
            "LTX Video 2 Distilled",
            "Use 'ltx-2-distilled' instead.",
        )
        registry["hunyuan-video"] = PlannedAdapter(
            "hunyuan-video",
            "Hunyuan Video",
            "Use 'hunyuan-video-1.5' instead.",
        )
        registry["wan-2.1"] = PlannedAdapter(
            "wan-2.1",
            "Wan 2.1",
            "Use 'wan2.2-ti2v-5b' instead.",
        )
        registry["cogvideox"] = PlannedAdapter(
            "cogvideox",
            "CogVideoX",
            "Use 'cogvideox-5b' instead.",
        )
        return registry

    def _ensure_backend_can_run(self, backend: str) -> None:
        if backend not in self._adapters:
            raise ValueError(f"Unknown backend '{backend}'.")
        info = self._adapters[backend].info()
        if not info.available:
            raise AdapterUnavailableError(info.notes or f"Backend '{backend}' is unavailable.")

    def _runtime(self, job_id: str) -> JobRuntime:
        runtime = self._jobs.get(job_id)
        if runtime is None:
            raise KeyError(job_id)
        return runtime

    def _next_job_id(self) -> str:
        self._job_counter += 1
        return f"job_{compact_timestamp()}_{self._job_counter:04d}"

    async def _worker_loop(self) -> None:
        while True:
            job_id = await self._queue.get()
            runtime = self._runtime(job_id)
            try:
                self._release_adapters(except_backend=runtime.snapshot.backend)
                await self._process_job(runtime)
            except Exception as exc:
                runtime.snapshot.status = "failed"
                runtime.snapshot.error = str(exc)
                runtime.snapshot.finishedAt = utc_now()
                runtime.snapshot.updatedAt = utc_now()
                self._write_snapshot(runtime)
                await self._broadcast_snapshot(runtime)
                await self._log(runtime, "error", f"Job failed: {exc}")
            finally:
                self._release_adapters()
                self._queue.task_done()

    async def _process_job(self, runtime: JobRuntime) -> None:
        runtime.snapshot.status = "running"
        runtime.snapshot.startedAt = utc_now()
        runtime.snapshot.updatedAt = utc_now()
        self._write_snapshot(runtime)
        await self._broadcast_snapshot(runtime)
        await self._log(runtime, "info", "Processing started.")

        adapter = self._adapters[runtime.snapshot.backend]
        result_doc: dict[str, Any] = {
            "schemaVersion": "video-pipeline.external-generation.output.v1",
            "generatedAt": utc_now().isoformat(),
            "videos": [],
            "errors": [],
        }

        for video in runtime.batch.videos:
            video_entry = {
                "videoId": video.videoId,
                "projectId": video.projectId,
                "runId": video.runId,
                "variants": [],
            }
            for variant in video.variants:
                variant_entry = {"key": variant.key, "segments": []}
                video_entry["variants"].append(variant_entry)
                if variant.manifest is None or not variant.manifestFound:
                    message = f"Variant '{variant.key}' has no manifest payload."
                    result_doc["errors"].append(
                        {
                            "videoId": video.videoId,
                            "variantKey": variant.key,
                            "status": "failed",
                            "error": message,
                        }
                    )
                    await self._log(runtime, "error", message)
                    runtime.snapshot.failedSegments += 1
                    runtime.snapshot.updatedAt = utc_now()
                    self._write_snapshot(runtime)
                    await self._broadcast_snapshot(runtime)
                    continue
                await self._process_variant(runtime, adapter, video, variant, variant_entry)
            result_doc["videos"].append(video_entry)

        result_doc["generatedAt"] = utc_now().isoformat()
        validation_errors = self._validate_result(runtime, result_doc)
        if validation_errors:
            result_doc["errors"].extend(validation_errors)
            for error in validation_errors:
                await self._log(runtime, "error", error["error"])

        self._write_json(runtime.result_path, result_doc)
        self._create_archive(runtime)
        runtime.snapshot.resultFile = runtime.result_path.relative_to(runtime.workspace_dir).as_posix()
        runtime.snapshot.archiveFile = runtime.archive_path.as_posix()
        runtime.snapshot.updatedAt = utc_now()
        runtime.snapshot.finishedAt = utc_now()
        runtime.snapshot.status = (
            "completed_with_errors"
            if runtime.snapshot.failedSegments > 0 or validation_errors
            else "completed"
        )
        self._write_snapshot(runtime)
        await self._broadcast_snapshot(runtime)
        await self._log(
            runtime,
            "info",
            f"Processing finished with status '{runtime.snapshot.status}'.",
        )

    async def _process_variant(
        self,
        runtime: JobRuntime,
        adapter: BaseGeneratorAdapter,
        video: VideoInfo,
        variant: VariantInfo,
        variant_entry: dict[str, Any],
    ) -> None:
        manifest = variant.manifest
        assert manifest is not None

        for segment in sorted(manifest.segments, key=lambda item: item.segmentIndex):
            request = self._build_segment_request(runtime, video, variant, manifest, segment)
            await self._log(
                runtime,
                "info",
                f"[{video.videoId}/{variant.key}] Segment {segment.segmentIndex} -> {segment.segmentId}",
            )
            try:
                artifact = await self._generate_segment(adapter, request)
                probed = probe_video(
                    artifact.outputPath,
                    fallback_width=request.width,
                    fallback_height=request.height,
                    fallback_fps=request.fps,
                    fallback_duration=request.durationSec,
                )
                self._assert_probe(artifact.outputPath, probed)

                video_rel_path = Path("videos") / str(video.videoId) / variant.key / f"{segment.segmentId}.mp4"
                metadata_path = (runtime.workspace_dir / video_rel_path).with_suffix(".json")
                metadata_doc = self._build_segment_metadata(
                    video=video,
                    variant=variant,
                    segment=segment,
                    request=request,
                    probed=probed,
                    video_rel_path=video_rel_path,
                    artifact_debug=artifact.debug,
                    model_name=artifact.modelName,
                    model_version=artifact.modelVersion,
                )
                self._write_json(metadata_path, metadata_doc)

                variant_entry["segments"].append(
                    {
                        "segmentId": segment.segmentId,
                        "videoFile": video_rel_path.as_posix(),
                        "durationSec": probed["durationSec"],
                        "width": probed["width"],
                        "height": probed["height"],
                        "fps": probed["fps"],
                    }
                )
                runtime.snapshot.completedSegments += 1
                runtime.snapshot.updatedAt = utc_now()
                self._write_snapshot(runtime)
                await self._broadcast_snapshot(runtime)
                await self._log(
                    runtime,
                    "info",
                    f"Segment {segment.segmentId} completed via {artifact.modelName}.",
                )
            except Exception as exc:
                variant_entry["segments"].append(
                    {
                        "segmentId": segment.segmentId,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                runtime.snapshot.failedSegments += 1
                runtime.snapshot.updatedAt = utc_now()
                self._write_snapshot(runtime)
                await self._broadcast_snapshot(runtime)
                await self._log(runtime, "error", f"Segment {segment.segmentId} failed: {exc}")

    async def _generate_segment(
        self,
        adapter: BaseGeneratorAdapter,
        request: SegmentGenerationRequest,
    ) -> GenerationArtifact:
        if isinstance(adapter, DiffusersVideoAdapter):
            return await self._generate_segment_in_subprocess(request)
        return await adapter.generate_segment(request)

    async def _generate_segment_in_subprocess(
        self,
        request: SegmentGenerationRequest,
    ) -> GenerationArtifact:
        request_path = request.outputPath.with_suffix(".request.json")
        artifact_path = request.outputPath.with_suffix(".artifact.json")
        request_path.parent.mkdir(parents=True, exist_ok=True)
        request_path.write_text(
            json.dumps(request.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "backend.app.cli",
            "run-segment",
            "--backend",
            request.backend,
            "--request-file",
            request_path.as_posix(),
            "--artifact-file",
            artifact_path.as_posix(),
            cwd=REPO_ROOT.as_posix(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(
                "\n".join(
                    item
                    for item in [
                        f"Generator subprocess failed with exit code {process.returncode}.",
                        stdout.decode("utf-8", errors="ignore").strip(),
                        stderr.decode("utf-8", errors="ignore").strip(),
                    ]
                    if item
                )
            )
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Missing artifact JSON from generator subprocess: {artifact_path.name}")
        return GenerationArtifact.model_validate_json(artifact_path.read_text(encoding="utf-8"))

    def _build_segment_metadata(
        self,
        *,
        video: VideoInfo,
        variant: VariantInfo,
        segment: ManifestSegment,
        request: SegmentGenerationRequest,
        probed: dict[str, float | int],
        video_rel_path: Path,
        artifact_debug: dict[str, Any],
        model_name: str,
        model_version: str | None,
    ) -> dict[str, Any]:
        return {
            "schemaVersion": "video-pipeline.external-generation.segment.v1",
            "generatedAt": utc_now().isoformat(),
            "videoId": video.videoId,
            "projectId": video.projectId,
            "runId": video.runId,
            "variantKey": variant.key,
            "segmentId": segment.segmentId,
            "segmentIndex": segment.segmentIndex,
            "timeline": segment.timeline.model_dump(mode="json"),
            "prompt": request.prompt,
            "negativePrompt": request.negativePrompt,
            "continuityNote": request.continuityNote,
            "shotGoal": request.shotGoal,
            "spokenText": request.spokenText,
            "subtitleText": request.subtitleText,
            "globalVisualDirection": request.globalVisualDirection,
            "globalNegativePrompt": request.globalNegativePrompt,
            "resolvedPrompt": request.resolvedPrompt,
            "resolvedNegativePrompt": request.resolvedNegativePrompt,
            "model": {"name": model_name, "version": model_version},
            "render": probed,
            "files": {"videoFile": video_rel_path.as_posix()},
            "debug": artifact_debug,
        }

    def _build_segment_request(
        self,
        runtime: JobRuntime,
        video: VideoInfo,
        variant: VariantInfo,
        manifest: VariantManifest,
        segment: ManifestSegment,
    ) -> SegmentGenerationRequest:
        width, height, fps, backend_params = self._resolve_render_settings(video, manifest)
        duration_sec = float(segment.timeline.generationDurationSec or manifest.segmentDurationSec or 8)
        output_path = runtime.workspace_dir / "videos" / str(video.videoId) / variant.key / f"{segment.segmentId}.mp4"
        prompt = segment.generation.prompt or ""
        negative_prompt = segment.generation.negativePrompt or ""
        resolved_prompt = "\n".join(
            item
            for item in [
                manifest.globalVisualDirection.strip(),
                prompt.strip(),
                segment.generation.continuityNote.strip(),
                segment.generation.shotGoal.strip(),
            ]
            if item
        )
        resolved_negative_prompt = "\n".join(
            item
            for item in [manifest.globalNegativePrompt.strip(), negative_prompt.strip()]
            if item
        )
        return SegmentGenerationRequest(
            jobId=runtime.snapshot.jobId,
            backend=runtime.snapshot.backend,
            videoId=video.videoId,
            projectId=video.projectId,
            runId=video.runId,
            videoTitle=video.title,
            variantKey=variant.key,
            variantLabel=variant.label,
            segmentId=segment.segmentId,
            segmentIndex=segment.segmentIndex,
            promptLanguage=manifest.promptLanguage,
            prompt=prompt,
            negativePrompt=negative_prompt,
            continuityNote=segment.generation.continuityNote,
            shotGoal=segment.generation.shotGoal,
            spokenText=segment.narration.spokenText,
            subtitleText=segment.narration.subtitleText,
            wordTimeline=[item.model_dump(mode="json") for item in segment.narration.wordTimeline],
            globalVisualDirection=manifest.globalVisualDirection,
            globalNegativePrompt=manifest.globalNegativePrompt,
            resolvedPrompt=resolved_prompt,
            resolvedNegativePrompt=resolved_negative_prompt,
            width=width,
            height=height,
            fps=fps,
            durationSec=duration_sec,
            outputPath=output_path,
            backendParams=backend_params,
            timeline=segment.timeline.model_dump(mode="json"),
        )

    def _resolve_render_settings(
        self, video: VideoInfo, manifest: VariantManifest
    ) -> tuple[int, int, float, dict[str, Any]]:
        profiles = self._collect_profiles(video.outputProfile, manifest.projectContext.outputProfile)
        width = int(self._pick_profile_value(profiles, ("width", "outputWidth", "videoWidth"), 720))
        height = int(self._pick_profile_value(profiles, ("height", "outputHeight", "videoHeight"), 1280))
        fps = float(self._pick_profile_value(profiles, ("fps", "frameRate"), 24.0))
        backend_params: dict[str, Any] = {}
        for profile in profiles:
            params = profile.get("backendParams")
            if isinstance(params, dict):
                backend_params.update(params)
        return width, height, fps, backend_params

    def _collect_profiles(self, *candidates: dict[str, Any]) -> list[dict[str, Any]]:
        profiles: list[dict[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            profiles.append(candidate)
            render = candidate.get("render")
            if isinstance(render, dict):
                profiles.append(render)
        return profiles

    def _pick_profile_value(
        self, profiles: list[dict[str, Any]], keys: tuple[str, ...], default: int | float
    ) -> int | float:
        for profile in profiles:
            for key in keys:
                value = profile.get(key)
                if value is None:
                    continue
                try:
                    return float(value) if isinstance(default, float) else int(value)
                except (TypeError, ValueError):
                    continue
        return default

    def _assert_probe(self, output_path: Path, probed: dict[str, float | int]) -> None:
        if not output_path.is_file():
            raise FileNotFoundError(output_path.as_posix())
        if probed["durationSec"] <= 0:
            raise ValueError(f"Output duration is invalid for {output_path.name}")
        if probed["width"] <= 0 or probed["height"] <= 0:
            raise ValueError(f"Output size is invalid for {output_path.name}")
        if probed["fps"] <= 0:
            raise ValueError(f"Output FPS is invalid for {output_path.name}")

    def _validate_result(self, runtime: JobRuntime, result_doc: dict[str, Any]) -> list[dict[str, str]]:
        errors: list[dict[str, str]] = []
        for video in result_doc["videos"]:
            for variant in video["variants"]:
                for segment in variant["segments"]:
                    if segment.get("status") == "failed":
                        continue
                    video_file = runtime.workspace_dir / segment["videoFile"]
                    metadata_file = video_file.with_suffix(".json")
                    if not video_file.is_file():
                        errors.append(
                            {
                                "segmentId": segment["segmentId"],
                                "status": "failed",
                                "error": f"Missing generated file: {segment['videoFile']}",
                            }
                        )
                        continue
                    if not metadata_file.is_file():
                        errors.append(
                            {
                                "segmentId": segment["segmentId"],
                                "status": "failed",
                                "error": (
                                    "Missing metadata JSON: "
                                    f"{metadata_file.relative_to(runtime.workspace_dir).as_posix()}"
                                ),
                            }
                        )
                        continue
                    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
                    if metadata.get("segmentId") != segment["segmentId"]:
                        errors.append(
                            {
                                "segmentId": segment["segmentId"],
                                "status": "failed",
                                "error": f"Metadata segmentId mismatch for {segment['segmentId']}",
                            }
                        )
        return errors

    def _create_archive(self, runtime: JobRuntime) -> None:
        runtime.archive_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(runtime.archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_handle:
            include_roots = [
                runtime.result_path,
                runtime.workspace_dir / "input",
                runtime.workspace_dir / "videos",
            ]
            for root in include_roots:
                if root.is_file():
                    zip_handle.write(root, arcname=root.relative_to(runtime.workspace_dir))
                    continue
                if not root.exists():
                    continue
                for path in root.rglob("*"):
                    if path.is_file():
                        zip_handle.write(path, arcname=path.relative_to(runtime.workspace_dir))

    async def _log(self, runtime: JobRuntime, level: str, message: str) -> None:
        entry = LogEntry(ts=utc_now(), level=level, message=message)
        runtime.logs.append(entry)
        runtime.logs_path.parent.mkdir(parents=True, exist_ok=True)
        with runtime.logs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.model_dump(mode="json"), ensure_ascii=False) + "\n")
        await self._broadcast(runtime, {"type": "log", "data": entry.model_dump(mode="json")})

    async def _broadcast_snapshot(self, runtime: JobRuntime) -> None:
        await self._broadcast(runtime, {"type": "snapshot", "data": runtime.snapshot.model_dump(mode="json")})

    async def _broadcast(self, runtime: JobRuntime, event: dict[str, Any]) -> None:
        stale: list[asyncio.Queue[dict[str, Any]]] = []
        for subscriber in runtime.subscribers:
            try:
                subscriber.put_nowait(event)
            except asyncio.QueueFull:
                stale.append(subscriber)
        for subscriber in stale:
            if subscriber in runtime.subscribers:
                runtime.subscribers.remove(subscriber)

    def _write_snapshot(self, runtime: JobRuntime) -> None:
        self._write_json(runtime.snapshot_path, runtime.snapshot.model_dump(mode="json"))

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _build_direct_batch_payload(self, request: DirectGenerationRequest) -> dict[str, Any]:
        exported_at = utc_now().isoformat()
        run_id = request.runId or compact_timestamp()
        segment_id = f"{run_id}_{request.variantKey}_s01"
        return {
            "schemaVersion": "video-pipeline.external-generation.batch.v1",
            "exportedAt": exported_at,
            "filters": {"projectId": request.projectId, "videoId": request.videoId, "status": "manual"},
            "totalVideos": 1,
            "totalVariants": 1,
            "videos": [
                {
                    "videoId": request.videoId,
                    "projectId": request.projectId,
                    "runId": run_id,
                    "title": request.title,
                    "status": "manual",
                    "createdAt": exported_at,
                    "updatedAt": exported_at,
                    "project": {"id": request.projectId, "name": "Manual project", "slug": "manual-project"},
                    "videoTemplate": {"id": 1, "key": "manual", "name": "Manual"},
                    "outputProfile": {
                        "width": request.width,
                        "height": request.height,
                        "fps": request.fps,
                        "backendParams": request.backendParams,
                    },
                    "deliveryProfile": {},
                    "subtitleStyle": {},
                    "requestSnapshot": {},
                    "variants": [
                        {
                            "key": request.variantKey,
                            "label": request.variantLabel,
                            "status": "manual",
                            "externalGenerationManifestPath": None,
                            "externalGenerationManifestUrl": None,
                            "manifestFound": True,
                            "manifest": {
                                "schemaVersion": "video-pipeline.external-generation.manifest.v1",
                                "mode": "deferred_generation",
                                "generatedAt": exported_at,
                                "runId": run_id,
                                "variantKey": request.variantKey,
                                "variantLabel": request.variantLabel,
                                "promptLanguage": request.promptLanguage,
                                "targetDurationSec": request.durationSec,
                                "speechDurationSec": request.durationSec,
                                "segmentDurationSec": request.durationSec,
                                "totalSegments": 1,
                                "projectContext": {
                                    "videoName": request.title,
                                    "mainRequest": request.prompt,
                                    "preferredStyle": "",
                                    "extraRequirements": "",
                                    "globalRules": "",
                                    "outputProfile": {
                                        "width": request.width,
                                        "height": request.height,
                                        "fps": request.fps,
                                        "backendParams": request.backendParams,
                                    },
                                    "deliveryProfile": {},
                                },
                                "globalVisualDirection": request.globalVisualDirection,
                                "globalNegativePrompt": request.globalNegativePrompt,
                                "sourceArtifacts": {},
                                "segments": [
                                    {
                                        "segmentId": segment_id,
                                        "segmentIndex": 1,
                                        "timeline": {
                                            "startSec": 0,
                                            "endSec": request.durationSec,
                                            "effectiveDurationSec": request.durationSec,
                                            "generationDurationSec": request.durationSec,
                                        },
                                        "narration": {
                                            "spokenText": request.spokenText,
                                            "subtitleText": request.subtitleText,
                                            "keywords": [],
                                            "wordTimeline": [],
                                            "subtitleSegments": [],
                                        },
                                        "generation": {
                                            "prompt": request.prompt,
                                            "negativePrompt": request.negativePrompt,
                                            "continuityNote": request.continuityNote,
                                            "shotGoal": request.shotGoal,
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                }
            ],
        }

    def _release_adapters(self, except_backend: str | None = None) -> None:
        for key, adapter in self._adapters.items():
            if except_backend is not None and key == except_backend:
                continue
            try:
                adapter.release()
            except Exception:
                continue
