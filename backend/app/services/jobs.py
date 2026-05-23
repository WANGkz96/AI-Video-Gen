from __future__ import annotations

import asyncio
import io
import json
import sys
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.app.adapters.base import AdapterUnavailableError, BaseGeneratorAdapter
from backend.app.adapters.comfyui import ComfyUiWorkflowAdapter
from backend.app.adapters.diffusers_video import DiffusersVideoAdapter
from backend.app.adapters.mock_gen import MockGenAdapter
from backend.app.adapters.planned import PlannedAdapter
from backend.app.adapters.registry import build_real_model_registry
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
from backend.app.services.provisioning import get_provisioning_status


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
    backend_params: dict[str, Any] = field(default_factory=dict)
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
        provisioning = get_provisioning_status(self._settings)
        return {
            "status": "ok" if provisioning["ready"] else provisioning["status"],
            "ready": provisioning["ready"],
            "defaultBackend": self._settings.generator_backend,
            "queuedJobs": self._queue.qsize(),
            "activeJobs": active_jobs,
            "frontendBuilt": self._settings.frontend_dist_dir.exists(),
            "provisioning": provisioning,
        }

    def provisioning(self) -> dict[str, Any]:
        return get_provisioning_status(self._settings)

    def list_backends(self, *, include_unavailable: bool = False) -> list[AdapterInfo]:
        backends = [adapter.info() for adapter in self._adapters.values()]
        if include_unavailable:
            return backends
        return [backend for backend in backends if backend.available and not backend.requiresDownload]

    async def create_job(
        self,
        batch_payload: dict[str, Any],
        backend: str | None = None,
        input_files: dict[str, bytes] | None = None,
        backend_params: dict[str, Any] | None = None,
    ) -> JobQueuedResponse:
        batch = BatchExport.model_validate(batch_payload)
        selected_backend = backend or self._settings.generator_backend
        job_backend_params = dict(backend_params or {})
        self._ensure_backend_can_run(selected_backend)
        generation_multiplier = self._settings.segment_variants
        total_videos = len(batch.videos)
        total_variants = sum(len(video.variants) for video in batch.videos)
        total_segments = sum(
            len(variant.manifest.segments)
            for video in batch.videos
            for variant in video.variants
            if variant.manifest is not None
        ) * generation_multiplier

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
            backend_params=job_backend_params,
        )
        self._jobs[job_id] = runtime
        self._write_json(input_path, batch.model_dump(mode="json"))
        for relative_path, content in (input_files or {}).items():
            target_path = self._resolve_input_archive_path(input_dir, relative_path)
            if target_path == input_path:
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(content)
        self._write_snapshot(runtime)
        await self._log(runtime, "info", f"Job {job_id} queued with backend '{selected_backend}'.")
        await self._queue.put(job_id)
        return JobQueuedResponse(jobId=job_id, status=snapshot.status)

    async def create_job_from_upload(
        self,
        *,
        filename: str,
        content: bytes,
        backend: str | None = None,
        backend_params: dict[str, Any] | None = None,
    ) -> JobQueuedResponse:
        batch_payload, input_files = self._parse_batch_upload(filename=filename, content=content)
        return await self.create_job(
            batch_payload,
            backend=backend,
            input_files=input_files,
            backend_params=backend_params,
        )

    async def create_direct_job(self, request: DirectGenerationRequest) -> JobQueuedResponse:
        payload = self._build_direct_batch_payload(request)
        return await self.create_job(payload, backend=request.backend)

    def get_job(self, job_id: str) -> JobSnapshot:
        runtime = self._jobs.get(job_id)
        if runtime is not None:
            return runtime.snapshot
        return self._read_snapshot(job_id)

    def list_jobs(self, *, limit: int = 20) -> list[JobSnapshot]:
        snapshots: dict[str, JobSnapshot] = {
            job_id: runtime.snapshot for job_id, runtime in self._jobs.items()
        }
        for snapshot_path in self._settings.jobs_dir.glob("*/job.json"):
            try:
                snapshot = JobSnapshot.model_validate_json(snapshot_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            snapshots.setdefault(snapshot.jobId, snapshot)
        return sorted(
            snapshots.values(),
            key=lambda snapshot: snapshot.updatedAt,
            reverse=True,
        )[: max(1, limit)]

    def get_logs(self, job_id: str) -> list[LogEntry]:
        runtime = self._jobs.get(job_id)
        if runtime is not None:
            return list(runtime.logs)
        return self._read_logs(job_id)

    def get_result(self, job_id: str) -> dict[str, Any]:
        result_path = self._result_path(job_id)
        if not result_path.is_file():
            raise FileNotFoundError("Job result.json is not ready yet.")
        return json.loads(result_path.read_text(encoding="utf-8"))

    def get_archive_path(self, job_id: str) -> Path:
        archive_path = self._archive_path(job_id)
        if not archive_path.is_file():
            raise FileNotFoundError("Job archive is not ready yet.")
        return archive_path

    def get_job_file(self, job_id: str, relative_path: str) -> Path:
        workspace = self._workspace_dir(job_id)
        target = (workspace / relative_path).resolve()
        workspace = workspace.resolve()
        if workspace not in target.parents and target != workspace:
            raise PermissionError("Requested path is outside the job workspace.")
        if not target.is_file():
            raise FileNotFoundError(relative_path)
        return target

    async def stream_events(self, job_id: str):
        runtime = self._jobs.get(job_id)
        if runtime is None:
            snapshot = self._read_snapshot(job_id)
            yield {"type": "snapshot", "data": snapshot.model_dump(mode="json")}
            for log in self._read_logs(job_id):
                yield {"type": "log", "data": log.model_dump(mode="json")}
            return

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
        comfy_adapter = ComfyUiWorkflowAdapter(self._settings)
        registry: dict[str, BaseGeneratorAdapter] = {
            comfy_adapter.key: comfy_adapter,
        }
        if self._settings.enable_mock_backend or self._settings.generator_backend == "mock-gen":
            registry["mock-gen"] = MockGenAdapter(self._settings.mock_media_dir)
        if self._settings.enable_legacy_backends:
            registry.update(build_real_model_registry(self._settings))
        registry["ltx-video-2"] = PlannedAdapter(
            "ltx-video-2",
            "LTX Video 2",
            "Use 'comfyui-ltx23' instead.",
        )
        registry["ltx-video-2-distilled"] = PlannedAdapter(
            "ltx-video-2-distilled",
            "LTX Video 2 Distilled",
            "Use 'comfyui-ltx23' instead.",
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

    def _workspace_dir(self, job_id: str) -> Path:
        runtime = self._jobs.get(job_id)
        if runtime is not None:
            return runtime.workspace_dir
        workspace = (self._settings.jobs_dir / job_id).resolve()
        jobs_root = self._settings.jobs_dir.resolve()
        if workspace != jobs_root and jobs_root not in workspace.parents:
            raise KeyError(job_id)
        if not (workspace / "job.json").is_file():
            raise KeyError(job_id)
        return workspace

    def _read_snapshot(self, job_id: str) -> JobSnapshot:
        snapshot_path = self._workspace_dir(job_id) / "job.json"
        return JobSnapshot.model_validate_json(snapshot_path.read_text(encoding="utf-8"))

    def _read_logs(self, job_id: str) -> list[LogEntry]:
        logs_path = self._workspace_dir(job_id) / "logs.jsonl"
        if not logs_path.is_file():
            return []
        entries: list[LogEntry] = []
        for line in logs_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entries.append(LogEntry.model_validate_json(line))
        return entries

    def _result_path(self, job_id: str) -> Path:
        runtime = self._jobs.get(job_id)
        if runtime is not None:
            return runtime.result_path
        workspace = self._workspace_dir(job_id)
        snapshot = self._read_snapshot(job_id)
        relative_path = snapshot.resultFile or "result.json"
        return self._resolve_job_relative_path(workspace, relative_path)

    def _archive_path(self, job_id: str) -> Path:
        runtime = self._jobs.get(job_id)
        if runtime is not None:
            return runtime.archive_path
        snapshot = self._read_snapshot(job_id)
        if snapshot.archiveFile:
            return Path(snapshot.archiveFile)
        return self._settings.archive_dir / f"{job_id}.zip"

    def _resolve_job_relative_path(self, workspace: Path, relative_path: str) -> Path:
        normalized = str(relative_path or "").replace("\\", "/").lstrip("./").strip()
        if (
            not normalized
            or normalized.startswith("/")
            or ":" in normalized.split("/", 1)[0]
            or any(part == ".." for part in normalized.split("/"))
        ):
            raise FileNotFoundError(relative_path)
        target = (workspace / normalized).resolve()
        workspace = workspace.resolve()
        if workspace not in target.parents and target != workspace:
            raise FileNotFoundError(relative_path)
        return target

    def _parse_batch_upload(
        self,
        *,
        filename: str,
        content: bytes,
    ) -> tuple[dict[str, Any], dict[str, bytes]]:
        name = str(filename or "").lower()
        if name.endswith(".zip") or content.startswith(b"PK\x03\x04"):
            with zipfile.ZipFile(io.BytesIO(content)) as zip_handle:
                names = [item for item in zip_handle.namelist() if not item.endswith("/")]
                batch_name = "batch.json" if "batch.json" in names else (
                    "input/batch.json" if "input/batch.json" in names else None
                )
                if batch_name is None:
                    raise ValueError("Input ZIP must contain batch.json.")
                batch_payload = json.loads(zip_handle.read(batch_name).decode("utf-8-sig"))
                input_files: dict[str, bytes] = {}
                for item in names:
                    normalized = self._normalize_archive_relative_path(item)
                    if normalized in {"batch.json", "input/batch.json"}:
                        continue
                    input_files[normalized] = zip_handle.read(item)
            return batch_payload, input_files

        return json.loads(content.decode("utf-8-sig")), {}

    def _normalize_archive_relative_path(self, relative_path: str) -> str:
        normalized = str(relative_path or "").replace("\\", "/").lstrip("./").strip()
        if (
            not normalized
            or normalized.startswith("/")
            or ":" in normalized.split("/", 1)[0]
            or any(part == ".." for part in normalized.split("/"))
        ):
            raise ValueError(f"Unsafe archive path: {relative_path}")
        if normalized.startswith("input/"):
            normalized = normalized[len("input/") :]
        return normalized

    def _resolve_input_archive_path(self, input_dir: Path, relative_path: str) -> Path:
        normalized = self._normalize_archive_relative_path(relative_path)
        target = (input_dir / normalized).resolve()
        root = input_dir.resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"Archive path escapes input directory: {relative_path}")
        return target

    def _resolve_input_file(self, runtime: JobRuntime, relative_path: str | None) -> Path | None:
        if not relative_path:
            return None
        normalized = self._normalize_archive_relative_path(relative_path)
        target = (runtime.workspace_dir / "input" / normalized).resolve()
        root = (runtime.workspace_dir / "input").resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"Input path escapes job input directory: {relative_path}")
        if not target.is_file():
            raise FileNotFoundError(f"Input asset is missing: {relative_path}")
        return target

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
            await self._log(
                runtime,
                "info",
                f"[{video.videoId}/{variant.key}] Segment {segment.segmentIndex} -> {segment.segmentId}",
            )
            segment_entry: dict[str, Any] = {
                "segmentId": segment.segmentId,
                "candidates": [],
            }
            variant_entry["segments"].append(segment_entry)
            try:
                for candidate_index in range(1, self._settings.segment_variants + 1):
                    request = self._build_segment_request(
                        runtime,
                        video,
                        variant,
                        manifest,
                        segment,
                        candidate_index=candidate_index,
                    )
                    artifact = await self._generate_segment(adapter, request)
                    probed = probe_video(
                        artifact.outputPath,
                        fallback_width=request.width,
                        fallback_height=request.height,
                        fallback_fps=request.fps,
                        fallback_duration=request.durationSec,
                    )
                    self._assert_probe(artifact.outputPath, probed)

                    video_rel_path = (
                        Path("videos")
                        / str(video.videoId)
                        / variant.key
                        / segment.segmentId
                        / f"c{candidate_index:02d}.mp4"
                    )
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
                        candidate_index=candidate_index,
                    )
                    self._write_json(metadata_path, metadata_doc)

                    candidate_entry = {
                        "candidateIndex": candidate_index,
                        "videoFile": video_rel_path.as_posix(),
                        "durationSec": probed["durationSec"],
                        "width": probed["width"],
                        "height": probed["height"],
                        "fps": probed["fps"],
                    }
                    segment_entry["candidates"].append(candidate_entry)
                    if candidate_index == 1:
                        segment_entry.update({
                            "videoFile": candidate_entry["videoFile"],
                            "durationSec": candidate_entry["durationSec"],
                            "width": candidate_entry["width"],
                            "height": candidate_entry["height"],
                            "fps": candidate_entry["fps"],
                        })
                    runtime.snapshot.completedSegments += 1
                    runtime.snapshot.updatedAt = utc_now()
                    self._write_snapshot(runtime)
                    await self._broadcast_snapshot(runtime)
                    await self._log(
                        runtime,
                        "info",
                        f"Segment {segment.segmentId} candidate {candidate_index} completed via {artifact.modelName}.",
                    )
            except Exception as exc:
                segment_entry.update({"status": "failed", "error": str(exc)})
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
        candidate_index: int,
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
            "candidateIndex": candidate_index,
            "timeline": segment.timeline.model_dump(mode="json"),
            "prompt": request.prompt,
            "imagePrompt": request.imagePrompt,
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
            "files": {
                "videoFile": video_rel_path.as_posix(),
                "imageFile": request.imageFile,
            },
            "debug": artifact_debug,
        }

    def _build_segment_request(
        self,
        runtime: JobRuntime,
        video: VideoInfo,
        variant: VariantInfo,
        manifest: VariantManifest,
        segment: ManifestSegment,
        candidate_index: int = 1,
    ) -> SegmentGenerationRequest:
        width, height, fps, backend_params = self._resolve_render_settings(video, manifest, runtime.backend_params)
        duration_sec = float(self._settings.video_duration_sec or segment.timeline.generationDurationSec or manifest.segmentDurationSec or 8)
        output_path = (
            runtime.workspace_dir
            / "videos"
            / str(video.videoId)
            / variant.key
            / segment.segmentId
            / f"c{candidate_index:02d}.mp4"
        )
        prompt = segment.generation.prompt or ""
        image_prompt = segment.generation.imagePrompt or prompt
        image = segment.generation.image if isinstance(segment.generation.image, dict) else {}
        image_file = segment.generation.imageFile or image.get("file") or image.get("path") or None
        image_mime_type = segment.generation.imageMimeType or image.get("mimeType") or image.get("mime_type") or None
        image_path = self._resolve_input_file(runtime, image_file) if image_file else None
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
            imagePrompt=image_prompt,
            imagePath=image_path,
            imageFile=image_file,
            imageMimeType=image_mime_type,
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
            backendParams={
                **backend_params,
                "candidateIndex": candidate_index,
                "segmentVariants": self._settings.segment_variants,
                "image": image_path.as_posix() if image_path else None,
                "sourceImage": image_file,
                "imageMimeType": image_mime_type,
            },
            timeline=segment.timeline.model_dump(mode="json"),
        )

    def _resolve_render_settings(
        self,
        video: VideoInfo,
        manifest: VariantManifest,
        job_backend_params: dict[str, Any] | None = None,
    ) -> tuple[int, int, float, dict[str, Any]]:
        profiles = self._collect_profiles(
            video.deliveryProfile,
            video.outputProfile,
            manifest.projectContext.deliveryProfile,
            manifest.projectContext.outputProfile,
        )
        width = int(
            self._pick_profile_value(
                profiles,
                ("width", "outputWidth", "videoWidth", "resolution.width"),
                720,
            )
        )
        height = int(
            self._pick_profile_value(
                profiles,
                ("height", "outputHeight", "videoHeight", "resolution.height"),
                1280,
            )
        )
        if width > height:
            width, height = self._settings.landscape_resolution
        elif height > width:
            width, height = self._settings.portrait_resolution
        backend_params: dict[str, Any] = {}
        for profile in profiles:
            params = profile.get("backendParams")
            if isinstance(params, dict):
                backend_params.update(params)
        if job_backend_params:
            backend_params.update(job_backend_params)
        fps = self._resolve_generation_fps(backend_params)
        return width, height, fps, backend_params

    def _resolve_generation_fps(self, backend_params: dict[str, Any]) -> float:
        for key in ("fps", "frameRate"):
            value = backend_params.get(key)
            if value is None:
                continue
            try:
                return max(1.0, float(value))
            except (TypeError, ValueError):
                continue
        return max(1.0, float(self._settings.video_fps))

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
                value = self._read_profile_value(profile, key)
                if value is None:
                    continue
                try:
                    return float(value) if isinstance(default, float) else int(value)
                except (TypeError, ValueError):
                    continue
        return default

    def _read_profile_value(self, profile: dict[str, Any], key: str) -> Any:
        if "." not in key:
            return profile.get(key)

        cursor: Any = profile
        for part in key.split("."):
            if not isinstance(cursor, dict):
                return None
            cursor = cursor.get(part)
            if cursor is None:
                return None
        return cursor

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
                    candidates = segment.get("candidates")
                    entries = candidates if isinstance(candidates, list) and candidates else [segment]
                    for candidate in entries:
                        video_file = runtime.workspace_dir / candidate["videoFile"]
                        metadata_file = video_file.with_suffix(".json")
                        if not video_file.is_file():
                            errors.append(
                                {
                                    "segmentId": segment["segmentId"],
                                    "status": "failed",
                                    "error": f"Missing generated file: {candidate['videoFile']}",
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
        direct_backend_params = {**request.backendParams, "fps": request.fps}
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
                        "backendParams": direct_backend_params,
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
                                        "backendParams": direct_backend_params,
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
