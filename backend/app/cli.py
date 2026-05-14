from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from pathlib import Path

from backend.app.adapters.base import AdapterUnavailableError, BaseGeneratorAdapter
from backend.app.adapters.registry import build_real_model_registry, get_downloadable_backend_keys
from backend.app.config import Settings
from backend.app.models import SegmentGenerationRequest
from backend.app.services.jobs import JobService, compact_timestamp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Video Generation Service helper CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-models", help="Show registered real-model backends.")

    download = subparsers.add_parser("download-models", help="Download one or more model repos into models/.")
    download.add_argument("--models", default="all", help="Comma-separated backend keys or 'all'.")

    smoke = subparsers.add_parser("smoke-test", help="Run one direct generation call through a backend.")
    smoke.add_argument("--backend", required=True)
    smoke.add_argument("--prompt", required=True)
    smoke.add_argument("--negative-prompt", default="")
    smoke.add_argument("--duration", type=float, default=3.0)
    smoke.add_argument("--width", type=int, default=832)
    smoke.add_argument("--height", type=int, default=480)
    smoke.add_argument("--fps", type=float, default=16.0)
    smoke.add_argument("--steps", type=int, default=12)

    run_segment = subparsers.add_parser("run-segment", help="Run one serialized segment request.")
    run_segment.add_argument("--backend", required=True)
    run_segment.add_argument("--request-file", required=True)
    run_segment.add_argument("--artifact-file", required=True)

    run_batch = subparsers.add_parser("run-batch", help="Run one batch JSON through the job service.")
    run_batch.add_argument("--batch-file", required=True)
    run_batch.add_argument("--backend", default=None)
    run_batch.add_argument("--timeout-sec", type=float, default=3600)
    run_batch.add_argument("--poll-sec", type=float, default=1)
    run_batch.add_argument("--copy-archive-to", default=None)
    return parser


def parse_model_selection(raw: str) -> list[str]:
    if raw == "all":
        return get_downloadable_backend_keys()
    keys = [item.strip() for item in raw.split(",") if item.strip()]
    known = set(get_downloadable_backend_keys())
    unknown = [key for key in keys if key not in known]
    if unknown:
        raise SystemExit(f"Unknown model keys: {', '.join(unknown)}")
    return keys


def build_real_model_adapter(settings: Settings, backend: str) -> BaseGeneratorAdapter:
    registry = build_real_model_registry(settings)
    try:
        return registry[backend]
    except KeyError as exc:
        raise SystemExit(f"Unknown backend '{backend}'.") from exc


async def run_smoke_test(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    adapter = build_real_model_adapter(settings, args.backend)
    smoke_dir = settings.temp_dir / "smoke-tests"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    output_path = smoke_dir / f"{args.backend}_{compact_timestamp()}.mp4"

    request = SegmentGenerationRequest(
        jobId=f"smoke_{compact_timestamp()}",
        backend=args.backend,
        videoId=1,
        projectId=1,
        runId=compact_timestamp(),
        videoTitle="Smoke test",
        variantKey="v01",
        variantLabel="Variant 1",
        segmentId=f"{compact_timestamp()}_v01_s01",
        segmentIndex=1,
        promptLanguage="en",
        prompt=args.prompt,
        negativePrompt=args.negative_prompt,
        continuityNote="",
        shotGoal="smoke test",
        spokenText="",
        subtitleText="",
        globalVisualDirection="",
        globalNegativePrompt="",
        resolvedPrompt=args.prompt,
        resolvedNegativePrompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        fps=args.fps,
        durationSec=args.duration,
        outputPath=output_path,
        backendParams={"num_inference_steps": args.steps},
    )
    artifact = await adapter.generate_segment(request)
    print(json.dumps(artifact.model_dump(mode="json"), ensure_ascii=False, indent=2))


async def run_serialized_segment(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    adapter = build_real_model_adapter(settings, args.backend)
    request = SegmentGenerationRequest.model_validate_json(
        Path(args.request_file).read_text(encoding="utf-8")
    )
    artifact = await adapter.generate_segment(request)
    Path(args.artifact_file).write_text(
        json.dumps(artifact.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


async def run_batch_job(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    service = JobService(settings)
    batch_path = Path(args.batch_file)
    await service.start()
    try:
        queued = await service.create_job_from_upload(
            filename=batch_path.name,
            content=batch_path.read_bytes(),
            backend=args.backend,
        )
        job_id = queued.jobId
        deadline = asyncio.get_running_loop().time() + max(1, args.timeout_sec)
        terminal_statuses = {"completed", "completed_with_errors", "failed"}

        while True:
            snapshot = service.get_job(job_id)
            if snapshot.status in terminal_statuses:
                break
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f"Timed out waiting for job {job_id}.")
            await asyncio.sleep(max(0.1, args.poll_sec))

        archive_path = None
        try:
            archive_path = service.get_archive_path(job_id)
        except FileNotFoundError:
            archive_path = None

        copied_archive_path = None
        if archive_path is not None and args.copy_archive_to:
            copy_target = Path(args.copy_archive_to)
            if copy_target.is_dir() or str(args.copy_archive_to).endswith(("/", "\\")):
                copy_target.mkdir(parents=True, exist_ok=True)
                copy_target = copy_target / archive_path.name
            else:
                copy_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(archive_path, copy_target)
            copied_archive_path = copy_target

        output = {
            "job": snapshot.model_dump(mode="json"),
            "archivePath": archive_path.as_posix() if archive_path else None,
            "copiedArchivePath": copied_archive_path.as_posix() if copied_archive_path else None,
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    finally:
        await service.stop()


def run_downloads(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    selected = parse_model_selection(args.models)
    for key in selected:
        adapter = build_real_model_adapter(settings, key)
        info = adapter.info()
        print(f"[{key}] {info.modelId} -> {info.localPath}")
        if not hasattr(adapter, "download_assets"):
            raise SystemExit(f"Backend '{key}' does not support asset downloads.")
        try:
            asyncio.run(adapter.download_assets())  # type: ignore[attr-defined]
        except AdapterUnavailableError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"[{key}] ready")


def list_models() -> None:
    settings = Settings.from_env()
    for key, adapter in build_real_model_registry(settings).items():
        info = adapter.info()
        print(
            json.dumps(
                {
                    "key": key,
                    "modelId": info.modelId,
                    "localPath": info.localPath,
                    "available": info.available,
                    "requiresDownload": info.requiresDownload,
                    "notes": info.notes,
                },
                ensure_ascii=False,
            )
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "list-models":
        list_models()
        return
    if args.command == "download-models":
        run_downloads(args)
        return
    if args.command == "smoke-test":
        asyncio.run(run_smoke_test(args))
        return
    if args.command == "run-segment":
        asyncio.run(run_serialized_segment(args))
        return
    if args.command == "run-batch":
        asyncio.run(run_batch_job(args))
        return
    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
