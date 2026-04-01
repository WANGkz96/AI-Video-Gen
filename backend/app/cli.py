from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from backend.app.adapters.catalog import get_diffusers_backend_specs
from backend.app.adapters.diffusers_video import DiffusersVideoAdapter
from backend.app.config import Settings
from backend.app.models import SegmentGenerationRequest
from backend.app.services.jobs import compact_timestamp


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
    return parser


def parse_model_selection(raw: str) -> list[str]:
    specs = get_diffusers_backend_specs()
    if raw == "all":
        return list(specs)
    keys = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [key for key in keys if key not in specs]
    if unknown:
        raise SystemExit(f"Unknown model keys: {', '.join(unknown)}")
    return keys


async def run_smoke_test(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    specs = get_diffusers_backend_specs()
    if args.backend not in specs:
        raise SystemExit(f"Unknown backend '{args.backend}'.")
    adapter = DiffusersVideoAdapter(settings, specs[args.backend])
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
    specs = get_diffusers_backend_specs()
    if args.backend not in specs:
        raise SystemExit(f"Unknown backend '{args.backend}'.")
    adapter = DiffusersVideoAdapter(settings, specs[args.backend])
    request = SegmentGenerationRequest.model_validate_json(
        Path(args.request_file).read_text(encoding="utf-8")
    )
    artifact = await adapter.generate_segment(request)
    Path(args.artifact_file).write_text(
        json.dumps(artifact.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_downloads(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    selected = parse_model_selection(args.models)
    for key in selected:
        adapter = DiffusersVideoAdapter(settings, get_diffusers_backend_specs()[key])
        info = adapter.info()
        print(f"[{key}] {info.modelId} -> {info.localPath}")
        asyncio.run(adapter.download_assets())
        print(f"[{key}] ready")


def list_models() -> None:
    settings = Settings.from_env()
    for key, spec in get_diffusers_backend_specs().items():
        info = DiffusersVideoAdapter(settings, spec).info()
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
    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
