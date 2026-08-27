#!/usr/bin/env python3
"""Prepare a one-scene, reproducible LongCat Avatar performance benchmark.

The script deliberately does not invoke the service or alter its production
settings.  It extracts a real dialogue scene from an exported batch and emits
the exact input/manifest format accepted by ``run_longcat_avatar_batch.py``.
This makes INT8/BF16 and runtime experiments compare the same prompt, image,
audio tracks, seed and 93-frame generation window.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROMPT_SUFFIX = " ".join(
    (
        "Preserve the exact two people, location and framing from the conditioning image.",
        "Natural multi-person lip sync: each mouth moves only for its own audio track.",
        "Use restrained physically coherent conversational gestures and reactions.",
        "Keep both faces visible. No cuts, subtitles, visible text, speech bubbles, zoom, push-in, dolly, reframing, duplicated people, or additional characters.",
    )
)


def _dialogue_scenes(batch: dict[str, Any]) -> list[dict[str, Any]]:
    variants = batch.get("variants")
    if not isinstance(variants, list):
        raise ValueError("Batch must contain variants[].")
    for variant in variants:
        manifest = variant.get("manifest") if isinstance(variant, dict) else None
        scenes = manifest.get("dialogueScenes") if isinstance(manifest, dict) else None
        if isinstance(scenes, list):
            return [scene for scene in scenes if isinstance(scene, dict)]
    raise ValueError("Batch contains no variants[].manifest.dialogueScenes[].")


def _required_file(root: Path, relative_path: str, label: str) -> Path:
    path = (root / relative_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path


def prepare_benchmark(
    *,
    batch_path: Path,
    scene_id: str,
    output_dir: Path,
    num_segments: int,
) -> dict[str, str]:
    """Create native LongCat input files and return their absolute paths."""
    batch_path = batch_path.resolve()
    root = batch_path.parent
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    scene = next((item for item in _dialogue_scenes(batch) if str(item.get("sceneId")) == scene_id), None)
    if scene is None:
        raise ValueError(f"Dialogue scene {scene_id!r} is not in {batch_path}")

    generation = scene.get("generation") or {}
    audio = scene.get("audio") or {}
    output = scene.get("output") or {}
    image_file = str(generation.get("imageFile") or generation.get("image", {}).get("file") or "")
    speaker1 = str(audio.get("speaker1File") or "")
    speaker2 = str(audio.get("speaker2File") or "")
    image_path = _required_file(root, image_file, "conditioning image")
    speaker1_path = _required_file(root, speaker1, "speaker1 audio")
    speaker2_path = _required_file(root, speaker2, "speaker2 audio")
    width, height = int(output.get("width") or 0), int(output.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError("Scene output width and height are required.")
    prompt = str(generation.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Scene generation prompt is required.")

    output_dir = output_dir.resolve()
    generated_dir = output_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "avatar_input.json"
    input_path.write_text(
        json.dumps(
            {
                "prompt": f"{prompt} {PROMPT_SUFFIX}",
                "prompt_schedule": [f"{prompt} {PROMPT_SUFFIX}"] * num_segments,
                "cond_image": image_path.as_posix(),
                "cond_audio": {
                    "person1": speaker1_path.as_posix(),
                    "person2": speaker2_path.as_posix(),
                },
                "audio_type": "para",
                "target_orientation": "portrait" if height > width else "landscape",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest_path = output_dir / "batch-input.json"
    manifest_path.write_text(
        json.dumps(
            {
                "scenes": [
                    {
                        "sceneId": scene_id,
                        "inputJson": input_path.as_posix(),
                        "outputDir": generated_dir.as_posix(),
                        "numSegments": num_segments,
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"input": input_path.as_posix(), "manifest": manifest_path.as_posix(), "outputDir": generated_dir.as_posix()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True, type=Path)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-segments", default=1, type=int)
    args = parser.parse_args()
    if args.num_segments < 1:
        raise ValueError("--num-segments must be positive")
    print(json.dumps(prepare_benchmark(
        batch_path=args.batch,
        scene_id=args.scene_id,
        output_dir=args.output_dir,
        num_segments=args.num_segments,
    ), ensure_ascii=False))


if __name__ == "__main__":
    main()
