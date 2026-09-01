#!/usr/bin/env python3
"""Render a LongCat Avatar job without reloading its weights for every scene.

The upstream demo accepts one input JSON and constructs the complete model
stack inside ``generate``.  Dialogue scenes are independent, but the model
stack is not: keeping it alive makes a multi-scene batch substantially faster
without changing the 480p INT8/distilled inference settings.

This runner intentionally supports one torchrun process.  The service uses one
GPU for Avatar jobs and the upstream context-parallel code is therefore not
needed here.  A future multi-GPU implementation should use the upstream demo
directly until its context-parallel lifecycle can be batched safely.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import PIL.Image
import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
from audio_separator.separator import Separator
from diffusers.utils import load_image
from transformers import AutoTokenizer, UMT5EncoderModel

from longcat_video.audio_process import get_audio_encoder, get_audio_feature_extractor
from longcat_video.audio_process.torch_utils import save_video_ffmpeg
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
from longcat_video.modules.quantization import load_quantized_dit
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline
from longcat_video.context_parallel import context_parallel_util

# Keep upstream audio preparation semantics byte-for-byte compatible with the
# original demo while changing only the model lifecycle.
from run_demo_avatar_multi_audio_to_video import (  # noqa: E402
    audio_prepare_multi,
    extract_vocal_from_speech,
    generate_random_uid,
)


NEGATIVE_PROMPT = (
    "Close-up, bright tones, overexposed, static, blurred details, subtitles, style, "
    "works, paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly "
    "drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, "
    "messy background, three legs, many people in the background, walking backwards"
)


@dataclass
class Runtime:
    checkpoint_dir: Path
    pipe: LongCatVideoAvatarPipeline
    vocal_separator: Separator
    audio_temp_dir: Path
    local_rank: int
    save_fps: int
    audio_stride: int
    num_frames: int
    num_cond_frames: int
    num_inference_steps: int
    text_guidance_scale: float
    audio_guidance_scale: float
    resolution: str
    use_distill: bool


def _resolution_size(resolution: str, target_orientation: str) -> tuple[int, int]:
    sizes = {
        "480p": (480, 832),
        "600p": (640, 960),
        "720p": (768, 1280),
    }
    height, width = sizes[resolution]
    if target_orientation == "portrait" and width > height:
        height, width = width, height
    elif target_orientation == "landscape" and height > width:
        height, width = width, height
    return height, width


def _release_scene_memory() -> None:
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def _load_runtime(args: argparse.Namespace) -> Runtime:
    if torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0):
        # Matches the tested Blackwell workaround in patch_longcat_runtime.py.
        torch.backends.cudnn.enabled = False

    rank = int(os.environ.get("RANK", "0"))
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "LongCat batch runner currently requires exactly one visible GPU; "
            "the remote Avatar adapter launches torchrun with --nproc_per_node=1."
        )
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=24))
    context_parallel_util.init_context_parallel(
        context_parallel_size=1,
        global_rank=dist.get_rank(),
        world_size=dist.get_world_size(),
    )
    cp_split_hw = context_parallel_util.get_optimal_split(1)

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    runtime_dir = checkpoint_dir.parent / "LongCat-Video"
    tokenizer = AutoTokenizer.from_pretrained(runtime_dir, subfolder="tokenizer", torch_dtype=torch.bfloat16)
    text_encoder = UMT5EncoderModel.from_pretrained(runtime_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(runtime_dir, subfolder="vae", torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        checkpoint_dir,
        subfolder="scheduler",
        torch_dtype=torch.bfloat16,
    )
    if args.use_int8:
        print("[INFO] Loading INT8 quantized DiT model once for the entire dialogue batch.", flush=True)
        dit = load_quantized_dit(checkpoint_dir, subfolder="base_model_int8", cp_split_hw=cp_split_hw)
    else:
        dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(
            checkpoint_dir,
            subfolder="base_model",
            cp_split_hw=cp_split_hw,
            torch_dtype=torch.bfloat16,
        )
    if args.use_distill:
        lora_path = checkpoint_dir / "lora" / "dmd_lora.safetensors"
        if lora_path.is_file():
            dit.load_lora(lora_path, "dmd", multiplier=1.0, lora_network_dim=128, lora_network_alpha=64)
            dit.enable_loras(["dmd"])

    audio_checkpoint = checkpoint_dir / "whisper-large-v3"
    audio_encoder = get_audio_encoder(audio_checkpoint, "avatar-v1.5").to(rank)
    audio_feature_extractor = get_audio_feature_extractor(audio_checkpoint, "avatar-v1.5")
    vocal_separator_path = checkpoint_dir / "vocal_separator" / "Kim_Vocal_2.onnx"
    audio_temp_dir = Path("./audio_temp_file").resolve()
    audio_temp_dir.mkdir(parents=True, exist_ok=True)
    vocal_separator = Separator(
        output_dir=audio_temp_dir / "vocals",
        output_single_stem="vocals",
        model_file_dir=vocal_separator_path.parent,
    )
    vocal_separator.load_model(vocal_separator_path.name)

    pipe = LongCatVideoAvatarPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
        audio_encoder=audio_encoder,
        audio_feature_extractor=audio_feature_extractor,
        model_type="avatar-v1.5",
    )
    pipe.to(rank)
    print("[INFO] LongCat models loaded; rendering scenes without further weight reloads.", flush=True)
    return Runtime(
        checkpoint_dir=checkpoint_dir,
        pipe=pipe,
        vocal_separator=vocal_separator,
        audio_temp_dir=audio_temp_dir,
        local_rank=rank,
        save_fps=25,
        audio_stride=1,
        num_frames=93,
        num_cond_frames=13,
        num_inference_steps=8 if args.use_distill else args.num_inference_steps,
        text_guidance_scale=1.0 if args.use_distill else args.text_guidance_scale,
        audio_guidance_scale=1.0 if args.use_distill else args.audio_guidance_scale,
        resolution=args.resolution,
        use_distill=args.use_distill,
    )


def _scene_prompt(input_data: dict[str, Any]) -> tuple[str, list[str] | None]:
    prompt = str(input_data["prompt"])
    schedule = input_data.get("prompt_schedule")
    if schedule is not None:
        if not isinstance(schedule, list) or not schedule:
            raise ValueError("prompt_schedule must be a non-empty list when provided")
        prompt = str(schedule[0])
        schedule = [str(item) for item in schedule]
    return prompt, schedule


def _make_audio_embeddings(
    runtime: Runtime,
    input_data: dict[str, Any],
    num_segments: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, str]:
    audio = input_data["cond_audio"]
    left_raw = audio.get("person1")
    right_raw = audio.get("person2")
    if not left_raw and not right_raw:
        raise ValueError("At least one dialogue speaker track is required.")
    generate_duration = runtime.num_frames / runtime.save_fps + (
        (num_segments - 1) * (runtime.num_frames - runtime.num_cond_frames) / runtime.save_fps
    )
    left_temp = runtime.audio_temp_dir / f"{generate_random_uid()}_left_temp_vocal.wav"
    right_temp = runtime.audio_temp_dir / f"{generate_random_uid()}_right_temp_vocal.wav"
    merge_path = f"/tmp/temp_speech_{generate_random_uid()}_{runtime.local_rank}_merge.wav"
    try:
        left_vocal = extract_vocal_from_speech(left_raw, left_temp, runtime.vocal_separator, runtime.audio_temp_dir)
        right_vocal = extract_vocal_from_speech(right_raw, right_temp, runtime.vocal_separator, runtime.audio_temp_dir)
        left_speech, right_speech, merged = audio_prepare_multi(
            left_vocal,
            right_vocal,
            generate_duration,
            left_raw,
            right_raw,
            sample_rate=16_000,
            audio_type=str(input_data.get("audio_type", "para")),
        )
        sf.write(merge_path, merged, 16_000)
        left_embedding = runtime.pipe.get_audio_embedding(
            left_speech,
            fps=runtime.save_fps * runtime.audio_stride,
            device=runtime.local_rank,
            sample_rate=16_000,
            model_type="avatar-v1.5",
        )
        right_embedding = runtime.pipe.get_audio_embedding(
            right_speech,
            fps=runtime.save_fps * runtime.audio_stride,
            device=runtime.local_rank,
            sample_rate=16_000,
            model_type="avatar-v1.5",
        )
        if torch.isnan(left_embedding).any() or torch.isnan(right_embedding).any():
            raise ValueError("LongCat produced a NaN audio embedding")
        return left_embedding, right_embedding, None, merge_path
    finally:
        for path in (left_temp, right_temp):
            if path.exists():
                path.unlink()


def _window_audio_embeddings(
    runtime: Runtime,
    left_full: torch.Tensor,
    right_full: torch.Tensor,
    start_idx: int,
) -> torch.Tensor:
    indices = torch.arange(5) - 2
    end_idx = start_idx + runtime.audio_stride * runtime.num_frames
    center = torch.arange(start_idx, end_idx, runtime.audio_stride).unsqueeze(1) + indices.unsqueeze(0)
    center = torch.clamp(center, min=0, max=left_full.shape[0] - 1)
    left = left_full[center][None, ...].to(runtime.local_rank)
    right = right_full[center][None, ...].to(runtime.local_rank)
    return torch.cat([left, right])


def _target_masks(image: PIL.Image.Image, input_data: dict[str, Any], device: int) -> torch.Tensor:
    src_width, src_height = image.size
    bbox = input_data.get("bbox", {})
    person1_bbox = bbox.get("person1")
    person2_bbox = bbox.get("person2")
    if person1_bbox is None and person2_bbox is None:
        avatar_layout = input_data.get("avatar_layout", {})
        person1_side = str(avatar_layout.get("person1", "left")).strip().lower()
        person2_side = str(avatar_layout.get("person2", "right")).strip().lower()
        if {person1_side, person2_side} != {"left", "right"}:
            raise ValueError("avatar_layout must put person1 and person2 on opposite left/right sides")
        face_scale = 0.1
        left_y_min, left_y_max = int(src_height * face_scale), int(src_height * (1 - face_scale))
        right_y_min, right_y_max = left_y_min, left_y_max
        half_width = src_width // 2
        left_x_min, left_x_max = int(half_width * face_scale), int(half_width * (1 - face_scale))
        right_x_min, right_x_max = int(half_width * face_scale + half_width), int(half_width * (1 - face_scale) + half_width)
        left_bbox = [left_y_min, left_x_min, left_y_max, left_x_max]
        right_bbox = [right_y_min, right_x_min, right_y_max, right_x_max]
        person1_bbox, person2_bbox = (left_bbox, right_bbox) if person1_side == "left" else (right_bbox, left_bbox)
    elif person1_bbox is not None and person2_bbox is not None:
        pass
    else:
        raise NotImplementedError("LongCat requires masks for both speakers or neither speaker")
    human_one = torch.zeros([src_height, src_width])
    human_two = torch.zeros([src_height, src_width])
    person1_y_min, person1_x_min, person1_y_max, person1_x_max = person1_bbox
    person2_y_min, person2_x_min, person2_y_max, person2_x_max = person2_bbox
    human_one[person1_y_min:person1_y_max, person1_x_min:person1_x_max] = 1
    human_two[person2_y_min:person2_y_max, person2_x_min:person2_x_max] = 1
    background = torch.where(human_one + human_two > 0, torch.tensor(0), torch.tensor(1))
    return torch.stack([human_one, human_two, background], dim=0).to(device)


def _render_scene(runtime: Runtime, item: dict[str, Any]) -> str:
    input_path = Path(item["inputJson"]).resolve()
    output_dir = Path(item["outputDir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    num_segments = max(1, int(item["numSegments"]))
    input_data = json.loads(input_path.read_text(encoding="utf-8"))
    prompt, prompt_schedule = _scene_prompt(input_data)
    height, width = _resolution_size(runtime.resolution, str(input_data.get("target_orientation", "")).lower())
    image = load_image(input_data["cond_image"])
    masks = _target_masks(image, input_data, runtime.local_rank)
    left_full, right_full, _background, merge_path = _make_audio_embeddings(runtime, input_data, num_segments)
    generator = torch.Generator(device=runtime.local_rank).manual_seed(42)
    try:
        audio_start = 0
        audio_emb = _window_audio_embeddings(runtime, left_full, right_full, audio_start)
        output, latent = runtime.pipe.generate_ai2v(
            image=image,
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            resolution=runtime.resolution,
            num_frames=runtime.num_frames,
            num_inference_steps=runtime.num_inference_steps,
            text_guidance_scale=runtime.text_guidance_scale,
            audio_guidance_scale=runtime.audio_guidance_scale,
            output_type="both",
            generator=generator,
            audio_emb=audio_emb,
            ref_target_masks=masks,
            use_distill=runtime.use_distill,
        )
        output = output[0]
        video = [PIL.Image.fromarray((output[index] * 255).astype(np.uint8)) for index in range(output.shape[0])]
        del output
        all_frames = video
        output_tensor = torch.from_numpy(np.array(all_frames))
        save_video_ffmpeg(output_tensor, (output_dir / "ai2v_demo_1").as_posix(), merge_path, fps=runtime.save_fps, quality=5)
        del output_tensor

        current_video = video
        reference_latent = latent[:, :, :1].clone()
        frame_width, frame_height = video[0].size
        for segment_index in range(1, num_segments):
            print(f"[SCENE {item.get('sceneId', input_path.stem)}] segment {segment_index + 1}/{num_segments}", flush=True)
            audio_start += runtime.audio_stride * (runtime.num_frames - runtime.num_cond_frames)
            audio_emb = _window_audio_embeddings(runtime, left_full, right_full, audio_start)
            segment_prompt = prompt if prompt_schedule is None else prompt_schedule[min(segment_index, len(prompt_schedule) - 1)]
            output, latent = runtime.pipe.generate_avc(
                video=current_video,
                video_latent=latent,
                prompt=segment_prompt,
                negative_prompt=NEGATIVE_PROMPT,
                height=frame_height,
                width=frame_width,
                num_frames=runtime.num_frames,
                num_cond_frames=runtime.num_cond_frames,
                num_inference_steps=runtime.num_inference_steps,
                text_guidance_scale=runtime.text_guidance_scale,
                audio_guidance_scale=runtime.audio_guidance_scale,
                generator=generator,
                output_type="both",
                use_kv_cache=True,
                offload_kv_cache=False,
                enhance_hf=not runtime.use_distill,
                audio_emb=audio_emb,
                ref_latent=reference_latent,
                ref_img_index=30,
                mask_frame_range=5,
                ref_target_masks=masks,
                use_distill=runtime.use_distill,
            )
            output = output[0]
            next_video = [PIL.Image.fromarray((output[index] * 255).astype(np.uint8)) for index in range(output.shape[0])]
            del output
            all_frames.extend(next_video[runtime.num_cond_frames:])
            current_video = next_video
            output_tensor = torch.from_numpy(np.array(all_frames))
            save_video_ffmpeg(
                output_tensor,
                (output_dir / f"video_continue_{segment_index + 1}").as_posix(),
                merge_path,
                fps=runtime.save_fps,
                quality=5,
            )
            del output_tensor
        final_name = f"video_continue_{num_segments}.mp4" if num_segments > 1 else "ai2v_demo_1.mp4"
        final_path = output_dir / final_name
        if not final_path.is_file():
            raise FileNotFoundError(f"LongCat did not write {final_path}")
        return final_path.as_posix()
    finally:
        if os.path.exists(merge_path):
            os.remove(merge_path)
        _release_scene_memory()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_manifest", required=True)
    parser.add_argument("--result_manifest", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--resolution", choices=["480p", "600p", "720p"], default="480p")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--text_guidance_scale", type=float, default=4.0)
    parser.add_argument("--audio_guidance_scale", type=float, default=4.0)
    parser.add_argument("--use_distill", action="store_true")
    parser.add_argument("--use_int8", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    batch_path = Path(args.batch_manifest).resolve()
    result_path = Path(args.result_manifest).resolve()
    scenes = json.loads(batch_path.read_text(encoding="utf-8")).get("scenes", [])
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("batch_manifest must contain a non-empty scenes array")
    results: list[dict[str, Any]] = []
    runtime: Runtime | None = None
    try:
        runtime = _load_runtime(args)
        for index, scene in enumerate(scenes, start=1):
            scene_id = str(scene.get("sceneId") or index)
            print(f"[BATCH] Starting scene {index}/{len(scenes)}: {scene_id}", flush=True)
            try:
                output_path = _render_scene(runtime, scene)
                results.append({"sceneId": scene_id, "status": "completed", "outputPath": output_path})
                print(f"[BATCH] Completed scene {index}/{len(scenes)}: {scene_id}", flush=True)
            except Exception as exc:  # preserve healthy scenes in a partially bad batch
                results.append({"sceneId": scene_id, "status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
                print(f"[BATCH] Failed scene {index}/{len(scenes)}: {scene_id}: {exc}", flush=True)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps({"scenes": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        if any(item["status"] == "failed" for item in results):
            raise RuntimeError("One or more LongCat dialogue scenes failed; see result_manifest.")
    finally:
        _release_scene_memory()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
