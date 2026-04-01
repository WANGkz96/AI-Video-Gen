from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from typing import Literal


@dataclass(slots=True)
class DiffusersBackendSpec:
    key: str
    label: str
    description: str
    model_id: str
    pipeline_candidates: tuple[str, ...]
    task: Literal["text-to-video", "text-image-to-video"] = "text-to-video"
    status: Literal["ready", "planned", "experimental"] = "experimental"
    minimum_vram_gb: int = 16
    default_width: int = 832
    default_height: int = 480
    default_fps: float = 16.0
    default_steps: int = 30
    default_guidance_scale: float = 6.0
    frame_multiple: int = 4
    frame_offset: int = 1
    resolution_multiple: int = 16
    recommended_dtype: Literal["auto", "float16", "bfloat16"] = "auto"
    notes: str | None = None
    allow_patterns: tuple[str, ...] = field(default_factory=tuple)
    uses_float32_vae: bool = False
    enable_vae_tiling: bool = True
    enable_vae_slicing: bool = True
    enable_attention_slicing: bool = True
    load_kwargs: dict[str, Any] = field(default_factory=dict)
    download_files: tuple[str, ...] = field(default_factory=tuple)
    static_call_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def storage_dir_name(self) -> str:
        return self.key


def get_diffusers_backend_specs() -> dict[str, DiffusersBackendSpec]:
    return {
        "sana-video-2b": DiffusersBackendSpec(
            key="sana-video-2b",
            label="SANA-Video 2B",
            description="SANA-Video 2B 480p Diffusers pipeline from NVIDIA/MIT HAN Lab.",
            model_id="Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
            pipeline_candidates=("SanaVideoPipeline",),
            task="text-to-video",
            status="experimental",
            minimum_vram_gb=24,
            default_width=832,
            default_height=480,
            default_fps=16.0,
            default_steps=30,
            default_guidance_scale=6.0,
            recommended_dtype="float16",
            resolution_multiple=32,
            notes="Official docs recommend BF16 for transformer weights; Turing fallback uses FP16/FP32.",
            uses_float32_vae=True,
            static_call_kwargs={"use_resolution_binning": False},
        ),
        "wan2.2-ti2v-5b": DiffusersBackendSpec(
            key="wan2.2-ti2v-5b",
            label="Wan2.2-TI2V-5B",
            description="Wan 2.2 text-image-to-video 5B Diffusers checkpoint.",
            model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            pipeline_candidates=("WanPipeline",),
            task="text-image-to-video",
            status="experimental",
            minimum_vram_gb=24,
            default_width=832,
            default_height=480,
            default_fps=16.0,
            default_steps=24,
            default_guidance_scale=5.0,
            recommended_dtype="float16",
            notes="Adapter currently uses text-only mode until image-conditioning is added to the API contract.",
            uses_float32_vae=True,
        ),
        "hunyuan-video-1.5": DiffusersBackendSpec(
            key="hunyuan-video-1.5",
            label="Hunyuan Video 1.5",
            description="HunyuanVideo 1.5 480p text-to-video Diffusers conversion.",
            model_id="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
            pipeline_candidates=(
                "HunyuanVideo15Pipeline",
                "HunyuanVideoPipeline",
            ),
            task="text-to-video",
            status="experimental",
            minimum_vram_gb=24,
            default_width=864,
            default_height=480,
            default_fps=16.0,
            default_steps=20,
            default_guidance_scale=6.0,
            recommended_dtype="float16",
            notes="Official docs use BF16 + Flash Attention; this adapter targets broader compatibility first.",
            load_kwargs={
                "low_cpu_mem_usage": True,
                "offload_state_dict": True,
                "device_map": "balanced",
            },
        ),
        "cogvideox-5b": DiffusersBackendSpec(
            key="cogvideox-5b",
            label="CogVideoX-5B",
            description="CogVideoX 5B text-to-video pipeline from THUDM.",
            model_id="THUDM/CogVideoX-5b",
            pipeline_candidates=("CogVideoXPipeline",),
            task="text-to-video",
            status="experimental",
            minimum_vram_gb=16,
            default_width=720,
            default_height=480,
            default_fps=8.0,
            default_steps=28,
            default_guidance_scale=6.0,
            recommended_dtype="float16",
            notes="Diffusers docs list quantized 5B inference at ~16GB VRAM.",
        ),
        "ltx-2-distilled": DiffusersBackendSpec(
            key="ltx-2-distilled",
            label="LTX 2 Distilled",
            description="LTX 2 distilled checkpoint family from Lightricks.",
            model_id="rootonchair/LTX-2-19b-distilled",
            pipeline_candidates=("LTX2Pipeline", "LTXPipeline"),
            task="text-to-video",
            status="experimental",
            minimum_vram_gb=24,
            default_width=704,
            default_height=480,
            default_fps=16.0,
            default_steps=8,
            default_guidance_scale=1.0,
            recommended_dtype="float16",
            notes="Uses the distilled Diffusers repo referenced by the official LTX-2 Diffusers docs for low-step inference.",
            load_kwargs={
                "low_cpu_mem_usage": True,
                "offload_state_dict": True,
                "device_map": "balanced",
            },
        ),
        "ltx-2.3-distilled": DiffusersBackendSpec(
            key="ltx-2.3-distilled",
            label="LTX 2.3 Distilled",
            description="LTX 2.3 distilled family from Lightricks.",
            model_id="Lightricks/LTX-2.3",
            pipeline_candidates=("LTX2Pipeline", "LTXPipeline"),
            task="text-to-video",
            status="planned",
            minimum_vram_gb=32,
            default_width=704,
            default_height=480,
            default_fps=16.0,
            default_steps=20,
            default_guidance_scale=4.0,
            recommended_dtype="float16",
            notes="Official model card says Diffusers support is coming soon; installer downloads the official distilled checkpoint for later native integration.",
            download_files=("ltx-2.3-22b-distilled.safetensors", "README.md"),
        ),
    }
