from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def _parse_origins(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_resolution(value: str, fallback: tuple[int, int]) -> tuple[int, int]:
    text = str(value or "").strip().lower().replace("×", "x")
    if "x" not in text:
        return fallback
    width_raw, height_raw = text.split("x", 1)
    try:
        width = int(width_raw.strip())
        height = int(height_raw.strip())
    except ValueError:
        return fallback
    if width <= 0 or height <= 0:
        return fallback
    return width, height


def _parse_output_upscale(value: str | None) -> float | None:
    text = str(value or "").strip().lower()
    if text in {"", "0", "off", "none", "false", "disabled"}:
        return None
    if text.endswith("x"):
        text = text[:-1].strip()
    try:
        scale = float(text)
    except ValueError:
        return None
    if scale in {1.5, 2.0}:
        return scale
    return None


def _parse_ltx_offload(value: str | None) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"", "0", "none", "off", "false", "disabled"}:
        return None
    if text in {"cpu", "disk"}:
        return text
    return None


def _parse_bool(value: str | None, fallback: bool = False) -> bool:
    text = str(value if value is not None else "").strip().lower()
    if not text:
        return fallback
    if text in {"1", "true", "yes", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disabled", "none"}:
        return False
    return fallback


@dataclass(slots=True)
class Settings:
    port: int
    workdir: Path
    jobs_dir: Path
    archive_dir: Path
    temp_dir: Path
    ltx_runtime_repo_dir: Path
    models_dir: Path
    generator_backend: str
    generator_api_url: str
    comfyui_t2v_workflow: Path
    comfyui_i2v_workflow: Path
    comfyui_output_prefix: str
    comfyui_strip_audio: bool
    comfyui_normalize_output: bool
    enable_legacy_backends: bool
    enable_mock_backend: bool
    max_parallel_segments: int
    segment_variants: int
    video_duration_sec: float
    portrait_resolution: tuple[int, int]
    landscape_resolution: tuple[int, int]
    output_upscale: float | None
    ltx_offload: str | None
    mock_media_dir: Path
    hf_token: str | None
    cors_origins: list[str]
    frontend_dist_dir: Path

    @classmethod
    def from_env(cls) -> "Settings":
        _load_env_file(REPO_ROOT / ".env")

        workdir = Path(os.getenv("WORKDIR", REPO_ROOT / "data")).resolve()
        jobs_dir = Path(os.getenv("JOBS_DIR", workdir / "jobs")).resolve()
        archive_dir = Path(os.getenv("ARCHIVE_DIR", workdir / "archives")).resolve()
        temp_dir = Path(os.getenv("TEMP_DIR", workdir / "tmp")).resolve()
        ltx_runtime_repo_dir = Path(
            os.getenv("LTX_RUNTIME_REPO_DIR", workdir / "runtime" / "ltx-2-official")
        ).resolve()
        models_dir = Path(os.getenv("MODELS_DIR", REPO_ROOT / "models")).resolve()
        frontend_dist_dir = Path(
            os.getenv("FRONTEND_DIST_DIR", REPO_ROOT / "frontend" / "dist")
        ).resolve()

        return cls(
            port=int(os.getenv("PORT", "3001")),
            workdir=workdir,
            jobs_dir=jobs_dir,
            archive_dir=archive_dir,
            temp_dir=temp_dir,
            ltx_runtime_repo_dir=ltx_runtime_repo_dir,
            models_dir=models_dir,
            generator_backend=os.getenv("GENERATOR_BACKEND", "comfyui-ltx23"),
            generator_api_url=os.getenv("GENERATOR_API_URL", "http://127.0.0.1:18188"),
            comfyui_t2v_workflow=Path(
                os.getenv(
                    "COMFYUI_T2V_WORKFLOW",
                    "/workspace/ComfyUI/blueprints/Text to Video (LTX-2.3).json",
                )
            ).resolve(),
            comfyui_i2v_workflow=Path(
                os.getenv(
                    "COMFYUI_I2V_WORKFLOW",
                    "/workspace/ComfyUI/blueprints/Image to Video (LTX-2.3).json",
                )
            ).resolve(),
            comfyui_output_prefix=os.getenv("COMFYUI_OUTPUT_PREFIX", "video/AI_Video_Gen"),
            comfyui_strip_audio=_parse_bool(os.getenv("COMFYUI_STRIP_AUDIO"), False),
            comfyui_normalize_output=_parse_bool(os.getenv("COMFYUI_NORMALIZE_OUTPUT"), False),
            enable_legacy_backends=_parse_bool(os.getenv("ENABLE_LEGACY_BACKENDS"), False),
            enable_mock_backend=_parse_bool(os.getenv("ENABLE_MOCK_BACKEND"), False),
            max_parallel_segments=max(1, int(os.getenv("MAX_PARALLEL_SEGMENTS", "1"))),
            segment_variants=max(1, int(os.getenv("SEGMENT_VARIANTS", "4"))),
            video_duration_sec=max(1.0, float(os.getenv("VIDEO_DURATION_SEC", "8"))),
            portrait_resolution=_parse_resolution(os.getenv("PORTRAIT_RESOLUTION", "720x1280"), (720, 1280)),
            landscape_resolution=_parse_resolution(os.getenv("LANDSCAPE_RESOLUTION", "1280x720"), (1280, 720)),
            output_upscale=_parse_output_upscale(
                os.getenv("OUTPUT_UPSCALE") or os.getenv("LTX_OUTPUT_UPSCALE")
            ),
            ltx_offload=_parse_ltx_offload(os.getenv("LTX_OFFLOAD")),
            mock_media_dir=Path(os.getenv("MOCK_MEDIA_DIR", REPO_ROOT / "mock-media")).resolve(),
            hf_token=os.getenv("HF_TOKEN") or None,
            cors_origins=_parse_origins(
                os.getenv(
                    "CORS_ORIGINS",
                    "http://localhost:5173,http://127.0.0.1:5173",
                )
            ),
            frontend_dist_dir=frontend_dist_dir,
        )

    def ensure_dirs(self) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.ltx_runtime_repo_dir.parent.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
