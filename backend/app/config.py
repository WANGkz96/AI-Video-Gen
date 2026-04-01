from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_origins(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(slots=True)
class Settings:
    port: int
    workdir: Path
    jobs_dir: Path
    archive_dir: Path
    temp_dir: Path
    models_dir: Path
    generator_backend: str
    generator_api_url: str
    max_parallel_segments: int
    mock_media_dir: Path
    hf_token: str | None
    cors_origins: list[str]
    frontend_dist_dir: Path

    @classmethod
    def from_env(cls) -> "Settings":
        workdir = Path(os.getenv("WORKDIR", REPO_ROOT / "data")).resolve()
        jobs_dir = Path(os.getenv("JOBS_DIR", workdir / "jobs")).resolve()
        archive_dir = Path(os.getenv("ARCHIVE_DIR", workdir / "archives")).resolve()
        temp_dir = Path(os.getenv("TEMP_DIR", workdir / "tmp")).resolve()
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
            models_dir=models_dir,
            generator_backend=os.getenv("GENERATOR_BACKEND", "mock-gen"),
            generator_api_url=os.getenv("GENERATOR_API_URL", "http://localhost:8188"),
            max_parallel_segments=max(1, int(os.getenv("MAX_PARALLEL_SEGMENTS", "1"))),
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
        self.models_dir.mkdir(parents=True, exist_ok=True)
