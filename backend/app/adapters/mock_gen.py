from __future__ import annotations

import asyncio
import itertools
import shutil
from pathlib import Path

from backend.app.adapters.base import AdapterUnavailableError, BaseGeneratorAdapter
from backend.app.models import AdapterInfo, GenerationArtifact, SegmentGenerationRequest


class MockGenAdapter(BaseGeneratorAdapter):
    def __init__(self, mock_media_dir: Path) -> None:
        self._mock_media_dir = mock_media_dir
        self._files = sorted(
            [path for path in mock_media_dir.glob("*.mp4") if path.is_file()],
            key=lambda item: item.name.lower(),
        )
        self._counter = itertools.count()
        self._lock = asyncio.Lock()

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            key="mock-gen",
            label="Mock Generator",
            description="Round-robin copier over local mock-media assets for pipeline testing.",
            status="ready",
            available=bool(self._files),
            notes=(
                None
                if self._files
                else f"No .mp4 files were found in {self._mock_media_dir.as_posix()}."
            ),
        )

    async def generate_segment(
        self, request: SegmentGenerationRequest
    ) -> GenerationArtifact:
        if not self._files:
            raise AdapterUnavailableError(
                f"Mock media directory is empty: {self._mock_media_dir.as_posix()}"
            )

        async with self._lock:
            source = self._files[next(self._counter) % len(self._files)]

        request.outputPath.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copy2, source, request.outputPath)

        return GenerationArtifact(
            modelName="mock-gen",
            modelVersion="simulated",
            outputPath=request.outputPath,
            debug={
                "sourceFile": source.name,
                "sourcePath": source.as_posix(),
            },
        )

