from __future__ import annotations

from backend.app.adapters.base import AdapterUnavailableError, BaseGeneratorAdapter
from backend.app.models import AdapterInfo, GenerationArtifact, SegmentGenerationRequest


class PlannedAdapter(BaseGeneratorAdapter):
    def __init__(self, key: str, label: str, description: str) -> None:
        self._key = key
        self._label = label
        self._description = description

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            key=self._key,
            label=self._label,
            description=self._description,
            status="planned",
            available=False,
            notes="Reserved slot in the adapter registry. Implementation is not connected yet.",
        )

    async def generate_segment(
        self, request: SegmentGenerationRequest
    ) -> GenerationArtifact:
        raise AdapterUnavailableError(f"Backend '{self._key}' is planned but not implemented yet.")

