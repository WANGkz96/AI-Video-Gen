from __future__ import annotations

from abc import ABC, abstractmethod

from backend.app.models import AdapterInfo, GenerationArtifact, SegmentGenerationRequest


class AdapterError(RuntimeError):
    pass


class AdapterUnavailableError(AdapterError):
    pass


class BaseGeneratorAdapter(ABC):
    @abstractmethod
    def info(self) -> AdapterInfo:
        raise NotImplementedError

    @abstractmethod
    async def generate_segment(
        self, request: SegmentGenerationRequest
    ) -> GenerationArtifact:
        raise NotImplementedError

