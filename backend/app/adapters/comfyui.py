from __future__ import annotations

from backend.app.adapters.base import AdapterUnavailableError, BaseGeneratorAdapter
from backend.app.models import AdapterInfo, GenerationArtifact, SegmentGenerationRequest


class ComfyUiWorkflowAdapter(BaseGeneratorAdapter):
    def __init__(self, api_url: str) -> None:
        self._api_url = api_url

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            key="comfyui-workflow",
            label="ComfyUI Workflow",
            description="Foundation adapter for external ComfyUI workflow execution over HTTP.",
            status="experimental",
            available=False,
            requiresRemote=True,
            notes=(
                "Workflow transport and payload scaffold are ready, but workflow mapping and "
                "artifact download are not wired yet."
            ),
        )

    def build_workflow_payload(self, request: SegmentGenerationRequest) -> dict:
        return {
            "meta": {
                "jobId": request.jobId,
                "segmentId": request.segmentId,
                "backend": request.backend,
            },
            "inputs": {
                "prompt": request.resolvedPrompt,
                "negative_prompt": request.resolvedNegativePrompt,
                "duration_sec": request.durationSec,
                "width": request.width,
                "height": request.height,
                "fps": request.fps,
                "spoken_text": request.spokenText,
                "timeline": request.timeline,
            },
            "backend_params": request.backendParams,
        }

    async def generate_segment(
        self, request: SegmentGenerationRequest
    ) -> GenerationArtifact:
        raise AdapterUnavailableError(
            "ComfyUI workflow adapter is scaffolded but not enabled yet. "
            f"Target API: {self._api_url}"
        )

