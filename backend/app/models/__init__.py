from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class BaseSchema(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class BatchFilters(BaseSchema):
    projectId: int | None = None
    videoId: int | None = None
    status: str | None = None


class ProjectInfo(BaseSchema):
    id: int
    name: str
    slug: str


class VideoTemplateInfo(BaseSchema):
    id: int
    key: str
    name: str


class WordTimelineItem(BaseSchema):
    word: str
    start: float
    end: float


class SubtitleSegmentItem(BaseSchema):
    index: int
    text: str
    start: float
    end: float


class SegmentTimeline(BaseSchema):
    startSec: float
    endSec: float
    effectiveDurationSec: float
    generationDurationSec: float


class SegmentNarration(BaseSchema):
    spokenText: str = ""
    subtitleText: str = ""
    keywords: list[str] = Field(default_factory=list)
    wordTimeline: list[WordTimelineItem] = Field(default_factory=list)
    subtitleSegments: list[SubtitleSegmentItem] = Field(default_factory=list)


class SegmentGeneration(BaseSchema):
    prompt: str = ""
    negativePrompt: str = ""
    continuityNote: str = ""
    shotGoal: str = ""


class ManifestSegment(BaseSchema):
    segmentId: str
    segmentIndex: int
    timeline: SegmentTimeline
    narration: SegmentNarration
    generation: SegmentGeneration


class ProjectContext(BaseSchema):
    videoName: str = ""
    mainRequest: str = ""
    preferredStyle: str = ""
    extraRequirements: str = ""
    globalRules: str = ""
    outputProfile: dict[str, Any] = Field(default_factory=dict)
    deliveryProfile: dict[str, Any] = Field(default_factory=dict)


class SourceArtifacts(BaseSchema):
    narrationTextPath: str | None = None
    alignmentJsonPath: str | None = None
    promptPreviewPath: str | None = None


class VariantManifest(BaseSchema):
    schemaVersion: str
    mode: str
    generatedAt: datetime
    runId: str
    variantKey: str
    variantLabel: str
    promptLanguage: str = "en"
    targetDurationSec: float
    speechDurationSec: float
    segmentDurationSec: float = 8
    totalSegments: int
    projectContext: ProjectContext = Field(default_factory=ProjectContext)
    globalVisualDirection: str = ""
    globalNegativePrompt: str = ""
    sourceArtifacts: SourceArtifacts = Field(default_factory=SourceArtifacts)
    segments: list[ManifestSegment] = Field(default_factory=list)


class VariantInfo(BaseSchema):
    key: str
    label: str
    status: str
    externalGenerationManifestPath: str | None = None
    externalGenerationManifestUrl: str | None = None
    manifestFound: bool = False
    manifest: VariantManifest | None = None


class VideoInfo(BaseSchema):
    videoId: int
    projectId: int
    runId: str
    title: str
    status: str
    createdAt: datetime
    updatedAt: datetime
    project: ProjectInfo
    videoTemplate: VideoTemplateInfo
    outputProfile: dict[str, Any] = Field(default_factory=dict)
    deliveryProfile: dict[str, Any] = Field(default_factory=dict)
    subtitleStyle: dict[str, Any] = Field(default_factory=dict)
    requestSnapshot: dict[str, Any] = Field(default_factory=dict)
    variants: list[VariantInfo] = Field(default_factory=list)


class BatchExport(BaseSchema):
    schemaVersion: str
    exportedAt: datetime
    filters: BatchFilters
    totalVideos: int
    totalVariants: int
    videos: list[VideoInfo] = Field(default_factory=list)


class AdapterInfo(BaseSchema):
    key: str
    label: str
    description: str
    status: Literal["ready", "planned", "experimental"]
    available: bool
    supportsBatch: bool = True
    supportsDirect: bool = True
    requiresRemote: bool = False
    notes: str | None = None


class LogEntry(BaseSchema):
    ts: datetime
    level: Literal["info", "warning", "error"]
    message: str


class JobSnapshot(BaseSchema):
    jobId: str
    status: str
    backend: str
    totalVideos: int
    totalVariants: int
    totalSegments: int
    completedSegments: int = 0
    failedSegments: int = 0
    startedAt: datetime | None = None
    updatedAt: datetime
    finishedAt: datetime | None = None
    inputFile: str | None = None
    resultFile: str | None = None
    archiveFile: str | None = None
    error: str | None = None


class JobQueuedResponse(BaseSchema):
    jobId: str
    status: str


class SegmentGenerationRequest(BaseSchema):
    jobId: str
    backend: str
    videoId: int
    projectId: int
    runId: str
    videoTitle: str
    variantKey: str
    variantLabel: str
    segmentId: str
    segmentIndex: int
    promptLanguage: str
    prompt: str
    negativePrompt: str
    continuityNote: str
    shotGoal: str
    spokenText: str
    subtitleText: str
    wordTimeline: list[dict[str, Any]] = Field(default_factory=list)
    globalVisualDirection: str = ""
    globalNegativePrompt: str = ""
    resolvedPrompt: str = ""
    resolvedNegativePrompt: str = ""
    width: int = 720
    height: int = 1280
    fps: float = 24.0
    durationSec: float = 8.0
    outputPath: Path
    backendParams: dict[str, Any] = Field(default_factory=dict)
    timeline: dict[str, Any] = Field(default_factory=dict)


class GenerationArtifact(BaseSchema):
    modelName: str
    modelVersion: str | None = None
    outputPath: Path
    debug: dict[str, Any] = Field(default_factory=dict)


class DirectGenerationRequest(BaseSchema):
    backend: str = "mock-gen"
    title: str = "Manual generation"
    promptLanguage: str = "en"
    prompt: str
    negativePrompt: str = ""
    continuityNote: str = ""
    shotGoal: str = ""
    spokenText: str = ""
    subtitleText: str = ""
    durationSec: float = 8.0
    width: int = 720
    height: int = 1280
    fps: float = 24.0
    videoId: int = 1
    projectId: int = 1
    runId: str | None = None
    variantKey: str = "v01"
    variantLabel: str = "Variant 1"
    globalVisualDirection: str = ""
    globalNegativePrompt: str = ""
    backendParams: dict[str, Any] = Field(default_factory=dict)

