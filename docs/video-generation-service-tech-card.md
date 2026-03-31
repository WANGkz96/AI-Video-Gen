# Batch Video Generation Service

## Purpose

The service accepts one JSON batch exported by the main platform, generates all requested video segments, and returns one archive that contains:

- generated segment video files;
- per-segment metadata JSON files;
- one batch result JSON for later import back into the main platform.

The service only:

1. accepts a batch JSON;
2. runs all segment generation tasks;
3. saves the generated files;
4. packs the result into one archive.

## Required Input

The service input is the exact JSON exported by the main platform from:

- `GET /api/videos/external-generation/export`

Top-level schema:

```json
{
  "schemaVersion": "video-pipeline.external-generation.batch.v1",
  "exportedAt": "2026-03-31T00:00:00.000Z",
  "filters": {
    "projectId": null,
    "videoId": null,
    "status": "waiting_media"
  },
  "totalVideos": 1,
  "totalVariants": 1,
  "videos": [
    {
      "videoId": 123,
      "projectId": 9,
      "runId": "20260331_000000_000",
      "title": "Video title",
      "status": "waiting_media",
      "createdAt": "2026-03-31T00:00:00.000Z",
      "updatedAt": "2026-03-31T00:00:00.000Z",
      "project": {
        "id": 9,
        "name": "Project name",
        "slug": "project-slug"
      },
      "videoTemplate": {
        "id": 2,
        "key": "youtube_shorts_test",
        "name": "Portrait"
      },
      "outputProfile": {},
      "deliveryProfile": {},
      "subtitleStyle": {},
      "requestSnapshot": {},
      "variants": [
        {
          "key": "v01",
          "label": "Variant 1",
          "status": "waiting_media",
          "externalGenerationManifestPath": ".../manifest.json",
          "externalGenerationManifestUrl": "/media/.../manifest.json",
          "manifestFound": true,
          "manifest": {
            "schemaVersion": "video-pipeline.external-generation.manifest.v1"
          }
        }
      ]
    }
  ]
}
```

## Required Manifest Fields

Each `variants[].manifest` object must be treated as the primary generation instruction.

Manifest schema:

```json
{
  "schemaVersion": "video-pipeline.external-generation.manifest.v1",
  "mode": "deferred_generation",
  "generatedAt": "2026-03-31T00:00:00.000Z",
  "runId": "20260331_000000_000",
  "variantKey": "v01",
  "variantLabel": "Variant 1",
  "promptLanguage": "en",
  "targetDurationSec": 88,
  "speechDurationSec": 86.2,
  "segmentDurationSec": 8,
  "totalSegments": 11,
  "projectContext": {
    "videoName": "...",
    "mainRequest": "...",
    "preferredStyle": "...",
    "extraRequirements": "...",
    "globalRules": "...",
    "outputProfile": {},
    "deliveryProfile": {}
  },
  "globalVisualDirection": "...",
  "globalNegativePrompt": "...",
  "sourceArtifacts": {
    "narrationTextPath": "...",
    "alignmentJsonPath": "...",
    "promptPreviewPath": "..."
  },
  "segments": [
    {
      "segmentId": "20260331_000000_000_v01_s01",
      "segmentIndex": 1,
      "timeline": {
        "startSec": 0,
        "endSec": 8,
        "effectiveDurationSec": 8,
        "generationDurationSec": 8
      },
      "narration": {
        "spokenText": "...",
        "subtitleText": "...",
        "keywords": ["..."],
        "wordTimeline": [
          { "word": "Hello", "start": 0.1, "end": 0.4 }
        ],
        "subtitleSegments": [
          { "index": 0, "text": "...", "start": 0, "end": 1.2 }
        ]
      },
      "generation": {
        "prompt": "...",
        "negativePrompt": "...",
        "continuityNote": "...",
        "shotGoal": "..."
      }
    }
  ]
}
```

## Processing Rules

The service must:

1. read the full batch JSON;
2. iterate through every `video`;
3. iterate through every `variant`;
4. iterate through every `manifest.segments[]` item in `segmentIndex` order;
5. generate one video file for each segment;
6. keep the original identifiers unchanged:
   - `videoId`
   - `projectId`
   - `runId`
   - `variantKey`
   - `segmentId`
7. save one metadata JSON next to each generated segment;
8. produce one final batch result JSON;
9. pack everything into one archive.

## Generation Rules

The generator must use:

- `manifest.globalVisualDirection` as a global visual instruction;
- `manifest.globalNegativePrompt` as a global negative instruction;
- `segment.generation.prompt` as the main prompt for that segment;
- `segment.generation.continuityNote` as a continuity hint;
- `segment.generation.shotGoal` as the local segment intent;
- `segment.narration.spokenText` and `segment.narration.wordTimeline` as speech timing context.

The generator must preserve exact segment duration targets:

- default segment duration is `8` seconds;
- each generated segment should match `timeline.generationDurationSec`;
- the service must not merge segments into one file.

## Recommended Internal API

Minimal API for the new service:

- `GET /api/health`
- `POST /api/jobs`
- `GET /api/jobs/:jobId`
- `GET /api/jobs/:jobId/result`
- `GET /api/jobs/:jobId/archive`

### `POST /api/jobs`

Accepts either:

- raw JSON body; or
- multipart upload with one `batch.json` file.

Recommended response:

```json
{
  "jobId": "job_20260331_000001",
  "status": "queued"
}
```

### `GET /api/jobs/:jobId`

Recommended response:

```json
{
  "jobId": "job_20260331_000001",
  "status": "running",
  "totalVideos": 12,
  "totalVariants": 12,
  "totalSegments": 132,
  "completedSegments": 48,
  "failedSegments": 0,
  "startedAt": "2026-03-31T00:00:00.000Z",
  "updatedAt": "2026-03-31T00:20:00.000Z"
}
```

## Required Output Archive

The service output must be one archive, recommended format: `zip`.

Recommended structure:

```text
batch-result.zip
  result.json
  input/
    batch.json
  videos/
    123/
      v01/
        20260331_000000_000_v01_s01.mp4
        20260331_000000_000_v01_s01.json
        20260331_000000_000_v01_s02.mp4
        20260331_000000_000_v01_s02.json
```

## Required `result.json`

Top-level output schema:

```json
{
  "schemaVersion": "video-pipeline.external-generation.output.v1",
  "generatedAt": "2026-03-31T01:00:00.000Z",
  "videos": [
    {
      "videoId": 123,
      "projectId": 9,
      "runId": "20260331_000000_000",
      "variants": [
        {
          "key": "v01",
          "segments": [
            {
              "segmentId": "20260331_000000_000_v01_s01",
              "videoFile": "videos/123/v01/20260331_000000_000_v01_s01.mp4",
              "durationSec": 8,
              "width": 720,
              "height": 1280,
              "fps": 24
            }
          ]
        }
      ]
    }
  ]
}
```

## Required Per-Segment Metadata JSON

Each segment JSON file must contain enough data for debugging and later import.

Recommended schema:

```json
{
  "schemaVersion": "video-pipeline.external-generation.segment.v1",
  "generatedAt": "2026-03-31T01:00:00.000Z",
  "videoId": 123,
  "projectId": 9,
  "runId": "20260331_000000_000",
  "variantKey": "v01",
  "segmentId": "20260331_000000_000_v01_s01",
  "segmentIndex": 1,
  "timeline": {
    "startSec": 0,
    "endSec": 8,
    "effectiveDurationSec": 8,
    "generationDurationSec": 8
  },
  "prompt": "...",
  "negativePrompt": "...",
  "continuityNote": "...",
  "shotGoal": "...",
  "spokenText": "...",
  "subtitleText": "...",
  "model": {
    "name": "ltx-video-2",
    "version": "..."
  },
  "render": {
    "width": 720,
    "height": 1280,
    "fps": 24,
    "durationSec": 8
  },
  "files": {
    "videoFile": "videos/123/v01/20260331_000000_000_v01_s01.mp4"
  }
}
```

## Validation Rules

Before final archive creation, the service must validate:

- every expected segment exists;
- every output file is readable;
- every segment has a matching metadata JSON;
- `segmentId` in JSON matches the file and manifest;
- output duration is present and positive;
- width, height, and fps are known.

If any segment fails:

- keep successful files;
- write failure information into the batch result JSON;
- mark the segment as failed instead of silently skipping it.

## Failure Reporting

Recommended segment failure entry inside `result.json`:

```json
{
  "segmentId": "20260331_000000_000_v01_s01",
  "status": "failed",
  "error": "model timeout"
}
```

## Minimal Deployment Requirements

The service should have:

- one HTTP API process;
- one background worker queue;
- one writable storage directory for:
  - incoming jobs;
  - generated segments;
  - finished archives;
- one configurable generator backend adapter.

Recommended environment variables:

```env
PORT=3001
WORKDIR=./data
JOBS_DIR=./data/jobs
OUTPUT_DIR=./data/output
ARCHIVE_DIR=./data/archives
TEMP_DIR=./data/tmp
GENERATOR_BACKEND=comfyui
GENERATOR_API_URL=http://localhost:8188
MAX_PARALLEL_SEGMENTS=1
```

## Required Determinism Rules

The service must never invent new IDs.

It must reuse exactly what came from the input:

- `videoId`
- `projectId`
- `runId`
- `variantKey`
- `segmentId`
- `segmentIndex`

Import on the main platform will depend on these identifiers.

## First Implementation Scope

The first implementation must support only:

- one input batch JSON;
- one generation backend;
- sequential segment processing;
- one output zip archive;
- no UI is required.

That is enough to make the service usable by the main platform.
