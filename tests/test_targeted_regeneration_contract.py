from backend.app.models import BatchExport


def test_targeted_regeneration_identity_is_accepted_by_batch_contract() -> None:
    batch = BatchExport.model_validate(
        {
            "schemaVersion": "video-pipeline.external-generation.batch.v1",
            "exportedAt": "2026-09-08T00:00:00Z",
            "filters": {"status": "waiting_media", "operation": "targeted_regeneration"},
            "totalVideos": 1,
            "totalVariants": 1,
            "videos": [
                {
                    "videoId": 10,
                    "projectId": 20,
                    "runId": "run-1",
                    "title": "Generic video",
                    "status": "waiting_media",
                    "createdAt": "2026-09-08T00:00:00Z",
                    "updatedAt": "2026-09-08T00:00:00Z",
                    "operation": "targeted_regeneration",
                    "regenerationRequestId": "request-1",
                    "project": {"id": 20, "name": "Generic", "slug": "generic"},
                    "videoTemplate": {"id": 1, "key": "landscape", "name": "Landscape"},
                    "variants": [
                        {
                            "key": "v01",
                            "label": "Variant 1",
                            "status": "waiting_media",
                            "manifestFound": True,
                            "manifest": {
                                "schemaVersion": "video-pipeline.external-generation.manifest.v1",
                                "mode": "deferred_generation",
                                "generatedAt": "2026-09-08T00:00:00Z",
                                "runId": "run-1",
                                "variantKey": "v01",
                                "variantLabel": "Variant 1",
                                "targetDurationSec": 8,
                                "speechDurationSec": 8,
                                "totalSegments": 1,
                                "segments": [
                                    {
                                        "segmentId": "run-1_v01_s01",
                                        "segmentIndex": 1,
                                        "timeline": {
                                            "startSec": 0,
                                            "endSec": 8,
                                            "effectiveDurationSec": 8,
                                            "generationDurationSec": 8,
                                        },
                                        "narration": {},
                                        "generation": {"prompt": "A generic cinematic shot"},
                                    }
                                ],
                            },
                        }
                    ],
                }
            ],
        }
    )

    assert batch.videos[0].operation == "targeted_regeneration"
    assert batch.videos[0].regenerationRequestId == "request-1"
