from pathlib import Path
import wave

import httpx

from backend.app.adapters.comfyui import ComfyUiWorkflowAdapter
from backend.app.adapters.longcat_avatar import LongCatAvatarAdapter
from backend.app.models import BatchExport
from scripts.patch_longcat_runtime import patch_source


def _write_pcm_wav(path: Path, samples: list[int], sample_rate: int = 24_000) -> None:
    payload = bytearray()
    for sample in samples:
        payload.extend(int(sample).to_bytes(2, "little", signed=True))
    with wave.open(path.as_posix(), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(payload)


def test_dialogue_only_manifest_is_valid() -> None:
    batch = BatchExport.model_validate(
        {
            "schemaVersion": "video-pipeline.external-generation.batch.v1",
            "exportedAt": "2026-08-21T00:00:00Z",
            "filters": {},
            "totalVideos": 1,
            "totalVariants": 1,
            "videos": [
                {
                    "videoId": 7,
                    "projectId": 2,
                    "runId": "20260821_000000",
                    "title": "Dialogue test",
                    "status": "waiting_media",
                    "createdAt": "2026-08-21T00:00:00Z",
                    "updatedAt": "2026-08-21T00:00:00Z",
                    "project": {"id": 2, "name": "The Obscura", "slug": "the-obscura"},
                    "videoTemplate": {"id": 1, "key": "shorts", "name": "Shorts"},
                    "variants": [
                        {
                            "key": "v01",
                            "label": "Variant 1",
                            "status": "waiting_media",
                            "manifestFound": True,
                            "manifest": {
                                "schemaVersion": "video-pipeline.external-generation.manifest.v1",
                                "mode": "deferred_generation",
                                "generatedAt": "2026-08-21T00:00:00Z",
                                "runId": "20260821_000000",
                                "variantKey": "v01",
                                "variantLabel": "Variant 1",
                                "targetDurationSec": 60,
                                "speechDurationSec": 60,
                                "totalSegments": 0,
                                "segments": [],
                                "dialogueScenes": [
                                    {
                                        "sceneId": "scene_01",
                                        "sceneIndex": 1,
                                        "timeline": {"startSec": 12, "endSec": 20, "durationSec": 8},
                                        "generation": {
                                            "prompt": "Two scientists exchange a skeptical look.",
                                            "image": {"file": "assets/scene.png"},
                                        },
                                        "audio": {
                                            "speaker1File": "assets/speaker_1.wav",
                                            "speaker2File": "assets/speaker_2.wav",
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                }
            ],
        }
    )

    manifest = batch.videos[0].variants[0].manifest
    assert manifest is not None
    assert manifest.segments == []
    assert manifest.dialogueScenes[0].timeline.durationSec == 8
    assert manifest.dialogueScenes[0].audio.speaker2File.endswith("speaker_2.wav")


def test_longcat_runtime_patch_is_idempotent_and_adds_prompt_schedule() -> None:
    pristine = """import torch

def generate(args):
    height, width = 480, 832
    with open(input_json, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    prompt = input_data['prompt']
    for segment_idx in range(2):
        output_tuple = pipe.generate_avc(
            prompt=prompt,
        )
"""
    runner = Path("run_demo_avatar_multi_audio_to_video.py")
    patched = patch_source(pristine, runner)
    patched_again = patch_source(patched, runner)

    assert patched_again == patched
    assert "torch.backends.cudnn.enabled = False" in patched
    assert "prompt_schedule = input_data.get('prompt_schedule')" in patched
    assert "prompt=segment_prompt" in patched
    assert patched.count("Using prompt schedule item") == 1
    assert "target_orientation = str(input_data.get('target_orientation'" in patched
    assert "height, width = width, height" in patched
    assert patched.count("Applying target orientation") == 2


def test_longcat_adapter_makes_fully_silent_track_separator_compatible(tmp_path: Path) -> None:
    source = tmp_path / "silent.wav"
    prepared = tmp_path / "prepared.wav"
    _write_pcm_wav(source, [0] * 2_400)

    assert LongCatAvatarAdapter._pcm_wav_is_fully_silent(source) is True
    LongCatAvatarAdapter._write_longcat_compatible_silence(source, prepared, 0.1)

    assert LongCatAvatarAdapter._pcm_wav_is_fully_silent(prepared) is False
    with wave.open(prepared.as_posix(), "rb") as result:
        assert result.getnframes() == 2_400
        assert result.getframerate() == 24_000


def test_longcat_adapter_keeps_spoken_track_untouched(tmp_path: Path) -> None:
    spoken = tmp_path / "spoken.wav"
    _write_pcm_wav(spoken, [0, 1, -1, 0])

    assert LongCatAvatarAdapter._pcm_wav_is_fully_silent(spoken) is False


def test_comfy_release_unloads_ltx_models(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return httpx.Response(200, request=httpx.Request("POST", url))

    adapter = object.__new__(ComfyUiWorkflowAdapter)
    adapter._api_url = "http://127.0.0.1:18188"
    monkeypatch.setattr(httpx, "post", fake_post)

    adapter.release()

    assert calls[0]["url"] == "http://127.0.0.1:18188/free"
    assert calls[0]["json"] == {"unload_models": True, "free_memory": True}
