import json
from pathlib import Path

from scripts.prepare_longcat_blackwell_benchmark import prepare_benchmark


def test_prepare_longcat_blackwell_benchmark_uses_real_scene_assets(tmp_path: Path) -> None:
    assets = tmp_path / "assets" / "scene"
    assets.mkdir(parents=True)
    for name in ("image.png", "speaker1.wav", "speaker2.wav"):
        (assets / name).write_bytes(b"test")
    batch = {
        "variants": [{"manifest": {"dialogueScenes": [{
            "sceneId": "scene_01",
            "generation": {"prompt": "Two researchers talk.", "imageFile": "assets/scene/image.png"},
            "audio": {"speaker1File": "assets/scene/speaker1.wav", "speaker2File": "assets/scene/speaker2.wav"},
            "output": {"width": 1920, "height": 1080},
        }]}}]
    }
    batch_path = tmp_path / "batch.json"
    batch_path.write_text(json.dumps(batch), encoding="utf-8")

    result = prepare_benchmark(
        batch_path=batch_path,
        scene_id="scene_01",
        output_dir=tmp_path / "benchmark",
        num_segments=1,
    )

    input_doc = json.loads(Path(result["input"]).read_text(encoding="utf-8"))
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert input_doc["target_orientation"] == "landscape"
    assert input_doc["cond_audio"]["person1"].endswith("speaker1.wav")
    assert "Natural multi-person lip sync" in input_doc["prompt"]
    assert manifest["scenes"][0]["numSegments"] == 1
