import json
from pathlib import Path
import wave

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


def test_prepare_longcat_blackwell_benchmark_accepts_export_batch_shape(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    for name in ("image.png", "speaker1.wav", "speaker2.wav"):
        (assets / name).write_bytes(b"test")
    batch = {
        "videos": [{"variants": [{"manifest": {"dialogueScenes": [{
            "sceneId": "scene_01",
            "generation": {"prompt": "Two researchers talk.", "imageFile": "assets/image.png"},
            "audio": {"speaker1File": "assets/speaker1.wav", "speaker2File": "assets/speaker2.wav"},
            "output": {"width": 1080, "height": 1920},
        }]}}]}]
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
    assert input_doc["target_orientation"] == "portrait"


def test_prepare_longcat_blackwell_benchmark_rewrites_digital_silence(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    image = assets / "image.png"
    image.write_bytes(b"image")
    for name, payload in (("speaker1.wav", b"\x01\x00" * 8000), ("speaker2.wav", b"\x00\x00" * 8000)):
        with wave.open((assets / name).as_posix(), "wb") as target:
            target.setnchannels(1)
            target.setsampwidth(2)
            target.setframerate(16_000)
            target.writeframes(payload)
    batch = {
        "variants": [{"manifest": {"dialogueScenes": [{
            "sceneId": "scene_01",
            "generation": {"prompt": "Two researchers talk.", "imageFile": "assets/image.png"},
            "audio": {"speaker1File": "assets/speaker1.wav", "speaker2File": "assets/speaker2.wav"},
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
    silent_path = Path(input_doc["cond_audio"]["person2"])
    assert silent_path.name == "speaker2_technical_silence.wav"
    with wave.open(silent_path.as_posix(), "rb") as output:
        assert output.readframes(1) == b"\x01\x00"
