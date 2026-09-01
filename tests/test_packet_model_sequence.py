from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from backend.app.services.jobs import JobService
from backend.app.services import provisioning
from scripts import download_comfy_ltx23_models as downloader


ROOT = Path(__file__).resolve().parents[1]


def _load_coordinator_module():
    path = ROOT / "scripts" / "coordinate_packet_model_branches.py"
    spec = importlib.util.spec_from_file_location("packet_sequence_coordinator", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_packet_deploy_serializes_mixed_model_branches_and_caps_downloads() -> None:
    script = (ROOT / "scripts" / "deploy_packet.sh").read_text(encoding="utf-8")

    assert "coordinate_packet_model_branches.py" in script
    assert 'MODEL_DOWNLOAD_CONCURRENCY="${AI_VIDEO_GEN_MODEL_DOWNLOAD_CONCURRENCY:-3}"' in script
    assert "--max-workers \"${MODEL_DOWNLOAD_CONCURRENCY}\"" in script
    assert "guard_packet_ltx_disk.sh" not in script
    assert "kill -STOP" not in script


def test_longcat_huggingface_downloads_are_limited_to_three_workers() -> None:
    script = (ROOT / "scripts" / "provision_longcat_avatar.sh").read_text(encoding="utf-8")

    assert 'MODEL_DOWNLOAD_CONCURRENCY="${AI_VIDEO_GEN_MODEL_DOWNLOAD_CONCURRENCY:-3}"' in script
    assert 'elif [ "${MODEL_DOWNLOAD_CONCURRENCY}" -gt 3 ]; then' in script
    assert script.count('--max-workers "${MODEL_DOWNLOAD_CONCURRENCY}"') == 2


def test_ltx_downloader_uses_configured_parallelism_without_exceeding_it(tmp_path: Path, monkeypatch) -> None:
    models = [
        downloader.ComfyModelFile(
            id=f"model-{index}",
            label=f"Model {index}",
            repo_id="owner/repo",
            repo_filename=f"model-{index}.safetensors",
            target_subdir="models",
            target_filename=f"model-{index}.safetensors",
        )
        for index in range(4)
    ]
    active = 0
    peak = 0
    lock = threading.Lock()
    completed: list[str] = []

    def fake_copy(model, *_args, **_kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            completed.append(model.id)
            active -= 1

    monkeypatch.setattr(downloader, "MODEL_FILES", models)
    monkeypatch.setattr(downloader, "discover_model_sizes", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(downloader, "copy_hf_file", fake_copy)
    monkeypatch.setattr(downloader, "write_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_comfy_ltx23_models.py",
            "--comfy-root",
            tmp_path.as_posix(),
            "--max-workers",
            "3",
        ],
    )

    downloader.main()

    assert peak == 3
    assert peak <= 3
    assert sorted(completed) == ["model-0", "model-1", "model-2", "model-3"]


def test_coordinator_starts_ltx_only_after_release_signal(tmp_path: Path, monkeypatch) -> None:
    coordinator = _load_coordinator_module()
    longcat_status = tmp_path / "longcat-status.json"
    ltx_status = tmp_path / "ltx-status.json"
    release_file = tmp_path / "longcat-released.json"
    longcat_weights = tmp_path / "LongCat-Video" / "weights"
    marker = tmp_path / "ltx-started.txt"

    longcat_status.write_text(json.dumps({"status": "ready"}), encoding="utf-8")
    release_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "coordinate_packet_model_branches.py",
            "--longcat-status-file",
            longcat_status.as_posix(),
            "--ltx-status-file",
            ltx_status.as_posix(),
            "--release-file",
            release_file.as_posix(),
            "--longcat-weights-dir",
            longcat_weights.as_posix(),
            "--",
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({marker.as_posix()!r}).write_text('started')",
        ],
    )

    assert coordinator.main() == 0
    assert marker.read_text(encoding="utf-8") == "started"
    assert json.loads(ltx_status.read_text(encoding="utf-8"))["status"] == "starting"


def test_coordinator_releases_failed_longcat_weights_then_starts_ltx(tmp_path: Path, monkeypatch) -> None:
    coordinator = _load_coordinator_module()
    longcat_status = tmp_path / "longcat-status.json"
    ltx_status = tmp_path / "ltx-status.json"
    release_file = tmp_path / "longcat-released.json"
    longcat_weights = tmp_path / "LongCat-Video" / "weights"
    marker = tmp_path / "ltx-started.txt"
    longcat_weights.mkdir(parents=True)
    (longcat_weights / "partial.safetensors.tmp").write_text("partial", encoding="utf-8")

    longcat_status.write_text(
        json.dumps({"status": "error", "error": "checkpoint download failed"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "coordinate_packet_model_branches.py",
            "--longcat-status-file",
            longcat_status.as_posix(),
            "--ltx-status-file",
            ltx_status.as_posix(),
            "--release-file",
            release_file.as_posix(),
            "--longcat-weights-dir",
            longcat_weights.as_posix(),
            "--poll-sec",
            "0.01",
            "--",
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({marker.as_posix()!r}).write_text('started')",
        ],
    )

    assert coordinator.main() == 0
    status = json.loads(ltx_status.read_text(encoding="utf-8"))
    assert status["status"] == "starting"
    assert marker.read_text(encoding="utf-8") == "started"
    assert not longcat_weights.exists()


def test_provisioning_keeps_ltx_unavailable_while_sequence_gate_is_closed(tmp_path: Path, monkeypatch) -> None:
    workflow = tmp_path / "workflow.json"
    workflow.write_text("{}", encoding="utf-8")
    settings = SimpleNamespace(
        enable_ltx=True,
        enable_longcat=True,
        comfyui_t2v_workflow=workflow,
        comfyui_i2v_workflow=workflow,
        generator_api_url="http://127.0.0.1:18188",
        provisioning_status_file=tmp_path / "ltx-status.json",
    )
    monkeypatch.setattr(provisioning, "list_comfy_ltx25_model_files", lambda _settings: [])
    monkeypatch.setattr(provisioning, "_check_comfy", lambda _settings: (True, None))
    monkeypatch.setattr(
        provisioning,
        "_check_longcat",
        lambda _settings: {"ready": True, "status": "ready", "message": "ready", "progressPercent": 100.0},
    )
    monkeypatch.setattr(
        provisioning,
        "_read_downloader_status",
        lambda _settings: {"status": "waiting_for_longcat", "message": "awaiting release"},
    )

    status = provisioning.get_provisioning_status(settings)

    assert status["branches"]["comfyui-ltx25"]["ready"] is False
    assert status["branches"]["comfyui-ltx25"]["status"] == "waiting_for_longcat"


def test_longcat_release_signal_is_optional_and_written_after_cleanup(tmp_path: Path) -> None:
    service = object.__new__(JobService)
    signal_path = tmp_path / "signals" / "longcat-released.json"
    service._settings = SimpleNamespace(longcat_branch_release_file=signal_path)

    assert service._signal_longcat_branch_release() == signal_path
    payload = json.loads(signal_path.read_text(encoding="utf-8"))
    assert payload["reason"] == "longcat_generation_branch_completed"

    service._settings = SimpleNamespace(longcat_branch_release_file=None)
    assert service._signal_longcat_branch_release() is None
