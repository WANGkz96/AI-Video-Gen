import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.app.adapters.base import AdapterUnavailableError
from backend.app.services import jobs as jobs_module
from backend.app.services.jobs import JobService


def _service() -> JobService:
    service = object.__new__(JobService)
    service._settings = SimpleNamespace(
        backend_ready_poll_sec=0.001,
        backend_ready_timeout_sec=1.0,
    )
    service._adapters = {}
    service._write_snapshot = lambda runtime: None
    service._broadcast_snapshot = AsyncMock()
    service._log = AsyncMock()
    return service


def _runtime():
    return SimpleNamespace(
        snapshot=SimpleNamespace(status="running", updatedAt=None),
    )


def _provisioning(*, ltx_ready: bool, longcat_ready: bool) -> dict:
    return {
        "branches": {
            "comfyui-ltx25": {
                "ready": ltx_ready,
                "status": "ready" if ltx_ready else "downloading",
                "progressPercent": 100 if ltx_ready else 45,
                "error": None,
            },
            "longcat-video-avatar": {
                "ready": longcat_ready,
                "status": "ready" if longcat_ready else "downloading",
                "progressPercent": 100 if longcat_ready else 35,
                "error": None,
            },
        }
    }


def test_longcat_starts_without_waiting_for_ltx(monkeypatch) -> None:
    states = iter(
        [
            _provisioning(ltx_ready=False, longcat_ready=False),
            _provisioning(ltx_ready=False, longcat_ready=True),
        ]
    )
    monkeypatch.setattr(jobs_module, "get_provisioning_status", lambda settings: next(states))

    selected = asyncio.run(
        _service()._wait_for_ready_backend(
            _runtime(),
            ["comfyui-ltx25", "longcat-video-avatar"],
        )
    )

    assert selected == "longcat-video-avatar"


def test_ltx_starts_without_waiting_for_longcat(monkeypatch) -> None:
    states = iter(
        [
            _provisioning(ltx_ready=False, longcat_ready=False),
            _provisioning(ltx_ready=True, longcat_ready=False),
        ]
    )
    monkeypatch.setattr(jobs_module, "get_provisioning_status", lambda settings: next(states))

    selected = asyncio.run(
        _service()._wait_for_ready_backend(
            _runtime(),
            ["comfyui-ltx25", "longcat-video-avatar"],
        )
    )

    assert selected == "comfyui-ltx25"


def test_failed_branch_does_not_block_other_branch_from_starting(monkeypatch) -> None:
    first = _provisioning(ltx_ready=False, longcat_ready=False)
    first["branches"]["longcat-video-avatar"].update(
        status="error",
        error="download failed",
    )
    states = iter([first, _provisioning(ltx_ready=True, longcat_ready=False)])
    monkeypatch.setattr(jobs_module, "get_provisioning_status", lambda settings: next(states))

    selected = asyncio.run(
        _service()._wait_for_ready_backend(
            _runtime(),
            ["comfyui-ltx25", "longcat-video-avatar"],
        )
    )

    assert selected == "comfyui-ltx25"


def test_generation_branches_are_isolated_on_the_single_gpu() -> None:
    service = _service()
    service._adapters = {
        "comfyui-ltx25": object(),
        "longcat-video-avatar": object(),
    }
    service._process_variant = AsyncMock()
    runtime = SimpleNamespace(snapshot=SimpleNamespace(backend="comfyui-ltx25"))
    work_items = [(object(), object(), {})]

    asyncio.run(
        service._process_generation_branch(runtime, "longcat-video-avatar", work_items)
    )
    asyncio.run(service._process_generation_branch(runtime, "comfyui-ltx25", work_items))

    first = service._process_variant.await_args_list[0].kwargs
    second = service._process_variant.await_args_list[1].kwargs
    assert first["process_segments"] is False
    assert first["process_dialogue_scenes"] is True
    assert second["process_segments"] is True
    assert second["process_dialogue_scenes"] is False


def test_configured_backend_can_queue_before_its_download_finishes() -> None:
    service = _service()
    service._settings.enable_ltx = True
    service._settings.enable_longcat = True
    unavailable = SimpleNamespace(
        available=False,
        requiresDownload=True,
        notes="still downloading",
    )
    service._adapters = {
        "comfyui-ltx25": SimpleNamespace(info=lambda: unavailable),
        "longcat-video-avatar": SimpleNamespace(info=lambda: unavailable),
    }

    service._ensure_backend_can_queue("comfyui-ltx25")
    service._ensure_backend_can_queue("longcat-video-avatar")

    service._settings.enable_longcat = False
    with pytest.raises(AdapterUnavailableError, match="still downloading"):
        service._ensure_backend_can_queue("longcat-video-avatar")


def test_turbo_runs_ltx_and_longcat_branches_concurrently() -> None:
    service = _service()
    service._wait_for_ready_backend = AsyncMock(side_effect=lambda _runtime, backends: backends[0])
    service._set_branch_state = AsyncMock()
    active = 0
    peak = 0

    async def process_branch(_runtime, _backend, _work_items):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)
        active -= 1

    service._process_generation_branch = process_branch
    runtime = SimpleNamespace(
        snapshot=SimpleNamespace(
            status="running",
            executionProfile={"effective": "turbo"},
            turboFallbackRecommended=False,
        )
    )
    result_doc = {"errors": []}

    asyncio.run(
        service._process_turbo_branches(
            runtime,
            ["comfyui-ltx25", "longcat-video-avatar"],
            [],
            result_doc,
        )
    )

    assert peak == 2
    assert result_doc["errors"] == []


def test_turbo_branch_failure_does_not_cancel_the_healthy_branch() -> None:
    service = _service()
    service._wait_for_ready_backend = AsyncMock(side_effect=lambda _runtime, backends: backends[0])
    service._set_branch_state = AsyncMock()
    completed = []

    async def process_branch(_runtime, backend, _work_items):
        if backend == "comfyui-ltx25":
            raise RuntimeError("LTX failed")
        await asyncio.sleep(0.01)
        completed.append(backend)

    service._process_generation_branch = process_branch
    runtime = SimpleNamespace(
        snapshot=SimpleNamespace(
            status="running",
            executionProfile={"effective": "turbo"},
            turboFallbackRecommended=False,
        )
    )
    result_doc = {"errors": []}

    asyncio.run(
        service._process_turbo_branches(
            runtime,
            ["comfyui-ltx25", "longcat-video-avatar"],
            [],
            result_doc,
        )
    )

    assert completed == ["longcat-video-avatar"]
    assert result_doc["errors"][0]["backend"] == "comfyui-ltx25"


def test_turbo_single_branch_effectively_remains_sequential(monkeypatch) -> None:
    service = _service()
    service._settings.execution_profile = "turbo"
    service._settings.turbo_max_concurrent_branches = 2
    service._settings.turbo_min_vram_gb = 160
    monkeypatch.setattr(
        jobs_module,
        "read_gpu_telemetry",
        lambda: {"available": True, "totalVramGb": 180},
    )

    profile, _telemetry, branch_state = service._resolve_execution_profile(
        has_regular_segments=True,
        has_dialogue_scenes=False,
    )

    assert profile["requested"] == "turbo"
    assert profile["effective"] == "standard"
    assert profile["maxConcurrentBranches"] == 1
    assert profile["degradedReason"] == "single_branch"
    assert list(branch_state) == ["comfyui-ltx25"]


def test_turbo_rejects_gpu_below_vram_floor(monkeypatch) -> None:
    service = _service()
    service._settings.execution_profile = "turbo"
    service._settings.turbo_max_concurrent_branches = 2
    service._settings.turbo_min_vram_gb = 160
    monkeypatch.setattr(
        jobs_module,
        "read_gpu_telemetry",
        lambda: {"available": True, "totalVramGb": 96},
    )

    with pytest.raises(AdapterUnavailableError, match="at least 160 GB VRAM"):
        service._resolve_execution_profile(
            has_regular_segments=True,
            has_dialogue_scenes=True,
        )


def test_turbo_memory_error_recommends_standard_retry() -> None:
    service = _service()
    runtime = SimpleNamespace(
        snapshot=SimpleNamespace(
            executionProfile={"effective": "turbo"},
            turboFallbackRecommended=False,
        )
    )

    service._mark_turbo_instability(runtime, RuntimeError("CUDA out of memory"))

    assert runtime.snapshot.turboFallbackRecommended is True
    assert runtime.snapshot.executionProfile["degradedReason"] == "gpu_memory_failure"
