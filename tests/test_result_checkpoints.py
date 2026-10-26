import asyncio
import ast
import json
from types import SimpleNamespace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from backend.app.adapters import comfyui as comfy_module
from backend.app.adapters.base import AdapterUnavailableError
from backend.app.adapters.comfyui import ComfyUiWorkflowAdapter
from backend.app.adapters.longcat_avatar import LongCatRuntimeFatalError
from backend.app.services.jobs import JobService


def test_running_result_is_readable_before_final_zip(tmp_path):
    service = object.__new__(JobService)
    doc = {"schemaVersion": "video-pipeline.external-generation.output.v1", "videos": [{"videoId": 914,
        "variants": [{"key": "v01", "segments": [], "dialogueScenes": [{"sceneId": "scene_01", "videoFile": "videos/914/scene.mp4"}]}]}]}
    runtime = SimpleNamespace(result_doc=doc, result_path=tmp_path / 'result.json', snapshot_path=tmp_path / 'job.json',
                              snapshot=SimpleNamespace(model_dump=lambda **_: {"status": "running", "completedSegments": 1}))
    service._jobs = {"test": runtime}
    service._write_snapshot(runtime)
    assert service.get_result('test') == doc
    assert json.loads(runtime.snapshot_path.read_text())["status"] == 'running'
    assert not (tmp_path / 'archive.zip').exists()
    assert not (tmp_path / 'result.json.tmp').exists()


def test_invalid_checkpoint_cannot_replace_previous_valid_result(tmp_path):
    service = object.__new__(JobService)
    path = tmp_path / 'result.json'
    service._write_json(path, {"completed": 1})
    with pytest.raises(TypeError):
        service._write_json(path, {"unserializable": object()})
    assert json.loads(path.read_text()) == {"completed": 1}


def test_fatal_cuda_finalizes_partial_work_without_starting_ltx():
    service = object.__new__(JobService)
    service._wait_for_ready_backend = AsyncMock(return_value='longcat-video-avatar')
    service._set_branch_state = AsyncMock()
    service._log = AsyncMock()
    service._release_adapters = lambda **_: None
    service._process_generation_branch = AsyncMock(side_effect=LongCatRuntimeFatalError('illegal memory access'))
    runtime = SimpleNamespace(snapshot=SimpleNamespace(status='running'))
    pending = ['longcat-video-avatar', 'comfyui-ltx25']
    asyncio.run(service._process_standard_branches(runtime, pending, []))
    assert service._process_generation_branch.await_count == 1
    assert pending == []
    assert any(c.args[1:3] == ('comfyui-ltx25', 'failed') for c in service._set_branch_state.await_args_list)


def test_cuda_failure_keeps_scenes_completed_before_worker_crash():
    service = object.__new__(JobService)
    service._log = AsyncMock()
    service._build_dialogue_scene_request = lambda _runtime, _video, _variant, _manifest, scene: SimpleNamespace(sceneId=scene.sceneId)
    artifact = object()
    error = LongCatRuntimeFatalError('illegal memory access', artifacts={'one': artifact})
    service._generate_dialogue_scenes_with_heartbeat = AsyncMock(side_effect=error)
    service._record_dialogue_scene_success = AsyncMock()
    service._record_dialogue_scene_failure = AsyncMock()
    scenes = [SimpleNamespace(sceneId='one', sceneIndex=1), SimpleNamespace(sceneId='two', sceneIndex=2)]
    with pytest.raises(LongCatRuntimeFatalError):
        asyncio.run(service._process_dialogue_scenes_batch(SimpleNamespace(), None, SimpleNamespace(videoId=915, operation=None, regenerationRequestId=None),
            SimpleNamespace(key='v01'), None, scenes, {'dialogueScenes': []}))
    assert service._record_dialogue_scene_success.await_count == 1
    assert service._record_dialogue_scene_failure.await_count == 1
    assert service._record_dialogue_scene_success.await_args.args[-2] is artifact


def test_child_worker_results_are_checkpointed_atomically(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'scripts/run_longcat_avatar_batch.py'
    helper = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
                  if isinstance(n, ast.FunctionDef) and n.name == '_write_batch_results')
    namespace = {'Path': Path, 'Any': Any, 'json': json}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(path), 'exec'), namespace)
    output = tmp_path / 'batch-result.json'
    namespace['_write_batch_results'](output, [{'sceneId': 'one', 'status': 'completed'}])
    assert json.loads(output.read_text())['scenes'][0]['sceneId'] == 'one'
    assert not output.with_name(output.name + '.tmp').exists()


def test_crashed_comfy_process_does_not_poll_for_two_hours(monkeypatch):
    adapter = object.__new__(ComfyUiWorkflowAdapter)
    adapter._api_url = 'http://comfy'
    adapter._history_request_timeout_sec = 1
    clock = [0.0]
    monkeypatch.setattr(comfy_module, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setenv('COMFYUI_UNAVAILABLE_TIMEOUT_SEC', '1')

    async def sleep(_):
        clock[0] += .6

    monkeypatch.setattr(comfy_module.asyncio, 'sleep', sleep)

    async def run():
        def unavailable(request):
            raise httpx.ConnectError('Comfy process exited', request=request)
        async with httpx.AsyncClient(transport=httpx.MockTransport(unavailable)) as client:
            await adapter._wait_for_history(client, 'lost-prompt')

    with pytest.raises(AdapterUnavailableError, match='became unavailable'):
        asyncio.run(run())
    assert clock[0] < 3


def test_short_comfy_connection_failure_can_recover(monkeypatch):
    adapter = object.__new__(ComfyUiWorkflowAdapter)
    adapter._api_url = 'http://comfy'
    adapter._history_request_timeout_sec = 1
    calls = []

    async def sleep(_):
        pass

    monkeypatch.setattr(comfy_module.asyncio, 'sleep', sleep)

    async def run():
        def recover(request):
            calls.append(request)
            if len(calls) == 1:
                raise httpx.ConnectError('temporary disconnect', request=request)
            return httpx.Response(200, json={'prompt': {'status': {'completed': True, 'status_str': 'success'}, 'outputs': {}}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(recover)) as client:
            return await adapter._wait_for_history(client, 'prompt')

    assert asyncio.run(run())['status']['completed'] is True
    assert len(calls) == 2
