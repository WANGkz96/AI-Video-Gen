import asyncio
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock
from unittest.mock import AsyncMock
from functools import wraps

import pytest

from backend.app.services.jobs import JobService
from backend.app.adapters.longcat_avatar import LongCatAvatarAdapter

ROOT = Path(__file__).resolve().parents[1]
JOB = 'job_20261026_220022_786491_0001'


def make_service(tmp_path, **changes):
    service = object.__new__(JobService)
    service._settings = SimpleNamespace(jobs_dir=tmp_path / 'jobs', archive_dir=tmp_path / 'archives')
    service._jobs = {}
    service._queue = asyncio.Queue()
    folder = service._settings.jobs_dir / JOB
    (folder / 'input').mkdir(parents=True)
    (folder / 'input/batch.json').write_text((ROOT / 'examples/sample-batch.json').read_text())
    snapshot = dict(jobId=JOB, status='running', backend='comfyui-ltx25', totalVideos=1,
                    totalVariants=1, totalSegments=2, updatedAt='2026-10-26T22:00:00Z', **changes)
    (folder / 'job.json').write_text(json.dumps(snapshot))
    return service, folder


def test_explicit_recovery_keeps_job_id_inputs_and_queue(tmp_path):
    service, folder = make_service(tmp_path)
    original = (folder / 'input/batch.json').read_bytes()
    asyncio.run(service.restore_empty_job(JOB))
    assert service._queue.get_nowait() == JOB
    runtime = service._jobs[JOB]
    assert runtime.snapshot.status == 'queued'
    assert runtime.archive_path.name == JOB + '.zip'
    assert runtime.snapshot.totalSegments == 2
    assert (folder / 'input/batch.json').read_bytes() == original
    assert len(runtime.logs) == 1


@pytest.mark.parametrize('counter', ['completedSegments', 'failedSegments'])
def test_processed_job_requires_archive_recovery(tmp_path, counter):
    service, _ = make_service(tmp_path, **{counter: 1})
    with pytest.raises(ValueError, match='missing-only'):
        asyncio.run(service.restore_empty_job(JOB))
    assert service._queue.empty()


def test_even_partial_mp4_is_not_overwritten(tmp_path):
    service, folder = make_service(tmp_path)
    p = folder / 'videos/915/scene.mp4'
    p.parent.mkdir(parents=True)
    p.write_bytes(b'partial')
    with pytest.raises(ValueError, match='Existing media'):
        asyncio.run(service.restore_empty_job(JOB))
    assert p.read_bytes() == b'partial'


def test_recovery_requires_idle_service_and_safe_job_id(tmp_path):
    service, _ = make_service(tmp_path)
    with pytest.raises(ValueError):
        asyncio.run(service.restore_empty_job('../job_escape'))
    service._jobs['other'] = object()
    with pytest.raises(RuntimeError, match='idle'):
        asyncio.run(service.restore_empty_job(JOB))


@pytest.mark.parametrize('enabled,cap,expected', [(False, (12, 0), False), (True, (8, 0), False), (True, (12, 0), True)])
def test_vae_preload_is_opt_in_and_only_for_sm120(monkeypatch, enabled, cap, expected):
    source = ast.parse((ROOT / 'scripts/run_longcat_avatar_batch.py').read_text())
    node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == '_preload_vae')
    cuda = SimpleNamespace(is_available=lambda: True, get_device_capability=lambda _: cap, synchronize=Mock())
    namespace = dict(os=os, torch=SimpleNamespace(cuda=cuda), Any=Any)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<preload>', 'exec'), namespace)
    monkeypatch.setenv('LONGCAT_PRELOAD_VAE', '1' if enabled else '0')
    vae = SimpleNamespace(to=Mock())
    namespace['_preload_vae'](vae, 0)
    assert vae.to.called is expected
    if expected:
        vae.to.assert_called_once_with(0, non_blocking=False)
        cuda.synchronize.assert_called_once_with(0)


@pytest.mark.parametrize('enabled,cap,serial', [(False, (12, 0), False), (True, (8, 0), False), (True, (12, 0), True)])
def test_serial_upload_preserves_components_lora_and_upstream_default(monkeypatch, enabled, cap, serial):
    source = ast.parse((ROOT / 'scripts/run_longcat_avatar_batch.py').read_text())
    node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == '_upload_pipeline')
    events = []
    def component(name):
        obj = SimpleNamespace()
        def move(rank, **kwargs):
            assert rank == 0 and kwargs == {'non_blocking': False}
            events.append(name)
            return obj
        obj.to = move
        return obj
    pipe = SimpleNamespace(dit=component('dit'), text_encoder=component('text'), vae=component('vae'), to=Mock())
    pipe.dit.lora_dict = {'dmd': SimpleNamespace(loras=[component('lora')])}
    cuda = SimpleNamespace(get_device_capability=lambda _: cap, synchronize=lambda rank: events.append('fence'))
    namespace = dict(os=os, torch=SimpleNamespace(cuda=cuda), Any=Any)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<upload>', 'exec'), namespace)
    monkeypatch.setenv('LONGCAT_SERIAL_MODEL_UPLOAD', '1' if enabled else '0')
    namespace['_upload_pipeline'](pipe, 0)
    if serial:
        assert events == ['dit', 'lora', 'fence', 'text', 'fence', 'vae', 'fence']
        assert pipe.device == 0
        pipe.to.assert_not_called()
    else:
        assert events == []
        pipe.to.assert_called_once_with(0)


def test_longcat_log_is_readable_before_process_completion(tmp_path):
    async def run():
        stream = asyncio.StreamReader()
        process = SimpleNamespace(stdout=stream, wait=AsyncMock(return_value=0))
        target = tmp_path / 'live.log'
        task = asyncio.create_task(LongCatAvatarAdapter._capture_process_log(process, target))
        stream.feed_data(b'uploading\n')
        await asyncio.sleep(0)
        assert target.read_bytes() == b'uploading\n'
        assert not task.done()
        stream.feed_data(b'done\n')
        stream.feed_eof()
        assert await task == 'uploading\ndone\n'
        process.wait.assert_awaited_once()
    asyncio.run(run())


def test_silent_longcat_worker_is_killed_and_log_retained(tmp_path, monkeypatch):
    async def run():
        stream = asyncio.StreamReader()
        process = SimpleNamespace(stdout=stream, pid=987654, kill=Mock(),
                                  wait=AsyncMock(return_value=-9),
                                  communicate=AsyncMock(return_value=(b'last diagnostic\n', None)))
        kill_group = Mock()
        monkeypatch.setattr(os, 'killpg', kill_group, raising=False)
        target = tmp_path / 'stalled.log'
        stream.feed_data(b'uploading\n')
        result = await LongCatAvatarAdapter._capture_process_log(process, target, timeout_sec=0.01)
        assert 'uploading' in result and 'LongCat worker produced no output' in result
        assert 'last diagnostic' in result
        if os.name == 'posix':
            kill_group.assert_called_once()
        else:
            process.kill.assert_called_once()
    asyncio.run(run())


def test_encoder_offload_preserves_embeddings_and_call_arguments(monkeypatch):
    source = ast.parse((ROOT / 'scripts/run_longcat_avatar_batch.py').read_text())
    node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == '_configure_encoder_offload')
    events = []
    text = SimpleNamespace(to=lambda device, **kw: events.append(('text', device)))
    audio = SimpleNamespace(to=lambda device, **kw: events.append(('audio', device)))
    embeddings = object()
    def encode(*args, **kwargs):
        events.append(('encode', args, kwargs))
        return embeddings
    pipe = SimpleNamespace(text_encoder=text, audio_encoder=audio, encode_prompt=encode, get_audio_embedding=encode)
    namespace = dict(os=os, Any=Any, wraps=wraps, torch=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)))
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<offload>', 'exec'), namespace)
    monkeypatch.setenv('LONGCAT_OFFLOAD_ENCODERS', '1')
    namespace['_configure_encoder_offload'](pipe, 0)
    assert events == [('text', 'cpu'), ('audio', 'cpu')]
    events.clear()
    assert pipe.encode_prompt('scene', device=0) is embeddings
    assert events == [('text', 0), ('encode', ('scene',), {'device': 0}), ('text', 'cpu')]
    events.clear()
    assert pipe.get_audio_embedding('audio') is embeddings
    assert events == [('audio', 0), ('encode', ('audio',), {}), ('audio', 'cpu')]


@pytest.mark.parametrize('device,providers,expected', [
    ('cuda', None, 'cpu'), ('cuda:0', ['CPUExecutionProvider'], 'cpu'),
    ('cuda', ['CUDAExecutionProvider', 'CPUExecutionProvider'], 'cuda'),
    ('mps', ['CoreMLExecutionProvider'], 'mps'),
])
def test_separator_fft_matches_actual_provider(device, providers, expected):
    source = ast.parse((ROOT / 'scripts/run_longcat_avatar_batch.py').read_text())
    node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == '_configure_separator_device')
    namespace = dict(Any=Any)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<separator>', 'exec'), namespace)
    separator = SimpleNamespace(torch_device=device, torch_device_cpu='cpu', onnx_execution_provider=providers)
    namespace['_configure_separator_device'](separator)
    assert separator.torch_device == expected
    if expected == 'cpu':
        assert separator.onnx_execution_provider == ['CPUExecutionProvider']
