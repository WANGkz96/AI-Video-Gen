import asyncio
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from backend.app.services.jobs import JobService

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
