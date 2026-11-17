import errno
from types import SimpleNamespace

import pytest

from scripts import download_comfy_ltx23_models as downloader


@pytest.fixture
def model(monkeypatch):
    item = downloader.ComfyModelFile("test", "Test", "owner/repo", "test.bin", "models", "test.bin")
    monkeypatch.setattr(downloader, "MODEL_FILES", [item])
    monkeypatch.setattr(downloader, "OBSOLETE_MODEL_FILES", ())
    monkeypatch.setattr(downloader, "DOWNLOAD_SPACE_RESERVE", 10)
    return item


@pytest.mark.parametrize("code", [errno.ENOSPC, getattr(errno, "EDQUOT", 122)])
def test_disk_capacity_failure_is_not_retried(tmp_path, monkeypatch, model, code):
    calls = []
    statuses = []
    partial = tmp_path / "models/test.bin.tmp"
    partial.parent.mkdir()
    partial.write_bytes(b"partial")

    def fail(*args, **kwargs):
        calls.append(1)
        raise OSError(code, "disk capacity exhausted")

    monkeypatch.setattr(downloader, "copy_hf_file_once", fail)
    monkeypatch.setattr(downloader, "write_status", lambda *a, **kw: statuses.append(kw))
    monkeypatch.setattr(downloader.time, "sleep", lambda _: pytest.fail("Must not sleep/retry"))
    with pytest.raises(OSError):
        downloader.copy_hf_file(
            model, tmp_path, force=False, status_file=None, current_index=1,
            max_attempts=60, retry_delay=20, connect_timeout=1, read_timeout=1,
            error_by_model={}, base_urls=["https://example.test"], expected_total_by_model={},
        )
    assert len(calls) == 1
    assert statuses[-1]["status"] == "error"
    assert partial.read_bytes() == b"partial"


def test_preflight_counts_resumable_partial_and_reserve(tmp_path, monkeypatch, model):
    partial = tmp_path / "models/test.bin.tmp"
    partial.parent.mkdir()
    partial.write_bytes(b"x" * 40)
    monkeypatch.setattr(downloader.shutil, "disk_usage", lambda _: SimpleNamespace(free=70))
    downloader.prepare_download_space(tmp_path, {model.id: 100}, force=False)
    monkeypatch.setattr(downloader.shutil, "disk_usage", lambda _: SimpleNamespace(free=69))
    with pytest.raises(OSError, match="insufficient disk space") as error:
        downloader.prepare_download_space(tmp_path, {model.id: 100}, force=False)
    assert error.value.errno == errno.ENOSPC
    assert partial.stat().st_size == 40


def test_preflight_checks_sum_of_parallel_downloads(tmp_path, monkeypatch, model):
    second = downloader.ComfyModelFile("second", "Second", "owner/repo", "second.bin", "models", "second.bin")
    monkeypatch.setattr(downloader, "MODEL_FILES", [model, second])
    monkeypatch.setattr(downloader.shutil, "disk_usage", lambda _: SimpleNamespace(free=150))
    with pytest.raises(OSError):
        downloader.prepare_download_space(tmp_path, {model.id: 100, second.id: 100}, force=False)


def test_migration_preserves_current_files_and_paths_outside_cache(tmp_path, monkeypatch, model):
    root = tmp_path / "ltx"
    (root / "models").mkdir(parents=True)
    active = root / "models/test.bin"
    active.write_bytes(b"current")
    obsolete = root / "models/old.bin"
    obsolete.write_bytes(b"old")
    other = tmp_path / "longcat.bin"
    other.write_bytes(b"longcat")
    monkeypatch.setattr(downloader, "OBSOLETE_MODEL_FILES", (
        "models/old.bin", "models/test.bin", "../longcat.bin",
    ))
    downloader.prepare_download_space(root, {model.id: 100}, force=False)
    assert not obsolete.exists()
    assert active.read_bytes() == b"current"
    assert other.read_bytes() == b"longcat"


def test_migration_does_not_follow_symlink(tmp_path, monkeypatch, model):
    root = tmp_path / "ltx"
    root.mkdir()
    other = tmp_path / "longcat.bin"
    other.write_bytes(b"longcat")
    link = root / "old.bin"
    try:
        link.symlink_to(other)
    except OSError:
        pytest.skip("Symlinks unavailable for this user")
    monkeypatch.setattr(downloader, "OBSOLETE_MODEL_FILES", ("old.bin",))
    downloader.prepare_download_space(root, {}, force=False)
    assert link.is_symlink()
    assert other.read_bytes() == b"longcat"


def test_transient_download_failure_still_retries(tmp_path, monkeypatch, model):
    calls = []
    def download(*args, **kwargs):
        calls.append(kwargs["base_url"])
        if len(calls) == 1:
            raise TimeoutError("network timeout")
    monkeypatch.setattr(downloader, "copy_hf_file_once", download)
    monkeypatch.setattr(downloader, "write_status", lambda *a, **kw: None)
    monkeypatch.setattr(downloader.time, "sleep", lambda _: None)
    downloader.copy_hf_file(
        model, tmp_path, force=False, status_file=None, current_index=1,
        max_attempts=2, retry_delay=20, connect_timeout=1, read_timeout=1,
        error_by_model={}, base_urls=["https://first.test", "https://second.test"], expected_total_by_model={},
    )
    assert calls == ["https://first.test", "https://second.test"]
