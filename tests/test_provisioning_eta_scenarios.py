from __future__ import annotations

import unittest
from collections import deque
from types import SimpleNamespace

from scripts import monitor_provisioning_eta as monitor


GIB = 1024**3
MODEL_TOTALS_GIB = [8.799, 27.144, 7.083, 0.585, 0.927]
DEFAULT_TOTAL_GIB = sum(MODEL_TOTALS_GIB)


def args(**overrides):
    values = {
        "interval_sec": 10.0,
        "window_sec": 600.0,
        "warmup_sec": 180.0,
        "min_window_bytes_mb": 500.0,
        "bad_eta_min": 60.0,
        "zero_progress_sec": 600.0,
        "transition_grace_sec": 300.0,
        "near_finish_percent": 90.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_payload(
    downloaded_gib: float,
    *,
    total_gib: float = DEFAULT_TOTAL_GIB,
    status: str = "downloading",
    current: bool = True,
) -> dict:
    remaining = int(downloaded_gib * GIB)
    model_totals = [int(value * GIB) for value in MODEL_TOTALS_GIB]
    if total_gib != DEFAULT_TOTAL_GIB:
        model_totals = [int(total_gib * GIB)]

    models = []
    current_model = None
    for index, total_bytes in enumerate(model_totals, start=1):
        bytes_downloaded = min(max(remaining, 0), total_bytes)
        remaining -= bytes_downloaded
        if status == "ready" or bytes_downloaded >= total_bytes:
            model_status = "ready"
        elif bytes_downloaded > 0 and current:
            model_status = status
        elif bytes_downloaded > 0:
            model_status = "ready"
        else:
            model_status = "missing"
        model = {
            "id": f"model_{index}",
            "label": f"Model {index}",
            "status": model_status,
            "bytesDownloaded": bytes_downloaded,
            "totalBytes": total_bytes,
        }
        models.append(model)
        if current and current_model is None and model_status not in {"ready", "missing"}:
            current_model = {
                "id": model["id"],
                "label": model["label"],
                "index": index,
                "total": len(model_totals),
                "bytesDownloaded": bytes_downloaded,
                "totalBytes": total_bytes,
            }

    downloaded_bytes = sum(int(model["bytesDownloaded"]) for model in models)
    total_bytes = sum(int(model["totalBytes"]) for model in models)
    ready_count = sum(1 for model in models if model["status"] == "ready")
    return {
        "status": status,
        "progressPercent": round((downloaded_bytes / total_bytes) * 100, 2) if total_bytes else 0,
        "modelFilesReady": ready_count,
        "modelFilesTotal": len(models),
        "models": models,
        "current": current_model if status != "error" else None,
    }


class ProvisioningEtaScenarioTests(unittest.TestCase):
    def run_points(self, points, *, payload_kwargs=None, monitor_args=None):
        samples = deque()
        reports = []
        payload_kwargs = payload_kwargs or {}
        monitor_args = monitor_args or args()
        original_time = monitor.time.time
        try:
            for ts, downloaded_gib in points:
                monitor.time.time = lambda value=ts: value
                payload = make_payload(downloaded_gib, **payload_kwargs)
                reports.append(monitor.evaluate(payload, samples, monitor_args))
        finally:
            monitor.time.time = original_time
        return reports

    def assert_no_recycle(self, reports):
        self.assertEqual([report["decision"] for report in reports if report["decision"] == "recycle"], [])

    def test_fast_download_keeps_instance(self):
        reports = self.run_points([(0, 0), (60, 2.1), (180, 6.3), (300, 10.5)])

        self.assert_no_recycle(reports)
        self.assertEqual(reports[-1]["decision"], "keep")
        self.assertLess(reports[-1]["etaSec"], 20 * 60)

    def test_medium_download_keeps_instance(self):
        reports = self.run_points([(0, 0), (180, 2.64), (600, 8.8)])

        self.assert_no_recycle(reports)
        self.assertEqual(reports[-1]["decision"], "keep")
        self.assertLess(reports[-1]["etaSec"], 60 * 60)

    def test_slow_but_acceptable_download_keeps_instance(self):
        reports = self.run_points([(0, 0), (180, 2.25), (600, 7.5)])

        self.assert_no_recycle(reports)
        self.assertEqual(reports[-1]["decision"], "keep")
        self.assertLess(reports[-1]["etaSec"], 60 * 60)

    def test_bursty_download_with_short_stalls_keeps_instance(self):
        reports = self.run_points(
            [(0, 0), (60, 2), (120, 4), (240, 4), (300, 5), (420, 9), (600, 12)]
        )

        self.assert_no_recycle(reports)
        self.assertEqual(reports[-1]["decision"], "keep")

    def test_too_slow_download_recycles_after_stable_eta(self):
        reports = self.run_points([(0, 0), (180, 0.54), (600, 1.76)])

        self.assertEqual(reports[-1]["decision"], "recycle")
        self.assertEqual(reports[-1]["reason"], "stable ETA exceeds configured limit")

    def test_large_download_with_bad_eta_recycles(self):
        reports = self.run_points(
            [(0, 0), (180, 2.1), (600, 7.03)],
            payload_kwargs={"total_gib": 100.0},
        )

        self.assertEqual(reports[-1]["decision"], "recycle")
        self.assertEqual(reports[-1]["reason"], "stable ETA exceeds configured limit")

    def test_no_byte_progress_recycles_after_threshold(self):
        reports = self.run_points([(0, 5), (660, 5)])

        self.assertEqual(reports[-1]["decision"], "recycle")
        self.assertEqual(reports[-1]["reason"], "no byte progress in rolling window")

    def test_short_transition_between_files_does_not_recycle(self):
        reports = self.run_points(
            [(0, 35.943), (240, 35.943)],
            payload_kwargs={"current": False},
        )

        self.assert_no_recycle(reports)
        self.assertEqual(reports[-1]["decision"], "keep")

    def test_long_transition_between_files_recycles(self):
        reports = self.run_points(
            [(0, 35.943), (360, 35.943)],
            payload_kwargs={"current": False},
        )

        self.assertEqual(reports[-1]["decision"], "recycle")
        self.assertEqual(reports[-1]["reason"], "transition between files exceeded grace window")

    def test_downloader_error_recycles_immediately(self):
        reports = self.run_points([(0, 0)], payload_kwargs={"status": "error", "current": False})

        self.assertEqual(reports[-1]["decision"], "recycle")
        self.assertEqual(reports[-1]["reason"], "downloader reported error")


if __name__ == "__main__":
    unittest.main()
