from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Watch AI_VIDEO_GEN_PROVISIONING_STATUS and estimate download ETA from rolling progress."
    )
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--interval-sec", type=float, default=5.0)
    parser.add_argument("--window-sec", type=float, default=600.0)
    parser.add_argument("--warmup-sec", type=float, default=180.0)
    parser.add_argument("--min-window-bytes-mb", type=float, default=500.0)
    parser.add_argument("--bad-eta-min", type=float, default=60.0)
    parser.add_argument("--zero-progress-sec", type=float, default=600.0)
    parser.add_argument("--transition-grace-sec", type=float, default=300.0)
    parser.add_argument("--near-finish-percent", type=float, default=90.0)
    parser.add_argument("--jsonl", action="store_true")
    parser.add_argument("--once", action="store_true", help="Read the status file once and exit.")
    return parser


def parse_ts(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def fmt_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "-"
    minutes = seconds / 60.0
    if minutes < 1:
        return f"{seconds:.0f}s"
    if minutes < 120:
        return f"{minutes:.1f}m"
    return f"{minutes / 60.0:.1f}h"


def total_downloaded(payload: dict[str, Any]) -> tuple[int, int | None, int, int]:
    downloaded = 0
    total_known = 0
    total_missing = False
    ready = 0
    total_files = 0
    for model in payload.get("models") or []:
        total_files += 1
        status = str(model.get("status") or "")
        bytes_downloaded = int(model.get("bytesDownloaded") or 0)
        total_bytes = model.get("totalBytes")
        if status == "ready":
            ready += 1
            downloaded += bytes_downloaded
            total_known += bytes_downloaded
        elif total_bytes:
            downloaded += min(bytes_downloaded, int(total_bytes))
            total_known += int(total_bytes)
        else:
            downloaded += bytes_downloaded
            total_missing = True
    return downloaded, None if total_missing else total_known, ready, total_files


def evaluate(payload: dict[str, Any], samples: deque[tuple[float, int]], args: argparse.Namespace) -> dict[str, Any]:
    now = time.time()
    status_updated = parse_ts(payload.get("updatedAt")) or now
    downloaded, total_bytes, ready_count, total_count = total_downloaded(payload)
    progress_percent = float(payload.get("progressPercent") or 0.0)
    status = str(payload.get("status") or "unknown")
    current = payload.get("current") or None

    samples.append((now, downloaded))
    retention_sec = max(args.window_sec, args.zero_progress_sec, args.transition_grace_sec)
    retention_sec += max(1.0, args.interval_sec) * 2
    while len(samples) > 2 and now - samples[0][0] > retention_sec:
        samples.popleft()

    speed_samples = [sample for sample in samples if now - sample[0] <= args.window_sec]
    if len(speed_samples) < 2 and len(samples) >= 2:
        speed_samples = list(samples)[-2:]
    first_sample = speed_samples[0] if speed_samples else (now, downloaded)
    window_sec = max(0.0, now - first_sample[0])
    window_bytes = max(0, downloaded - first_sample[1])
    speed_bps = window_bytes / window_sec if window_sec > 0 else 0.0
    remaining_bytes = max(0, total_bytes - downloaded) if total_bytes is not None else None
    eta_sec = remaining_bytes / speed_bps if remaining_bytes is not None and speed_bps > 0 else None

    last_progress_at = None
    if len(samples) <= 1:
        last_progress_at = samples[-1][0] if downloaded > 0 else None
    else:
        previous_bytes = samples[0][1]
        for ts, bytes_value in list(samples)[1:]:
            if bytes_value > previous_bytes:
                last_progress_at = ts
            previous_bytes = bytes_value
        if downloaded > 0 and last_progress_at is None:
            last_progress_at = samples[0][0]
    zero_progress_for = now - last_progress_at if last_progress_at is not None else None

    observed_sec = max(0.0, now - samples[0][0]) if samples else 0.0
    stable_window = window_sec >= args.warmup_sec or window_bytes >= args.min_window_bytes_mb * 1024 * 1024
    warmup = observed_sec < args.warmup_sec or not stable_window
    transitioning = current is None and status in {"verifying", "retrying", "downloading"} and ready_count < total_count
    near_finish = progress_percent >= args.near_finish_percent

    decision = "keep"
    reason = "progress is acceptable"
    confidence = "medium"
    if status == "ready":
        decision = "ready"
        reason = "all files are ready"
        confidence = "high"
    elif status == "error":
        decision = "recycle"
        reason = "downloader reported error"
        confidence = "high"
    elif transitioning and zero_progress_for and zero_progress_for > args.transition_grace_sec:
        decision = "recycle"
        reason = "transition between files exceeded grace window"
        confidence = "high"
    elif zero_progress_for and zero_progress_for > args.zero_progress_sec:
        decision = "recycle"
        reason = "no byte progress in rolling window"
        confidence = "high"
    elif warmup:
        decision = "warming_up"
        reason = "not enough stable download samples yet"
        confidence = "low"
    elif eta_sec and eta_sec > args.bad_eta_min * 60 and not near_finish:
        decision = "recycle"
        reason = "stable ETA exceeds configured limit"
        confidence = "high"
    elif eta_sec and eta_sec > args.bad_eta_min * 60 and near_finish:
        decision = "keep_grace"
        reason = "ETA is high but progress is near finish"
        confidence = "medium"

    return {
        "ts": datetime.now(UTC).isoformat(),
        "status": status,
        "decision": decision,
        "reason": reason,
        "confidence": confidence,
        "progressPercent": round(progress_percent, 2),
        "readyFiles": ready_count,
        "totalFiles": total_count,
        "downloadedBytes": downloaded,
        "totalBytes": total_bytes,
        "windowSec": round(window_sec, 1),
        "windowBytes": window_bytes,
        "speedBps": round(speed_bps, 2),
        "etaSec": round(eta_sec, 1) if eta_sec else None,
        "zeroProgressForSec": round(zero_progress_for, 1) if zero_progress_for is not None else None,
        "statusAgeSec": round(time.time() - status_updated, 1),
        "current": current,
    }


def print_report(report: dict[str, Any], *, jsonl: bool) -> None:
    if jsonl:
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return
    print(
        " | ".join(
            [
                report["decision"],
                report["reason"],
                f"progress={report['progressPercent']}%",
                f"files={report['readyFiles']}/{report['totalFiles']}",
                f"speed={report['speedBps'] / 1024 / 1024:.2f} MB/s",
                f"eta={fmt_duration(report['etaSec'])}",
                f"window={fmt_duration(report['windowSec'])}/{report['windowBytes'] / 1024 / 1024:.1f} MB",
            ]
        ),
        flush=True,
    )


def main() -> None:
    args = build_parser().parse_args()
    status_file = Path(args.status_file).expanduser().resolve()
    samples: deque[tuple[float, int]] = deque()
    if args.once:
        payload = json.loads(status_file.read_text(encoding="utf-8"))
        print_report(evaluate(payload, samples, args), jsonl=args.jsonl)
        return
    while True:
        try:
            payload = json.loads(status_file.read_text(encoding="utf-8"))
            report = evaluate(payload, samples, args)
            print_report(report, jsonl=args.jsonl)
        except FileNotFoundError:
            print(f"waiting for {status_file}", flush=True)
        except Exception as exc:
            print(f"monitor error: {exc}", flush=True)
        time.sleep(max(1.0, args.interval_sec))


if __name__ == "__main__":
    main()
