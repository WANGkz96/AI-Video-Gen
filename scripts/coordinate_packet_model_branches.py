"""Run LTX provisioning only after a mixed Packet job releases LongCat weights.

Packet's 150 GB ephemeral disk cannot safely hold the LongCat Avatar and LTX
2.5 model packs at the same time.  This coordinator is deliberately small:
it waits for JobService's post-LongCat release signal, then starts the normal
resumable LTX downloader.  It never stops or kills another downloader.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_ltx_status(path: Path, *, status: str, message: str, error: str | None = None) -> None:
    """Write the subset of the downloader status needed before LTX starts."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "schemaVersion": "ai-video-gen.provisioning.v1",
                "updatedAt": _utc_now(),
                "status": status,
                "message": message,
                "progressPercent": 0.0,
                "modelFilesReady": 0,
                "modelFilesTotal": 0,
                "models": [],
                "current": None,
                "error": error,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate the LTX download until LongCat's branch released its model weights."
    )
    parser.add_argument("--longcat-status-file", required=True)
    parser.add_argument("--ltx-status-file", required=True)
    parser.add_argument("--release-file", required=True)
    parser.add_argument(
        "--longcat-weights-dir",
        required=True,
        help="Exact LongCat weights directory that may be removed after a failed first branch.",
    )
    parser.add_argument("--poll-sec", type=float, default=2.0)
    parser.add_argument(
        "--preserve-longcat-weights",
        action="store_true",
        help="Keep LongCat weights after a failed provision when they live in durable model storage.",
    )
    parser.add_argument(
        "download_command",
        nargs=argparse.REMAINDER,
        help="Command for the normal LTX downloader, placed after --.",
    )
    args = parser.parse_args()
    if args.download_command and args.download_command[0] == "--":
        args.download_command = args.download_command[1:]
    if not args.download_command:
        parser.error("a downloader command is required after --")
    return args


def _release_failed_longcat_weights(path: Path) -> None:
    """Remove only the known model payload after LongCat provisioning failed."""

    if path.name != "weights":
        raise ValueError(f"Refusing to remove unexpected LongCat weights path: {path}")
    if path.exists():
        shutil.rmtree(path)


def _start_ltx_download(args: argparse.Namespace, status_file: Path, message: str) -> int:
    _write_ltx_status(status_file, status="starting", message=message)
    print("Packet model sequence: starting LTX downloader.", flush=True)
    return subprocess.run(args.download_command, check=False).returncode


def main() -> int:
    args = _parse_args()
    longcat_status_file = Path(args.longcat_status_file).expanduser().resolve()
    ltx_status_file = Path(args.ltx_status_file).expanduser().resolve()
    release_file = Path(args.release_file).expanduser().resolve()
    longcat_weights_dir = Path(args.longcat_weights_dir).expanduser().resolve()
    poll_sec = max(0.25, float(args.poll_sec))

    _write_ltx_status(
        ltx_status_file,
        status="waiting_for_longcat",
        message="Waiting for the LongCat generation branch to release its model weights.",
    )
    print("Packet model sequence: waiting for LongCat branch release before LTX download.", flush=True)

    while not release_file.exists():
        longcat_status = _read_json(longcat_status_file) or {}
        state = str(longcat_status.get("status") or "").strip().lower()
        error = longcat_status.get("error")
        if state in {"error", "failed"} or error:
            detail = str(error or longcat_status.get("message") or "LongCat provisioning failed.")
            if not args.preserve_longcat_weights:
                try:
                    _release_failed_longcat_weights(longcat_weights_dir)
                except Exception as exc:
                    cleanup_error = f"LongCat provisioning failed ({detail}); unable to release its weights: {exc}"
                    _write_ltx_status(
                        ltx_status_file,
                        status="error",
                        message="LTX download was not started because LongCat cleanup failed.",
                        error=cleanup_error,
                    )
                    print(f"Packet model sequence aborted: {cleanup_error}", file=sys.stderr, flush=True)
                    return 1
            return _start_ltx_download(
                args,
                ltx_status_file,
                (
                    "LongCat provisioning failed; its durable model cache was retained for resume. "
                    "Starting LTX 2.5 download."
                    if args.preserve_longcat_weights
                    else "LongCat provisioning failed and its partial weights were released; starting LTX 2.5 download."
                ),
            )
        time.sleep(poll_sec)

    return _start_ltx_download(
        args,
        ltx_status_file,
        "LongCat weights were released; starting LTX 2.5 model download.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
