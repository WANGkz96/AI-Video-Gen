from __future__ import annotations

import argparse
import mimetypes
import random
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Serve a local HuggingFace-compatible file mirror with optional "
            "bandwidth limits and download turbulence."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--mirror-root", required=True, help="Root with repo_id/repo_file layout.")
    parser.add_argument("--rate-mbps", type=float, default=0.0, help="Average per-connection speed. 0 = unlimited.")
    parser.add_argument("--jitter", type=float, default=0.0, help="0..1 random delay variation around rate limit.")
    parser.add_argument("--chunk-kb", type=int, default=1024)
    parser.add_argument("--latency-ms", type=float, default=0.0, help="Fixed latency before every response.")
    parser.add_argument("--stall-every-mb", type=float, default=0.0, help="Pause after each N MB sent. 0 = disabled.")
    parser.add_argument("--stall-duration-sec", type=float, default=0.0)
    parser.add_argument("--fail-after-mb", type=float, default=0.0, help="Close connection after N MB sent. 0 = disabled.")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def safe_resolve(root: Path, relative: str) -> Path | None:
    candidate = (root / relative).resolve()
    root = root.resolve()
    if candidate == root or root in candidate.parents:
        return candidate
    return None


def resolve_hf_path(mirror_root: Path, request_path: str) -> tuple[Path | None, str]:
    parts = [unquote(part) for part in request_path.strip("/").split("/") if part]
    try:
        resolve_index = parts.index("resolve")
    except ValueError:
        return None, "Path must include /resolve/main/."
    if resolve_index < 2 or len(parts) <= resolve_index + 2:
        return None, "Path must look like /owner/repo/resolve/main/file."
    revision = parts[resolve_index + 1]
    if revision != "main":
        return None, "Only /resolve/main/ is supported by the local test mirror."
    repo_id = "/".join(parts[:resolve_index])
    repo_file = "/".join(parts[resolve_index + 2 :])
    target = safe_resolve(mirror_root, f"{repo_id}/{repo_file}")
    return target, ""


class LocalMirrorHandler(BaseHTTPRequestHandler):
    server: "LocalMirrorServer"

    def log_message(self, format: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} {format % args}", flush=True)

    def do_HEAD(self) -> None:
        self._serve(send_body=False)

    def do_GET(self) -> None:
        self._serve(send_body=True)

    def _serve(self, *, send_body: bool) -> None:
        config = self.server.config
        if config.latency_ms > 0:
            time.sleep(config.latency_ms / 1000.0)

        parsed = urlparse(self.path)
        target, error = resolve_hf_path(config.mirror_root, parsed.path)
        if target is None:
            self.send_error(HTTPStatus.BAD_REQUEST, error)
            return
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, f"Mirror file not found: {target}")
            return

        file_size = target.stat().st_size
        start = 0
        end = file_size - 1
        range_header = self.headers.get("Range", "").strip()
        if range_header:
            start, end = self._parse_range(range_header, file_size)
            if start is None or end is None:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return

        status = HTTPStatus.PARTIAL_CONTENT if range_header else HTTPStatus.OK
        content_length = max(0, end - start + 1)
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(content_length))
        if range_header:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()

        if not send_body or content_length <= 0:
            return

        self._send_file(target, start=start, end=end)

    def _parse_range(self, value: str, file_size: int) -> tuple[int | None, int | None]:
        if not value.startswith("bytes="):
            return None, None
        raw = value[len("bytes=") :].split(",", 1)[0].strip()
        if "-" not in raw:
            return None, None
        left, right = raw.split("-", 1)
        if left == "":
            suffix = int(right)
            start = max(0, file_size - suffix)
            end = file_size - 1
        else:
            start = int(left)
            end = int(right) if right else file_size - 1
        if start < 0 or end < start or start >= file_size:
            return None, None
        return start, min(end, file_size - 1)

    def _send_file(self, path: Path, *, start: int, end: int) -> None:
        config = self.server.config
        chunk_size = max(1, config.chunk_kb * 1024)
        rate_bps = config.rate_mbps * 1024 * 1024 / 8 if config.rate_mbps > 0 else 0.0
        stall_every_bytes = int(config.stall_every_mb * 1024 * 1024) if config.stall_every_mb > 0 else 0
        fail_after_bytes = int(config.fail_after_mb * 1024 * 1024) if config.fail_after_mb > 0 else 0
        sent = 0
        next_stall_at = stall_every_bytes if stall_every_bytes > 0 else 0
        with path.open("rb") as file:
            file.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                data = file.read(min(chunk_size, remaining))
                if not data:
                    return
                before = time.monotonic()
                try:
                    self.wfile.write(data)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                sent += len(data)
                remaining -= len(data)

                if fail_after_bytes > 0 and sent >= fail_after_bytes:
                    self.close_connection = True
                    return

                if stall_every_bytes > 0 and sent >= next_stall_at:
                    time.sleep(max(0.0, config.stall_duration_sec))
                    next_stall_at += stall_every_bytes

                if rate_bps > 0:
                    target_delay = len(data) / rate_bps
                    jitter = 1.0 + self.server.random.uniform(-config.jitter, config.jitter)
                    target_delay = max(0.0, target_delay * jitter)
                    elapsed = time.monotonic() - before
                    if target_delay > elapsed:
                        time.sleep(target_delay - elapsed)


class LocalMirrorServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], config: argparse.Namespace):
        super().__init__(server_address, LocalMirrorHandler)
        self.config = config
        self.random = random.Random(config.seed)


def main() -> None:
    config = build_parser().parse_args()
    config.mirror_root = Path(config.mirror_root).expanduser().resolve()
    config.mirror_root.mkdir(parents=True, exist_ok=True)
    server = LocalMirrorServer((config.host, config.port), config)
    base_url = f"http://{config.host}:{config.port}"
    print(f"[mirror] serving {config.mirror_root} at {base_url}", flush=True)
    print("[mirror] use downloader base URL:", base_url, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[mirror] stopped", flush=True)


if __name__ == "__main__":
    main()
