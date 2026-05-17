from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx


def env_path(key: str, default: str) -> Path:
    return Path(os.getenv(key, default)).expanduser().resolve()


def main() -> None:
    api_url = os.getenv("GENERATOR_API_URL", "http://127.0.0.1:18188").rstrip("/")
    timeout_sec = float(os.getenv("COMFYUI_READY_TIMEOUT_SEC", "900"))
    poll_sec = max(1.0, float(os.getenv("COMFYUI_READY_POLL_SEC", "5")))
    workflow_paths = [
        env_path(
            "COMFYUI_T2V_WORKFLOW",
            "/workspace/ComfyUI/blueprints/Text to Video (LTX-2.3).json",
        ),
        env_path(
            "COMFYUI_I2V_WORKFLOW",
            "/workspace/ComfyUI/blueprints/Image to Video (LTX-2.3).json",
        ),
    ]
    deadline = time.monotonic() + timeout_sec
    last_error = ""

    while time.monotonic() < deadline:
        try:
            missing = [path for path in workflow_paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "Missing ComfyUI workflow blueprint(s): "
                    + ", ".join(path.as_posix() for path in missing)
                )

            with httpx.Client(timeout=30.0) as client:
                stats = client.get(f"{api_url}/system_stats")
                stats.raise_for_status()

                for workflow_path in workflow_paths:
                    payload = json.loads(workflow_path.read_text(encoding="utf-8"))
                    response = client.post(f"{api_url}/workflow/convert", json=payload)
                    response.raise_for_status()
                    converted = response.json()
                    if not isinstance(converted, dict) or not converted:
                        raise RuntimeError(
                            f"Unexpected /workflow/convert response for {workflow_path}"
                        )

            print(f"[ready] ComfyUI API and workflow converter are ready at {api_url}")
            return
        except Exception as exc:
            last_error = str(exc)
            print(f"[wait] ComfyUI is not ready yet: {last_error}", flush=True)
            time.sleep(poll_sec)

    raise SystemExit(
        f"ComfyUI did not become ready within {timeout_sec:.0f}s at {api_url}. "
        f"Last error: {last_error}"
    )


if __name__ == "__main__":
    main()
