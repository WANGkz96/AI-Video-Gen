from __future__ import annotations

import subprocess
from typing import Any


def read_gpu_telemetry() -> dict[str, Any]:
    """Return a small, non-sensitive snapshot for runtime admission and logs."""

    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "available": False,
            "error": f"Unable to query nvidia-smi: {exc}",
        }

    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not rows:
        return {"available": False, "error": "nvidia-smi returned no GPUs."}

    gpus: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        parts = [part.strip() for part in row.split(",")]
        if len(parts) < 5:
            continue
        try:
            total_mib = float(parts[1])
            used_mib = float(parts[2])
            free_mib = float(parts[3])
            utilization_percent = float(parts[4])
        except ValueError:
            continue
        gpus.append(
            {
                "index": index,
                "name": parts[0],
                "totalVramGb": round(total_mib / 1024, 2),
                "usedVramGb": round(used_mib / 1024, 2),
                "freeVramGb": round(free_mib / 1024, 2),
                "utilizationPercent": round(utilization_percent, 2),
            }
        )

    if not gpus:
        return {"available": False, "error": "Unable to parse nvidia-smi output."}
    return {
        "available": True,
        "gpuCount": len(gpus),
        "gpus": gpus,
        **gpus[0],
    }
