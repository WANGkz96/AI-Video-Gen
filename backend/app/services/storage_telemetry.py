from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from backend.app.config import Settings


def read_storage_telemetry(settings: Settings) -> dict[str, Any]:
    """Return disk pressure for the model filesystem without exposing secrets."""

    persistent_model_cache_dir = getattr(settings, "persistent_model_cache_dir", None)
    storage_profile = getattr(
        settings,
        "storage_profile",
        "persistent" if persistent_model_cache_dir else "ephemeral",
    )
    ephemeral_storage_gb = getattr(settings, "ephemeral_storage_gb", None)
    target = persistent_model_cache_dir or Path("/workspace")
    try:
        usage = shutil.disk_usage(target)
    except OSError as exc:
        return {
            "available": False,
            "profile": storage_profile,
            "path": target.as_posix(),
            "declaredCapacityGb": ephemeral_storage_gb,
            "error": f"Unable to query model storage: {exc}",
        }

    divisor = 1024 ** 3
    return {
        "available": True,
        "profile": storage_profile,
        "path": target.as_posix(),
        "declaredCapacityGb": ephemeral_storage_gb,
        "totalGb": round(usage.total / divisor, 2),
        "usedGb": round(usage.used / divisor, 2),
        "freeGb": round(usage.free / divisor, 2),
        "usedPercent": round((usage.used / usage.total) * 100, 2) if usage.total else 0.0,
    }
