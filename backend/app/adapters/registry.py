from __future__ import annotations

from backend.app.adapters.base import BaseGeneratorAdapter
from backend.app.adapters.catalog import get_diffusers_backend_specs
from backend.app.adapters.diffusers_video import DiffusersVideoAdapter
from backend.app.adapters.ltx_native import LtxNativeAdapter, get_ltx_native_backend_specs
from backend.app.config import Settings


def build_real_model_registry(settings: Settings) -> dict[str, BaseGeneratorAdapter]:
    registry: dict[str, BaseGeneratorAdapter] = {}
    for spec in get_diffusers_backend_specs().values():
        registry[spec.key] = DiffusersVideoAdapter(settings, spec)
    for spec in get_ltx_native_backend_specs().values():
        registry[spec.key] = LtxNativeAdapter(settings, spec)
    return registry


def get_downloadable_backend_keys() -> list[str]:
    return [*get_diffusers_backend_specs().keys(), *get_ltx_native_backend_specs().keys()]
