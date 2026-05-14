from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app.adapters.base import AdapterUnavailableError, BaseGeneratorAdapter
from backend.app.models import AdapterInfo, GenerationArtifact, SegmentGenerationRequest


def _natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def _normalize_prompt(value: str | None) -> str:
    if not value:
        return ""

    text = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]

    normalized: list[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if not previous_blank:
                normalized.append("")
            previous_blank = True
            continue

        normalized.append(line)
        previous_blank = False

    return "\n".join(normalized).strip()


def _prompt_hash(value: str) -> str | None:
    if not value:
        return None
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _orientation_for_size(width: Any, height: Any) -> str:
    try:
        w = int(width)
        h = int(height)
    except (TypeError, ValueError):
        return "unknown"
    if w > h:
        return "landscape"
    if h > w:
        return "portrait"
    return "square"


def _request_orientation(request: SegmentGenerationRequest) -> str:
    return _orientation_for_size(request.width, request.height)


class MockGenAdapter(BaseGeneratorAdapter):
    def __init__(self, mock_media_dir: Path) -> None:
        self._mock_media_dir = mock_media_dir
        self._index_path = mock_media_dir / "mock-media.json"
        self._state_path = mock_media_dir / "mock-gen-state.json"
        self._index_mtime_ns: int | None = None
        self._prompt_groups: dict[str, dict[str, Any]] = {}
        self._fallback_files: list[dict[str, Any]] = []
        self._legacy_files: list[Path] = []
        self._prompt_cursors: defaultdict[str, int] = defaultdict(int)
        self._fallback_cursor = 0
        self._legacy_cursor = 0
        self._lock = asyncio.Lock()
        self._load_state()
        self._refresh_index(force=True)

    def info(self) -> AdapterInfo:
        self._refresh_index()
        available = self._has_any_media()
        return AdapterInfo(
            key="mock-gen",
            label="Mock Generator",
            description=(
                "Prompt-aware round-robin copier over mock-media assets, "
                "with fallback videos for unmatched prompts."
            ),
            status="ready",
            available=available,
            notes=(
                None
                if available
                else (
                    "No indexed prompt videos or fallback .mp4 files were found. "
                    f"Expected index at {self._index_path.as_posix()}."
                )
            ),
        )

    async def generate_segment(
        self, request: SegmentGenerationRequest
    ) -> GenerationArtifact:
        async with self._lock:
            self._refresh_index()
            source, selection = self._select_source(request)
            self._write_state()

        request.outputPath.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copy2, source, request.outputPath)

        return GenerationArtifact(
            modelName="mock-gen",
            modelVersion="simulated",
            outputPath=request.outputPath,
            debug={
                **selection,
                "sourceFile": source.name,
                "sourcePath": source.as_posix(),
            },
        )

    def _has_any_media(self) -> bool:
        return bool(self._prompt_groups or self._fallback_files or self._legacy_files)

    def _load_state(self) -> None:
        if not self._state_path.is_file():
            return

        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception:
            return

        prompt_cursors = data.get("promptCursors")
        if isinstance(prompt_cursors, dict):
            for key, value in prompt_cursors.items():
                try:
                    cursor = int(value)
                except (TypeError, ValueError):
                    continue
                if cursor >= 0:
                    self._prompt_cursors[str(key)] = cursor

        try:
            self._fallback_cursor = max(0, int(data.get("fallbackCursor", 0)))
        except (TypeError, ValueError):
            self._fallback_cursor = 0

        try:
            self._legacy_cursor = max(0, int(data.get("legacyCursor", 0)))
        except (TypeError, ValueError):
            self._legacy_cursor = 0

    def _write_state(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schemaVersion": "ai-video-gen.mock-gen-state.v1",
            "updatedAtUtc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "promptCursors": dict(sorted(self._prompt_cursors.items())),
            "fallbackCursor": self._fallback_cursor,
            "legacyCursor": self._legacy_cursor,
        }
        temp_path = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp_path.replace(self._state_path)

    def _refresh_index(self, *, force: bool = False) -> None:
        self._legacy_files = sorted(
            [path for path in self._mock_media_dir.glob("*.mp4") if path.is_file()],
            key=lambda item: _natural_key(item.name),
        )

        try:
            mtime_ns = self._index_path.stat().st_mtime_ns
        except FileNotFoundError:
            if force or self._index_mtime_ns is not None:
                self._index_mtime_ns = None
                self._prompt_groups = {}
                self._fallback_files = self._scan_fallback_dir()
            return

        if not force and self._index_mtime_ns == mtime_ns:
            return

        data = json.loads(self._index_path.read_text(encoding="utf-8"))
        self._prompt_groups = self._load_prompt_groups(data)
        self._fallback_files = self._load_fallback_files(data)
        self._index_mtime_ns = mtime_ns

    def _load_prompt_groups(self, data: dict[str, Any]) -> dict[str, dict[str, Any]]:
        prompt_groups: dict[str, dict[str, Any]] = {}
        groups = data.get("groups") if isinstance(data.get("groups"), list) else []

        for group in groups:
            if not isinstance(group, dict):
                continue

            prompt = _normalize_prompt(group.get("prompt"))
            if not prompt:
                continue

            files = self._resolve_group_files(group)
            if not files:
                continue

            bucket = prompt_groups.setdefault(
                prompt,
                {
                    "promptId": group.get("promptId") or _prompt_hash(prompt),
                    "prompt": prompt,
                    "files": [],
                },
            )
            bucket["files"].extend(files)

        for group in prompt_groups.values():
            seen: set[Path] = set()
            unique_files: list[dict[str, Any]] = []
            for file_entry in group["files"]:
                file_path = file_entry["path"]
                if file_path in seen:
                    continue
                seen.add(file_path)
                unique_files.append(file_entry)
            group["files"] = sorted(unique_files, key=lambda item: _natural_key(item["path"].name))

        return prompt_groups

    def _resolve_group_files(self, group: dict[str, Any]) -> list[dict[str, Any]]:
        raw_videos = group.get("videos")
        if isinstance(raw_videos, list):
            items = [item for item in raw_videos if isinstance(item, dict) and isinstance(item.get("file"), str)]
        else:
            items = [{"file": item} for item in group.get("files", []) if isinstance(item, str)]

        files: list[dict[str, Any]] = []
        for item in items:
            name = item.get("file")
            candidate = (self._mock_media_dir / name).resolve()
            if candidate.is_file():
                files.append({
                    "path": candidate,
                    "orientation": item.get("orientation") or _orientation_for_size(item.get("width"), item.get("height")),
                    "width": item.get("width"),
                    "height": item.get("height"),
                })
        return files

    def _load_fallback_files(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        fallback = data.get("fallback") if isinstance(data.get("fallback"), dict) else {}
        raw_videos = fallback.get("videos")
        if isinstance(raw_videos, list):
            items = [item for item in raw_videos if isinstance(item, dict) and isinstance(item.get("file"), str)]
        else:
            items = [{"file": item} for item in fallback.get("files", []) if isinstance(item, str)]

        files: list[dict[str, Any]] = []
        for item in items:
            name = item.get("file")
            candidate = (self._mock_media_dir / "fallback" / name).resolve()
            if candidate.is_file():
                files.append({
                    "path": candidate,
                    "orientation": item.get("orientation") or _orientation_for_size(item.get("width"), item.get("height")),
                    "width": item.get("width"),
                    "height": item.get("height"),
                })

        return sorted(files, key=lambda item: _natural_key(item["path"].name)) or self._scan_fallback_dir()

    def _scan_fallback_dir(self) -> list[dict[str, Any]]:
        fallback_dir = self._mock_media_dir / "fallback"
        return [
            {"path": path, "orientation": "unknown", "width": None, "height": None}
            for path in sorted(
                [path for path in fallback_dir.glob("*.mp4") if path.is_file()],
                key=lambda item: _natural_key(item.name),
            )
        ]

    def _select_source(self, request: SegmentGenerationRequest) -> tuple[Path, dict[str, Any]]:
        for match_type, raw_prompt in (
            ("resolvedPrompt", request.resolvedPrompt),
            ("prompt", request.prompt),
        ):
            prompt = _normalize_prompt(raw_prompt)
            if not prompt:
                continue

            group = self._prompt_groups.get(prompt)
            if not group:
                continue

            requested_orientation = _request_orientation(request)
            files = self._filter_by_orientation(group["files"], requested_orientation)
            prompt_id = str(group.get("promptId") or _prompt_hash(prompt) or prompt)
            cursor_key = f"{prompt_id}:{requested_orientation}"
            cursor = self._prompt_cursors[cursor_key]
            selection_index = cursor % len(files)
            self._prompt_cursors[cursor_key] = cursor + 1
            source_entry = files[selection_index]
            source = source_entry["path"]
            return source, {
                "mockMode": "prompt-index",
                "fallbackUsed": False,
                "matchType": match_type,
                "requestedPromptHash": _prompt_hash(prompt),
                "matchedPromptId": prompt_id,
                "cursorKey": cursor_key,
                "requestedOrientation": requested_orientation,
                "selectedOrientation": source_entry.get("orientation"),
                "selectionIndex": selection_index,
                "selectionCount": len(files),
                "promptCursor": cursor,
                "stateFile": self._state_path.as_posix(),
            }

        if self._fallback_files:
            requested_orientation = _request_orientation(request)
            fallback_files = self._filter_by_orientation(self._fallback_files, requested_orientation)
            cursor = self._fallback_cursor
            selection_index = cursor % len(fallback_files)
            self._fallback_cursor = cursor + 1
            source_entry = fallback_files[selection_index]
            source = source_entry["path"]
            requested_prompt = _normalize_prompt(request.resolvedPrompt or request.prompt)
            return source, {
                "mockMode": "fallback",
                "fallbackUsed": True,
                "matchType": "fallback",
                "requestedPromptHash": _prompt_hash(requested_prompt),
                "requestedOrientation": requested_orientation,
                "selectedOrientation": source_entry.get("orientation"),
                "selectionIndex": selection_index,
                "selectionCount": len(fallback_files),
                "fallbackCursor": cursor,
                "stateFile": self._state_path.as_posix(),
            }

        if self._legacy_files and not self._index_path.exists():
            cursor = self._legacy_cursor
            selection_index = cursor % len(self._legacy_files)
            self._legacy_cursor = cursor + 1
            source = self._legacy_files[selection_index]
            return source, {
                "mockMode": "legacy-round-robin",
                "fallbackUsed": False,
                "matchType": "legacy",
                "selectionIndex": selection_index,
                "selectionCount": len(self._legacy_files),
                "legacyCursor": cursor,
                "stateFile": self._state_path.as_posix(),
            }

        requested = _normalize_prompt(request.resolvedPrompt or request.prompt)
        requested_preview = requested[:240].replace("\n", "\\n")
        raise AdapterUnavailableError(
            "No mock video matched the request prompt and no fallback videos are available. "
            f"Add .mp4 files to {(self._mock_media_dir / 'fallback').as_posix()} and rebuild "
            f"{self._index_path.name}. Requested prompt hash: {_prompt_hash(requested)}; "
            f"preview: {requested_preview}"
        )

    def _filter_by_orientation(
        self,
        files: list[dict[str, Any]],
        requested_orientation: str,
    ) -> list[dict[str, Any]]:
        if requested_orientation in {"portrait", "landscape"}:
            filtered = [
                item for item in files
                if item.get("orientation") == requested_orientation
            ]
            if filtered:
                return filtered
        return files
