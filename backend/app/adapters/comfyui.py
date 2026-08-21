from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import secrets
import shutil
import subprocess
import time
import uuid
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlencode

import httpx

from backend.app.adapters.base import AdapterUnavailableError, BaseGeneratorAdapter
from backend.app.config import REPO_ROOT, Settings
from backend.app.models import AdapterInfo, GenerationArtifact, SegmentGenerationRequest
from backend.app.services.provisioning import missing_comfy_ltx23_model_files


class ComfyUiWorkflowAdapter(BaseGeneratorAdapter):
    """Run the production LTX 2.3 ComfyUI blueprints through ComfyUI's HTTP API."""

    key = "comfyui-ltx23"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._api_url = settings.generator_api_url.rstrip("/")
        self._workflow_cache: dict[Path, tuple[int, dict[str, dict]]] = {}
        self._http_timeout_sec = self._env_float("COMFYUI_HTTP_TIMEOUT_SEC", 120.0, minimum=1.0)
        self._history_request_timeout_sec = self._env_float(
            "COMFYUI_HISTORY_REQUEST_TIMEOUT_SEC",
            30.0,
            minimum=1.0,
        )
        self._output_download_timeout_sec = self._env_float(
            "COMFYUI_OUTPUT_DOWNLOAD_TIMEOUT_SEC",
            600.0,
            minimum=1.0,
        )

    @staticmethod
    def _env_float(name: str, fallback: float, *, minimum: float) -> float:
        try:
            value = float(os.getenv(name, str(fallback)))
        except (TypeError, ValueError):
            value = fallback
        return max(minimum, value)

    @staticmethod
    def _timeout(seconds: float, *, connect_cap: float = 30.0) -> httpx.Timeout:
        return httpx.Timeout(seconds, connect=max(1.0, min(connect_cap, seconds)))

    def info(self) -> AdapterInfo:
        missing = [
            path.as_posix()
            for path in [self._settings.comfyui_t2v_workflow, self._settings.comfyui_i2v_workflow]
            if not path.is_file()
        ]
        missing_models = [path.as_posix() for path in missing_comfy_ltx23_model_files(self._settings)]
        comfy_error = self._check_comfyui()
        available = not missing and not missing_models and comfy_error is None
        notes = None
        if missing:
            notes = "Missing ComfyUI workflow blueprint(s): " + ", ".join(missing)
        elif missing_models:
            notes = "Missing required ComfyUI model file(s): " + ", ".join(missing_models)
        elif comfy_error:
            notes = comfy_error

        return AdapterInfo(
            key=self.key,
            label="ComfyUI LTX 2.3",
            description=(
                "Production backend that executes the ComfyUI video_ltx2_3_t2v and "
                "video_ltx2_3_i2v workflows through the local ComfyUI API."
            ),
            status="ready",
            available=available,
            supportsBatch=True,
            supportsDirect=True,
            requiresRemote=True,
            requiresDownload=bool(missing_models),
            modelId="ComfyUI workflow: video_ltx2_3_t2v / video_ltx2_3_i2v",
            localPath=self._settings.comfyui_t2v_workflow.parent.as_posix(),
            minimumVramGb=32,
            notes=notes,
        )

    def release(self) -> None:
        """Ask ComfyUI to unload resident LTX models before another GPU backend runs."""
        try:
            response = httpx.post(
                f"{self._api_url}/free",
                json={"unload_models": True, "free_memory": True},
                timeout=self._timeout(30.0),
            )
            response.raise_for_status()
        except Exception:
            # Releasing cached models is best-effort during shutdown. The explicit
            # transition before LongCat is logged by JobService and LongCat still
            # reports a useful generation error if memory could not be reclaimed.
            return

    async def generate_segment(self, request: SegmentGenerationRequest) -> GenerationArtifact:
        request.outputPath.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=self._timeout(self._http_timeout_sec)) as client:
            prompt, workflow_kind, uploaded_image, debug = await self._build_prompt(client, request)
            prompt_id = await self._queue_prompt(client, prompt)
            history = await self._wait_for_history(client, prompt_id)
            output_info = self._extract_video_output(history)
            await self._download_output(client, output_info, request.outputPath)

        if self._settings.comfyui_normalize_output or self._settings.comfyui_strip_audio:
            await asyncio.to_thread(self._postprocess_output, request.outputPath, request, debug)

        debug.update(
            {
                "apiUrl": self._api_url,
                "promptId": prompt_id,
                "workflowKind": workflow_kind,
                "uploadedImage": uploaded_image,
                "comfyOutput": output_info,
                "elapsedSec": round(time.perf_counter() - started, 3),
            }
        )
        return GenerationArtifact(
            modelName=self.key,
            modelVersion=workflow_kind,
            outputPath=request.outputPath,
            debug=debug,
        )

    def build_workflow_payload(self, request: SegmentGenerationRequest) -> dict:
        return {
            "meta": {
                "jobId": request.jobId,
                "segmentId": request.segmentId,
                "backend": request.backend,
            },
            "inputs": {
                "prompt": request.resolvedPrompt,
                "negative_prompt": request.resolvedNegativePrompt,
                "duration_sec": request.durationSec,
                "width": request.width,
                "height": request.height,
                "fps": request.fps,
                "image": request.imagePath.as_posix() if request.imagePath else None,
                "timeline": request.timeline,
            },
            "backend_params": request.backendParams,
        }

    def _check_comfyui(self) -> str | None:
        try:
            response = httpx.get(f"{self._api_url}/system_stats", timeout=2.0)
            response.raise_for_status()
        except Exception as exc:
            return f"ComfyUI API is unavailable at {self._api_url}: {exc}"
        return None

    async def _build_prompt(
        self,
        client: httpx.AsyncClient,
        request: SegmentGenerationRequest,
    ) -> tuple[dict[str, dict], str, dict[str, object] | None, dict[str, object]]:
        use_i2v = bool(request.imagePath and request.imagePath.is_file())
        workflow_path = (
            self._settings.comfyui_i2v_workflow if use_i2v else self._settings.comfyui_t2v_workflow
        )
        workflow_kind = "video_ltx2_3_i2v" if use_i2v else "video_ltx2_3_t2v"
        prompt = deepcopy(await self._load_api_workflow(client, workflow_path))
        seed = self._resolve_seed(request)
        output_prefix = self._build_output_prefix(request)

        self._set_prompt_text(prompt, request.resolvedPrompt or request.prompt)
        self._set_negative_prompt(prompt, request.resolvedNegativePrompt or request.negativePrompt)
        self._set_primitive_int(prompt, "Width", request.width)
        self._set_primitive_int(prompt, "Height", request.height)
        self._set_primitive_int(prompt, "Duration", max(1, int(round(request.durationSec))))
        self._set_primitive_int(prompt, "Frame Rate", max(1, int(round(request.fps))))
        self._set_text_to_video_switch(prompt, enabled=not use_i2v)
        self._set_noise_seed(prompt, seed)
        self._add_save_video_node(prompt, output_prefix)

        uploaded_image = None
        image_wiring = None
        if use_i2v:
            assert request.imagePath is not None
            uploaded_image = await self._upload_image(client, request)
            image_wiring = self._wire_uploaded_image(prompt, uploaded_image["loadImageValue"])

        debug = {
            "workflowPath": workflow_path.as_posix(),
            "workflowKind": workflow_kind,
            "seed": seed,
            "outputPrefix": output_prefix,
            "requested": {
                "width": request.width,
                "height": request.height,
                "fps": request.fps,
                "durationSec": request.durationSec,
            },
            "injectedControls": self._collect_injected_controls(prompt),
            "imageWiring": image_wiring,
            "payload": self.build_workflow_payload(request),
        }
        return prompt, workflow_kind, uploaded_image, debug

    async def _load_api_workflow(
        self,
        client: httpx.AsyncClient,
        workflow_path: Path,
    ) -> dict[str, dict]:
        if not workflow_path.is_file():
            raise AdapterUnavailableError(f"ComfyUI workflow file does not exist: {workflow_path}")

        mtime_ns = workflow_path.stat().st_mtime_ns
        cached = self._workflow_cache.get(workflow_path)
        if cached and cached[0] == mtime_ns:
            return cached[1]

        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        response = await client.post(f"{self._api_url}/workflow/convert", json=workflow)
        if response.status_code == 404:
            raise AdapterUnavailableError(
                "ComfyUI /workflow/convert endpoint is unavailable. Install or enable "
                "comfyui-workflow-to-api-converter-endpoint in the ComfyUI template."
            )
        response.raise_for_status()
        api_workflow = response.json()
        if not isinstance(api_workflow, dict):
            raise AdapterUnavailableError(f"Unexpected workflow conversion response for {workflow_path}")
        self._workflow_cache[workflow_path] = (mtime_ns, api_workflow)
        return api_workflow

    async def _upload_image(
        self,
        client: httpx.AsyncClient,
        request: SegmentGenerationRequest,
    ) -> dict[str, object]:
        image_path = request.imagePath
        if not image_path or not image_path.is_file():
            raise FileNotFoundError(f"Missing first-frame image for I2V segment {request.segmentId}.")
        suffix = image_path.suffix or ".png"
        filename = f"aivg_{request.jobId}_{request.segmentId}_c{int(request.backendParams.get('candidateIndex', 1)):02d}{suffix}"
        filename = self._safe_filename(filename)
        mime_type = request.imageMimeType or mimetypes.guess_type(filename)[0] or "image/png"

        with image_path.open("rb") as file:
            response = await client.post(
                f"{self._api_url}/upload/image",
                data={"type": "input", "overwrite": "true"},
                files={"image": (filename, file, mime_type)},
            )
        response.raise_for_status()
        payload = response.json()
        name = str(payload.get("name") or filename)
        subfolder = str(payload.get("subfolder") or "")
        load_image_value = f"{subfolder}/{name}" if subfolder else name
        return {
            "sourcePath": image_path.as_posix(),
            "name": name,
            "subfolder": subfolder,
            "type": payload.get("type"),
            "loadImageValue": load_image_value,
        }

    async def _queue_prompt(self, client: httpx.AsyncClient, prompt: dict[str, dict]) -> str:
        response = await client.post(
            f"{self._api_url}/prompt",
            json={"prompt": prompt, "client_id": f"ai-video-gen-{uuid.uuid4()}"},
        )
        if response.status_code >= 400:
            raise AdapterUnavailableError(f"ComfyUI rejected prompt: {response.text}")
        payload = response.json()
        prompt_id = payload.get("prompt_id")
        if not prompt_id:
            raise AdapterUnavailableError(f"ComfyUI did not return prompt_id: {payload}")
        return str(prompt_id)

    async def _wait_for_history(
        self,
        client: httpx.AsyncClient,
        prompt_id: str,
    ) -> dict:
        timeout_sec = float(os.getenv("COMFYUI_POLL_TIMEOUT_SEC", "7200"))
        poll_sec = max(0.25, float(os.getenv("COMFYUI_POLL_SEC", "2")))
        deadline = time.monotonic() + timeout_sec
        last_payload: dict | None = None

        while time.monotonic() < deadline:
            try:
                response = await client.get(
                    f"{self._api_url}/history/{prompt_id}",
                    timeout=self._timeout(self._history_request_timeout_sec, connect_cap=10.0),
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                last_payload = {"pollError": str(exc)}
                await asyncio.sleep(poll_sec)
                continue
            payload = response.json()
            if isinstance(payload, dict) and prompt_id in payload:
                history = payload[prompt_id]
                status = history.get("status") or {}
                if status.get("status_str") == "error" or status.get("completed") is False:
                    messages = status.get("messages") or []
                    raise AdapterUnavailableError(
                        f"ComfyUI prompt {prompt_id} failed: {json.dumps(messages, ensure_ascii=False)}"
                    )
                return history
            last_payload = payload if isinstance(payload, dict) else None
            await asyncio.sleep(poll_sec)

        raise TimeoutError(f"Timed out waiting for ComfyUI prompt {prompt_id}: {last_payload}")

    def _extract_video_output(self, history: dict) -> dict[str, str]:
        outputs = history.get("outputs") or {}
        for node_output in outputs.values():
            for item in node_output.get("images") or []:
                filename = str(item.get("filename") or "")
                if filename.lower().endswith((".mp4", ".mov", ".webm", ".mkv")):
                    return {
                        "filename": filename,
                        "subfolder": str(item.get("subfolder") or ""),
                        "type": str(item.get("type") or "output"),
                    }
            for item in node_output.get("videos") or []:
                filename = str(item.get("filename") or "")
                if filename:
                    return {
                        "filename": filename,
                        "subfolder": str(item.get("subfolder") or ""),
                        "type": str(item.get("type") or "output"),
                    }
        raise FileNotFoundError(f"ComfyUI history did not contain a video output: {history.keys()}")

    async def _download_output(
        self,
        client: httpx.AsyncClient,
        output_info: dict[str, str],
        output_path: Path,
    ) -> None:
        query = urlencode(output_info)
        response = await client.get(
            f"{self._api_url}/view?{query}",
            timeout=self._timeout(self._output_download_timeout_sec),
        )
        response.raise_for_status()
        tmp_path = output_path.with_suffix(f".download{output_path.suffix}")
        tmp_path.write_bytes(response.content)
        tmp_path.replace(output_path)

    def _set_prompt_text(self, prompt: dict[str, dict], text: str) -> None:
        found = False
        for node in prompt.values():
            if node.get("class_type") == "PrimitiveStringMultiline":
                title = (node.get("_meta") or {}).get("title")
                if title == "Prompt":
                    node.setdefault("inputs", {})["value"] = text
                    found = True
        if not found:
            raise AdapterUnavailableError("ComfyUI workflow does not expose a Prompt node.")

    def _set_negative_prompt(self, prompt: dict[str, dict], text: str) -> None:
        if not text:
            return
        for node in prompt.values():
            if node.get("class_type") == "CLIPTextEncode":
                inputs = node.setdefault("inputs", {})
                if isinstance(inputs.get("text"), str):
                    inputs["text"] = text

    def _set_primitive_int(self, prompt: dict[str, dict], title: str, value: int) -> None:
        found = False
        for node in prompt.values():
            if node.get("class_type") == "PrimitiveInt" and (node.get("_meta") or {}).get("title") == title:
                node.setdefault("inputs", {})["value"] = int(value)
                found = True
        if not found:
            raise AdapterUnavailableError(f"ComfyUI workflow does not expose PrimitiveInt '{title}'.")

    def _set_text_to_video_switch(self, prompt: dict[str, dict], enabled: bool) -> None:
        for node in prompt.values():
            if node.get("class_type") != "PrimitiveBoolean":
                continue
            title = str((node.get("_meta") or {}).get("title") or "")
            if "Text to Video" in title:
                node.setdefault("inputs", {})["value"] = bool(enabled)

    def _set_noise_seed(self, prompt: dict[str, dict], seed: int) -> None:
        offset = 0
        for node in prompt.values():
            if node.get("class_type") == "RandomNoise":
                node.setdefault("inputs", {})["noise_seed"] = int(seed + offset)
                offset += 1
        if offset == 0:
            raise AdapterUnavailableError("ComfyUI workflow does not expose RandomNoise seed nodes.")

    def _collect_injected_controls(self, prompt: dict[str, dict]) -> dict[str, object]:
        controls: dict[str, object] = {}
        for node in prompt.values():
            title = (node.get("_meta") or {}).get("title")
            inputs = node.get("inputs") or {}
            if title in {"Width", "Height", "Duration", "Frame Rate", "Prompt"}:
                controls[str(title)] = dict(inputs)
            if node.get("class_type") == "PrimitiveBoolean" and "Text to Video" in str(title or ""):
                controls[str(title)] = dict(inputs)
            if node.get("class_type") == "RandomNoise":
                controls.setdefault("RandomNoise", []).append(inputs.get("noise_seed"))
        return controls

    def _wire_uploaded_image(self, prompt: dict[str, dict], load_image_value: str) -> dict[str, object]:
        load_node_id = self._next_node_id(prompt)
        prompt[load_node_id] = {
            "inputs": {"image": load_image_value},
            "class_type": "LoadImage",
            "_meta": {"title": "AI Video Gen First Frame"},
        }
        wired_nodes: list[dict[str, object]] = []
        for node_id, node in prompt.items():
            if node.get("class_type") == "ResizeImageMaskNode":
                node.setdefault("inputs", {})["input"] = [load_node_id, 0]
                wired_nodes.append(
                    {
                        "nodeId": node_id,
                        "title": (node.get("_meta") or {}).get("title"),
                        "input": [load_node_id, 0],
                    }
                )
        if wired_nodes:
            return {
                "loadImageNodeId": load_node_id,
                "loadImageValue": load_image_value,
                "wiredResizeNodes": wired_nodes,
            }
        raise AdapterUnavailableError("ComfyUI I2V workflow does not expose ResizeImageMaskNode input.")

    def _add_save_video_node(self, prompt: dict[str, dict], output_prefix: str) -> None:
        create_node_id = None
        for node_id, node in prompt.items():
            if node.get("class_type") == "CreateVideo":
                create_node_id = node_id
                break
        if not create_node_id:
            raise AdapterUnavailableError("ComfyUI workflow does not expose CreateVideo output.")
        save_node_id = self._next_node_id(prompt)
        prompt[save_node_id] = {
            "inputs": {
                "video": [create_node_id, 0],
                "filename_prefix": output_prefix,
                "format": "mp4",
                "codec": "h264",
            },
            "class_type": "SaveVideo",
            "_meta": {"title": "AI Video Gen Save Video"},
        }

    def _resolve_seed(self, request: SegmentGenerationRequest) -> int:
        params = request.backendParams or {}
        if params.get("seed") not in {None, ""}:
            base_seed = int(params["seed"])
            candidate_index = int(params.get("candidateIndex") or 0)
            return base_seed + ((max(1, int(request.segmentIndex)) - 1) * 1_000) + max(0, candidate_index - 1)
        if os.environ.get("COMFYUI_SEED") not in {None, ""}:
            base_seed = int(os.environ["COMFYUI_SEED"])
            candidate_index = int(params.get("candidateIndex") or 0)
            return base_seed + (int(request.videoId) * 100_000) + (int(request.segmentIndex) * 1_000) + candidate_index
        candidate_index = int(params.get("candidateIndex") or 0)
        return secrets.randbelow(900_000_000_000_000) + candidate_index

    def _build_output_prefix(self, request: SegmentGenerationRequest) -> str:
        candidate_index = int((request.backendParams or {}).get("candidateIndex") or 1)
        parts = [
            self._settings.comfyui_output_prefix.strip("/"),
            self._safe_filename(request.jobId),
            f"{self._safe_filename(request.segmentId)}_c{candidate_index:02d}",
        ]
        return "/".join(part for part in parts if part)

    def _next_node_id(self, prompt: dict[str, dict]) -> str:
        used = set(prompt)
        index = 900000
        while str(index) in used:
            index += 1
        return str(index)

    def _safe_filename(self, value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)[:180]

    def _postprocess_output(
        self,
        output_path: Path,
        request: SegmentGenerationRequest,
        debug: dict[str, object],
    ) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            debug["outputPostprocess"] = {"enabled": False, "reason": "ffmpeg_not_found"}
            return

        filters: list[str] = []
        if self._settings.comfyui_normalize_output:
            filters.append(
                f"scale={int(request.width)}:{int(request.height)}:flags=lanczos"
            )
            filters.append(f"fps={max(1, float(request.fps))}")
            filters.append("setsar=1")

        tmp_path = output_path.with_name(f"{output_path.stem}.post.tmp{output_path.suffix}")
        command = [
            ffmpeg,
            "-y",
            "-i",
            output_path.as_posix(),
            "-map",
            "0:v:0",
        ]
        if not self._settings.comfyui_strip_audio:
            command.extend(["-map", "0:a?"])
        if filters:
            command.extend([
                "-vf",
                ",".join(filters),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
            ])
        else:
            command.extend(["-c:v", "copy"])
        if self._settings.comfyui_strip_audio:
            command.append("-an")
        else:
            command.extend(["-c:a", "copy"])
        command.extend(["-movflags", "+faststart", tmp_path.as_posix()])
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(
                "\n".join(
                    item
                    for item in [
                        f"Output postprocess failed for {output_path.name}.",
                        completed.stdout.strip(),
                        completed.stderr.strip(),
                    ]
                    if item
                )
            )
        tmp_path.replace(output_path)
        debug["outputPostprocess"] = {
            "enabled": True,
            "ffmpeg": ffmpeg,
            "normalizeOutput": self._settings.comfyui_normalize_output,
            "targetWidth": request.width,
            "targetHeight": request.height,
            "targetFps": request.fps,
            "stripAudio": self._settings.comfyui_strip_audio,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
        debug["audioStripped"] = {
            "enabled": self._settings.comfyui_strip_audio,
            "via": "outputPostprocess",
        }
