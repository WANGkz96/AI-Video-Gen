from __future__ import annotations

import json
import hmac
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.app.adapters.base import AdapterUnavailableError
from backend.app.config import Settings
from backend.app.models import DirectGenerationRequest, JobQueuedResponse, JobSnapshot, LogEntry
from backend.app.services.jobs import JobService


settings = Settings.from_env()
job_service = JobService(settings)

PUBLIC_READ_API_ROUTES = {
    "/api/health",
    "/api/provisioning",
    "/api/backends",
    "/api/jobs",
}


def _is_public_read_api_route(request: Request) -> bool:
    if request.method.upper() != "GET":
        return False
    path = request.url.path.rstrip("/") or "/"
    if path in PUBLIC_READ_API_ROUTES:
        return True
    if not path.startswith("/api/jobs/"):
        return False
    suffix = path.removeprefix("/api/jobs/").strip("/")
    if not suffix:
        return False
    parts = suffix.split("/")
    if len(parts) == 1:
        return True
    return len(parts) == 2 and parts[1] in {"logs", "events"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    await job_service.start()
    yield
    await job_service.stop()


app = FastAPI(title="AI Video Generation Service", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _is_authorized_api_request(request: Request) -> bool:
    if not settings.ai_video_gen_auth_required:
        return True
    if request.method.upper() == "OPTIONS":
        return True
    if not request.url.path.startswith("/api/"):
        return True
    if _is_public_read_api_route(request):
        return True
    expected_token = settings.ai_video_gen_api_token or ""
    header = request.headers.get("authorization", "")
    prefix = "Bearer "
    if header.startswith(prefix):
        supplied_token = header[len(prefix):].strip()
    else:
        supplied_token = (request.query_params.get("token") or "").strip()
    return bool(supplied_token) and hmac.compare_digest(supplied_token, expected_token)


@app.middleware("http")
async def api_bearer_auth(request: Request, call_next):
    if not _is_authorized_api_request(request):
        return JSONResponse({"detail": "Unauthorized."}, status_code=401)
    return await call_next(request)


def get_jobs() -> JobService:
    return job_service


@app.get("/api/health")
async def health(jobs: JobService = Depends(get_jobs)):
    return jobs.health()


@app.get("/api/provisioning")
async def provisioning(jobs: JobService = Depends(get_jobs)):
    return jobs.provisioning()


@app.get("/api/backends")
async def list_backends(
    include_unavailable: bool = Query(default=False, alias="includeUnavailable"),
    jobs: JobService = Depends(get_jobs),
):
    return jobs.list_backends(include_unavailable=include_unavailable)


def _parse_backend_params(raw_value: str | None) -> dict[str, object] | None:
    if raw_value is None or not raw_value.strip():
        return None
    parsed = json.loads(raw_value)
    if not isinstance(parsed, dict):
        raise ValueError("backendParams must be a JSON object.")
    return parsed


@app.post("/api/jobs", response_model=JobQueuedResponse)
async def create_job(
    request: Request,
    batch_file: UploadFile | None = File(default=None, alias="batch"),
    backend_query: str | None = Query(default=None, alias="backend"),
    backend_form: str | None = Form(default=None, alias="backend"),
    backend_params_form: str | None = Form(default=None, alias="backendParams"),
    jobs: JobService = Depends(get_jobs),
):
    try:
        content_type = request.headers.get("content-type", "")
        if "multipart/form-data" in content_type:
            if batch_file is None:
                raise HTTPException(status_code=400, detail="Multipart request requires a 'batch' file.")
            backend = backend_form or backend_query
            backend_params = _parse_backend_params(backend_params_form)
            return await jobs.create_job_from_upload(
                filename=batch_file.filename or "batch.zip",
                content=await batch_file.read(),
                backend=backend,
                backend_params=backend_params,
            )
        payload = await request.json()
        backend = backend_query
        backend_params = payload.pop("backendParams", None)
        if backend_params is not None and not isinstance(backend_params, dict):
            raise ValueError("backendParams must be a JSON object.")
        return await jobs.create_job(payload, backend=backend, backend_params=backend_params)
    except AdapterUnavailableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {exc}") from exc


@app.get("/api/jobs", response_model=list[JobSnapshot])
async def list_jobs(
    limit: int = Query(default=20, ge=1, le=100),
    jobs: JobService = Depends(get_jobs),
):
    return jobs.list_jobs(limit=limit)


@app.post("/api/jobs/direct", response_model=JobQueuedResponse)
async def create_direct_job(
    payload: DirectGenerationRequest,
    jobs: JobService = Depends(get_jobs),
):
    try:
        return await jobs.create_direct_job(payload)
    except AdapterUnavailableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}", response_model=JobSnapshot)
async def get_job(job_id: str, jobs: JobService = Depends(get_jobs)):
    try:
        return jobs.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc


@app.get("/api/jobs/{job_id}/logs", response_model=list[LogEntry])
async def get_logs(job_id: str, jobs: JobService = Depends(get_jobs)):
    try:
        return jobs.get_logs(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str, jobs: JobService = Depends(get_jobs)):
    try:
        async def event_stream():
            async for event in jobs.stream_events(job_id):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc


@app.get("/api/jobs/{job_id}/result")
async def get_result(job_id: str, jobs: JobService = Depends(get_jobs)):
    try:
        return JSONResponse(jobs.get_result(job_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}/archive")
async def get_archive(job_id: str, jobs: JobService = Depends(get_jobs)):
    try:
        archive_path = jobs.get_archive_path(job_id)
        return FileResponse(archive_path, media_type="application/zip", filename=archive_path.name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}/files/{relative_path:path}")
async def get_job_file(job_id: str, relative_path: str, jobs: JobService = Depends(get_jobs)):
    try:
        target = jobs.get_job_file(job_id, relative_path)
        return FileResponse(target, filename=target.name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/")
async def root():
    if settings.frontend_dist_dir.exists():
        index_path = settings.frontend_dist_dir / "index.html"
        if index_path.is_file():
            return FileResponse(index_path)
    return {"service": "ai-video-generation-service", "status": "running", "docs": "/api/health"}


if settings.frontend_dist_dir.exists():
    app.mount("/assets", StaticFiles(directory=settings.frontend_dist_dir / "assets"), name="frontend-assets")


@app.get("/{full_path:path}")
async def spa_fallback(full_path: str):
    if not settings.frontend_dist_dir.exists():
        raise HTTPException(status_code=404, detail="Not found.")
    requested_path = settings.frontend_dist_dir / full_path
    if requested_path.is_file():
        return FileResponse(requested_path)
    index_path = settings.frontend_dist_dir / "index.html"
    if index_path.is_file():
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Not found.")
