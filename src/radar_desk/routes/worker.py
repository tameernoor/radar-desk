"""The pull worker's API under /worker. A worker token is the auth; every job call carries the lease.

The worker never talks to the bucket: the source is streamed from storage and each artefact PUT is
spooled to a temp file under DATA_DIR and then stored, so any storage backend works.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Annotated, Any

import anyio
from fastapi import APIRouter, Body, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from radar_desk.auth import require_worker
from radar_desk.records import JobError, WorkerToken
from radar_desk.routes import TOO_LARGE, Svc, spool_request
from radar_desk.services import ServiceError

router = APIRouter(prefix="/worker", tags=["worker"], dependencies=[Depends(require_worker)])

TEMP_DIR = "worker-uploads"
MAX_VERSIONS_BYTES = 4000

Token = Annotated[WorkerToken, Depends(require_worker)]


class WorkerInfo(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    hostname: str | None = Field(default=None, max_length=200)
    gpu_name: str | None = Field(default=None, max_length=200)
    device: str | None = Field(default=None, max_length=200)
    versions: dict[str, Any] = Field(default_factory=dict)

    @field_validator("versions")
    @classmethod
    def _small_versions(cls, value: dict[str, Any]) -> dict[str, Any]:
        if len(json.dumps(value).encode()) > MAX_VERSIONS_BYTES:
            raise ValueError(f"versions is over {MAX_VERSIONS_BYTES} bytes as JSON")
        return value


class ClaimBody(BaseModel):
    worker: WorkerInfo


class LeaseBody(BaseModel):
    lease: str


class HeartbeatBody(LeaseBody):
    progress: str | None = Field(default=None, max_length=200)


class FailBody(LeaseBody):
    error: JobError


@router.get("/me")
def me(token: Token) -> dict:
    return {"token": {"id": token.id, "name": token.name}}


@router.post("/claim", response_model=None)
def claim(body: ClaimBody, token: Token, svc: Svc) -> dict | Response:
    claimed = svc.workers.claim(token, body.worker.model_dump())
    return Response(status_code=204) if claimed is None else claimed


@router.post("/jobs/{job_id}/heartbeat")
def heartbeat(job_id: str, body: HeartbeatBody, svc: Svc) -> dict:
    return svc.workers.heartbeat(job_id, body.lease, body.progress)


@router.get("/jobs/{job_id}/source")
def source(job_id: str, lease: str, svc: Svc) -> StreamingResponse:
    size, stream = svc.workers.source_stream(job_id, lease)
    return StreamingResponse(stream, media_type="application/gzip", headers={"Content-Length": str(size)})


@router.put("/jobs/{job_id}/artefacts/{name}", status_code=204)
async def put_artefact(job_id: str, name: str, lease: str, request: Request, svc: Svc) -> Response:
    limit = svc.settings.max_upload_bytes
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise HTTPException(413, TOO_LARGE)
    await anyio.to_thread.run_sync(svc.workers.artefact_key, job_id, lease, name)
    temp_dir = Path(svc.settings.data_dir) / TEMP_DIR
    temp_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=temp_dir, prefix=".upload-")
    try:
        await spool_request(request, limit, fd)
        await anyio.to_thread.run_sync(svc.workers.put_artefact, job_id, lease, name, Path(tmp),
                                       request.headers.get("content-type"))
    finally:
        Path(tmp).unlink(missing_ok=True)
    return Response(status_code=204)


@router.post("/jobs/{job_id}/complete")
def complete(job_id: str, data: Annotated[dict[str, Any], Body()], svc: Svc) -> dict:
    lease = data.get("lease")
    if not isinstance(lease, str) or not lease:
        raise ServiceError(422, "lease: field required")
    return svc.workers.complete(job_id, lease, data).model_dump()


@router.post("/jobs/{job_id}/fail")
def fail(job_id: str, body: FailBody, svc: Svc) -> dict:
    return svc.workers.fail(job_id, body.lease, body.error).model_dump()


@router.post("/jobs/{job_id}/release")
def release(job_id: str, body: LeaseBody, svc: Svc) -> dict:
    return svc.workers.release(job_id, body.lease).model_dump()
