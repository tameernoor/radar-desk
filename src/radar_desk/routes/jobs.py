"""Jobs: list, get, live state over SSE, logs, cancel, retry."""

from __future__ import annotations

import asyncio
import json
from typing import Annotated

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse

from radar_desk.records import TERMINAL_JOB_STATES
from radar_desk.routes import OWNER, SSE_HEADERS, Svc
from radar_desk.services import ServiceError

router = APIRouter(prefix="/jobs", tags=["jobs"], dependencies=OWNER)


@router.get("")
def list_jobs(svc: Svc, state: str | None = None, scan_id: str | None = None,
              limit: Annotated[int, Query(ge=1, le=1000)] = 100) -> dict:
    return {"jobs": [j.model_dump() for j in svc.jobs.list(state=state, scan_id=scan_id, limit=limit)]}


@router.get("/{job_id}")
def get_job(job_id: str, svc: Svc) -> dict:
    return svc.jobs.get(job_id).model_dump()


@router.get("/{job_id}/events")
async def job_events(job_id: str, request: Request, svc: Svc) -> StreamingResponse:
    """Server-sent `state` events with the job JSON, one per interval, until the job is terminal."""
    svc.jobs.get(job_id)
    interval = request.app.state.sse_interval_s

    async def stream():
        while True:
            try:
                job = svc.jobs.get(job_id)
            except ServiceError as exc:
                yield f"event: error\ndata: {json.dumps({'detail': exc.detail})}\n\n"
                return
            yield f"event: state\ndata: {json.dumps(job.model_dump(), ensure_ascii=False)}\n\n"
            if job.state in TERMINAL_JOB_STATES or await request.is_disconnected():
                return
            await asyncio.sleep(interval)

    return StreamingResponse(stream(), media_type="text/event-stream", headers=SSE_HEADERS)


@router.get("/{job_id}/logs")
def job_logs(job_id: str, svc: Svc, lines: Annotated[int, Query(ge=1, le=5000)] = 200) -> dict:
    return {"text": svc.jobs.logs(job_id, lines)}


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str, svc: Svc) -> dict:
    return svc.jobs.cancel(job_id).model_dump()


@router.post("/{job_id}/retry")
def retry_job(job_id: str, svc: Svc) -> dict:
    return svc.jobs.retry(job_id).model_dump()
