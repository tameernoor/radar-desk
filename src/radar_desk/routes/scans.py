"""Scans: list, get, delete, queue scoring, and the CT for the viewer."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Response
from fastapi.responses import RedirectResponse

from radar_desk.routes import OWNER, Svc

router = APIRouter(prefix="/scans", tags=["scans"], dependencies=OWNER)


@router.get("")
def list_scans(svc: Svc, state: str | None = None,
               limit: Annotated[int, Query(ge=1, le=1000)] = 100) -> dict:
    return {"scans": [svc.scan_view(s) for s in svc.scans.list(state=state, limit=limit)]}


@router.get("/{scan_id}")
def get_scan(scan_id: str, svc: Svc) -> dict:
    return svc.scan_view(scan_id)


@router.delete("/{scan_id}", status_code=204)
def delete_scan(scan_id: str, svc: Svc) -> Response:
    svc.scans.delete(scan_id)
    return Response(status_code=204)


@router.post("/{scan_id}/jobs")
def create_job(scan_id: str, svc: Svc) -> dict:
    """Queue scoring, or return the job already queued or running for this scan."""
    return svc.jobs.create(scan_id).model_dump()


@router.get("/{scan_id}/source.nii.gz")
def source(scan_id: str, svc: Svc) -> RedirectResponse:
    """307 to a one-hour signed GET of the CT."""
    return RedirectResponse(svc.scans.view_url(scan_id), status_code=307)
