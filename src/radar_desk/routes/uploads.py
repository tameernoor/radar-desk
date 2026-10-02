"""POST /uploads and POST /uploads/{scan_id}/complete."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from radar_desk.routes import OWNER, Svc

router = APIRouter(prefix="/uploads", tags=["uploads"], dependencies=OWNER)


class UploadRequest(BaseModel):
    filename: str
    size_bytes: int
    research_only_confirmed: bool = False


@router.post("")
def begin_upload(body: UploadRequest, svc: Svc) -> dict:
    """A signed PUT URL for the scan, valid for 15 minutes."""
    return svc.scans.begin_upload(body.filename, body.size_bytes, body.research_only_confirmed).model_dump()


@router.post("/{scan_id}/complete")
def complete_upload(scan_id: str, svc: Svc) -> dict:
    """Check the uploaded file and return the scan as ready or rejected."""
    svc.scans.complete_upload(scan_id)
    return svc.scan_view(scan_id)
