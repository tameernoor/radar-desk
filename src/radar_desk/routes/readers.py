"""The Readers lens: NV-Reason-CT's job for a scan (later: the reads, the rows and the questions)."""

from __future__ import annotations

from fastapi import APIRouter

from radar_desk.routes import OWNER, Svc

router = APIRouter(prefix="/scans", tags=["readers"], dependencies=OWNER)


@router.post("/{scan_id}/nvreason")
def start_nvreason(scan_id: str, svc: Svc) -> dict:
    """Queue NV-Reason-CT for the scan, or return its queued, running or finished job."""
    return svc.nvreason.start(scan_id).model_dump()
