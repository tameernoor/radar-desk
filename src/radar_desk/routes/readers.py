"""The Readers lens: NV-Reason-CT's job for a scan (later: the reads, the rows and the questions)."""

from __future__ import annotations

from fastapi import APIRouter

from radar_desk.routes import OWNER, Svc

router = APIRouter(prefix="/scans", tags=["readers"], dependencies=OWNER)


@router.post("/{scan_id}/nvreason")
def start_nvreason(scan_id: str, svc: Svc) -> dict:
    """Queue NV-Reason-CT for the scan, or return its queued, running or finished job."""
    return svc.nvreason.start(scan_id).model_dump()


@router.post("/{scan_id}/readers/refresh")
def refresh_readers(scan_id: str, svc: Svc) -> dict:
    """Run the reads again: stale ones, and failed ones too (the tab's retry)."""
    svc.scans.get(scan_id)
    started = svc.readers.kick(scan_id, force=True)
    return {"started": started, **svc.readers.status(scan_id)}


@router.get("/{scan_id}/readers")
def get_readers(scan_id: str, svc: Svc) -> dict:
    """The Readers tab: one row per label with the report, RADAR and NV-Reason-CT side by side."""
    return svc.readers.view(scan_id)
