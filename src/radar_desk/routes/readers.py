"""The Readers lens: NV-Reason-CT's job for a scan (later: the reads, the rows and the questions)."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

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


class AskBody(BaseModel):
    question: str | None = None
    query_id: str | None = None


@router.post("/{scan_id}/nvreason/ask")
def ask_nvreason(scan_id: str, body: AskBody, svc: Svc) -> dict:
    """Ask NV-Reason-CT one question, prepared (query_id) or free. Waits for the answer, up to 4 minutes cold."""
    return svc.readers.ask(scan_id, question=body.question, query_id=body.query_id)
