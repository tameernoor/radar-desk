"""Exports in upstream's formats, and redirects to the stored artefacts."""

from __future__ import annotations

from fastapi import APIRouter, Response
from fastapi.responses import RedirectResponse

from radar_desk.routes import OWNER, Svc

router = APIRouter(tags=["exports"], dependencies=OWNER)

CSV_TYPE = "text/csv; charset=utf-8"


def _csv(data: bytes, filename: str) -> Response:
    return Response(data, media_type=CSV_TYPE,
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.get("/jobs/{job_id}/scores.csv")
def scores_csv(job_id: str, svc: Svc) -> Response:
    return _csv(svc.exports.scores_csv(job_id), f"radar-{job_id}.csv")


@router.get("/jobs/{job_id}/scores.json")
def scores_json(job_id: str, svc: Svc) -> dict:
    return svc.exports.scores_json(job_id)


@router.get("/jobs/{job_id}/mask.nii.gz")
def mask(job_id: str, svc: Svc) -> RedirectResponse:
    return RedirectResponse(svc.exports.artefact_url(job_id, "mask"), status_code=307)


@router.get("/jobs/{job_id}/trace.json")
def trace(job_id: str, svc: Svc) -> RedirectResponse:
    return RedirectResponse(svc.exports.artefact_url(job_id, "trace"), status_code=307)


@router.get("/export/scores.csv")
def export_all(svc: Svc) -> Response:
    """One row per done job."""
    return _csv(svc.exports.export_all_csv(), "radar-desk-scores.csv")
