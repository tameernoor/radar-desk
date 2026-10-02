"""Results: the stored scores, one organ, and a comparison with another job or the fixture reference."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from radar_desk.routes import OWNER, Svc

router = APIRouter(prefix="/jobs", tags=["results"], dependencies=OWNER)


@router.get("/{job_id}/result")
def get_result(job_id: str, svc: Svc) -> dict:
    return svc.results.get(job_id).model_dump()


@router.get("/{job_id}/organs/{organ}")
def get_organ(job_id: str, organ: str, svc: Svc) -> dict:
    return svc.results.get_organ(job_id, organ)


@router.get("/{job_id}/compare")
def compare(job_id: str, against: Annotated[str, Query(description='another job id, or "fixture"')],
            svc: Svc) -> dict:
    return svc.results.compare(job_id, against)
