"""The 146 findings and the 36 labels with the 18 scored organs."""

from __future__ import annotations

from fastapi import APIRouter

from radar_desk.routes import OWNER, Svc

router = APIRouter(prefix="/catalog", tags=["catalog"], dependencies=OWNER)


@router.get("/findings")
def findings(svc: Svc) -> dict:
    return {"findings": svc.catalog.FINDINGS}


@router.get("/labels")
def labels(svc: Svc) -> dict:
    return {"labels": svc.catalog.LABELS, "scored_organs": svc.catalog.SCORED_ORGANS}
