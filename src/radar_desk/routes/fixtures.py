"""GET /fixtures: the known reference scans and which reference files are present."""

from __future__ import annotations

from fastapi import APIRouter

from radar_desk.routes import OWNER, Svc

router = APIRouter(tags=["catalog"], dependencies=OWNER)


@router.get("/fixtures")
def fixtures(svc: Svc) -> dict:
    return {"fixtures": svc.fixtures.list()}
