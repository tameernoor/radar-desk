"""GET /gpu/status."""

from __future__ import annotations

from fastapi import APIRouter

from radar_desk.routes import OWNER, Svc

router = APIRouter(prefix="/gpu", tags=["gpu"], dependencies=OWNER)


@router.get("/status")
def gpu_status(svc: Svc) -> dict:
    return svc.gpu_status()
