"""GET /health, open to all, for the platform's checks."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from fastapi import APIRouter

from radar_desk.routes import Svc

router = APIRouter(tags=["health"])

try:
    VERSION = version("radar-desk")
except PackageNotFoundError:
    VERSION = "unknown"


@router.get("/health")
def health(svc: Svc) -> dict:
    return {"ok": True, "backend": svc.compute.mode, "version": VERSION}
