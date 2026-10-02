"""The compute choice and the RunPod pod, for the owner (plan.md, Compute switch B)."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from radar_desk.routes import OWNER, Svc

router = APIRouter(prefix="/compute", tags=["compute"], dependencies=OWNER)


class ModeBody(BaseModel):
    mode: str


@router.get("")
def compute_status(svc: Svc) -> dict:
    return svc.compute.status()


@router.put("")
def set_mode(body: ModeBody, svc: Svc) -> dict:
    return svc.compute.set_mode(body.mode)


@router.post("/pod/start", status_code=202)
def start_pod(svc: Svc) -> dict:
    return svc.compute.start_now()


@router.post("/pod/stop")
def stop_pod(svc: Svc) -> dict:
    return svc.compute.stop_now()
