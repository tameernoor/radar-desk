"""The owner's view of pull workers: tokens and the workers that have reported in."""

from __future__ import annotations

from fastapi import APIRouter, Response
from pydantic import BaseModel, ConfigDict, Field

from radar_desk.routes import OWNER, Svc

router = APIRouter(prefix="/workers", tags=["workers"], dependencies=OWNER)


class TokenBody(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=100)


@router.get("/tokens")
def list_tokens(svc: Svc) -> dict:
    return {"tokens": [t.model_dump(exclude={"token_hash"}) for t in svc.workers.list_tokens()]}


@router.post("/tokens", status_code=201)
def create_token(body: TokenBody, svc: Svc) -> dict:
    token, plaintext = svc.workers.create_token(body.name)
    return {"id": token.id, "name": token.name, "created_at": token.created_at, "token": plaintext}


@router.delete("/tokens/{token_id}", status_code=204)
def revoke_token(token_id: str, svc: Svc) -> Response:
    svc.workers.revoke_token(token_id)
    return Response(status_code=204)


@router.get("")
def list_workers(svc: Svc) -> dict:
    return {"workers": svc.workers.list_workers(), "app_url": svc.settings.public_base_url,
            "image": svc.settings.worker_image, "lease_s": svc.settings.worker_lease_s}
