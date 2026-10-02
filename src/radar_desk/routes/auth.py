"""POST /auth/login, POST /auth/logout, GET /auth/me."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from radar_desk.auth import client_ip, require_owner

router = APIRouter(prefix="/auth", tags=["auth"])


class Login(BaseModel):
    token: str


@router.post("/login", status_code=204)
def login(body: Login, request: Request) -> Response:
    auth = request.app.state.auth
    ip = client_ip(request)
    if auth.limiter.blocked(ip):
        raise HTTPException(429, "too many failed logins; try again in ten minutes")
    if not auth.token_ok(body.token):
        auth.limiter.fail(ip)
        raise HTTPException(401, "wrong token")
    auth.limiter.reset(ip)
    response = Response(status_code=204)
    auth.set_cookie(response)
    return response


@router.post("/logout", status_code=204)
def logout(request: Request) -> Response:
    response = Response(status_code=204)
    request.app.state.auth.clear_cookie(response)
    return response


@router.get("/me")
def me(via: Annotated[str, Depends(require_owner)]) -> dict:
    return {"owner": True, "via": via}
