"""HTTP routes. Each one is a thin call into `services/`; the rules live there."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from radar_desk.auth import require_owner
from radar_desk.services import Services

OWNER = [Depends(require_owner)]
SSE_HEADERS = {"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"}


def get_services(request: Request) -> Services:
    return request.app.state.services


Svc = Annotated[Services, Depends(get_services)]
