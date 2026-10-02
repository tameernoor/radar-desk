"""HTTP routes. Each one is a thin call into `services/`; the rules live there."""

from __future__ import annotations

import os
from typing import Annotated

import anyio
import anyio.from_thread
from fastapi import Depends, HTTPException, Request
from starlette.requests import ClientDisconnect

from radar_desk.auth import require_owner
from radar_desk.services import Services

OWNER = [Depends(require_owner)]
SSE_HEADERS = {"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"}
TOO_LARGE = "the body is larger than the upload limit"


def get_services(request: Request) -> Services:
    return request.app.state.services


Svc = Annotated[Services, Depends(get_services)]


class _TooLarge(Exception):
    pass


async def spool_request(request: Request, limit: int, fd: int) -> None:
    """Write the request body to the open file descriptor `fd`, which is closed afterwards. A body over
    `limit` bytes is a 413 and a client that goes away mid-body is a 400."""
    body = request.stream().__aiter__()

    def spool() -> None:
        seen = 0
        with os.fdopen(fd, "wb") as fh:
            while True:
                try:
                    chunk = anyio.from_thread.run(body.__anext__)
                except StopAsyncIteration:
                    return
                seen += len(chunk)
                if seen > limit:
                    raise _TooLarge
                fh.write(chunk)

    try:
        await anyio.to_thread.run_sync(spool)
    except _TooLarge:
        raise HTTPException(413, TOO_LARGE) from None
    except ClientDisconnect:
        raise HTTPException(400, "the client disconnected before the upload finished") from None
