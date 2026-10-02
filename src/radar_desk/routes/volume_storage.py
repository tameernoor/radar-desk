"""PUT and GET /_volume/{key}: the browser's way to a Modal Volume, through the API.

Mounted only when the storage is ModalVolumeStorage. The owner's cookie or bearer is the auth;
the browser's upload and the viewer's fetch are same-origin, so the cookie goes along. A PUT is
streamed to a temp file under DATA_DIR and then uploaded from disk, so a scan never sits in memory.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import anyio
import anyio.from_thread
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from starlette.requests import ClientDisconnect

from radar_desk.routes import OWNER
from radar_desk.storage import ModalVolumeStorage, ObjectExists, ObjectMissing, StorageError, validate_key

router = APIRouter(prefix="/_volume", tags=["storage"], dependencies=OWNER, include_in_schema=False)

TEMP_DIR = "volume-uploads"


class _TooLarge(Exception):
    pass


def _storage(request: Request, key: str) -> ModalVolumeStorage:
    try:
        validate_key(key)
    except StorageError as exc:
        raise HTTPException(400, str(exc)) from None
    return request.app.state.services.storage


@router.put("/{key:path}", status_code=204)
async def put_object(key: str, request: Request) -> Response:
    storage = _storage(request, key)
    settings = request.app.state.services.settings
    limit = settings.max_upload_bytes
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise HTTPException(413, "the body is larger than the upload limit")
    if await anyio.to_thread.run_sync(storage.exists, key):
        raise HTTPException(409, "the object already exists; objects are never overwritten")
    temp_dir = Path(settings.data_dir) / TEMP_DIR
    temp_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=temp_dir, prefix=".upload-")
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
        await anyio.to_thread.run_sync(storage.put_file, key, Path(tmp))
    except _TooLarge:
        raise HTTPException(413, "the body is larger than the upload limit") from None
    except ClientDisconnect:
        raise HTTPException(400, "the client disconnected before the upload finished") from None
    except ObjectExists:
        raise HTTPException(409, "the object already exists; objects are never overwritten") from None
    finally:
        Path(tmp).unlink(missing_ok=True)
    return Response(status_code=204)


@router.get("/{key:path}")
def get_object(key: str, request: Request) -> StreamingResponse:
    storage = _storage(request, key)
    try:
        size = storage.size(key)
        stream = storage.open_stream(key)
    except ObjectMissing:
        raise HTTPException(404, "no such object") from None
    media = "application/gzip" if key.endswith(".gz") else "application/octet-stream"
    return StreamingResponse(stream, media_type=media, headers={"Content-Length": str(size)})
