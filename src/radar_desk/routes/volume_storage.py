"""PUT and GET /_volume/{key}: the browser's way to a volume, through the API.

Mounted for any adapter the browser reaches through the API (Modal Volume, RunPod volume), the ones
with `browser_via_api = True`. The owner's cookie or bearer is the auth; the browser's upload and the
viewer's fetch are same-origin, so the cookie goes along. A PUT is streamed to a temp file under
DATA_DIR and then uploaded from disk, so a scan never sits in memory.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import anyio
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from radar_desk.routes import OWNER, TOO_LARGE, spool_request
from radar_desk.storage import ObjectExists, ObjectMissing, Storage, StorageError, validate_key

router = APIRouter(prefix="/_volume", tags=["storage"], dependencies=OWNER, include_in_schema=False)

TEMP_DIR = "volume-uploads"


def _storage(request: Request, key: str) -> Storage:
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
        raise HTTPException(413, TOO_LARGE)
    if await anyio.to_thread.run_sync(storage.exists, key):
        raise HTTPException(409, "the object already exists; objects are never overwritten")
    temp_dir = Path(settings.data_dir) / TEMP_DIR
    temp_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=temp_dir, prefix=".upload-")
    try:
        await spool_request(request, limit, fd)
        await anyio.to_thread.run_sync(storage.put_file, key, Path(tmp))
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
