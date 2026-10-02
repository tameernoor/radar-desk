"""PUT and GET /_storage/{key}: the local stand-in for the bucket's signed URLs.

No login here; the signature in the query is the auth, as with a presigned S3 URL. Mounted only
when the storage is LocalStorage.
"""

from __future__ import annotations

import anyio
import anyio.from_thread
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from starlette.requests import ClientDisconnect

from radar_desk.storage import LocalStorage, ObjectMissing

router = APIRouter(prefix="/_storage", tags=["storage"], include_in_schema=False)


class _TooLarge(Exception):
    pass


def _storage(request: Request) -> LocalStorage:
    return request.app.state.services.storage


def _check(request: Request, method: str, key: str) -> LocalStorage:
    storage = _storage(request)
    exp, sig = request.query_params.get("exp"), request.query_params.get("sig")
    if not exp or not sig or not storage.verify(method, key, exp, sig):
        raise HTTPException(403, "the signature is invalid or has expired")
    return storage


@router.put("/{key:path}", status_code=204)
async def put_object(key: str, request: Request) -> Response:
    storage = _check(request, "PUT", key)
    if storage.exists(key):
        raise HTTPException(409, "the object already exists; objects are never overwritten")
    limit = request.app.state.services.settings.max_upload_bytes
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise HTTPException(413, "the body is larger than the upload limit")
    body = request.stream().__aiter__()

    def chunks():
        seen = 0
        while True:
            try:
                chunk = anyio.from_thread.run(body.__anext__)
            except StopAsyncIteration:
                return
            seen += len(chunk)
            if seen > limit:
                raise _TooLarge
            if chunk:
                yield chunk

    try:
        await anyio.to_thread.run_sync(storage.write_stream, key, chunks())
    except _TooLarge:
        raise HTTPException(413, "the body is larger than the upload limit") from None
    except ClientDisconnect:
        raise HTTPException(400, "the client disconnected before the upload finished") from None
    return Response(status_code=204)


@router.get("/{key:path}")
def get_object(key: str, request: Request) -> StreamingResponse:
    storage = _check(request, "GET", key)
    try:
        size = storage.size(key)
        stream = storage.open_stream(key)
    except ObjectMissing:
        raise HTTPException(404, "no such object") from None
    media = "application/gzip" if key.endswith(".gz") else "application/octet-stream"
    return StreamingResponse(stream, media_type=media, headers={"Content-Length": str(size)})
