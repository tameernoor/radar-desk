"""persona chat endpoints: dispatch and resume, both streaming wire frames."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from radar_desk.chat.agent import ChatNotFound
from radar_desk.routes import OWNER, SSE_HEADERS

router = APIRouter(prefix="/chat", tags=["chat"], dependencies=OWNER)


async def _body(request: Request) -> dict:
    try:
        body = await request.json()
    except ValueError as exc:
        raise HTTPException(422, "body must be JSON") from exc
    if not isinstance(body, dict):
        raise HTTPException(422, "body must be a JSON object")
    return body


def _stream(frames) -> StreamingResponse:
    return StreamingResponse(frames, media_type="text/event-stream", headers=SSE_HEADERS)


@router.post("")
async def chat(request: Request) -> StreamingResponse:
    body = await _body(request)
    return _stream(request.app.state.chat_agent.dispatch(body))


@router.post("/resume")
async def resume(request: Request) -> StreamingResponse:
    body = await _body(request)
    try:
        frames = request.app.state.chat_agent.resume(body)
    except ChatNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    return _stream(frames)
