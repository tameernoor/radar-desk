"""A streaming client for OpenAI-compatible chat completions (OpenRouter by default).

`stream_chat` yields TextDelta for each content fragment, then one ToolCall per tool call
(assembled from its streamed fragments by index), then Done with the finish reason.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx


class LlmError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Done:
    finish_reason: str


LlmEvent = TextDelta | ToolCall | Done


class LLM(Protocol):
    def stream_chat(self, messages: list[dict], tools: list[dict]) -> AsyncIterator[LlmEvent]: ...


def parse_arguments(raw: str) -> dict:
    """Tool arguments from their JSON text. Empty means {}; anything not a JSON object is kept as _raw."""
    if not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except ValueError:
        return {"_raw": raw}
    return value if isinstance(value, dict) else {"_raw": raw}


class OpenAICompatibleLLM:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        client: httpx.AsyncClient | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.client = client
        self.timeout_s = timeout_s

    def request_body(self, messages: list[dict], tools: list[dict]) -> dict:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "stream": True}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        return body

    async def stream_chat(self, messages: list[dict], tools: list[dict]) -> AsyncIterator[LlmEvent]:
        if self.client is not None:
            async for event in self._stream(self.client, messages, tools):
                yield event
            return
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            async for event in self._stream(client, messages, tools):
                yield event

    async def _stream(
        self, client: httpx.AsyncClient, messages: list[dict], tools: list[dict]
    ) -> AsyncIterator[LlmEvent]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "text/event-stream",
            "X-Title": "radar-desk",
        }
        try:
            async with client.stream(
                "POST", self.url, json=self.request_body(messages, tools), headers=headers
            ) as resp:
                if resp.status_code >= 300:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    raise LlmError(
                        f"the model endpoint returned {resp.status_code}: {error_message(body)}",
                        resp.status_code,
                    )
                async for event in parse_stream(resp.aiter_lines()):
                    yield event
        except httpx.HTTPError as exc:
            raise LlmError(f"could not reach the model endpoint: {type(exc).__name__}: {exc}") from exc


def error_message(body: str) -> str:
    """The provider's own message from an error body shaped {"error": {"code", "message"}}, else the raw text."""
    try:
        err = json.loads(body).get("error")
    except (ValueError, AttributeError):
        err = None
    return _message_of(err, body)


def _message_of(err: Any, fallback: str) -> str:
    """err["message"] when it is a non-empty string, else the fallback text; capped at 300 characters."""
    if isinstance(err, dict) and isinstance(err.get("message"), str) and err["message"]:
        return err["message"][:300]
    return fallback[:300]


async def parse_stream(lines: AsyncIterator[str]) -> AsyncIterator[LlmEvent]:
    """OpenAI-style SSE lines to events. Tool calls are yielded once the stream ends."""
    calls: dict[int, dict[str, Any]] = {}
    finish: str | None = None
    async for line in lines:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except ValueError as exc:
            raise LlmError(f"unreadable stream chunk: {data[:200]}") from exc
        if not isinstance(chunk, dict):
            raise LlmError(f"unexpected stream chunk: {data[:200]}")
        if chunk.get("error"):
            err = chunk["error"]
            code = err.get("code") if isinstance(err, dict) else None
            status = code if isinstance(code, int) and not isinstance(code, bool) else None
            message = _message_of(err, data if isinstance(err, dict) else str(err))
            prefix = f"the model endpoint reported an error {status}" if status else "the model endpoint reported an error"
            raise LlmError(f"{prefix}: {message}", status)
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                yield TextDelta(delta["content"])
            for frag in delta.get("tool_calls") or []:
                slot = calls.setdefault(frag.get("index", 0), {"id": None, "name": "", "arguments": ""})
                if frag.get("id"):
                    slot["id"] = frag["id"]
                fn = frag.get("function") or {}
                if fn.get("name") and not slot["name"]:
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
            if choice.get("finish_reason") == "error":
                raise LlmError("the model endpoint ended the stream with an error and gave no details")
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    for index in sorted(calls):
        slot = calls[index]
        yield ToolCall(
            id=slot["id"] or f"call_{uuid.uuid4().hex[:24]}",
            name=slot["name"],
            arguments=parse_arguments(slot["arguments"]),
        )
    yield Done(finish or ("tool_calls" if calls else "stop"))
