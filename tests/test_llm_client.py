from __future__ import annotations

import json

import httpx
import pytest

from radar_desk.chat.llm import Done, LlmError, OpenAICompatibleLLM, TextDelta, ToolCall

BASE = "https://llm.test/v1/openai"


def sse(*chunks) -> str:
    lines = [f"data: {json.dumps(c) if not isinstance(c, str) else c}" for c in chunks]
    return "\n\n".join(lines) + "\n\n"


def chunk(delta: dict, finish: str | None = None) -> dict:
    return {"id": "x", "object": "chat.completion.chunk", "choices": [
        {"index": 0, "delta": delta, "finish_reason": finish}]}


TEXT_STREAM = sse(
    chunk({"role": "assistant", "content": ""}),
    chunk({"content": "Hel"}),
    chunk({"content": "lo"}),
    chunk({}, "stop"),
    "[DONE]",
)

TOOL_STREAM = sse(
    chunk({"role": "assistant", "content": None, "tool_calls": [
        {"index": 0, "id": "call_a", "type": "function", "function": {"name": "get_scores", "arguments": ""}}]}),
    chunk({"tool_calls": [{"index": 0, "function": {"arguments": "{\"job_id\": "}}]}),
    chunk({"tool_calls": [
        {"index": 1, "id": "call_b", "type": "function", "function": {"name": "jump_to_organ", "arguments": "{\"or"}}]}),
    chunk({"tool_calls": [{"index": 0, "function": {"arguments": "\"job_1\"}"}}]}),
    chunk({"tool_calls": [{"index": 1, "function": {"arguments": "gan\": \"Liver\"}"}}]}),
    chunk({"tool_calls": [
        {"index": 2, "id": "call_c", "type": "function", "function": {"name": "gpu_status", "arguments": "{not json"}}]}),
    chunk({}, "tool_calls"),
    "[DONE]",
)


def make_llm(handler) -> tuple[OpenAICompatibleLLM, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(wrapped))
    return OpenAICompatibleLLM(BASE, "sk-test", "some/model", client=client), seen


async def collect(llm, messages=None, tools=None):
    return [e async for e in llm.stream_chat(messages or [{"role": "user", "content": "hi"}], tools or [])]


async def test_text_stream_and_request_body():
    llm, seen = make_llm(lambda r: httpx.Response(200, text=TEXT_STREAM, headers={"content-type": "text/event-stream"}))
    tools = [{"type": "function", "function": {"name": "t", "description": "d", "parameters": {"type": "object"}}}]
    events = await collect(llm, tools=tools)
    assert events == [TextDelta("Hel"), TextDelta("lo"), Done("stop")]
    req = seen[0]
    assert str(req.url) == BASE + "/chat/completions"
    assert req.headers["authorization"] == "Bearer sk-test"
    assert req.headers["x-title"] == "radar-desk"
    body = json.loads(req.content)
    assert body["stream"] is True
    assert body["model"] == "some/model"
    assert body["tools"] == tools
    assert body["tool_choice"] == "auto"
    assert body["messages"] == [{"role": "user", "content": "hi"}]


async def test_tool_call_fragments_assembled_by_index():
    llm, _ = make_llm(lambda r: httpx.Response(200, text=TOOL_STREAM))
    events = await collect(llm)
    assert events == [
        ToolCall("call_a", "get_scores", {"job_id": "job_1"}),
        ToolCall("call_b", "jump_to_organ", {"organ": "Liver"}),
        ToolCall("call_c", "gpu_status", {"_raw": "{not json"}),
        Done("tool_calls"),
    ]


async def test_4xx_raises_llm_error():
    llm, _ = make_llm(lambda r: httpx.Response(401, text='{"detail": "bad key"}'))
    with pytest.raises(LlmError) as e:
        await collect(llm)
    assert e.value.status == 401
    assert "bad key" in str(e.value)


async def test_provider_error_body_message_and_status():
    # OpenRouter error shape: {"error": {"code", "message", "metadata"?}}, openrouter.ai/docs/api-reference/errors
    body = {"error": {"code": 402, "message": "Insufficient credits. Add more using https://openrouter.ai/credits"}}
    llm, _ = make_llm(lambda r: httpx.Response(402, json=body))
    with pytest.raises(LlmError) as e:
        await collect(llm)
    assert e.value.status == 402
    assert str(e.value) == (
        "the model endpoint returned 402: Insufficient credits. Add more using https://openrouter.ai/credits"
    )


async def test_transport_error_and_stream_error_raise_llm_error():
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    llm, _ = make_llm(refuse)
    with pytest.raises(LlmError):
        await collect(llm)
    llm, _ = make_llm(lambda r: httpx.Response(200, text=sse({"error": {"message": "overloaded"}})))
    with pytest.raises(LlmError, match="overloaded"):
        await collect(llm)


async def test_mid_stream_error_in_openrouter_shape():
    # OpenRouter sends errors after a 200 as an SSE chunk, openrouter.ai/docs/api-reference/errors
    err = {"error": {"code": 502, "message": "Provider returned error", "metadata": {"error_type": "provider_error"}},
           "choices": [{"finish_reason": "error"}]}
    llm, _ = make_llm(lambda r: httpx.Response(200, text=sse(chunk({"content": "Hel"}), err)))
    with pytest.raises(LlmError) as e:
        await collect(llm)
    assert e.value.status == 502
    assert str(e.value) == "the model endpoint reported an error 502: Provider returned error"


async def test_error_messages_capped_at_300_characters():
    long = "x" * 1000
    llm, _ = make_llm(lambda r: httpx.Response(402, json={"error": {"code": 402, "message": long}}))
    with pytest.raises(LlmError) as e:
        await collect(llm)
    assert str(e.value) == "the model endpoint returned 402: " + "x" * 300
    llm, _ = make_llm(lambda r: httpx.Response(200, text=sse({"error": long})))
    with pytest.raises(LlmError) as e:
        await collect(llm)
    assert str(e.value) == "the model endpoint reported an error: " + "x" * 300
    llm, _ = make_llm(lambda r: httpx.Response(200, text=sse({"error": {"code": 500, "detail": long}})))
    with pytest.raises(LlmError) as e:
        await collect(llm)
    assert e.value.status == 500 and "None" not in str(e.value)
    assert len(str(e.value)) == len("the model endpoint reported an error 500: ") + 300


async def test_finish_reason_error_without_error_object_raises():
    llm, _ = make_llm(lambda r: httpx.Response(200, text=sse(chunk({"content": "Hel"}), chunk({}, "error"))))
    with pytest.raises(LlmError, match="ended the stream with an error"):
        await collect(llm)


async def test_stream_without_done_marker_still_finishes():
    text = sse(chunk({"content": "ok"}), chunk({}, "length"))
    llm, _ = make_llm(lambda r: httpx.Response(200, text=text))
    assert await collect(llm) == [TextDelta("ok"), Done("length")]


async def test_non_object_chunk_raises_llm_error():
    llm, _ = make_llm(lambda r: httpx.Response(200, text=sse("[1, 2]")))
    with pytest.raises(LlmError, match="unexpected stream chunk"):
        await collect(llm)
