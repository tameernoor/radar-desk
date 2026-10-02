"""Chat model providers: presets, make_llm, error shapes, Ollama streams and the status checks.

Every HTTP call goes to an httpx.MockTransport; nothing here touches the network.
"""

from __future__ import annotations

import json

import httpx
import pytest

from radar_desk.chat.agent import ChatAgent, make_llm
from radar_desk.chat.llm import Done, LlmError, OpenAICompatibleLLM, ToolCall
from radar_desk.chat.providers import (
    PROVIDERS,
    StatusCache,
    check_provider,
    get_provider,
    llm_configured,
    ollama_origin,
)
from radar_desk.chat.wire import parse_frames
from test_agent import FakeLLM, text_turn


def sse(*chunks) -> str:
    return "".join(f"data: {c if isinstance(c, str) else json.dumps(c)}\n\n" for c in chunks)


def mock_client(handler, seen: list | None = None) -> httpx.AsyncClient:
    def wrapped(request):
        if seen is not None:
            seen.append(request)
        return handler(request)

    return httpx.AsyncClient(transport=httpx.MockTransport(wrapped))


async def collect(llm) -> list:
    return [e async for e in llm.stream_chat([{"role": "user", "content": "hi"}], [])]


# Registry and settings


def test_presets():
    assert set(PROVIDERS) == {"openrouter", "ollama", "openai"}
    assert get_provider(None).name == "openrouter"
    assert PROVIDERS["openrouter"].default_base_url == "https://openrouter.ai/api/v1"
    assert PROVIDERS["openrouter"].extra_headers == {"X-Title": "radar-desk"}
    assert PROVIDERS["ollama"].default_base_url == "http://localhost:11434/v1"
    assert PROVIDERS["openai"].default_base_url == "https://api.openai.com/v1"
    assert [p.key_required for p in PROVIDERS.values()] == [True, False, True]
    assert PROVIDERS["ollama"].extra_headers == {} == PROVIDERS["openai"].extra_headers
    with pytest.raises(KeyError):
        get_provider("nope")
    assert ollama_origin("http://localhost:11434/v1/") == "http://localhost:11434"
    assert ollama_origin("http://box:11434") == "http://box:11434"


def test_base_url_resolution_and_configured(make_services):
    s = make_services(llm_provider="ollama").settings
    assert s.resolved_llm_base_url == "http://localhost:11434/v1"
    assert not llm_configured(s)  # no model
    s = make_services(llm_provider="ollama", llm_base_url="http://gpu-box:11434/v1", chat_model="qwen3:8b").settings
    assert s.resolved_llm_base_url == "http://gpu-box:11434/v1"
    assert llm_configured(s)
    s = make_services(llm_base_url="https://openrouter.ai/api/v1", chat_model="m").settings
    assert s.resolved_llm_base_url == "https://openrouter.ai/api/v1" and not llm_configured(s)


@pytest.mark.parametrize(
    ("provider", "key", "configured", "url"),
    [
        ("openrouter", None, False, None),
        ("openrouter", "k", True, "https://openrouter.ai/api/v1/chat/completions"),
        ("ollama", None, True, "http://localhost:11434/v1/chat/completions"),
        ("openai", None, False, None),
        ("openai", "k", True, "https://api.openai.com/v1/chat/completions"),
    ],
)
def test_make_llm_per_provider(make_services, provider, key, configured, url):
    extra = {"llm_api_key": key} if key else {}
    llm = make_llm(make_services(llm_provider=provider, chat_model="m", **extra).settings)
    if not configured:
        assert llm is None
        return
    assert isinstance(llm, OpenAICompatibleLLM)
    assert llm.url == url and llm.provider.name == provider


def test_make_llm_needs_a_model_even_for_ollama(make_services):
    assert make_llm(make_services(llm_provider="ollama").settings) is None


# Client headers and error shapes


async def test_ollama_request_has_no_auth_and_no_title():
    seen: list[httpx.Request] = []
    client = mock_client(lambda r: httpx.Response(200, text=sse("[DONE]")), seen)
    llm = OpenAICompatibleLLM("http://localhost:11434/v1", None, "qwen3:8b", client=client,
                              provider=PROVIDERS["ollama"])
    assert await collect(llm) == [Done("stop")]
    assert "authorization" not in seen[0].headers and "x-title" not in seen[0].headers


async def test_openrouter_request_keeps_title_and_bearer():
    seen: list[httpx.Request] = []
    client = mock_client(lambda r: httpx.Response(200, text=sse("[DONE]")), seen)
    await collect(OpenAICompatibleLLM("https://openrouter.ai/api/v1", "sk-or", "m", client=client))
    assert seen[0].headers["authorization"] == "Bearer sk-or"
    assert seen[0].headers["x-title"] == "radar-desk"


async def test_openrouter_numeric_code_error():
    body = {"error": {"code": 402, "message": "Insufficient credits", "metadata": {"provider_name": "x"}}}
    llm = OpenAICompatibleLLM("https://o/api/v1", "k", "m", client=mock_client(lambda r: httpx.Response(402, json=body)))
    with pytest.raises(LlmError) as e:
        await collect(llm)
    assert e.value.status == 402 and str(e.value) == "the model endpoint returned 402: Insufficient credits"


async def test_ollama_error_shapes():
    # Ollama's openai.go Error: {"message", "type", "param", "code": string or null}
    not_found = {"error": {"message": "model 'qwen3:8b' not found", "type": "not_found_error",
                           "param": None, "code": None}}
    llm = OpenAICompatibleLLM("http://localhost:11434/v1", None, "qwen3:8b", provider=PROVIDERS["ollama"],
                              client=mock_client(lambda r: httpx.Response(404, json=not_found)))
    with pytest.raises(LlmError) as e:
        await collect(llm)
    assert e.value.status == 404
    assert str(e.value) == (
        "the model endpoint returned 404: model 'qwen3:8b' not found (run `ollama pull` for the model)"
    )
    bad = {"error": {"message": "invalid tool", "type": "invalid_request_error", "param": None, "code": "bad_tool"}}
    llm = OpenAICompatibleLLM("http://localhost:11434/v1", None, "m", provider=PROVIDERS["ollama"],
                              client=mock_client(lambda r: httpx.Response(400, json=bad)))
    with pytest.raises(LlmError, match="returned 400: invalid tool$"):
        await collect(llm)
    # A string code mid-stream is no HTTP status.
    llm = OpenAICompatibleLLM("http://localhost:11434/v1", None, "m", provider=PROVIDERS["ollama"],
                              client=mock_client(lambda r: httpx.Response(200, text=sse(bad))))
    with pytest.raises(LlmError) as e:
        await collect(llm)
    assert e.value.status is None and str(e.value) == "the model endpoint reported an error: invalid tool"


# Ollama-style tool call streams


def _chunk(delta: dict, finish: str | None = None) -> dict:
    return {"id": "chatcmpl-1", "object": "chat.completion.chunk", "model": "qwen3:8b",
            "system_fingerprint": "fp_ollama", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


async def test_ollama_whole_tool_calls_sharing_index_zero():
    first = {"id": "call_aaa", "index": 0, "type": "function",
             "function": {"name": "get_scores", "arguments": "{\"job_id\":\"job_1\"}"}}
    second = {"id": "call_bbb", "index": 0, "type": "function",
              "function": {"name": "jump_to_organ", "arguments": "{\"organ\":\"Liver\"}"}}
    text = sse(
        _chunk({"role": "assistant", "content": "", "tool_calls": [first]}),
        _chunk({"role": "assistant", "content": "", "tool_calls": [second]}),
        _chunk({}, "tool_calls"),
        "[DONE]",
    )
    llm = OpenAICompatibleLLM("http://localhost:11434/v1", None, "qwen3:8b", provider=PROVIDERS["ollama"],
                              client=mock_client(lambda r: httpx.Response(200, text=text)))
    assert await collect(llm) == [
        ToolCall("call_aaa", "get_scores", {"job_id": "job_1"}),
        ToolCall("call_bbb", "jump_to_organ", {"organ": "Liver"}),
        Done("tool_calls"),
    ]


async def test_ollama_two_calls_in_one_delta_with_their_own_indexes():
    calls = [
        {"id": "call_a", "index": 0, "type": "function", "function": {"name": "gpu_status", "arguments": "{}"}},
        {"id": "call_b", "index": 1, "type": "function", "function": {"name": "get_job", "arguments": "{\"job_id\":\"j\"}"}},
    ]
    text = sse(_chunk({"tool_calls": calls}), _chunk({}, "tool_calls"), "[DONE]")
    llm = OpenAICompatibleLLM("http://x/v1", None, "m", provider=PROVIDERS["ollama"],
                              client=mock_client(lambda r: httpx.Response(200, text=text)))
    events = await collect(llm)
    assert [e.id for e in events if isinstance(e, ToolCall)] == ["call_a", "call_b"]
    assert events[-1] == Done("tool_calls")


# Status checks


def settings_for(make_services, provider="openrouter", key="sk-test", model="m", **extra):
    kw = {"llm_provider": provider, "chat_model": model, **extra}
    if key:
        kw["llm_api_key"] = key
    return make_services(**kw).settings


async def status_with(settings, handler, seen=None):
    async with mock_client(handler, seen) as client:
        return await check_provider(settings, client)


async def test_openrouter_status_ok_and_detail(make_services):
    seen: list[httpx.Request] = []
    data = {"data": {"label": "sk-or-v1-abc...", "limit": 10, "limit_remaining": 4.5, "usage": 5.5,
                     "is_free_tier": False, "usage_daily": 0.1}}
    st = await status_with(settings_for(make_services), lambda r: httpx.Response(200, json=data), seen)
    assert str(seen[0].url) == "https://openrouter.ai/api/v1/key"
    assert seen[0].headers["authorization"] == "Bearer sk-test"
    assert st.ok and st.configured and st.problem is None
    assert "label" not in st.detail  # a key-derived label never leaves the server
    assert st.detail == {"limit": 10, "limit_remaining": 4.5, "usage": 5.5,
                         "is_free_tier": False}
    assert st.base_url_host == "openrouter.ai" and st.provider == "openrouter" and st.model == "m"
    assert "sk-test" not in json.dumps(st.to_dict())


async def test_openrouter_status_problems(make_services):
    s = settings_for(make_services)
    st = await status_with(s, lambda r: httpx.Response(401, json={"error": {"code": 401, "message": "No auth"}}))
    assert not st.ok and st.problem == "the OpenRouter key was rejected"
    st = await status_with(s, lambda r: httpx.Response(200, json={"data": {"limit": 5, "limit_remaining": 0}}))
    assert not st.ok and st.problem == "this key's OpenRouter credit limit is used up"
    st = await status_with(s, lambda r: httpx.Response(200, json={"data": {"limit": None, "limit_remaining": None}}))
    assert st.ok  # unlimited key


async def test_ollama_status(make_services):
    tags = {"models": [{"name": "qwen3:8b", "model": "qwen3:8b"}, {"name": "llama3.2:latest", "model": "llama3.2:latest"}]}

    def ollama(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json=tags)
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.12.3"})
        return httpx.Response(404)

    seen: list[httpx.Request] = []
    st = await status_with(settings_for(make_services, "ollama", key=None, model="qwen3:8b"), ollama, seen)
    assert st.ok and st.detail == {"models": 2, "version": "0.12.3"}
    assert str(seen[0].url) == "http://localhost:11434/api/tags"
    assert st.base_url_host == "localhost:11434"
    st = await status_with(settings_for(make_services, "ollama", key=None, model="llama3.2"), ollama)
    assert st.ok  # name without :latest
    st = await status_with(settings_for(make_services, "ollama", key=None, model="qwen3:32b"), ollama)
    assert not st.ok and st.problem == "model qwen3:32b is not pulled; run `ollama pull qwen3:32b`"

    def down(request):
        raise httpx.ConnectError("connection refused", request=request)

    st = await status_with(settings_for(make_services, "ollama", key=None, model="qwen3:8b"), down)
    assert not st.ok and st.problem == "Ollama is not running at localhost:11434"


async def test_openai_status(make_services):
    seen: list[httpx.Request] = []
    s = settings_for(make_services, "openai", llm_base_url="https://llm.example.com/v1")
    st = await status_with(s, lambda r: httpx.Response(200, json={"data": []}), seen)
    assert st.ok and str(seen[0].url) == "https://llm.example.com/v1/models"
    st = await status_with(s, lambda r: httpx.Response(401, json={"error": {"message": "bad key"}}))
    assert not st.ok and st.problem == "the key was rejected by llm.example.com"


async def test_status_timeout_and_unconfigured(make_services):
    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)

    st = await status_with(settings_for(make_services), slow)
    assert not st.ok and st.problem == "openrouter.ai did not answer within 5 s"

    def boom(request):
        raise AssertionError("no request expected")

    st = await status_with(settings_for(make_services, key=None), boom)
    assert not st.configured and not st.ok
    assert st.problem == "Chat is not configured: set CHAT_MODEL and, for OpenRouter, LLM_API_KEY."
    st = await status_with(settings_for(make_services, "ollama", key=None, model=None), boom)
    assert st.problem == "Chat is not configured: set CHAT_MODEL."


async def test_status_cache_keeps_result_for_30_seconds(make_services):
    calls = []
    now = [100.0]

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"data": {"limit_remaining": 1}})

    cache = StatusCache(settings_for(make_services), transport=httpx.MockTransport(handler), clock=lambda: now[0])
    first = await cache.get()
    now[0] += 29
    assert await cache.get() is first and len(calls) == 1
    now[0] += 2
    assert await cache.get() is not first and len(calls) == 2


# Dispatch surfaces the provider problem


async def test_dispatch_reports_provider_problem(make_services):
    svc = make_services(llm_provider="ollama", chat_model="qwen3:8b")

    def down(request):
        raise httpx.ConnectError("refused", request=request)

    llm = FakeLLM([text_turn("never")])
    agent = ChatAgent(svc, llm, status=StatusCache(svc.settings, transport=httpx.MockTransport(down)))
    body = {"messages": [{"role": "user", "content": "hi"}], "context": {}}
    frames = parse_frames("".join([f async for f in agent.dispatch(body)]))
    assert [f["event"] for f in frames] == ["execution_start", "execution_error"]
    assert frames[1]["data"]["error"] == {"code": "provider_unavailable",
                                          "message": "Ollama is not running at localhost:11434"}
    assert llm.calls == []


async def test_dispatch_runs_when_provider_is_ok(make_services):
    svc = make_services(llm_provider="ollama", chat_model="qwen3:8b")
    tags = {"models": [{"name": "qwen3:8b"}]}
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=tags if r.url.path == "/api/tags" else {}))
    agent = ChatAgent(svc, FakeLLM([text_turn("hello")]), status=StatusCache(svc.settings, transport=transport))
    body = {"messages": [{"role": "user", "content": "hi"}], "context": {}}
    frames = parse_frames("".join([f async for f in agent.dispatch(body)]))
    assert frames[-1]["event"] == "execution_complete"


# Review follow-ups


@pytest.mark.parametrize(
    ("base_url", "provider", "expected"),
    [
        (None, None, "openrouter"),
        ("https://openrouter.ai/api/v1", None, "openrouter"),
        ("http://localhost:11434/v1", None, "ollama"),
        ("http://127.0.0.1:11434/v1", None, "ollama"),
        ("http://localhost:8080/v1", None, "openai"),
        ("https://api.deepinfra.com/v1/openai", None, "openai"),
        ("https://api.deepinfra.com/v1/openai", "openrouter", "openrouter"),
        ("http://gpu-box:11434/v1", "ollama", "ollama"),
    ],
)
def test_provider_inferred_from_base_url(make_services, base_url, provider, expected):
    extra = {k: v for k, v in (("llm_base_url", base_url), ("llm_provider", provider)) if v}
    assert make_services(**extra).settings.resolved_llm_provider == expected


def test_pre_provider_setup_keeps_working(make_services):
    s = make_services(llm_base_url="http://localhost:11434/v1", chat_model="qwen3:8b").settings
    llm = make_llm(s)
    assert llm is not None and llm.provider.name == "ollama"  # no key needed, inferred from the URL
    s = make_services(llm_base_url="https://api.deepinfra.com/v1/openai", chat_model="m", llm_api_key="k").settings
    llm = make_llm(s)
    assert llm.provider.name == "openai" and llm.url == "https://api.deepinfra.com/v1/openai/chat/completions"


async def test_openai_status_without_models_endpoint(make_services):
    s = settings_for(make_services, "openai", llm_base_url="https://llm.example.com/v1")
    for code in (404, 405):
        st = await status_with(s, lambda r, c=code: httpx.Response(c))
        assert st.ok and st.detail == {"models_endpoint": "absent"}
    st = await status_with(s, lambda r: httpx.Response(403))
    assert not st.ok and st.problem == "the key was rejected by llm.example.com"
    st = await status_with(s, lambda r: httpx.Response(503, json={"error": {"message": "down"}}))
    assert not st.ok and st.problem == "llm.example.com answered 503: down"


async def test_openrouter_server_error_names_status_not_key(make_services):
    s = settings_for(make_services, key="sk-or-secret")
    st = await status_with(s, lambda r: httpx.Response(500, json={"error": {"code": 500, "message": "oops"}}))
    assert not st.ok and st.problem == "OpenRouter answered 500: oops"
    assert "sk-or-secret" not in json.dumps(st.to_dict())


async def test_ollama_version_optional_and_bad_json(make_services):
    s = settings_for(make_services, "ollama", key=None, model="qwen3:8b")

    def no_version(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]})
        raise httpx.ConnectError("gone", request=request)

    st = await status_with(s, no_version)
    assert st.ok and st.detail == {"models": 1}

    def html_version(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]})
        return httpx.Response(200, text="<html>not json</html>")

    st = await status_with(s, html_version)
    assert st.ok and "version" not in st.detail
    st = await status_with(s, lambda r: httpx.Response(200, text="<html>proxy page</html>"))
    assert not st.ok and st.problem == "localhost:11434 did not answer like Ollama (no model list)"


async def test_ollama_custom_host_in_not_running_message(make_services):
    s = settings_for(make_services, "ollama", key=None, model="qwen3:8b", llm_base_url="http://gpu-box:11500/v1")

    def down(request):
        raise httpx.ConnectError("refused", request=request)

    st = await status_with(s, down)
    assert st.problem == "Ollama is not running at gpu-box:11500"


async def test_status_cache_keeps_a_failure_for_5_seconds(make_services):
    calls = []
    now = [100.0]

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json={"data": {}})

    cache = StatusCache(settings_for(make_services), transport=httpx.MockTransport(handler), clock=lambda: now[0])
    assert not (await cache.get()).ok
    now[0] += 4
    assert not (await cache.get()).ok and len(calls) == 1
    now[0] += 2
    assert (await cache.get()).ok and len(calls) == 2


async def test_whole_calls_without_ids_sharing_index_zero():
    first = {"index": 0, "type": "function", "function": {"name": "gpu_status", "arguments": "{}"}}
    second = {"index": 0, "type": "function", "function": {"name": "get_job", "arguments": "{\"job_id\":\"j\"}"}}
    text = sse(_chunk({"tool_calls": [first]}), _chunk({"tool_calls": [second]}), _chunk({}, "tool_calls"), "[DONE]")
    llm = OpenAICompatibleLLM("http://x/v1", None, "m", provider=PROVIDERS["ollama"],
                              client=mock_client(lambda r: httpx.Response(200, text=text)))
    events = await collect(llm)
    calls = [e for e in events if isinstance(e, ToolCall)]
    assert [(c.name, c.arguments) for c in calls] == [("gpu_status", {}), ("get_job", {"job_id": "j"})]
    assert calls[0].id != calls[1].id
