"""POST /chat and POST /chat/resume through the API, with a scripted LLM."""

from __future__ import annotations

from radar_desk.chat.llm import ToolCall
from radar_desk.chat.wire import parse_frames
from test_agent import FakeLLM, text_turn, tool_turn
from test_api import api, make_api  # noqa: F401  (fixtures)

PAGE_TOOL = {"name": "jump_to_organ", "description": "move the view",
             "parametersSchema": {"type": "object", "properties": {"organ": {"type": "string"}}},
             "origin": "webmcp"}


def frames(resp):
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/event-stream")
    return parse_frames(resp.text)


def test_chat_needs_login(api):
    assert api.anon.post("/chat", json={"messages": []}).status_code == 401


def test_chat_without_a_key_streams_unconfigured(api):
    out = frames(api.client.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]}))
    assert [f["event"] for f in out] == ["execution_start", "execution_error"]
    assert out[-1]["data"]["error"]["code"] == "chat_unconfigured"


def test_chat_body_must_be_an_object(api):
    assert api.client.post("/chat", content=b"[]", headers={"content-type": "application/json"}).status_code == 422


def test_text_turn_round_trip(api):
    api.app.state.chat_agent.llm = FakeLLM([text_turn("Liver 12%.")])
    out = frames(api.client.post("/chat", json={"messages": [{"role": "user", "content": "liver?"}]}))
    events = [f["event"] for f in out]
    assert events[0] == "execution_start" and events[-1] == "execution_complete"
    assert "".join(f["data"]["delta"] for f in out if f["event"] == "text_delta") == "Liver 12%."


def test_page_tool_await_and_resume(api):
    api.app.state.chat_agent.llm = FakeLLM([
        tool_turn(ToolCall("call_1", "jump_to_organ", {"organ": "Liver"})),
        text_turn("Done."),
    ])
    body = {"messages": [{"role": "user", "content": "show the liver"}], "clientTools": [PAGE_TOOL]}
    out = frames(api.client.post("/chat", json=body))
    waits = [f for f in out if f["event"] == "await"]
    assert len(waits) == 1 and waits[0]["data"]["toolName"] == "jump_to_organ"
    assert waits[0]["data"]["origin"] == "webmcp"
    assert out[-1]["event"] == "await"
    execution_id = waits[0]["data"]["executionId"]

    resumed = frames(api.client.post("/chat/resume", json={
        "executionId": execution_id,
        "toolOutputs": {"call_1": {"content": [{"type": "text", "text": "{\"ok\": true}"}]}},
        "streamResponse": True,
    }))
    assert all(f["data"]["executionId"] == execution_id for f in resumed)
    assert resumed[0]["event"] == "turn_start"
    assert resumed[0]["data"]["iteration"] == 2
    assert resumed[-1]["event"] == "execution_complete"


def test_resume_unknown_execution_is_404(api):
    resp = api.client.post("/chat/resume", json={"executionId": "exec_nope", "toolOutputs": {}})
    assert resp.status_code == 404
