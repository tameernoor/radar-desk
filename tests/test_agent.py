from __future__ import annotations

import copy
import json

import pytest

from radar_desk.chat.agent import ChatAgent, ChatNotFound, make_llm
from radar_desk.chat.llm import Done, LlmError, OpenAICompatibleLLM, TextDelta, ToolCall
from radar_desk.chat.prompt import system_prompt
from radar_desk.chat.tools import TOOLS
from radar_desk.chat.wire import parse_frames
from radar_desk.gpu.fake import canned_result
from radar_desk.radar import catalog
from test_chat_tools import done_job, ready_scan

PAGE_TOOLS = [
    {"name": "jump_to_organ", "description": "Move the viewer", "origin": "webmcp",
     "parametersSchema": {"type": "object", "properties": {"organ": {"type": "string"}}, "required": ["organ"]}},
    {"name": "set_window", "description": "Window preset", "origin": "webmcp"},
]


class FakeLLM:
    """Scripted turns: each turn is a list of events, or an exception to raise."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls: list[dict] = []

    async def stream_chat(self, messages, tools):
        self.calls.append({"messages": copy.deepcopy(messages), "tools": tools})
        turn = self.turns.pop(0) if self.turns else [TextDelta("again"), Done("stop")]
        if isinstance(turn, Exception):
            raise turn
        for event in turn:
            yield event


def text_turn(*parts):
    return [*(TextDelta(p) for p in parts), Done("stop")]


def tool_turn(*calls):
    return [*calls, Done("tool_calls")]


async def run(gen) -> list[dict]:
    return parse_frames("".join([f async for f in gen]))


def events(frames):
    return [f["event"] for f in frames]


def body(scan_id="", job_id=None, text="What does the liver show?", client_tools=PAGE_TOOLS):
    return {
        "agent": "radar-desk",
        "messages": [{"id": "m1", "role": "user", "content": [{"type": "text", "text": text}]}],
        "context": {"scan_id": scan_id, "job_id": job_id},
        "clientTools": client_tools,
    }


@pytest.fixture
def seeded(make_services, tmp_path):
    svc = make_services()
    scan = ready_scan(svc, tmp_path)
    result = canned_result("job_one")
    result["organs_scored"] = [
        {"organ": "Liver", "label": 21, "how": "window", "window_index": 2, "box_mm": [[0, 0, 0], [1, 1, 1]]},
        {"organ": "Spleen", "label": 30, "how": "centered_crop", "window_index": None,
         "box_mm": [[0, 0, 0], [1, 1, 1]]},
    ]
    for f in result["findings"]:
        if f["organ"] == "Gallbladder":
            f["prob"] = None
    result["organs_not_found"] = ["Gallbladder"]
    job = done_job(svc, scan.id, "job_one", result=result)
    return svc, scan, job


async def test_text_only_turn_golden_sequence(seeded):
    svc, scan, job = seeded
    llm = FakeLLM([text_turn("The liver ", "scores are low.")])
    frames = await run(ChatAgent(svc, llm).dispatch(body(scan.id, job.id)))
    assert events(frames) == [
        "execution_start", "turn_start", "text_start", "text_delta", "text_delta",
        "text_complete", "turn_complete", "execution_complete",
    ]
    ids = {f["data"]["executionId"] for f in frames}
    assert len(ids) == 1
    assert [f["data"]["seq"] for f in frames] == list(range(len(frames)))
    assert frames[1]["data"]["iteration"] == 1
    ex = svc.db.get_execution(ids.pop())
    assert ex.state == "done" and ex.scan_id == scan.id and ex.job_id == job.id
    assert [m["role"] for m in ex.messages] == ["system", "user", "assistant"]
    assert ex.messages[-1]["content"] == "The liver scores are low."
    sent_tools = [t["function"]["name"] for t in llm.calls[0]["tools"]]
    assert sent_tools == [t.name for t in TOOLS] + ["jump_to_organ", "set_window"]
    assert llm.calls[0]["tools"][-1]["function"]["parameters"] == {"type": "object", "properties": {}}


async def test_history_is_flattened_and_replays_skipped(seeded):
    svc, scan, job = seeded
    llm = FakeLLM([text_turn("ok")])
    b = body(scan.id, job.id)
    b["messages"] = [
        {"role": "system", "content": "client system text"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
        {"role": "assistant", "content": [{"type": "tool-call", "toolCallId": "c1", "toolName": "jump_to_organ"}]},
        {"role": "tool", "content": [{"type": "tool-result", "toolCallId": "c1", "output": "done"}]},
        {"role": "user", "content": [{"type": "text", "text": "second"}, {"type": "image", "url": "x"}]},
    ]
    await run(ChatAgent(svc, llm).dispatch(b))
    sent = llm.calls[0]["messages"]
    assert sent[0]["role"] == "system" and "client system text" not in sent[0]["content"]
    assert sent[1:] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "second"},
    ]


async def test_server_tool_runs_in_process_and_loop_continues(seeded):
    svc, scan, job = seeded
    llm = FakeLLM([
        tool_turn(ToolCall("call_1", "get_scores", {"job_id": job.id, "organ": "Liver"})),
        text_turn("Done."),
    ])
    frames = await run(ChatAgent(svc, llm).dispatch(body(scan.id, job.id)))
    assert events(frames) == [
        "execution_start", "turn_start", "tool_start", "tool_complete",
        "text_start", "text_delta", "text_complete", "turn_complete", "execution_complete",
    ]
    start, complete = frames[2]["data"], frames[3]["data"]
    assert start["toolCallId"] == "call_1" and start["toolName"] == "get_scores"
    assert complete["success"] is True
    assert all(f["organ"] == "Liver" for f in complete["result"]["findings"])
    second = llm.calls[1]["messages"]
    assert second[-2]["tool_calls"][0]["id"] == "call_1"
    assert second[-1]["role"] == "tool" and second[-1]["tool_call_id"] == "call_1"
    assert json.loads(second[-1]["content"])["job_id"] == job.id


async def test_tool_error_is_unsuccessful_and_loop_continues(seeded):
    svc, scan, job = seeded
    llm = FakeLLM([tool_turn(ToolCall("call_1", "get_job", {"job_id": "job_nope"})), text_turn("Sorry.")])
    frames = await run(ChatAgent(svc, llm).dispatch(body(scan.id, job.id)))
    complete = next(f["data"] for f in frames if f["event"] == "tool_complete")
    assert complete["success"] is False
    assert complete["result"]["error"]["message"] == "no job job_nope"
    assert events(frames)[-1] == "execution_complete"


async def test_page_tool_awaits_then_resume_after_restart(seeded, make_services):
    svc, scan, job = seeded
    llm = FakeLLM([
        [TextDelta("Jumping."), ToolCall("call_p", "jump_to_organ", {"organ": "Liver"}), Done("tool_calls")],
    ])
    frames = await run(ChatAgent(svc, llm).dispatch(body(scan.id, job.id)))
    assert events(frames) == [
        "execution_start", "turn_start", "text_start", "text_delta", "text_complete", "await",
    ]
    awaited = frames[-1]["data"]
    assert awaited["toolName"] == "jump_to_organ"
    assert awaited["toolCallId"] == "call_p"
    assert awaited["origin"] == "webmcp"
    assert awaited["parameters"] == {"organ": "Liver"}
    execution_id = awaited["executionId"]
    ex = svc.db.get_execution(execution_id)
    assert ex.state == "awaiting"
    assert ex.pending_tool_calls == [{"id": "call_p", "name": "jump_to_organ", "arguments": {"organ": "Liver"}}]

    # A fresh services and agent over the same database, as after a restart.
    svc2 = make_services()
    llm2 = FakeLLM([text_turn("The viewer is on the liver.")])
    agent2 = ChatAgent(svc2, llm2)
    frames = await run(agent2.resume({
        "executionId": execution_id,
        "toolOutputs": {"call_p": {"content": [{"type": "text", "text": "jumped to Liver"}]}},
        "streamResponse": True,
    }))
    assert events(frames) == [
        "turn_start", "text_start", "text_delta", "text_complete", "turn_complete", "execution_complete",
    ]
    assert {f["data"]["executionId"] for f in frames} == {execution_id}
    assert frames[0]["data"]["iteration"] == 2
    assert frames[-2]["data"]["iteration"] == 2
    sent = llm2.calls[0]["messages"]
    assert sent[-1] == {"role": "tool", "tool_call_id": "call_p", "content": "jumped to Liver"}
    assert sent[-2]["tool_calls"][0]["function"] == {"name": "jump_to_organ", "arguments": '{"organ": "Liver"}'}
    ex = svc2.db.get_execution(execution_id)
    assert ex.state == "done" and ex.iteration == 2 and ex.pending_tool_calls == []

    with pytest.raises(ChatNotFound):
        agent2.resume({"executionId": execution_id, "toolOutputs": {}})
    with pytest.raises(ChatNotFound):
        agent2.resume({"executionId": "exec_unknown", "toolOutputs": {}})


async def test_two_page_tools_two_awaits_one_resume(seeded):
    svc, scan, job = seeded
    llm = FakeLLM([
        tool_turn(
            ToolCall("call_a", "jump_to_organ", {"organ": "Liver"}),
            ToolCall("call_s", "gpu_status", {}),
            ToolCall("call_b", "set_window", {"preset": "liver"}),
        ),
        text_turn("Both done."),
    ])
    agent = ChatAgent(svc, llm)
    frames = await run(agent.dispatch(body(scan.id, job.id)))
    assert events(frames) == [
        "execution_start", "turn_start", "tool_start", "tool_complete", "await", "await",
    ]
    assert [f["data"]["toolCallId"] for f in frames if f["event"] == "await"] == ["call_a", "call_b"]
    execution_id = frames[0]["data"]["executionId"]
    assert [c["id"] for c in svc.db.get_execution(execution_id).pending_tool_calls] == ["call_a", "call_b"]

    frames = await run(agent.resume({
        "executionId": execution_id,
        "toolOutputs": {
            "call_a": {"content": [{"type": "text", "text": "ok a"}]},
            "call_b": {"content": [{"type": "text", "text": "no such preset"}], "isError": True},
        },
    }))
    assert events(frames)[0] == "turn_start" and events(frames)[-1] == "execution_complete"
    tool_msgs = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["call_s", "call_a", "call_b"]
    assert tool_msgs[1]["content"] == "ok a"
    assert json.loads(tool_msgs[2]["content"])["error"]["message"] == "no such preset"


async def test_resume_with_missing_output_sends_error_tool_message(seeded):
    svc, scan, job = seeded
    llm = FakeLLM([tool_turn(ToolCall("call_a", "jump_to_organ", {"organ": "Liver"})), text_turn("ok")])
    agent = ChatAgent(svc, llm)
    frames = await run(agent.dispatch(body(scan.id, job.id)))
    await run(agent.resume({"executionId": frames[0]["data"]["executionId"], "toolOutputs": {}}))
    tool_msg = llm.calls[1]["messages"][-1]
    assert json.loads(tool_msg["content"])["error"]["class"] == "missing_output"


@pytest.mark.parametrize(
    ("exc", "code"), [(LlmError("the model endpoint returned 500: x", 500), "llm_error"), (ValueError("bad"), "agent_error")]
)
async def test_exception_becomes_execution_error(seeded, exc, code):
    svc, scan, job = seeded
    frames = await run(ChatAgent(svc, FakeLLM([exc])).dispatch(body(scan.id, job.id)))
    assert events(frames) == ["execution_start", "execution_error"]
    err = frames[-1]["data"]["error"]
    assert err["code"] == code
    if code == "llm_error":
        assert err["message"] == "the model endpoint returned 500: x"
    assert svc.db.get_execution(frames[0]["data"]["executionId"]).state == "error"


async def test_llm_error_on_resume_marks_error(seeded):
    svc, scan, job = seeded
    llm = FakeLLM([tool_turn(ToolCall("call_a", "jump_to_organ", {})), LlmError("down")])
    agent = ChatAgent(svc, llm)
    frames = await run(agent.dispatch(body(scan.id, job.id)))
    execution_id = frames[0]["data"]["executionId"]
    frames = await run(agent.resume({"executionId": execution_id, "toolOutputs": {}}))
    assert events(frames) == ["execution_error"]
    assert svc.db.get_execution(execution_id).state == "error"
    with pytest.raises(ChatNotFound):
        agent.resume({"executionId": execution_id, "toolOutputs": {}})


async def test_llm_call_cap_stops_loops(seeded):
    svc, scan, job = seeded
    turns = [tool_turn(ToolCall(f"call_{i}", "gpu_status", {})) for i in range(20)]
    llm = FakeLLM(turns)
    frames = await run(ChatAgent(svc, llm, max_llm_calls=8).dispatch(body(scan.id, job.id)))
    assert len(llm.calls) == 8
    assert events(frames)[-1] == "execution_complete"
    assert "too many tool steps" in "".join(f["data"]["delta"] for f in frames if f["event"] == "text_delta")


async def test_unconfigured_chat(seeded):
    svc, scan, job = seeded
    frames = await run(ChatAgent(svc, None).dispatch(body(scan.id, job.id)))
    assert events(frames) == ["execution_start", "execution_error"]
    assert frames[1]["data"]["error"] == {
        "code": "chat_unconfigured",
        "message": "Chat is not configured: set CHAT_MODEL and, for OpenRouter, LLM_API_KEY.",
    }


def test_make_llm(make_services):
    assert make_llm(make_services().settings) is None
    assert make_llm(make_services(llm_api_key="k").settings) is None
    llm = make_llm(make_services(llm_api_key="k", chat_model="org/model").settings)
    assert isinstance(llm, OpenAICompatibleLLM)
    assert llm.model == "org/model" and llm.url == "https://openrouter.ai/api/v1/chat/completions"


def test_system_prompt_with_scores(seeded):
    svc, scan, job = seeded
    text = system_prompt(svc, scan.id, job.id)
    for sentence in (
        "research use, not diagnosis",
        "never from imagination",
        "closed at 146",
        "say its score and its organ",
        "50% is a display threshold, not a calibrated one",
        "not that the organ is normal",
        "Offer to jump the viewer",
    ):
        assert sentence in text
    assert "24 x 20 x 10 voxels" in text and scan.filename in text
    assert "- Liver: scored in window 2" in text
    assert "- Spleen: scored in centred crop" in text
    assert "- Gallbladder: not found" in text
    rows = [line for line in text.splitlines() if line.startswith("  ") and " | " in line]
    assert len(rows) == len(catalog.FINDINGS)
    first = catalog.FINDINGS[0]
    assert any(r.strip().startswith(f"{first['key']} | {first['finding']} | ") and r.endswith("%") for r in rows)
    assert "synthetic" in text  # canned results carry versions.gpu = "fake"


def test_system_prompt_without_job_or_scan(make_services, tmp_path):
    svc = make_services()
    assert "No scan is open" in system_prompt(svc, None, None)
    scan = ready_scan(svc, tmp_path)
    text = system_prompt(svc, scan.id, None)
    assert "not been scored yet" in text and " | " not in text
    job = svc.jobs.create(scan.id)
    assert f"Job {job.id} is queued" in system_prompt(svc, scan.id, job.id)


async def test_get_job_on_done_job_is_success(seeded):
    svc, scan, job = seeded
    llm = FakeLLM([tool_turn(ToolCall("call_1", "get_job", {"job_id": job.id})), text_turn("ok")])
    frames = await run(ChatAgent(svc, llm).dispatch(body(scan.id, job.id)))
    complete = next(f["data"] for f in frames if f["event"] == "tool_complete")
    assert complete["success"] is True
    assert complete["result"]["state"] == "done" and complete["result"]["error"] is None


async def test_awaiting_is_saved_before_the_first_await_frame(seeded):
    svc, scan, job = seeded
    llm = FakeLLM([tool_turn(ToolCall("call_a", "jump_to_organ", {"organ": "Liver"}))])
    gen = ChatAgent(svc, llm).dispatch(body(scan.id, job.id))
    seen = ""
    async for frame in gen:
        seen += frame
        if "event: await" in frame:
            break
    await gen.aclose()
    execution_id = parse_frames(seen)[0]["data"]["executionId"]
    ex = svc.db.get_execution(execution_id)
    assert ex.state == "awaiting" and [c["id"] for c in ex.pending_tool_calls] == ["call_a"]


async def test_bare_string_page_output_is_kept(seeded):
    svc, scan, job = seeded
    llm = FakeLLM([tool_turn(ToolCall("call_a", "jump_to_organ", {"organ": "Liver"})), text_turn("ok")])
    agent = ChatAgent(svc, llm)
    frames = await run(agent.dispatch(body(scan.id, job.id)))
    await run(agent.resume({"executionId": frames[0]["data"]["executionId"], "toolOutputs": {"call_a": "moved"}}))
    assert llm.calls[1]["messages"][-1] == {"role": "tool", "tool_call_id": "call_a", "content": "moved"}


def test_system_prompt_says_when_scores_come_from_an_older_job(seeded):
    svc, scan, job = seeded
    queued = svc.jobs.create(scan.id)
    text = system_prompt(svc, scan.id, queued.id)
    assert f"Job {queued.id} on screen is queued; the scores below are from job {job.id}." in text
    assert "on screen" not in system_prompt(svc, scan.id, job.id)


def test_system_prompt_view_rules_and_hu_table(make_services):
    text = system_prompt(make_services(), None, None)
    assert 'or "which slice is this", call get_view_state first.' in text
    assert "be labelled as a guess from density, never a diagnosis" in text
    assert "Outlines are RADAR's segmentation, not confirmed anatomy." in text
    assert "nothing is loaded in the viewer" in text
    assert "the CT is loaded but RADAR's outlines are not there yet" in text
    assert "- Air: about -1000" in text
    assert "- Cancellous bone: 300 to 400" in text
    assert "- Cortical bone: 500 to 1900" in text


VIEW_STATE = {
    "scan_id": "s", "job_id": "j", "plane": "axial",
    "slice": {"axis": "axial", "index": 41, "number": 42, "count": 120},
    "organs_on_slice": [
        {"organ": "Liver", "label": 21, "pixels": 9000, "percent_of_mask": 81.8, "scored": True},
        {"organ": "Spleen", "label": 30, "pixels": 2000, "percent_of_mask": 18.2, "scored": True},
    ],
    "crosshair": {"mm": [10.0, -40.0, 55.0], "hu": 58, "label": 21, "organ": "Liver"},
    "nearest_organ": None,
}


async def test_what_am_i_looking_at_reads_view_state_first(seeded):
    svc, scan, job = seeded
    llm = FakeLLM([
        tool_turn(ToolCall("call_v", "get_view_state", {})),
        text_turn("Axial slice 42 of 120. Liver and spleen are outlined."),
    ])
    tools = [*PAGE_TOOLS, {"name": "get_view_state", "description": "Read the viewer", "origin": "webmcp"}]
    agent = ChatAgent(svc, llm)
    frames = await run(agent.dispatch(body(scan.id, job.id, text="What am I looking at?", client_tools=tools)))
    assert events(frames)[-1] == "await"
    awaited = frames[-1]["data"]
    assert awaited["toolName"] == "get_view_state" and awaited["toolCallId"] == "call_v"
    output = {"content": [{"type": "text", "text": json.dumps(VIEW_STATE)}]}
    frames = await run(agent.resume({"executionId": awaited["executionId"], "toolOutputs": {"call_v": output}}))
    assert events(frames)[0] == "turn_start" and frames[0]["data"]["iteration"] == 2
    assert events(frames)[-2:] == ["turn_complete", "execution_complete"]
    tool_msg = llm.calls[1]["messages"][-1]
    assert tool_msg["tool_call_id"] == "call_v"
    assert json.loads(tool_msg["content"])["slice"] == {"axis": "axial", "index": 41, "number": 42, "count": 120}
    assert svc.db.get_execution(awaited["executionId"]).state == "done"
