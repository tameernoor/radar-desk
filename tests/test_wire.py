"""Tests for the persona wire emitter. Golden files hold the exact frames for four scenarios."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import pytest

from radar_desk.chat.wire import Emitter, parse_frames

GOLDEN = Path(__file__).parent / "golden"
CLOCK = "2026-10-01T00:00:00Z"
EXEC = "exec_1"


def fixed_ids():
    counters: dict[str, int] = defaultdict(int)

    def make(prefix: str) -> str:
        counters[prefix] += 1
        return f"{prefix}_{counters[prefix]}"

    return make


def emitter(**kwargs) -> Emitter:
    return Emitter(EXEC, id_factory=fixed_ids(), clock=lambda: CLOCK, **kwargs)


def text_turn() -> str:
    e = emitter()
    return e.start() + e.text_delta("Liver ") + e.text_delta("score is 12%.") + e.complete()


def server_tool_turn() -> str:
    e = emitter()
    out = e.start()
    out += e.text_delta("Checking the scores.")
    out += e.tool_call("get_scores", {"organ": "liver"}, {"scores": [{"key": "liver_lesion", "p": 0.12}]})
    out += e.text_delta("Liver lesion is 12%.")
    out += e.complete()
    return out


def page_tool_await() -> str:
    e = emitter()
    out = e.start()
    out += e.text_delta("Moving the view.")
    out += e.await_page_tool("call_page_1", "jump_to_organ", {"organ": "liver"})
    out += e.complete()
    return out


def resume_turn() -> str:
    e = emitter(iteration=2, announce_start=False)
    out = e.start()
    out += e.text_delta("The view is on the liver.")
    out += e.complete()
    return out


SCENARIOS = {
    "wire_text_turn.txt": text_turn,
    "wire_server_tool.txt": server_tool_turn,
    "wire_page_tool_await.txt": page_tool_await,
    "wire_resume_turn.txt": resume_turn,
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_golden(name: str) -> None:
    actual = SCENARIOS[name]()
    path = GOLDEN / name
    if os.environ.get("UPDATE_GOLDEN"):
        path.write_text(actual, encoding="utf-8")
    assert actual == path.read_text(encoding="utf-8")


def events(text: str) -> list[str]:
    return [f["event"] for f in parse_frames(text)]


def test_text_turn_order() -> None:
    assert events(text_turn()) == [
        "execution_start",
        "turn_start",
        "text_start",
        "text_delta",
        "text_delta",
        "text_complete",
        "turn_complete",
        "execution_complete",
    ]


def test_execution_start_payload() -> None:
    first = parse_frames(text_turn())[0]["data"]
    assert first["kind"] == "agent"
    assert first["agentId"] == "radar-desk"
    assert first["agentName"] == "radar-desk"
    assert first["startedAt"] == CLOCK


def test_text_complete_before_tool_frame() -> None:
    ev = events(server_tool_turn())
    i = ev.index("tool_start")
    assert ev[i - 1] == "text_complete"
    assert ev[i + 1] == "tool_complete"


def test_turn_opens_lazily() -> None:
    e = emitter()
    assert events(e.start()) == ["execution_start"]
    assert events(e.text_delta("hi"))[:2] == ["turn_start", "text_start"]


def test_await_frame_and_no_completion_after() -> None:
    frames = parse_frames(page_tool_await())
    ev = [f["event"] for f in frames]
    assert ev[-1] == "await"
    assert ev[-2] == "text_complete"
    assert "turn_complete" not in ev and "execution_complete" not in ev
    data = frames[-1]["data"]
    assert data["toolName"] == "jump_to_organ"
    assert data["origin"] == "webmcp"
    assert data["toolCallId"] == data["toolId"] == "call_page_1"
    assert data["parameters"] == {"organ": "liver"}
    assert data["awaitedAt"] == CLOCK


def test_paused_property_and_parallel_awaits() -> None:
    e = emitter()
    assert not e.paused
    out = e.await_page_tool("c1", "get_view_state", {}) + e.await_page_tool("c2", "set_window", {"p": "x"})
    assert e.paused
    assert events(out) == ["turn_start", "await", "await"]
    assert e.complete() == ""


def test_resume_has_no_execution_start() -> None:
    frames = parse_frames(resume_turn())
    assert frames[0]["event"] == "turn_start"
    assert frames[0]["data"]["iteration"] == 2
    assert "execution_start" not in [f["event"] for f in frames]


def test_error_then_complete_emits_nothing() -> None:
    e = emitter()
    out = e.start() + e.error("llm_error", "upstream 500")
    frames = parse_frames(out)
    assert frames[-1]["event"] == "execution_error"
    assert frames[-1]["data"]["error"] == {"code": "llm_error", "message": "upstream 500"}
    assert frames[-1]["data"]["kind"] == "agent"
    assert e.complete() == ""
    assert e.error("x", "y") == ""


def test_complete_is_idempotent() -> None:
    e = emitter()
    e.text_delta("a")
    assert e.complete() != ""
    assert e.complete() == ""


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_frame_properties(name: str) -> None:
    text = SCENARIOS[name]()
    raw_blocks = text.split("\n\n")
    assert raw_blocks[-1] == ""
    for block in raw_blocks[:-1]:
        lines = block.split("\n")
        assert len(lines) == 2
        assert lines[0].startswith("event: ")
        assert lines[1].startswith("data: ")
        data = json.loads(lines[1][len("data: ") :])
        assert data["type"] == lines[0][len("event: ") :]
        assert data["executionId"] == EXEC
    seqs = [f["data"]["seq"] for f in parse_frames(text)]
    assert seqs == list(range(len(seqs)))


def test_tool_failure_frame() -> None:
    e = emitter()
    out = e.tool_start("call_9", "score_scan", {"scan_id": "s"}) + e.tool_complete(
        "call_9", False, {"error": {"class": "ValueError", "message": "bad"}}
    )
    frames = parse_frames(out)
    assert [f["event"] for f in frames] == ["turn_start", "tool_start", "tool_complete"]
    assert frames[1]["data"]["toolType"] == "local"
    assert frames[2]["data"]["success"] is False


def test_unicode_is_not_escaped() -> None:
    e = emitter()
    assert "lever å" in e.text_delta("lever å")


def test_default_ids_and_clock() -> None:
    e = Emitter("exec_x")
    frames = parse_frames(e.start() + e.text_delta("a"))
    assert frames[1]["data"]["id"].startswith("turn_")
    assert frames[2]["data"]["id"].startswith("text_")
    assert frames[0]["data"]["startedAt"].endswith("Z")
    tc = parse_frames(e.tool_call("t", {}, {}))
    assert tc[1]["data"]["toolCallId"].startswith("call_")
