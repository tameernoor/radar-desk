"""Asking NV-Reason-CT: one conversation per scan and region, seeded with its report, every answer in qa.jsonl."""

from __future__ import annotations

import threading
import time

import pytest

from api_client import api_client
from radar_desk.chat.agent import ChatAgent
from radar_desk.chat.llm import ToolCall
from radar_desk.chat.prompt import system_prompt
from radar_desk.chat.tools import TOOLS, ToolSpec, active_tools, run_tool
from radar_desk.chat.wire import parse_frames
from radar_desk.gpu.nvreason_backend import NvAskTimeout
from radar_desk.services import ServiceError
from radar_desk.services.nvreason import COLD_NOTICE, NV_REPORT_PROMPT, NV_SCALEDOWN_S, OFF, nv_keys, qa_key
from radar_desk.services.readers import _thread_runner
from readers_seed import NV_READ, QUERIES
from test_agent import FakeLLM, body, text_turn, tool_turn

YES_NO = "Is there free fluid in the abdomen or pelvis? Answer only Yes or No."


def nv_done(readers_world):
    """A scan with RADAR and NV done, NV's report read and the questions prepared."""
    w = readers_world([NV_READ, QUERIES])
    w.nv.delay_ticks = 1
    w.svc.jobs.create(w.scan.id)
    for _ in range(4):
        w.poller.tick()
    return w


def test_ask_starts_from_nv_report_and_logs_the_answer(readers_world):
    w = nv_done(readers_world)
    job = w.svc.nvreason.latest_done(w.scan.id)
    out = w.svc.readers.ask(w.scan.id, question=YES_NO)
    assert out["answer"] == "Yes." and out["label"] == "ascites" and out["warm"] == "warm"
    assert out["conversation"] == f"{job.id}:abdomen" and out["job_id"] == job.id
    sent = w.nv.asks[0]
    assert sent["enable_thinking"] is False and sent["max_new_tokens"] == 16
    assert sent["region"] == "abdomen" and sent["timeout_s"] == 240
    assert sent["source_ref"].split("?")[0].endswith(f"/scans/{w.scan.id}/source.nii.gz")
    assert [m["role"] for m in sent["messages"]] == ["user", "assistant", "user"]
    assert sent["messages"][0]["content"] == [{"type": "image"}, {"type": "text", "text": NV_REPORT_PROMPT}]
    assert sent["messages"][1]["content"] == [{"type": "text", "text": w.svc.nvreason.report(job.id)["text"]}]
    entries = w.svc.nvreason.qa_entries(job.id)
    assert len(entries) == 1
    assert entries[0]["question"] == YES_NO and entries[0]["thinking"] == "" and entries[0]["label"] == "ascites"


def test_a_follow_up_carries_the_history_with_thinking_on(readers_world):
    w = nv_done(readers_world)
    w.svc.readers.ask(w.scan.id, question=YES_NO)
    out = w.svc.readers.ask(w.scan.id, question="Where is it?")
    sent = w.nv.asks[1]
    assert [m["role"] for m in sent["messages"]] == ["user", "assistant", "user", "assistant", "user"]
    assert sent["messages"][3]["content"] == [{"type": "text", "text": "Yes."}]
    assert sent["enable_thinking"] is True and sent["max_new_tokens"] == 2048
    assert out["label"] is None and out["thinking"] == "Fake thinking."


def test_a_prepared_question_by_id_shows_in_its_row(readers_world):
    w = nv_done(readers_world)
    client = api_client(w.svc)
    view = w.svc.readers.view(w.scan.id)
    q = next(q for r in view["rows"] if r["id"] == "ascites" for q in r["queries"] if q["kind"] == "yes_no")
    r = client.post(f"/scans/{w.scan.id}/nvreason/ask", json={"query_id": q["id"]})
    assert r.status_code == 202, r.text
    assert r.json() == {"asked": q["id"]}
    row = next(r for r in w.svc.readers.view(w.scan.id)["rows"] if r["id"] == "ascites")
    assert row["asked"][0]["answer"] == "Yes." and row["asked"][0]["differs"] is True
    assert row["nv"]["state"] == "absent"  # the dot stays the report read
    assert client.post(f"/scans/{w.scan.id}/nvreason/ask", json={"query_id": "q_missing"}).status_code == 404
    assert client.post(f"/scans/{w.scan.id}/nvreason/ask", json={}).status_code == 422


def test_cold_and_warm(readers_world):
    w = nv_done(readers_world)
    assert w.svc.nvreason.cold_start_notice(w.scan.id) is None  # the job just finished
    w.clock.advance(NV_SCALEDOWN_S)
    assert w.svc.nvreason.cold_start_notice(w.scan.id) == COLD_NOTICE
    assert w.svc.readers.ask(w.scan.id, question="Describe the liver.")["warm"] == "cold"
    assert w.svc.nvreason.cold_start_notice(w.scan.id) is None


def test_not_read_yet(readers_world):
    w = readers_world(llm=False)
    with pytest.raises(ServiceError) as err:
        w.svc.readers.ask(w.scan.id, question=YES_NO)
    assert err.value.status == 409 and "has not read this scan" in err.value.detail
    assert w.svc.nvreason.cold_start_notice(w.scan.id) is None


def test_refused_when_off(readers_world):
    w = readers_world(llm=False, enabled=False)
    with pytest.raises(ServiceError) as err:
        w.svc.readers.ask(w.scan.id, question=YES_NO)
    assert err.value.status == 409 and err.value.detail == OFF


def test_a_timeout_is_a_504_and_logs_nothing(readers_world):
    w = nv_done(readers_world)
    w.nv.ask_error = NvAskTimeout("NV-Reason-CT did not answer within 240 s")
    with pytest.raises(ServiceError) as err:
        w.svc.readers.ask(w.scan.id, question=YES_NO)
    assert err.value.status == 504 and "240 s" in err.value.detail
    assert w.svc.nvreason.qa_entries(w.svc.nvreason.latest_done(w.scan.id).id) == []


def test_two_asks_at_once_keep_both_answers(readers_world):
    """Review Focus 4: a double click on a chip."""
    w = nv_done(readers_world)
    real_ask = w.nv.ask
    gate = threading.Barrier(2)

    def slow_ask(*args):
        gate.wait(timeout=5)
        time.sleep(0.05)
        return real_ask(*args)

    w.nv.ask = slow_ask
    errors: list[Exception] = []

    def ask():
        try:
            w.svc.readers.ask(w.scan.id, question=YES_NO)
        except Exception as exc:  # noqa: BLE001 - collected for the assert
            errors.append(exc)

    threads = [threading.Thread(target=ask) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert errors == []
    assert len(w.svc.nvreason.qa_entries(w.svc.nvreason.latest_done(w.scan.id).id)) == 2


def test_chat_tool_and_its_notice(readers_world):
    w = nv_done(readers_world)
    out = run_tool(w.svc, "ask_nv_reason", {"scan_id": w.scan.id, "question": YES_NO})
    assert out["answer"] == "Yes." and out["label"] == "ascites"
    spec = next(t for t in TOOLS if t.name == "ask_nv_reason")
    assert spec.notice(w.svc, {"scan_id": w.scan.id}) is None
    w.clock.advance(NV_SCALEDOWN_S)
    assert spec.notice(w.svc, {"scan_id": w.scan.id}) == COLD_NOTICE
    assert run_tool(w.svc, "ask_nv_reason", {"scan_id": w.scan.id})["error"]["class"] == "invalid_arguments"


async def test_the_agent_shows_the_notice_and_keeps_a_slow_tool_alive(readers_world):
    w = readers_world(llm=False)

    def slow(services, args):
        time.sleep(0.3)
        return {"answer": "Yes."}

    tool = ToolSpec("slow_tool", "A slow tool.", {"type": "object", "properties": {}, "required": [],
                                                   "additionalProperties": False},
                    slow, notice=lambda services, args: COLD_NOTICE)
    llm = FakeLLM([tool_turn(ToolCall("call_1", "slow_tool", {})), text_turn("Done.")])
    agent = ChatAgent(w.svc, llm, tools=[tool], keepalive_s=0.05)
    raw = "".join([f async for f in agent.dispatch(body(w.scan.id, None, client_tools=[]))])
    assert ": keepalive\n\n" in raw
    frames = [f for f in parse_frames(raw) if f["event"]]
    names = [f["event"] for f in frames]
    deltas = [f["data"]["delta"] for f in frames if f["event"] == "text_delta"]
    assert deltas[0] == COLD_NOTICE + "\n\n"
    assert names.index("text_delta") < names.index("tool_start") < names.index("tool_complete")
    assert names[-1] == "execution_complete"


def test_the_chat_rules_name_the_tool(readers_world):
    w = readers_world(llm=False)
    text = system_prompt(w.svc, None, None)
    assert "ask_nv_reason asks NVIDIA's NV-Reason-CT" in text
    assert "never paraphrase it" in text
    assert "Never tell NV-Reason-CT what RADAR scored" in text


def test_with_the_valve_off_the_chat_has_no_nv_tool_or_rule(make_services):
    svc = make_services()
    assert svc.nvreason.unavailable() is not None
    text = system_prompt(svc, None, None)
    assert "ask_nv_reason" not in text and "NV-Reason-CT" not in text
    assert "ask_nv_reason" not in [t.name for t in active_tools(svc, TOOLS)]


def test_with_the_valve_on_the_chat_offers_the_nv_tool(readers_world):
    w = readers_world(llm=False)
    assert "ask_nv_reason" in [t.name for t in active_tools(w.svc, TOOLS)]


def test_a_leading_chat_question_is_refused_without_calling_the_backend(readers_world):
    w = nv_done(readers_world)
    for q in ("RADAR scored the liver at 80%. Is there a lesion in the liver?",
              "The radiologist report says there is ascites. Do you agree?",
              "Has the cyst grown since the prior scan?"):
        out = run_tool(w.svc, "ask_nv_reason", {"scan_id": w.scan.id, "question": q})
        assert out["error"]["class"] == "ServiceError" and "about the scan only" in out["error"]["message"]
    assert w.nv.asks == []
    assert w.svc.nvreason.qa_entries(w.svc.nvreason.latest_done(w.scan.id).id) == []


def test_a_neutral_chat_question_goes_through_even_about_an_organ_outside_the_cube(readers_world):
    w = nv_done(readers_world)
    out = run_tool(w.svc, "ask_nv_reason", {"scan_id": w.scan.id, "question": "What do you see in the lungs?"})
    assert "error" not in out and len(w.nv.asks) == 1


def test_a_chip_question_still_goes_through_by_id(readers_world):
    w = nv_done(readers_world)
    out = w.svc.readers.ask(w.scan.id, query_id=_yes_no_query(w)["id"])
    assert out["answer"] == "Yes." and len(w.nv.asks) == 1


def test_scan_delete_removes_the_nv_crop_but_keeps_its_report(readers_world):
    w = nv_done(readers_world)
    job = w.svc.nvreason.latest_done(w.scan.id)
    keys = nv_keys(job.id)
    assert w.svc.storage.exists(keys["crop_nii"])
    w.svc.scans.delete(w.scan.id)
    assert not w.svc.storage.exists(keys["crop_nii"])
    assert w.svc.storage.exists(keys["nv_report"])


def test_an_answer_that_lands_after_the_scan_was_deleted_is_not_logged(readers_world):
    w = nv_done(readers_world)
    job = w.svc.nvreason.latest_done(w.scan.id)
    real_ask = w.nv.ask

    def ask_then_delete(*args):
        out = real_ask(*args)
        w.svc.db.delete_scan(w.scan.id)
        return out

    w.nv.ask = ask_then_delete
    w.svc.nvreason.ask(w.scan.id, "Where is the fluid?")
    assert w.svc.nvreason.qa_entries(job.id) == []
    assert not w.svc.storage.exists(qa_key(job.id))


def _yes_no_query(w):
    view = w.svc.readers.view(w.scan.id)
    return next(q for r in view["rows"] if r["id"] == "ascites" for q in r["queries"] if q["kind"] == "yes_no")


def test_the_chip_route_returns_at_once_and_the_view_follows_the_ask(readers_world):
    w = nv_done(readers_world)
    w.svc.readers.runner = _thread_runner
    q = _yes_no_query(w)
    started, release = threading.Event(), threading.Event()
    real_ask = w.nv.ask

    def held_ask(*args):
        started.set()
        release.wait(timeout=5)
        return real_ask(*args)

    w.nv.ask = held_ask
    t0 = time.monotonic()
    r = api_client(w.svc).post(f"/scans/{w.scan.id}/nvreason/ask", json={"query_id": q["id"]})
    assert r.status_code == 202 and time.monotonic() - t0 < 2
    assert started.wait(timeout=5)
    view = w.svc.readers.view(w.scan.id)
    assert view["nv"]["asking"] == [q["id"]] and view["nv"]["ask_errors"] == {}
    assert next(r for r in view["rows"] if r["id"] == "ascites")["asked"] == []
    # Pressing it again while it runs does not ask twice.
    assert api_client(w.svc).post(f"/scans/{w.scan.id}/nvreason/ask", json={"query_id": q["id"]}).status_code == 202
    release.set()
    for _ in range(100):
        if not w.svc.nvreason.ask_state(w.scan.id)["asking"]:
            break
        time.sleep(0.05)
    view = w.svc.readers.view(w.scan.id)
    assert view["nv"]["asking"] == []
    assert [a["answer"] for a in next(r for r in view["rows"] if r["id"] == "ascites")["asked"]] == ["Yes."]
    assert len(w.nv.asks) == 1


def test_a_failed_background_ask_shows_in_the_view_and_clears_in_flight(readers_world):
    w = nv_done(readers_world)
    q = _yes_no_query(w)
    w.nv.ask_error = NvAskTimeout("NV-Reason-CT did not answer within 240 s")
    r = api_client(w.svc).post(f"/scans/{w.scan.id}/nvreason/ask", json={"query_id": q["id"]})
    assert r.status_code == 202
    view = w.svc.readers.view(w.scan.id)
    assert view["nv"]["asking"] == []
    assert view["nv"]["ask_errors"] == {q["id"]: "could not ask NV-Reason-CT, try again"}
    w.nv.ask_error = None  # pressing it again clears the failure
    api_client(w.svc).post(f"/scans/{w.scan.id}/nvreason/ask", json={"query_id": q["id"]})
    assert w.svc.readers.view(w.scan.id)["nv"]["ask_errors"] == {}


def test_the_chip_route_still_refuses_up_front(readers_world):
    w = readers_world(llm=False)
    r = api_client(w.svc).post(f"/scans/{w.scan.id}/nvreason/ask", json={"query_id": "q_x"})
    assert r.status_code == 404
