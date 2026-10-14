"""NV-Reason-CT backends: the Modal one against a stub `modal` module, and the fake one."""

from __future__ import annotations

import builtins
import gzip
import sys
import types

import pytest

from radar_desk.config import Settings
from radar_desk.gpu.backend import Errored, Finished, Pending
from radar_desk.gpu.nvreason_backend import ModalNvReasonBackend, NvAskTimeout
from radar_desk.gpu.nvreason_fake import FAKE_REPORT_TEXT, FakeNvReasonBackend, synthesize_nv_result
from radar_desk.records import Job
from test_chat_tools import ready_scan


class StubCall:
    def __init__(self, value=None, object_id="fc-nv-1"):
        self.object_id = object_id
        self.value = value
        self.cancelled = False
        self.timeouts = []

    def get(self, timeout=None):
        self.timeouts.append(timeout)
        if isinstance(self.value, BaseException):
            raise self.value
        return self.value

    def cancel(self):
        self.cancelled = True


class StubMethod:
    def __init__(self, call):
        self.call = call
        self.args = None

    def spawn(self, *args):
        self.args = args
        return self.call


@pytest.fixture
def stub(monkeypatch):
    state = types.SimpleNamespace(names=None, report=StubMethod(StubCall()), ask=StubMethod(StubCall({"answer": "Yes."})))

    class StubCls:
        @staticmethod
        def from_name(app_name, class_name, **kw):
            state.names = (app_name, class_name, kw)
            return lambda: types.SimpleNamespace(report=state.report, ask=state.ask)

    mod = types.ModuleType("modal")
    exc = types.ModuleType("modal.exception")
    exc.OutputExpiredError = type("OutputExpiredError", (Exception,), {})
    exc.FunctionTimeoutError = type("FunctionTimeoutError", (Exception,), {})
    mod.exception = exc
    mod.Cls = StubCls
    monkeypatch.setitem(sys.modules, "modal", mod)
    monkeypatch.setitem(sys.modules, "modal.exception", exc)
    return state


def backend():
    return ModalNvReasonBackend(Settings(_env_file=None, owner_token="t", session_secret="s"))


def test_spawn_report_calls_the_deployed_class(stub):
    call_id = backend().spawn_report("job_nv", "https://bucket.example/s?sig=1")
    assert call_id == "fc-nv-1"
    assert stub.names == ("radar-desk-nvreason", "NvReason", {})
    assert stub.report.args == ("job_nv", "https://bucket.example/s?sig=1")


def test_ask_waits_up_to_the_timeout(stub):
    out = backend().ask("ref", "abdomen", [{"role": "user"}], False, 16, 240)
    assert out == {"answer": "Yes."}
    assert stub.ask.args == ("ref", "abdomen", [{"role": "user"}], False, 16)
    assert stub.ask.call.timeouts == [240]


def test_ask_timeout_cancels_the_call(stub):
    stub.ask.call.value = builtins.TimeoutError()
    with pytest.raises(NvAskTimeout, match="did not answer within 240 s"):
        backend().ask("ref", "abdomen", [], True, 2048, 240)
    assert stub.ask.call.cancelled


def test_ask_refuses_a_non_dict(stub):
    stub.ask.call.value = "Yes"
    with pytest.raises(TypeError, match="not a dict"):
        backend().ask("ref", "abdomen", [], False, 16, 240)


def test_synthesized_result_is_marked_fake_and_cuts_the_top_third(make_services, tmp_path):
    svc = make_services()
    scan = ready_scan(svc, tmp_path)
    job = Job(id="job_nv", scan_id=scan.id, kind="nvreason")
    out = synthesize_nv_result(job, scan)
    assert out["ok"] is True and out["versions"]["gpu"] == "fake" and out["nv_report"]["fake"] is True
    assert out["nv_report"]["text"] == FAKE_REPORT_TEXT
    assert "<answer>" not in out["nv_report"]["answer"]
    assert out["nv_report"]["answer_tag"] == "Liver mass or non-cystic lesion, Renal cyst"
    (lo, hi) = out["crop"]["box_mm"]
    assert lo[2] < hi[2] and out["crop"]["cut_above_mm"] > 0 and out["crop"]["fallback"] is False
    assert gzip.decompress(out["crop_gz"])[:4]


def test_fake_poll_delay_result_and_errors():
    fake = FakeNvReasonBackend(synthesize=lambda job_id: {"ok": True, "job_id": job_id}, delay_ticks=1)
    call = fake.spawn_report("job_a", "ref")
    assert call == "fc-nvfake-1"
    assert isinstance(fake.poll(call), Pending)
    assert fake.poll(call) == Finished({"ok": True, "job_id": "job_a"})
    fake.results["job_b"] = RuntimeError("boom")
    fake.delay_ticks = 0
    assert fake.poll(fake.spawn_report("job_b", "ref")) == Errored("RuntimeError", "boom")
    assert isinstance(fake.poll("fc-missing"), Errored)


def test_fake_ask_answers_yes_no_and_free_questions():
    fake = FakeNvReasonBackend()
    q = [{"role": "user", "content": [{"type": "text", "text": "Is there ascites? Answer only Yes or No."}]}]
    assert fake.ask("ref", "abdomen", q, False, 16, 240)["answer"] == "Yes."
    free = [{"role": "user", "content": [{"type": "text", "text": "Describe the liver."}]}]
    out = fake.ask("ref", "abdomen", free, True, 2048, 240)
    assert out["answer"] == "Fake NV-Reason-CT answer to: Describe the liver." and out["thinking"]
    assert len(fake.asks) == 2


def test_modal_nv_backend_cannot_start_a_radar_run(stub):
    with pytest.raises(NotImplementedError):
        backend().spawn(Job(id="job_x", scan_id="scan_x"), "https://x", {})
