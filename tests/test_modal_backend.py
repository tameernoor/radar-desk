"""ModalGpuBackend against a stub `modal` module; the real package is never imported."""

from __future__ import annotations

import sys
import types

import pytest

from radar_desk.config import Settings
from radar_desk.gpu.backend import Errored, Finished, Pending
from radar_desk.gpu.modal_backend import ModalGpuBackend
from radar_desk.records import Job


class StubError(Exception):
    pass


class FunctionTimeoutError(StubError):
    pass


class OutputExpiredError(StubError):
    pass


class StubConnectionError(StubError):
    pass


class StubCall:
    def __init__(self, outcome=None, object_id="fc-1"):
        self.object_id = object_id
        self.outcome = outcome
        self.cancelled = False
        self.get_timeouts = []
        self.logs = types.SimpleNamespace(tail=self._tail)
        self.tail_entries = None

    def get(self, timeout=None):
        self.get_timeouts.append(timeout)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome

    def cancel(self):
        self.cancelled = True

    def _tail(self, entries=100):
        self.tail_entries = entries
        return [types.SimpleNamespace(message="line one\n"), types.SimpleNamespace(message="line two\n")]


@pytest.fixture
def stub(monkeypatch):
    mod = types.ModuleType("modal")
    exc = types.ModuleType("modal.exception")
    exc.FunctionTimeoutError = FunctionTimeoutError
    exc.OutputExpiredError = OutputExpiredError
    exc.ConnectionError = StubConnectionError
    mod.exception = exc
    state = types.SimpleNamespace(from_name=[], spawned=[], calls={}, from_id=[], clients=[])

    class Function:
        @staticmethod
        def from_name(app, name, **kw):
            state.from_name.append((app, name, kw))
            fn = types.SimpleNamespace()

            def spawn(*args):
                state.spawned.append(args)
                return StubCall(object_id="fc-abc")

            fn.spawn = spawn
            return fn

    class FunctionCall:
        @staticmethod
        def from_id(call_id, **kw):
            state.from_id.append((call_id, kw))
            return state.calls[call_id]

    class Client:
        @staticmethod
        def from_credentials(tid, tsecret):
            state.clients.append((tid, tsecret))
            return "client-object"

    mod.Function, mod.FunctionCall, mod.Client = Function, FunctionCall, Client
    monkeypatch.setitem(sys.modules, "modal", mod)
    monkeypatch.setitem(sys.modules, "modal.exception", exc)
    return state


def settings(**kw):
    return Settings(_env_file=None, owner_token="o", session_secret="s", gpu_backend="modal", **kw)


def test_spawn_calls_deployed_function_and_returns_object_id(stub):
    backend = ModalGpuBackend(settings())
    job = Job(id="job_1", scan_id="scan_1")
    call_id = backend.spawn(job, "https://get", {"mask": "https://put"})
    assert call_id == "fc-abc"
    assert stub.from_name == [("radar-desk", "score", {})]
    job_id, source, urls, keys = stub.spawned[0]
    assert (job_id, source, urls) == ("job_1", "https://get", {"mask": "https://put"})
    assert keys["mask"] == "jobs/job_1/mask.nii.gz"


def test_spawn_uses_token_from_settings(stub):
    backend = ModalGpuBackend(settings(modal_token_id="ak-1", modal_token_secret="as-2"))
    backend.spawn(Job(id="job_1", scan_id="s"), "u", {})
    assert stub.clients == [("ak-1", "as-2")]
    assert stub.from_name[0][2] == {"client": "client-object"}


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (TimeoutError(), Pending()),
        ({"ok": True, "job_id": "job_1"}, Finished({"ok": True, "job_id": "job_1"})),
        (FunctionTimeoutError("took too long"), Errored("FunctionTimeoutError", "took too long")),
        (OutputExpiredError("gone"), Errored("expired", "gone")),
        (RuntimeError("CUDA error"), Errored("RuntimeError", "CUDA error")),
    ],
)
def test_poll_maps_outcomes(stub, outcome, expected):
    stub.calls["fc-1"] = call = StubCall(outcome)
    assert ModalGpuBackend(settings()).poll("fc-1") == expected
    assert call.get_timeouts == [0]


def test_cancel_and_logs(stub):
    stub.calls["fc-1"] = call = StubCall()
    backend = ModalGpuBackend(settings())
    backend.cancel("fc-1")
    assert call.cancelled
    assert backend.logs("fc-1", lines=7) == "line one\nline two\n"
    assert call.tail_entries == 7


@pytest.mark.parametrize("error", [StubConnectionError("no route"), OSError("network down")])
def test_poll_reraises_transport_errors(stub, error):
    stub.calls["fc-1"] = StubCall(error)
    with pytest.raises(type(error)):
        ModalGpuBackend(settings()).poll("fc-1")
