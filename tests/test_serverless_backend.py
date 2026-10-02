"""ServerlessGpuBackend against a fake RunPod endpoint over httpx.MockTransport; nothing talks to RunPod."""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from radar_desk.compute.runpod import RunPodError
from radar_desk.config import Settings
from radar_desk.db import Database
from radar_desk.gpu.backend import Errored, Finished, Pending, artefact_keys
from radar_desk.gpu.serverless_backend import ServerlessGpuBackend
from radar_desk.records import Job, Scan
from radar_desk.services.costs import parse_iso
from radar_desk.storage.local import LocalStorage
from radar_desk.storage.runpod_volume import RunPodVolumeStorage
from test_volume_routes import FakeS3

KEY = "rp-secret-key-123"
ENDPOINT = "ep123"
CALL = "rq-1"
JOB = "job_000000000001"
SCAN = "scan_000000000001"
HEALTH = {"jobs": {"completed": 1, "failed": 0, "inProgress": 0, "inQueue": 1, "retried": 0},
          "workers": {"idle": 0, "running": 0}}


class Clock:
    def __init__(self) -> None:
        self.t = parse_iso("2026-10-15T12:00:00.000000Z")

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class FakeEndpoint:
    """The four job routes of one endpoint. Each answer is a (status, json) pair or an exception to raise."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.run: tuple[int, object] = (200, {"id": CALL, "status": "IN_QUEUE"})
        self.status: tuple[int, object] | Exception = (200, {"id": CALL, "status": "IN_QUEUE"})
        self.health: tuple[int, object] | Exception = (200, HEALTH)
        self.cancel: tuple[int, object] = (200, {"id": CALL, "status": "CANCELLED"})

    def paths(self) -> list[str]:
        return [f"{r.method} {r.url.path}" for r in self.requests]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert request.url.host == "api.runpod.ai"
        prefix = f"/v2/{ENDPOINT}"
        assert request.url.path.startswith(prefix)
        path = request.url.path.removeprefix(prefix)
        if request.method == "POST" and path == "/run":
            answer = self.run
        elif request.method == "GET" and path == f"/status/{CALL}":
            answer = self.status
        elif request.method == "GET" and path == "/health":
            answer = self.health
        elif request.method == "POST" and path == f"/cancel/{CALL}":
            answer = self.cancel
        else:
            answer = (404, {"error": "not found"})
        if isinstance(answer, Exception):
            raise answer
        code, body = answer
        return httpx.Response(code, json=body) if body is not None else httpx.Response(code)


def completed(result, execution: int | None = 61000, wrap: bool = True) -> tuple[int, dict]:
    """A COMPLETED answer; the handler's result arrives under "result" unless `wrap` is False."""
    return 200, {"id": CALL, "status": "COMPLETED", "delayTime": 31618, "executionTime": execution,
                 "output": {"result": result} if wrap else result}


def ok_output() -> dict:
    return {"ok": True, "job_id": JOB, "timings": {"total_s": 50.0}, "versions": {"gpu": "NVIDIA L4"}}


class World:
    def __init__(self, tmp_path, storage=None, **overrides) -> None:
        self.clock = Clock()
        self.endpoint = FakeEndpoint()
        base = {"owner_token": "o", "session_secret": "s", "data_dir": tmp_path / "data",
                "runpod_api_key": KEY, "runpod_endpoint_id": ENDPOINT}
        self.settings = Settings(_env_file=None, **{**base, **overrides})
        self.storage = storage or LocalStorage(tmp_path / "store", "http://127.0.0.1:8000", "secret")
        self.db = Database(tmp_path / "radar.db")
        self.db.insert_scan(Scan(id=SCAN, filename="a.nii.gz", size_bytes=12345, state="ready"))
        self.db.insert_job(Job(id=JOB, scan_id=SCAN))
        self.backend = ServerlessGpuBackend(self.settings, self.storage, self.db,
                                            client=httpx.Client(transport=httpx.MockTransport(self.endpoint)),
                                            clock=self.clock)

    def submitted(self) -> None:
        """The job as the poller leaves it after a spawn."""
        self.db.transition(self.db.get_job(JOB), "submitted", modal_call_id=CALL, backend="serverless",
                           submitted_at="2026-10-15T12:00:00.000000Z")

    def spawn(self) -> str:
        keys = artefact_keys(JOB)
        refs = {name: self.storage.worker_ref(key, "PUT", 3600) for name, key in keys.items()}
        source = self.storage.worker_ref(f"scans/{SCAN}/source.nii.gz", "GET", 3600)
        return self.backend.spawn(self.db.get_job(JOB), source, refs, keys)


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def volume_world(tmp_path) -> World:
    storage = RunPodVolumeStorage("vol1", "EU-RO-1", "user_x", "rps_x", 300 * 1024 * 1024, client=FakeS3())
    return World(tmp_path, storage=storage, storage_backend="runpod_volume")


# Spawn


def test_spawn_posts_the_run_body_on_the_volume(tmp_path):
    w = volume_world(tmp_path)
    for key in [*artefact_keys(JOB).values(), f"jobs/{JOB}/result.json"]:
        w.storage.put_bytes(key, b"old")
    assert w.spawn() == CALL
    assert w.endpoint.paths() == [f"POST /v2/{ENDPOINT}/run"]
    body = json.loads(w.endpoint.requests[0].content)
    keys = artefact_keys(JOB)
    assert body == {
        "input": {"job_id": JOB, "source": f"volume://scans/{SCAN}/source.nii.gz",
                  "artefacts": {name: f"volume://{key}" for name, key in keys.items()},
                  "artefact_keys": keys, "result": f"volume://jobs/{JOB}/result.json", "expected_size": 12345},
        "policy": {"executionTimeout": 1800 * 1000, "ttl": (2 * 1800 + 600) * 1000},
    }
    assert w.storage.client.files == {}  # the six keys were deleted before /run


def test_spawn_on_the_volume_when_the_six_keys_do_not_exist(tmp_path):
    """A new job's keys are missing; FakeS3 answers 404 to their DeleteObject and the spawn still posts /run."""
    w = volume_world(tmp_path)
    assert w.spawn() == CALL
    assert w.endpoint.paths() == [f"POST /v2/{ENDPOINT}/run"]
    assert w.storage.client.files == {}


def test_spawn_on_url_storage_sends_a_presigned_result_url(world):
    world.spawn()
    body = json.loads(world.endpoint.requests[0].content)["input"]
    assert body["result"].startswith(f"http://127.0.0.1:8000/_storage/jobs/{JOB}/result.json?exp=")
    assert "sig=" in body["result"] and body["source"].startswith("http://127.0.0.1:8000/_storage/scans/")


def test_every_call_carries_bearer_and_user_agent(world):
    world.spawn()
    world.submitted()
    world.backend.poll(CALL)
    world.backend.cancel(CALL)
    assert len(world.endpoint.requests) == 4
    for request in world.endpoint.requests:
        assert request.headers["authorization"] == f"Bearer {KEY}"
        assert request.headers["user-agent"].startswith("radar-desk-compute/")


def test_spawn_errors_never_carry_the_key(world):
    world.endpoint.run = (404, None)
    with pytest.raises(RunPodError) as info:
        world.spawn()
    assert "RUNPOD_ENDPOINT_ID" in str(info.value) and KEY not in str(info.value)
    world.endpoint.run = (401, {"detail": "bad key"})
    with pytest.raises(RunPodError) as info:
        world.spawn()
    assert info.value.status == 401 and KEY not in str(info.value)
    world.endpoint.run = (200, {"status": "IN_QUEUE"})
    with pytest.raises(RunPodError, match="returned no id"):
        world.spawn()


def test_gpu_requested_is_the_pool_list(tmp_path):
    w = World(tmp_path, runpod_serverless_gpus="ADA_24, AMPERE_24")
    assert w.backend.gpu_requested == ["ADA_24", "AMPERE_24"]
    assert w.backend.name == "serverless" and w.backend.priced is True and not hasattr(w.backend, "pull")


# Poll


@pytest.mark.parametrize("status", ["IN_QUEUE", "IN_PROGRESS", "RUNNING"])
def test_waiting_statuses_are_pending(world, status):
    world.submitted()
    world.endpoint.status = (200, {"id": CALL, "status": status})
    assert world.backend.poll(CALL) == Pending()
    assert world.backend.last_status(CALL) == (status, "2026-10-15T12:00:00.000000Z")


def test_an_unknown_status_is_pending_and_logged_once(world, caplog):
    world.submitted()
    world.endpoint.status = (200, {"id": CALL, "status": "THROTTLED"})
    with caplog.at_level(logging.WARNING):
        assert world.backend.poll(CALL) == Pending()
        assert world.backend.poll(CALL) == Pending()
    assert sum("THROTTLED" in r.message for r in caplog.records) == 1


@pytest.mark.parametrize("answer,expected", [
    ({"status": "FAILED", "error": "CUDA error"}, Errored("runpod_failed", "CUDA error")),
    ({"status": "CANCELLED"}, Errored("cancelled", "cancelled outside the app")),
])
def test_failed_and_cancelled(world, answer, expected):
    world.submitted()
    world.endpoint.status = (200, {"id": CALL, **answer})
    assert world.backend.poll(CALL) == expected


def test_timed_out(world):
    world.submitted()
    world.endpoint.status = (200, {"id": CALL, "status": "TIMED_OUT"})
    outcome = world.backend.poll(CALL)
    assert isinstance(outcome, Errored) and outcome.klass == "timeout" and "1800" in outcome.message


@pytest.mark.parametrize(("output", "shape"), [
    ("oops", "str"),
    ({"ok": True, "job_id": JOB}, "a dict without a dict result"),
    ({"result": "oops"}, "a dict without a dict result"),
])
def test_an_output_without_a_dict_result_is_bad_result(world, output, shape):
    world.submitted()
    world.endpoint.status = completed(output, wrap=False)
    assert world.backend.poll(CALL) == Errored("bad_result", f'RunPod returned {shape}, not {{"result": dict}}')


def test_ok_false_settles_at_once_without_the_mask(world):
    world.submitted()
    world.endpoint.status = completed({"ok": False, "error": {"class": "input_error", "message": "x"}})
    outcome = world.backend.poll(CALL)
    assert isinstance(outcome, Finished) and outcome.result["ok"] is False
    assert outcome.result["runpod"] == {"id": CALL, "delayTime": 31618, "executionTime": 61000}


def test_completed_waits_for_the_mask_then_settles(world):
    world.submitted()
    world.endpoint.status = completed(ok_output())
    assert world.backend.poll(CALL) == Pending()
    world.clock.advance(10)
    assert world.backend.poll(CALL) == Pending()
    world.storage.put_bytes(artefact_keys(JOB)["mask"], b"mask")
    world.clock.advance(10)
    outcome = world.backend.poll(CALL)
    assert isinstance(outcome, Finished)
    assert outcome.result["runpod"] == {"id": CALL, "delayTime": 31618, "executionTime": 61000}
    assert outcome.result["timings"] == {"total_s": 50.0, "runpod_execution_s": 61.0}
    assert world.backend.completed_at == {} and world.backend.statuses == {}


def test_completed_settles_after_the_visibility_limit_with_a_warning(world, caplog):
    world.submitted()
    world.endpoint.status = completed(ok_output(), execution=None)
    assert world.backend.poll(CALL) == Pending()
    world.clock.advance(59)
    assert world.backend.poll(CALL) == Pending()
    world.clock.advance(1)
    with caplog.at_level(logging.WARNING):
        outcome = world.backend.poll(CALL)
    assert isinstance(outcome, Finished) and "runpod_execution_s" not in outcome.result["timings"]
    assert any("not visible after 60" in r.message for r in caplog.records)


def test_a_404_reads_result_json_from_storage(world):
    world.submitted()
    world.endpoint.status = (404, None)
    outcome = world.backend.poll(CALL)
    assert outcome == Errored("lost", "RunPod no longer knows the job; results expire 30 min after completion "
                                      "and the TTL was 4200 s")
    world.storage.put_bytes(f"jobs/{JOB}/result.json", json.dumps(ok_output()).encode())
    assert world.backend.poll(CALL) == Finished(ok_output())  # result.json is the unwrapped result


def test_a_404_with_a_truncated_result_json_is_bad_result(world):
    world.submitted()
    world.endpoint.status = (404, None)
    world.storage.put_bytes(f"jobs/{JOB}/result.json", json.dumps(ok_output()).encode()[:20])
    outcome = world.backend.poll(CALL)
    assert isinstance(outcome, Errored) and outcome.klass == "bad_result"
    assert outcome.message.startswith("result.json does not parse")


@pytest.mark.parametrize("answer", [(500, None), (429, None), (401, {"detail": "bad key"}),
                                    httpx.ConnectError("refused")])
def test_server_and_transport_errors_raise(world, answer):
    world.submitted()
    world.endpoint.status = answer
    with pytest.raises((RunPodError, httpx.HTTPError)):
        world.backend.poll(CALL)


# Health, cancel, logs


def test_health_is_read_once_per_poll_and_kept_on_failure(world):
    world.submitted()
    world.backend.poll(CALL)
    assert world.endpoint.paths() == [f"GET /v2/{ENDPOINT}/health", f"GET /v2/{ENDPOINT}/status/{CALL}"]
    assert world.backend.health == {**HEALTH, "at": "2026-10-15T12:00:00.000000Z"}
    assert world.backend.health_failed_since is None
    world.endpoint.health = (503, None)
    world.clock.advance(10)
    assert world.backend.poll(CALL) == Pending()
    assert world.backend.health["at"] == "2026-10-15T12:00:00.000000Z"
    assert world.backend.health_failed_since == world.clock()
    world.endpoint.health = httpx.ConnectError("refused")
    world.clock.advance(10)
    world.backend.poll(CALL)
    assert world.backend.health_failed_since == world.clock() - 10
    world.endpoint.health = (200, HEALTH)
    world.backend.poll(CALL)
    assert world.backend.health_failed_since is None


def test_cancel(world):
    world.backend.cancel(CALL)
    assert world.endpoint.paths() == [f"POST /v2/{ENDPOINT}/cancel/{CALL}"]
    world.endpoint.cancel = (500, None)
    with pytest.raises(RunPodError):
        world.backend.cancel(CALL)


def test_logs_from_the_artefact(world):
    world.submitted()
    world.storage.put_bytes(artefact_keys(JOB)["log"], b"one\ntwo\nthree\n")
    assert world.backend.logs(CALL, lines=2) == "two\nthree\n"


def test_logs_from_the_status_line(world):
    world.submitted()
    assert world.backend.logs(CALL) == f"RunPod job {CALL}: status unknown; no /health yet\n"
    world.backend.poll(CALL)
    assert world.backend.logs(CALL) == (
        f"RunPod job {CALL}: IN_QUEUE since 2026-10-15T12:00:00.000000Z; "
        "workers idle 0 running 0, jobs in queue 1 in progress 0\n")
