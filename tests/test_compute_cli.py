"""The compute command as a front end to the app's Compute routes, against a fake app (MockTransport)."""

from __future__ import annotations

import json
import os

import httpx
import pytest

from radar_desk.compute.cli import Deps, compute_lines, main

OWNER = "owner-secret-1"
ENV = f"# radar desk\nOWNER_TOKEN={OWNER}\nGPU_BACKEND=fake\nLAST=1"
POD = {"id": "pod_ab12", "runpod_id": "rp123", "phase": "ready", "gpu": "NVIDIA L4", "image": "ghcr.io/x/radar-worker:0.1",
       "cost_per_hr": 0.39, "created_at": "x", "started_at": "x", "ready_at": "x", "up_s": 754, "idle_s": 30,
       "idle_delete_s": 600, "worker_id": "runpod-abc123", "tunnel_url": "https://a-b.trycloudflare.com",
       "tunnel_alive": True, "job_id": None}


class FakeApp:
    def __init__(self) -> None:
        self.mode = "modal"
        self.pod: dict | None = None
        self.problem: str | None = None
        self.calls: list[tuple[str, str, dict | None]] = []
        self.refuse: str | None = None
        self.workers = [{"id": "runpod-abc123", "online": True, "job_id": "job_1"}]
        self.serverless: dict = {"configured": False, "endpoint_id": None, "gpus": ["AMPERE_24", "ADA_24"],
                                 "idle_s": 60, "price_per_s": 0.00031, "health": None, "job": None}

    def body(self) -> dict:
        return {"mode": self.mode, "changeable": True, "changed_at": None, "tunnel_mode": "managed",
                "public_url": None,
                "runpod": {"configured": True, "datacenter": "EU-RO-1", "max_pod_hours": 3,
                           "gpus": ["NVIDIA L4", "NVIDIA GeForce RTX 4090"], "idle_delete_s": 600,
                           "app_lost_delete_s": 600},
                "serverless": self.serverless,
                "pod": self.pod, "in_flight": None, "queued": 0, "held": [], "problem": self.problem,
                "last_event": None, "spend_month_usd": 1.234, "budget_usd": 10.0, "month": "2026-10",
                "storage": {"backend": "s3", "name": "Tigris bucket radar-desk-data", "modes": {
                    "modal": {"available": True, "reason": None, "note": None},
                    "worker": {"available": True, "reason": None, "note": None},
                    "runpod": {"available": True, "reason": None, "note": None},
                    "serverless": {"available": False, "reason": "x", "note": "no endpoint"}}}}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        payload = json.loads(request.content) if request.content else None
        self.calls.append((method, path, payload))
        if path == "/health":
            return httpx.Response(200, json={"ok": True, "backend": self.mode, "version": "1.2.3"})
        if request.headers.get("authorization") != f"Bearer {OWNER}":
            return httpx.Response(401, json={"detail": "Not logged in"})
        if self.refuse and path.startswith("/compute") and method != "GET":
            return httpx.Response(409, json={"detail": self.refuse})
        if method == "GET" and path == "/compute":
            return httpx.Response(200, json=self.body())
        if method == "PUT" and path == "/compute":
            self.mode = payload["mode"]
            return httpx.Response(200, json=self.body())
        if method == "POST" and path == "/compute/pod/start":
            self.pod = {**POD, "phase": "tunnel", "runpod_id": None, "gpu": None, "image": None,
                        "cost_per_hr": None, "up_s": None}
            return httpx.Response(202, json=self.body())
        if method == "POST" and path == "/compute/pod/stop":
            self.pod = None
            return httpx.Response(200, json=self.body())
        if method == "GET" and path == "/workers":
            return httpx.Response(200, json={"workers": self.workers, "app_url": "x", "image": "x", "lease_s": 120})
        return httpx.Response(404, json={"detail": "Not Found"})


@pytest.fixture
def world(tmp_path):
    env = tmp_path / ".env"
    env.write_text(ENV)
    os.chmod(env, 0o600)
    app = FakeApp()
    deps = Deps(env_path=env, environ={}, app_client=httpx.Client(transport=httpx.MockTransport(app)))

    def run(capsys, *argv: str) -> tuple[int, str, str]:
        code = main(list(argv), deps)
        captured = capsys.readouterr()
        assert OWNER not in captured.out + captured.err
        return code, captured.out, captured.err

    yield app, run, deps
    assert env.read_text() == ENV
    assert env.stat().st_mode & 0o777 == 0o600
    assert sorted(p.name for p in tmp_path.iterdir()) == [".env"]


def test_runpod_sets_the_mode(world, capsys):
    app, run, _ = world
    code, out, _ = run(capsys, "runpod")
    assert code == 0
    assert app.calls == [("PUT", "/compute", {"mode": "runpod"})]
    assert "compute: mode runpod (RunPod pod), tunnel managed" in out
    assert "pod: none" in out
    assert "spend: $1.23 of $10.00 in 2026-10" in out


def test_workers_sets_the_mode(world, capsys):
    app, run, _ = world
    code, out, _ = run(capsys, "workers")
    assert code == 0
    assert app.calls == [("PUT", "/compute", {"mode": "worker"})]
    assert "compute: mode worker (Own GPU workers), tunnel managed" in out


def test_runpod_start_asks_for_a_pod(world, capsys):
    app, run, _ = world
    code, out, _ = run(capsys, "runpod", "--start")
    assert code == 0
    assert [(m, p) for m, p, _ in app.calls] == [("PUT", "/compute"), ("POST", "/compute/pod/start")]
    assert "pod: none yet, tunnel, ?, $?/h, image ?, not deployed yet" in out


def test_modal_sets_the_mode(world, capsys):
    app, run, _ = world
    app.mode = "runpod"
    code, out, _ = run(capsys, "modal")
    assert code == 0
    assert app.calls == [("PUT", "/compute", {"mode": "modal"})]
    assert "compute: mode modal (Modal)" in out


def test_stop_stops_the_pod(world, capsys):
    app, run, _ = world
    app.pod = POD
    code, out, _ = run(capsys, "stop")
    assert code == 0
    assert app.calls == [("POST", "/compute/pod/stop", None)]
    assert "pod: none" in out


def test_status_prints_health_compute_and_workers(world, capsys):
    app, run, _ = world
    app.mode, app.pod, app.problem = "runpod", POD, "RunPod did not answer"
    code, out, _ = run(capsys, "status")
    assert code == 0
    assert out.splitlines() == [
        "app: backend runpod, version 1.2.3",
        "compute: mode runpod (RunPod pod), tunnel managed",
        "gpus: NVIDIA L4, NVIDIA GeForce RTX 4090 in EU-RO-1",
        "pod: rp123, ready, NVIDIA L4, $0.390/h, image ghcr.io/x/radar-worker:0.1, up 12 min",
        "serverless: not configured",
        "problem: RunPod did not answer",
        "spend: $1.23 of $10.00 in 2026-10",
        "storage: Tigris bucket radar-desk-data, s3; modal yes, worker yes, runpod yes, serverless no",
        "worker: runpod-abc123, online, job job_1",
    ]
    assert all(m == "GET" for m, _, _ in app.calls)


def test_a_409_prints_the_detail_and_exits_1(world, capsys):
    app, run, _ = world
    app.refuse = "A pod is already open"
    code, out, err = run(capsys, "runpod", "--start")
    assert code == 1
    assert "A pod is already open" in err
    assert out == ""


def test_the_app_down_exits_1(world, capsys):
    _, run, deps = world

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    deps.app_client = httpx.Client(transport=httpx.MockTransport(down))
    for argv in (["status"], ["workers"], ["runpod"], ["modal"], ["serverless"], ["stop"]):
        code, _, err = run(capsys, *argv, "--app", "http://127.0.0.1:9")
        assert code == 1
        assert "does not answer at http://127.0.0.1:9" in err


def test_the_environment_wins_and_a_missing_token_is_named(world, capsys):
    _, run, deps = world
    deps.environ = {"OWNER_TOKEN": "other"}
    code, _, err = run(capsys, "modal")
    assert code == 1
    assert "Not logged in" in err
    deps.environ = {}
    deps.env_path = deps.env_path.parent / "missing.env"
    code, _, err = run(capsys, "status")
    assert code == 1
    assert "OWNER_TOKEN is not set" in err


def test_an_older_body_without_gpus_prints_no_gpus_line():
    body = {"mode": "worker", "changeable": True, "tunnel_mode": "managed", "runpod": {"datacenter": "EU-RO-1"},
            "pod": None, "spend_month_usd": 0.0, "budget_usd": 10.0, "month": "2026-10"}
    assert not any(line.startswith("gpus:") for line in compute_lines(body))
    body["runpod"] = {"gpus": ["NVIDIA L4"]}
    assert "gpus: NVIDIA L4" in compute_lines(body)


def test_serverless_sets_the_mode(world, capsys):
    app, run, _ = world
    app.serverless = {**app.serverless, "configured": True, "endpoint_id": "ep123"}
    code, out, _ = run(capsys, "serverless")
    assert code == 0
    assert app.calls == [("PUT", "/compute", {"mode": "serverless"})]
    assert "compute: mode serverless" in out
    assert "serverless: endpoint ep123, AMPERE_24, ADA_24, no job" in out


def test_the_serverless_line_in_its_three_forms():
    body = {"mode": "serverless", "changeable": True, "tunnel_mode": "managed", "pod": None,
            "spend_month_usd": 0.0, "budget_usd": 10.0, "month": "2026-10",
            "serverless": {"configured": False, "gpus": ["AMPERE_24"], "job": None}}
    assert "serverless: not configured" in compute_lines(body)
    body["serverless"] = {"configured": True, "endpoint_id": "ep123", "gpus": ["AMPERE_24", "ADA_24"], "job": None}
    assert "serverless: endpoint ep123, AMPERE_24, ADA_24, no job" in compute_lines(body)
    job = {"job_id": "job_ab12cd34ef56", "status": None, "submitted_at": "2026-10-15T12:00:00.000000Z",
           "status_at": None}
    body["serverless"]["job"] = job
    assert ("serverless: endpoint ep123, AMPERE_24, ADA_24, job job_ab12 status pending "
            "since 2026-10-15T12:00:00.000000Z") in compute_lines(body)
    job["status"] = "IN_QUEUE"
    assert ("serverless: endpoint ep123, AMPERE_24, ADA_24, job job_ab12 IN_QUEUE "
            "since 2026-10-15T12:00:00.000000Z") in compute_lines(body)
    del body["serverless"]
    assert not any(line.startswith("serverless:") for line in compute_lines(body))
