"""The compute choice and the pod lifecycle with a fake RunPod, tunnel, probe and clock."""

from __future__ import annotations

import json

import httpx
import pytest

from radar_desk.compute.runpod import RunPod
from radar_desk.compute.tunnel import Tunnel, TunnelError
from radar_desk.config import Settings
from radar_desk.gpu.backend import artefact_keys
from radar_desk.gpu.fake import FakeGpuBackend, canned_result
from radar_desk.gpu.poller import Poller
from radar_desk.gpu.serverless_backend import ServerlessGpuBackend
from radar_desk.records import Job
from radar_desk.services import build_services
from radar_desk.services.compute import (
    FIXED,
    HEALTH_DOWN,
    MIGRATED,
    MODAL_NEEDS_STORAGE,
    MODAL_ON_RUNPOD_VOLUME,
    POD_FAILED,
    QUEUE_WARNING,
    RUNPOD_NEEDS_CONFIG,
    SERVERLESS_NEEDS_CONFIG,
    SERVERLESS_NEEDS_STORAGE,
    TUNNEL_DIED,
    UNREACHABLE,
    mode_availability,
    mode_refusal,
)
from radar_desk.services.costs import parse_iso
from radar_desk.services.errors import ServiceError
from radar_desk.services.scans import source_key
from radar_desk.storage import make_storage
from synth import make_nifti
from test_serverless_backend import CALL, ENDPOINT, FakeEndpoint, completed

OWNER = "test-owner"
IMAGE = "ghcr.io/example/radar-worker:test"
PUBLIC = "https://desk.example"


class Clock:
    def __init__(self, iso: str = "2026-10-15T12:00:00.000000Z") -> None:
        self.t = parse_iso(iso)

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class FakeRunPod:
    """RunPod's v2 pod endpoints over httpx.MockTransport."""

    def __init__(self) -> None:
        self.pods: list[dict] = []
        self.deployed: list[dict] = []
        self.requested: list[str] = []
        self.deleted: list[str] = []
        self.stock = False
        self.down = False
        self.lose_answer = False  # the pod is created but the answer is a 504
        self.fail_deletes = 0  # answer 500 to this many deletes
        self.no_price = False  # the create answer has no cost

    def add(self, pod_id: str, name: str = "radar-worker") -> None:
        self.pods.append({"id": pod_id, "name": name, "image": IMAGE, "cost": 0.39, "status": "RUNNING",
                          "runtime": None, "gpu": {"id": "NVIDIA L4", "count": 1, "vcpuCount": 4, "memory": 24}})

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["user-agent"].startswith("radar-desk-compute/")
        assert request.url.path.startswith("/v2/pods")
        if self.down:
            return httpx.Response(503, text="down")
        pod_id = request.url.path.removeprefix("/v2/pods").strip("/")
        if request.method == "DELETE":
            if self.fail_deletes:
                self.fail_deletes -= 1
                return httpx.Response(500)
            self.deleted.append(pod_id)
            before = len(self.pods)
            self.pods = [p for p in self.pods if p["id"] != pod_id]
            if len(self.pods) < before:
                return httpx.Response(204)
            return httpx.Response(404, json={"title": "Not Found", "status": 404, "detail": "pod not found"})
        if request.method == "POST":
            body = json.loads(request.content)
            self.requested.append(body["gpu"]["id"])
            if self.stock:
                return httpx.Response(400, json={"title": "Bad Request", "status": 400,
                                                 "detail": "no instances available"})
            self.deployed.append(body)
            self.add(f"rp{len(self.deployed)}", body["name"])
            if self.lose_answer:
                return httpx.Response(504, text="gateway timeout")
            answer = {k: v for k, v in self.pods[-1].items() if k != "cost"} if self.no_price else self.pods[-1]
            return httpx.Response(201, json=answer)
        if pod_id:
            found = next((p for p in self.pods if p["id"] == pod_id), None)
            if found is None:
                return httpx.Response(404, json={"title": "Not Found", "status": 404, "detail": "pod not found"})
            return httpx.Response(200, json=found)
        return httpx.Response(200, json={"pods": self.pods, "pagination": {"nextCursor": None, "hasNextPage": False}})


class FakeTunnel:
    """The tunnel module's three calls, with no process."""

    def __init__(self) -> None:
        self.started: list[tuple[int, str]] = []
        self.stopped: list[int] = []
        self.live: set[int] = set()
        self.error: Exception | None = None
        self._pid = 1000

    def start(self, port, log, binary="cloudflared", **kw) -> Tunnel:
        if self.error is not None:
            raise self.error
        self._pid += 1
        self.started.append((port, str(log)))
        self.live.add(self._pid)
        url = f"https://t{self._pid}.trycloudflare.com"
        return Tunnel(pid=self._pid, url=url, log=str(log), binary=binary)

    def alive(self, pid, binary) -> bool:
        return pid in self.live

    def stop(self, pid, binary, grace_s=5) -> None:
        self.stopped.append(pid)
        self.live.discard(pid)


class Probe:
    def __init__(self) -> None:
        self.up = True
        self.calls: list[str] = []

    def __call__(self, url: str) -> dict | None:
        self.calls.append(url)
        return {"ok": True, "backend": "runpod", "version": "x"} if self.up else None


class World:
    def __init__(self, tmp_path, **overrides) -> None:
        self.tmp = tmp_path
        self.clock = Clock()
        self.rp = FakeRunPod()
        self.tunnel = FakeTunnel()
        self.probe = Probe()
        self.sleeps: list[float] = []
        # s3 in the settings so the modal mode is allowed; the bytes still go to local storage.
        base = {"owner_token": OWNER, "session_secret": "s", "data_dir": tmp_path / "data",
                "gpu_backend": "runpod",
                "runpod_api_key": "rp-key", "runpod_volume_id": "vol-1", "runpod_registry_auth_id": "reg-1",
                "worker_image": IMAGE, "storage_backend": "s3", "s3_bucket": "b"}
        self.settings = Settings(_env_file=None, **{**base, **overrides})
        storage = make_storage(self.settings.model_copy(update={"storage_backend": "local"}))
        runpod = RunPod("rp-key", client=httpx.Client(transport=httpx.MockTransport(self.rp)),
                        sleep=self.sleeps.append)
        self.svc = build_services(self.settings, storage=storage, runpod=runpod, tunnel=self.tunnel,
                                  probe=self.probe, clock=self.clock)
        self.compute = self.svc.compute
        self._n = 0

    def queue(self) -> str:
        self._n += 1
        path = make_nifti(self.tmp / f"s{self._n}.nii.gz", shape=(24, 20, 10))
        ticket = self.svc.scans.begin_upload(path.name, path.stat().st_size, True)
        self.svc.storage.put_bytes(source_key(ticket.scan_id), path.read_bytes())
        self.svc.scans.complete_upload(ticket.scan_id)
        return self.svc.jobs.create(ticket.scan_id).id

    def tick(self, advance: float = 0) -> None:
        self.clock.advance(advance)
        self.compute.tick()

    @property
    def pod(self):
        return self.svc.db.open_pod()

    def claim(self) -> dict | None:
        """The pod's worker reports in and claims the oldest queued job."""
        pod = self.pod
        token = self.svc.db.get_worker_token(pod.token_id)
        return self.svc.workers.claim(token, {"id": pod.worker_id, "gpu_name": "NVIDIA L4"})

    def idle(self, seconds: float) -> None:
        """Tick through `seconds` while the pod's worker polls for work once a minute, as a real one does."""
        while seconds > 0:
            step = min(60, seconds)
            assert self.claim() is None
            self.tick(step)
            seconds -= step

    def finish(self, claim: dict) -> None:
        result = canned_result(claim["job_id"], gpu="NVIDIA L4")
        self.svc.workers.complete(claim["job_id"], claim["lease"], result)

    def ready(self) -> dict:
        """Queue a job and bring a pod to ready with the worker holding that job."""
        self.queue()
        self.tick()
        self.tick()
        claim = self.claim()
        self.tick()
        assert self.pod.phase == "ready"
        return claim

    def job(self, job_id: str):
        return self.svc.db.get_job(job_id)

    def serverless(self) -> FakeEndpoint:
        """Serve the serverless backend from a fake endpoint; the settings need RUNPOD_ENDPOINT_ID=ENDPOINT."""
        endpoint = FakeEndpoint()
        client = httpx.Client(transport=httpx.MockTransport(endpoint))
        self.compute.backends["serverless"] = lambda: ServerlessGpuBackend(
            self.compute.settings, self.svc.storage, self.svc.db, client=client, clock=self.clock)
        return endpoint

    def last(self):
        return self.svc.db.list_pods()[0]


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


# Mode


def test_mode_is_seeded_from_gpu_backend_and_set_mode_refuses(tmp_path, make_services):
    w = World(tmp_path)
    assert w.compute.mode == "runpod" and w.compute.changeable
    assert w.svc.db.get_setting("mode") == "runpod"
    body = w.compute.set_mode("modal")
    assert body["mode"] == "modal" and body["changed_at"] == "2026-10-15T12:00:00.000000Z"
    assert w.svc.backend.name == "modal" and w.svc.workers.mode() == "modal"
    with pytest.raises(ServiceError) as info:
        w.compute.set_mode("cloud")
    assert info.value.status == 409

    local = World(tmp_path / "local", storage_backend=None, s3_bucket=None)
    with pytest.raises(ServiceError) as info:
        local.compute.set_mode("modal")
    assert info.value.status == 409 and info.value.detail == MODAL_NEEDS_STORAGE

    modal = World(tmp_path / "modal", gpu_backend="modal")
    assert modal.compute.mode == "modal"
    own = World(tmp_path / "own", gpu_backend="worker")
    assert own.compute.mode == "worker" and own.svc.db.get_setting("mode") == "worker"

    fake = make_services()
    assert fake.compute.mode == "fake" and not fake.compute.changeable
    with pytest.raises(ServiceError) as info:
        fake.compute.set_mode("worker")
    assert info.value.status == 409 and info.value.detail == FIXED


def test_backends_are_built_on_first_use(world):
    assert world.compute._built == {}
    assert world.svc.backend.name == "worker"
    assert set(world.compute._built) == {"runpod"}


# Start paths


def test_managed_start_from_a_queued_job_to_ready(world):
    job_id = world.queue()
    world.probe.up = False
    world.tick()
    pod = world.pod
    assert pod.phase == "tunnel" and pod.tunnel_url == "https://t1001.trycloudflare.com"
    assert world.tunnel.started == [(8000, str(world.tmp / "data" / "cloudflared.log"))]
    assert pod.tunnel == {"pid": 1001, "url": pod.tunnel_url, "log": world.tunnel.started[0][1],
                          "binary": "cloudflared"}
    assert pod.worker_id.startswith("runpod-") and len(pod.worker_id) == len("runpod-") + 6
    world.tick(10)
    assert world.pod.phase == "tunnel" and world.rp.deployed == []

    world.probe.up = True
    world.tick(10)
    pod = world.pod
    assert pod.phase == "starting" and pod.runpod_id == "rp1" and pod.gpu == "NVIDIA L4" and pod.cost_per_hr == 0.39
    assert world.probe.calls[-1] == pod.tunnel_url
    [inp] = world.rp.deployed
    env = inp["env"]
    assert env["RADAR_DESK_URL"] == pod.tunnel_url and env["RADAR_WORKER_ID"] == pod.worker_id
    assert env["RADAR_IMAGE"] == IMAGE and env["RADAR_WORKER_TOKEN"].startswith("rdw_")
    assert env["RADAR_POD_IDLE_DELETE_S"] == "600" and env["RADAR_POD_APP_LOST_DELETE_S"] == "600"
    assert inp["gpu"] == {"id": "NVIDIA L4", "count": 1} and inp["dataCenterIds"] == ["EU-RO-1"]
    assert inp["mounts"] == {"network": [{"volumeId": "vol-1", "path": "/workspace"}]}
    token = world.svc.db.get_worker_token(pod.token_id)
    assert token.name == f"pod {pod.id}" and token.revoked_at is None
    assert world.svc.workers.authenticate(env["RADAR_WORKER_TOKEN"]).id == pod.token_id

    world.tick(10)
    assert world.pod.phase == "starting"
    claim = world.claim()
    assert claim["job_id"] == job_id and world.job(job_id).backend == "worker"
    world.tick(10)
    assert world.pod.phase == "ready" and world.pod.ready_at == "2026-10-15T12:00:40.000000Z"
    view = world.compute.status()["pod"]
    assert view["job_id"] == job_id and view["tunnel_alive"] is True and view["idle_delete_s"] == 600
    assert world.compute.status()["in_flight"] == {"job_id": job_id, "backend": "worker"}


def test_external_start_uses_the_public_url(tmp_path):
    w = World(tmp_path, worker_public_url=PUBLIC, radar_pod_idle_delete_s=900, radar_pod_app_lost_delete_s=300.7)
    w.queue()
    w.tick()
    assert w.pod.phase == "tunnel" and w.pod.tunnel_url == PUBLIC and w.pod.tunnel is None
    assert w.tunnel.started == []
    w.tick()
    assert w.pod.phase == "starting" and w.probe.calls == [PUBLIC]
    env = w.rp.deployed[0]["env"]
    assert env["RADAR_DESK_URL"] == PUBLIC
    assert env["RADAR_POD_IDLE_DELETE_S"] == "900" and env["RADAR_POD_APP_LOST_DELETE_S"] == "300"
    assert w.compute.status()["pod"]["tunnel_alive"] is None
    assert w.compute.status()["tunnel_mode"] == "external"


def test_a_tunnel_that_never_answers_holds_the_queue(world):
    job_id = world.queue()
    world.probe.up = False
    world.tick()
    world.tick(119)
    assert world.pod.phase == "tunnel"
    world.tick(1)
    assert world.pod is None and world.last().reason == "tunnel_timeout"
    assert world.tunnel.stopped == [1001]
    assert world.job(job_id).hold_reason == "tunnel_unreachable"
    assert world.compute.status()["problem"] == UNREACHABLE


def test_a_tunnel_that_does_not_start_is_a_problem_and_retried(world):
    world.queue()
    world.tunnel.error = TunnelError("cloudflared is not installed or not on PATH")
    world.tick()
    assert world.pod is None
    assert world.compute.status()["problem"] == "cloudflared is not installed or not on PATH"
    world.tunnel.error = None
    world.tick(30)
    assert world.pod is None
    world.tick(30)
    assert world.pod.phase == "tunnel"


# Stopping


def test_the_app_does_not_stop_an_idle_pod(world):
    """The pod deletes itself when idle; the app only closes it at the hour cap."""
    claim = world.ready()
    world.tick(60)
    world.finish(claim)
    pod = world.pod
    world.idle(600)
    view = world.compute.status()["pod"]
    assert view["idle_s"] == 600 and view["idle_delete_s"] == 600
    world.idle(3600)
    assert world.pod is not None and world.rp.deleted == []
    world.idle(3 * 3600 - 60 - 600 - 3600 - 60)
    assert world.pod is not None
    world.idle(60)
    assert world.pod is None
    stopped = world.svc.db.get_pod(pod.id)
    assert stopped.reason == "cap" and stopped.phase == "stopped"
    assert world.rp.deleted == ["rp1"] and world.tunnel.stopped == [1001]


def test_a_pod_that_deleted_itself(world):
    claim = world.ready()
    world.tick(60)
    world.finish(claim)
    world.idle(600)
    pod = world.pod
    world.rp.pods.clear()  # the pod deleted itself, RunPod no longer lists it
    world.tick(30)
    assert world.pod is None
    closed = world.svc.db.get_pod(pod.id)
    assert closed.reason == "vanished" and world.rp.deleted == []
    assert world.tunnel.stopped == [1001]
    assert world.svc.db.get_worker_token(pod.token_id).revoked_at is not None
    assert world.compute.status()["pod"] is None

    world.queue()
    world.tick()
    assert world.pod.phase == "tunnel" and world.pod.id != pod.id and world.pod.tunnel["pid"] == 1002
    world.tick()
    assert world.pod.phase == "starting" and world.pod.runpod_id == "rp2"


def test_switch_to_modal_stops_the_pod_once_its_job_finishes(world):
    claim = world.ready()
    world.compute.set_mode("modal")
    world.tick(5)
    assert world.pod is not None
    world.finish(claim)
    world.tick(5)
    assert world.pod is None and world.last().reason == "mode"


def test_runpod_gpus_sets_the_order_tried(tmp_path):
    w = World(tmp_path, runpod_gpus="NVIDIA GeForce RTX 4090,NVIDIA L4")
    w.queue()
    w.rp.stock = True
    w.tick()
    w.tick()
    assert w.rp.requested == ["NVIDIA GeForce RTX 4090", "NVIDIA L4"]
    assert w.compute.status()["runpod"]["gpus"] == ["NVIDIA GeForce RTX 4090", "NVIDIA L4"]


def test_the_hour_cap_stops_a_busy_pod(world):
    world.ready()
    world.tick(3 * 3600 - 30)
    assert world.pod is not None
    world.tick(30)
    assert world.pod is None and world.last().reason == "cap"


def test_stock_error_holds_no_gpu_and_retries_after_a_minute(world):
    job_id = world.queue()
    world.rp.stock = True
    world.tick()
    world.tick()
    assert world.rp.requested == ["NVIDIA L4", "NVIDIA GeForce RTX 4090"] and world.sleeps == []
    assert world.pod.phase == "tunnel" and world.job(job_id).hold_reason == "no_gpu"
    assert world.compute.status()["problem"] == (
        "No NVIDIA L4 or NVIDIA GeForce RTX 4090 in EU-RO-1, retrying in 60 s")
    assert all(t.revoked_at for t in world.svc.db.list_worker_tokens())
    world.rp.stock = False
    world.tick(59)
    assert world.rp.deployed == []
    world.tick(1)
    assert world.pod.phase == "starting" and world.job(job_id).hold_reason is None
    assert world.compute.status()["problem"] is None


def test_other_deploy_errors_are_redacted(world, monkeypatch):
    world.queue()
    world.tick()

    def fail(spec, **kw):
        from radar_desk.compute.runpod import RunPodError

        raise RunPodError(f"bad input {spec.env['RADAR_WORKER_TOKEN']}")

    monkeypatch.setattr(world.compute.runpod, "deploy", fail)
    world.tick()
    problem = world.compute.status()["problem"]
    assert problem == "bad input [token]" and "rdw_" not in world.pod.model_dump_json()


def test_two_start_timeouts_hold_pod_failed(world):
    job_id = world.queue()
    world.tick()
    world.tick()
    first = world.pod
    world.tick(600)
    assert world.svc.db.get_pod(first.id).reason == "start_timeout"
    assert world.pod.phase == "tunnel" and world.job(job_id).hold_reason is None  # a new start follows
    world.tick()
    assert world.pod.phase == "starting"
    world.tick(600)
    assert world.pod is None and world.job(job_id).hold_reason == "pod_failed"
    assert world.compute.status()["problem"] == POD_FAILED
    assert world.rp.deleted == ["rp1", "rp2"]


def test_budget_hold(tmp_path):
    w = World(tmp_path, gpu_monthly_budget_usd=2.0)  # one pod is 3 h at the L4 price, 2.40
    job_id = w.queue()
    w.tick()
    assert w.pod is None and w.job(job_id).hold_reason == "budget"
    assert w.compute.status()["problem"] == "Budget reached"


def test_a_tunnel_that_dies_under_a_running_pod_stops_it(world):
    world.ready()
    world.tunnel.live.clear()
    world.tick(10)
    assert world.pod is None and world.last().reason == "tunnel_died"
    assert world.rp.deleted == ["rp1"] and world.compute.status()["problem"] == TUNNEL_DIED


def test_a_vanished_pod_is_closed_and_strays_are_deleted(world):
    world.ready()
    world.rp.pods.clear()
    world.rp.add("rp-other")
    world.rp.add("someone-elses", name="training")
    world.tick(30)
    assert world.last().reason == "vanished" and world.tunnel.stopped == [1001]
    assert world.svc.db.get_worker_token(world.last().token_id).revoked_at is not None
    assert world.rp.deleted == ["rp-other"]
    assert "rp-other" in world.compute.status()["last_event"]


def test_runpod_down_is_a_problem(world):
    world.queue()
    world.rp.down = True
    world.tick()
    assert world.pod is None and world.compute.status()["problem"] == "RunPod did not answer"
    world.rp.down = False
    world.tick(30)
    assert world.pod is not None and world.compute.status()["problem"] is None


# Reconcile at start


def test_reconcile_keeps_a_good_row_and_deletes_unknown_pods(world):
    world.ready()
    world.rp.add("rp-stray")
    world.compute.reconcile_at_start()
    assert world.pod.runpod_id == "rp1" and world.rp.deleted == ["rp-stray"]


def test_reconcile_stops_a_row_whose_pod_is_gone(world):
    world.ready()
    world.rp.pods.clear()
    world.compute.reconcile_at_start()
    assert world.pod is None and world.last().reason == "orphan" and world.rp.deleted == []
    assert world.tunnel.stopped == [1001]


def test_reconcile_stops_a_pod_whose_managed_tunnel_died(world):
    world.ready()
    world.tunnel.live.clear()
    world.compute.reconcile_at_start()
    assert world.pod is None and world.last().reason == "orphan" and world.rp.deleted == ["rp1"]


def test_reconcile_keeps_a_tunnel_row_with_a_live_tunnel(world):
    world.queue()
    world.probe.up = False
    world.tick()
    world.compute.reconcile_at_start()
    assert world.pod.phase == "tunnel"
    world.tunnel.live.clear()
    world.compute.reconcile_at_start()
    assert world.pod is None and world.last().reason == "orphan"


# Owner actions


def test_start_now_and_stop_now(world):
    world.compute.start_now()
    with pytest.raises(ServiceError):
        world.compute.stop_now()
    world.tick()
    assert world.pod.phase == "tunnel"
    with pytest.raises(ServiceError) as info:
        world.compute.start_now()
    assert info.value.status == 409
    body = world.compute.stop_now()
    assert body["pod"] is None and world.last().reason == "owner"
    with pytest.raises(ServiceError):
        world.compute.stop_now()
    world.tick(60)
    assert world.pod is None  # the request was used up

    world.compute.set_mode("modal")
    with pytest.raises(ServiceError):
        world.compute.start_now()


def test_start_now_needs_runpod(tmp_path):
    w = World(tmp_path, runpod_api_key=None)
    with pytest.raises(ServiceError) as info:
        w.compute.start_now()
    assert "RUNPOD_API_KEY" in info.value.detail
    w.queue()
    w.tick()
    assert w.pod is None


def test_start_now_releases_the_start_holds(world):
    job_id = world.queue()
    world.compute.hold_queued("pod_failed")
    world.compute.start_now()
    assert world.job(job_id).hold_reason is None


# Spend


def test_pod_spend_in_gpu_status(world):
    claim = world.ready()
    world.finish(claim)
    world.tick(600)
    closed = world.last().cost_usd
    world.ready()
    world.tick(3600)
    status = world.svc.gpu_status(world.clock())
    assert status["compute_mode"] == "runpod" and status["pod"]["runpod_id"] == "rp2"
    assert status["price_per_hour_usd"] == 0.39 and status["runpod_configured"] is True
    assert status["problem"] is None
    assert status["spend_month_usd"] == pytest.approx(closed + 0.39, abs=1e-6)
    assert world.compute.status()["spend_month_usd"] == status["spend_month_usd"]


# The poller across modes


def modal_stand_in(world) -> FakeGpuBackend:
    backend = FakeGpuBackend(clock=world.clock)
    backend.name = "modal"
    world.compute.backends["modal"] = lambda: backend
    return backend


def test_a_modal_job_is_collected_while_the_mode_is_runpod(world):
    modal = modal_stand_in(world)
    world.compute.set_mode("modal")
    job_id = world.queue()
    poller = Poller(world.svc, clock=world.clock)
    poller.tick()
    assert world.job(job_id).state == "submitted" and world.job(job_id).backend == "modal"
    world.compute.set_mode("runpod")
    modal.set_result(job_id, canned_result(job_id, gpu="NVIDIA L4"))
    world.clock.advance(100)
    poller.tick()
    job = world.job(job_id)
    assert job.state == "done" and job.cost_estimate_usd > 0
    assert world.pod is None  # nothing queued, so no pod


def test_a_worker_job_finishes_while_the_mode_is_modal(world):
    modal = modal_stand_in(world)
    claim = world.ready()
    second = world.queue()
    world.compute.set_mode("modal")
    poller = Poller(world.svc, clock=world.clock)
    poller.tick()
    assert world.job(claim["job_id"]).state == "submitted" and modal.calls == {}
    assert world.job(second).state == "queued"  # one job at a time
    world.finish(claim)
    assert world.job(claim["job_id"]).cost_estimate_usd is None
    poller.tick()
    assert world.pod is None and world.last().reason == "mode"
    assert world.job(second).state == "submitted" and world.job(second).backend == "modal"


def test_a_claim_is_refused_in_mode_modal(world):
    claim = world.ready()
    world.finish(claim)
    job_id = world.queue()
    world.compute.set_mode("modal")
    assert world.claim() is None and world.job(job_id).state == "queued"


def test_an_older_job_row_is_read_by_its_call_id(world):
    from radar_desk.records import Job
    from radar_desk.services.compute import job_backend

    assert job_backend(Job(id="j", scan_id="s", modal_call_id="wk_1")) == "worker"
    assert job_backend(Job(id="j", scan_id="s", modal_call_id="fc-1")) == "modal"
    assert world.compute.backend_for(Job(id="j", scan_id="s", modal_call_id="wk_1")).name == "worker"


# Review fixes


def test_a_probe_through_the_same_app_does_not_deadlock(world):
    from fastapi.testclient import TestClient

    from radar_desk.app import create_app

    client = TestClient(create_app(world.svc.settings, world.svc, start_poller=False))
    world.compute.probe = lambda url: client.get("/health").json()
    world.queue()
    world.tick()
    world.tick()
    assert world.pod.phase == "starting" and len(world.rp.deployed) == 1


def test_a_lost_deploy_answer_is_deleted_after_a_switch_to_modal(world):
    job_id = world.queue()
    world.rp.lose_answer = True
    world.tick()
    world.tick()
    assert world.pod.runpod_id is None and [p["id"] for p in world.rp.pods] == ["rp1"]
    assert all(t.revoked_at for t in world.svc.db.list_worker_tokens())
    assert world.job(job_id).hold_reason == "no_gpu"
    world.compute.set_mode("modal")
    world.tick()
    assert world.pod is None and world.last().reason == "mode"
    world.tick(30)
    assert world.rp.deleted == ["rp1"] and world.rp.pods == []


def test_strays_are_deleted_while_the_mode_is_modal(world):
    world.compute.set_mode("modal")
    world.rp.add("rp-stray")
    world.tick()
    assert world.rp.deleted == ["rp-stray"]


def test_a_failed_delete_is_retried_on_the_next_tick(world):
    world.ready()
    world.rp.fail_deletes = 1
    world.compute.stop_now()
    assert world.pod is not None and world.compute.status()["problem"].startswith("Could not delete pod rp1")
    world.compute.set_mode("modal")  # nothing waits, so the next tick stops it as mode
    world.finish({"job_id": world.compute.status()["pod"]["job_id"], "lease": world.job(
        world.compute.status()["pod"]["job_id"]).modal_call_id})
    world.tick(10)
    assert world.pod is None and world.rp.deleted == ["rp1"]


def test_stop_now_holds_the_queue_until_start_now(world):
    job_id = world.queue()
    world.tick()
    world.compute.stop_now()
    assert world.job(job_id).hold_reason == "owner"
    world.tick(10)
    world.tick(10)
    assert world.pod is None
    world.compute.start_now()
    assert world.job(job_id).hold_reason is None
    world.tick()
    assert world.pod.phase == "tunnel"


def test_a_silent_worker_stops_a_ready_pod_even_with_work_queued(world):
    claim = world.ready()
    world.finish(claim)
    world.queue()  # waits, but the worker no longer claims
    world.tick(120)
    assert world.pod is not None
    world.tick(1)
    lost = world.svc.db.list_pods()[1]
    assert lost.reason == "worker_lost" and world.rp.deleted == ["rp1"]
    assert world.pod.phase == "tunnel"  # the queued job starts a new one


def test_an_open_row_is_still_managed_when_runpod_is_not_fully_configured(world):
    world.ready()
    world.compute.settings = world.settings.model_copy(update={"runpod_volume_id": None})
    world.tick(10)
    assert world.compute.status()["problem"] == "RunPod is not fully configured, the pod is managed but no new one starts"
    world.compute.stop_now()
    assert world.pod is None and world.rp.deleted == ["rp1"]
    world.queue()
    world.tick(10)
    assert world.pod is None


def test_a_malformed_deploy_answer_revokes_the_token(world, monkeypatch):
    job_id = world.queue()
    world.tick()
    monkeypatch.setattr(world.compute.runpod, "deploy", lambda spec, **kw: (_ for _ in ()).throw(KeyError("id")))
    world.tick()
    assert world.pod.phase == "tunnel" and world.job(job_id).hold_reason == "no_gpu"
    assert all(t.revoked_at for t in world.svc.db.list_worker_tokens())
    world.tick(10)
    assert len(world.svc.db.list_worker_tokens()) == 1  # no second try before RETRY_S


def test_a_claim_waits_while_a_modal_job_runs(world):
    modal_stand_in(world)
    world.compute.set_mode("modal")
    world.queue()
    Poller(world.svc, clock=world.clock).tick()
    world.compute.set_mode("runpod")
    world.queue()
    world.compute.start_now()
    world.tick()
    world.tick()
    assert world.claim() is None


def test_the_budget_counts_modal_jobs_in_flight(tmp_path):
    w = World(tmp_path, gpu_monthly_budget_usd=3.0)  # a pod needs 2.40, one Modal worst case 0.82
    modal_stand_in(w)
    w.compute.set_mode("modal")
    w.queue()
    Poller(w.svc, clock=w.clock).tick()
    w.compute.set_mode("runpod")
    job_id = w.queue()
    w.tick()
    assert w.pod is None and w.job(job_id).hold_reason == "budget"


def test_a_tunnel_row_with_nothing_to_do_stops_as_idle(world):
    job_id = world.queue()
    world.rp.stock = True
    world.tick()
    world.tick()
    assert world.pod.phase == "tunnel"
    world.tick(10)
    assert world.pod is not None  # held as no_gpu, still waiting for stock
    world.svc.jobs.cancel(job_id)
    world.tick(10)
    assert world.pod is None and world.last().reason == "idle" and world.tunnel.stopped == [1001]


def test_a_deploy_answer_without_a_price_takes_the_listed_one(world):
    world.rp.no_price = True
    world.ready()
    assert world.pod.cost_per_hr == 0
    world.tick(30)
    assert world.pod.cost_per_hr == 0.39


# RunPod serverless

SLS = 0.00031
WORST_SLS = (2 * 1800 + 60) * SLS


def serverless_world(tmp_path, **overrides) -> tuple[World, FakeEndpoint, Poller]:
    w = World(tmp_path, runpod_endpoint_id=ENDPOINT, **overrides)
    endpoint = w.serverless()
    w.compute.set_mode("serverless")
    return w, endpoint, Poller(w.svc, clock=w.clock)


def test_serverless_is_seeded_from_gpu_backend(tmp_path):
    w = World(tmp_path, gpu_backend="serverless", runpod_endpoint_id=ENDPOINT)
    assert w.compute.mode == "serverless" and w.svc.db.get_setting("mode") == "serverless"


def test_set_mode_serverless_refusals(tmp_path):
    cases = [({}, SERVERLESS_NEEDS_CONFIG),
             ({"runpod_endpoint_id": ENDPOINT, "storage_backend": None, "s3_bucket": None}, SERVERLESS_NEEDS_STORAGE),
             ({"runpod_endpoint_id": ENDPOINT, "storage_backend": "modal_volume"}, SERVERLESS_NEEDS_STORAGE)]
    for n, (overrides, detail) in enumerate(cases):
        w = World(tmp_path / str(n), **overrides)
        with pytest.raises(ServiceError) as info:
            w.compute.set_mode("serverless")
        assert info.value.status == 409 and info.value.detail == detail
        assert w.compute.mode == "runpod"
    w = World(tmp_path / "volume", runpod_endpoint_id=ENDPOINT, storage_backend="runpod_volume")
    with pytest.raises(ServiceError) as info:
        w.compute.set_mode("modal")
    assert info.value.status == 409 and info.value.detail == MODAL_ON_RUNPOD_VOLUME
    assert w.compute.set_mode("serverless")["mode"] == "serverless"
    with pytest.raises(ServiceError) as info:
        w.compute.set_mode("cloud")
    assert info.value.detail == "unknown mode 'cloud'; use modal, worker, runpod or serverless"


def test_the_poller_spawns_and_settles_through_serverless(tmp_path):
    w, endpoint, poller = serverless_world(tmp_path)
    job_id = w.queue()
    poller.tick()
    job = w.job(job_id)
    assert job.state == "submitted" and job.backend == "serverless" and job.modal_call_id == CALL
    assert job.gpu_requested == ["AMPERE_24", "ADA_24"]
    assert endpoint.paths() == [f"POST /v2/{ENDPOINT}/run"]
    assert w.pod is None and w.rp.deployed == []

    endpoint.status = completed(canned_result(job_id, gpu="NVIDIA L4"))
    w.svc.storage.put_bytes(artefact_keys(job_id)["mask"], b"mask")
    w.clock.advance(100)
    poller.tick()
    job = w.job(job_id)
    assert job.state == "done" and job.cost_estimate_usd == pytest.approx((61 + 60) * SLS)


def test_no_serverless_spawn_while_a_modal_job_is_submitted(tmp_path):
    w = World(tmp_path, runpod_endpoint_id=ENDPOINT)
    endpoint = w.serverless()
    modal_stand_in(w).delay_ticks = 100
    w.compute.set_mode("modal")
    first = w.queue()
    poller = Poller(w.svc, clock=w.clock)
    poller.tick()
    assert w.job(first).backend == "modal"
    w.compute.set_mode("serverless")
    second = w.queue()
    poller.tick()
    assert w.job(first).state == "submitted"
    assert w.job(second).state == "queued" and endpoint.requests == []


def test_a_serverless_job_finishes_after_a_switch_to_runpod(tmp_path):
    w, endpoint, poller = serverless_world(tmp_path)
    job_id = w.queue()
    poller.tick()
    w.compute.set_mode("runpod")
    endpoint.status = completed(canned_result(job_id, gpu="NVIDIA L4"))
    w.svc.storage.put_bytes(artefact_keys(job_id)["mask"], b"mask")
    w.clock.advance(100)
    poller.tick()
    assert w.job(job_id).state == "done" and w.job(job_id).cost_estimate_usd > 0
    assert w.pod is None  # nothing queued, so no pod


def test_switch_to_serverless_stops_the_pod_once_its_job_finishes(tmp_path):
    w = World(tmp_path, runpod_endpoint_id=ENDPOINT)
    claim = w.ready()
    w.compute.set_mode("serverless")
    w.tick(5)
    assert w.pod is not None
    w.finish(claim)
    w.tick(5)
    assert w.pod is None and w.last().reason == "mode"


def test_the_budget_holds_at_the_serverless_worst_case(tmp_path):
    w, endpoint, poller = serverless_world(tmp_path, gpu_monthly_budget_usd=1.0)  # Modal's worst case fits
    job_id = w.queue()
    poller.tick()
    assert w.job(job_id).hold_reason == "budget" and endpoint.requests == []
    w.compute.settings.gpu_monthly_budget_usd = WORST_SLS + 0.01
    poller.tick()
    assert w.job(job_id).state == "submitted"
    assert w.svc.costs.in_flight() == pytest.approx(WORST_SLS)


def test_the_serverless_block_of_status(tmp_path):
    w = World(tmp_path)
    assert w.compute.status()["serverless"] == {
        "configured": False, "endpoint_id": None, "gpus": ["AMPERE_24", "ADA_24"], "idle_s": 60,
        "price_per_s": SLS, "health": None, "job": None}

    w, _endpoint, poller = serverless_world(tmp_path / "sls")
    block = w.compute.status()["serverless"]
    assert block["configured"] is True and block["endpoint_id"] == ENDPOINT and block["job"] is None
    assert "serverless" not in w.compute._built  # status never builds the backend
    job_id = w.queue()
    poller.tick()
    assert w.compute.status()["serverless"]["job"] == {
        "job_id": job_id, "status": None, "submitted_at": "2026-10-15T12:00:00.000000Z", "status_at": None}
    w.clock.advance(10)
    poller.tick()
    body = w.compute.status()
    block = body["serverless"]
    assert block["job"]["status"] == "IN_QUEUE" and block["job"]["status_at"] == "2026-10-15T12:00:10.000000Z"
    assert block["health"]["workers"] == {"idle": 0, "running": 0}
    assert block["health"]["at"] == "2026-10-15T12:00:10.000000Z"
    assert body["in_flight"] == {"job_id": job_id, "backend": "serverless"}


def test_the_queue_warning(tmp_path):
    w, endpoint, poller = serverless_world(tmp_path)
    w.queue()
    poller.tick()
    w.clock.advance(300)
    poller.tick()
    assert w.compute.status()["problem"] is None
    w.clock.advance(1)
    poller.tick()
    assert w.compute.status()["problem"] == QUEUE_WARNING.format(n=5, datacenter="EU-RO-1")
    assert w.compute.status()["problem"].startswith("No worker has started in 5 min; EU-RO-1 stock")
    endpoint.health = (200, {"workers": {"idle": 0, "running": 1}, "jobs": {}})
    w.clock.advance(10)
    poller.tick()
    assert w.compute.status()["problem"] is None
    endpoint.health = (200, {"workers": {"idle": 0, "running": 0}, "jobs": {}})
    endpoint.status = (200, {"id": CALL, "status": "IN_PROGRESS"})
    poller.tick()
    assert w.compute.status()["problem"] is None


def test_health_down_is_a_problem_after_a_minute(tmp_path):
    w, endpoint, poller = serverless_world(tmp_path)
    w.queue()
    poller.tick()
    endpoint.health = (503, None)
    w.clock.advance(10)
    poller.tick()
    w.clock.advance(59)
    poller.tick()
    assert w.compute.status()["problem"] is None
    w.clock.advance(1)
    assert w.compute.status()["problem"] == HEALTH_DOWN
    endpoint.health = (200, {"workers": {"idle": 1, "running": 0}, "jobs": {}})
    poller.tick()
    assert w.compute.status()["problem"] is None


def test_health_down_clears_once_the_job_has_settled(tmp_path):
    """Only a poll reads /health, so a failure on the last tick must not stay on GET /compute."""
    w, endpoint, poller = serverless_world(tmp_path)
    job_id = w.queue()
    poller.tick()
    endpoint.health = (503, None)
    w.clock.advance(10)
    poller.tick()
    w.clock.advance(60)
    assert w.compute.status()["problem"] == HEALTH_DOWN
    endpoint.status = completed(canned_result(job_id, gpu="NVIDIA L4"))
    w.svc.storage.put_bytes(artefact_keys(job_id)["mask"], b"mask")
    poller.tick()
    assert w.job(job_id).state != "submitted"
    assert w.compute._built["serverless"].health_failed_since is not None
    assert w.compute.status()["problem"] is None


def test_gpu_status_prices_serverless_by_the_pool(tmp_path):
    w, _endpoint, _poller = serverless_world(tmp_path)
    body = w.svc.gpu_status()
    assert body["backend"] == "serverless" and body["compute_mode"] == "serverless"
    assert body["price_per_hour_usd"] == round(SLS * 3600, 4)


def test_gpu_status_carries_the_serverless_block_and_problem(tmp_path):
    w, _endpoint, poller = serverless_world(tmp_path)
    w.queue()
    poller.tick()
    w.clock.advance(301)
    poller.tick()
    body = w.svc.gpu_status(w.clock())
    assert body["serverless"] == w.compute.status()["serverless"] == w.compute.serverless_view()
    assert body["serverless"]["job"]["status"] == "IN_QUEUE"
    assert body["problem"] == w.compute.status()["problem"] == QUEUE_WARNING.format(n=5, datacenter="EU-RO-1")


def test_serverless_attempt_cost(tmp_path):
    w, endpoint, poller = serverless_world(tmp_path)
    costs, now = w.svc.costs, w.clock()
    job = Job(id="j", scan_id="s", backend="serverless", submitted_at="2026-10-15T11:50:00.000000Z")
    assert costs.attempt_cost(job, {"total_s": 50.0, "runpod_execution_s": 61.0}, None, now) == pytest.approx(
        (61 + 60) * SLS)
    assert costs.attempt_cost(job, {"total_s": 70.0}, None, now) == pytest.approx((70 + 60) * SLS)
    assert costs.attempt_cost(job, None, None, now) == pytest.approx((600 + 60) * SLS)

    job_id = w.queue()
    poller.tick()
    w.clock.advance(100)
    cancelled = w.svc.jobs.cancel(job_id)
    assert endpoint.paths()[-1] == f"POST /v2/{ENDPOINT}/cancel/{CALL}"
    assert cancelled.state == "cancelled" and cancelled.cost_estimate_usd == pytest.approx((100 + 60) * SLS)


# Own workers (mode worker)


def own_world(tmp_path, **overrides) -> World:
    return World(tmp_path, gpu_backend="worker", **overrides)


def own_claim(w: World, name: str = "mine") -> dict | None:
    token, _plaintext = w.svc.workers.create_token(name)
    return w.svc.workers.claim(token, {"id": "box-1", "gpu_name": "NVIDIA RTX 4090"})


def test_mode_worker_never_deploys(tmp_path):
    w = own_world(tmp_path)
    w.rp.add("rp-stray")
    job_id = w.queue()
    for _ in range(5):
        w.tick(60)
    assert w.rp.deleted == ["rp-stray"]  # the tick ran its reconcile
    assert w.pod is None and w.svc.db.list_pods() == []
    assert w.tunnel.started == [] and w.rp.deployed == [] and w.rp.requested == []
    assert w.compute.status()["problem"] is None
    assert w.job(job_id).state == "queued" and w.job(job_id).hold_reason is None


@pytest.mark.parametrize("mode", ["worker", "runpod"])
def test_an_own_worker_claims_in_both_pull_modes(tmp_path, mode):
    w = World(tmp_path, gpu_backend=mode)
    job_id = w.queue()
    claim = own_claim(w)
    assert claim["job_id"] == job_id and w.rp.deployed == []
    w.svc.workers.complete(job_id, claim["lease"], canned_result(job_id, gpu="NVIDIA RTX 4090"))
    assert w.job(job_id).state == "done" and w.job(job_id).backend == "worker"


def test_switch_runpod_to_worker_drains_and_stops_the_pod(world):
    claim = world.ready()
    token_id = world.pod.token_id
    world.compute.set_mode("worker")
    world.tick(5)
    assert world.pod is not None  # its job is in flight
    second = world.queue()
    assert world.claim() is None  # the pod's worker no longer claims
    world.finish(claim)
    assert world.claim() is None and world.job(second).state == "queued"
    world.tick(5)
    assert world.pod is None and world.last().reason == "mode" and world.rp.deleted == ["rp1"]
    assert world.svc.db.get_worker_token(token_id).revoked_at is not None
    world.tick(60)
    world.tick(60)
    assert world.pod is None and len(world.rp.deployed) == 1
    assert world.job(second).state == "queued" and world.job(second).hold_reason is None


def test_an_own_worker_job_does_not_keep_a_draining_pod(world):
    claim = world.ready()
    world.compute.set_mode("worker")
    second = world.queue()
    world.finish(claim)
    own = own_claim(world)
    assert own["job_id"] == second
    assert world.compute.pod_view()["job_id"] is None
    world.tick(5)
    assert world.pod is None and world.last().reason == "mode" and world.rp.deleted == ["rp1"]
    assert world.job(second).state == "submitted"


def test_an_own_worker_claims_while_a_pod_drains(world):
    claim = world.ready()
    world.compute.set_mode("worker")
    second = world.queue()
    own = own_claim(world)  # the pod row is open and its job still submitted
    assert own["job_id"] == second
    assert world.compute.pod_view()["job_id"] == claim["job_id"]
    world.tick(5)
    assert world.pod is not None  # the pod's own job is in flight
    world.finish(claim)
    assert world.compute.pod_view()["job_id"] is None
    world.tick(5)
    assert world.pod is None and world.last().reason == "mode"


def test_a_silent_pod_worker_stops_while_an_own_worker_scores(world):
    claim = world.ready()
    world.finish(claim)
    second = world.queue()
    assert own_claim(world)["job_id"] == second
    world.tick(120)
    assert world.pod is not None
    world.tick(1)
    assert world.pod is None and world.last().reason == "worker_lost"
    assert world.job(second).state == "submitted"


@pytest.mark.parametrize("missing", ["runpod_api_key", "worker_image"])
def test_set_mode_runpod_needs_runpod(tmp_path, missing):
    w = own_world(tmp_path, **{missing: None})
    with pytest.raises(ServiceError) as info:
        w.compute.set_mode("runpod")
    assert info.value.status == 409 and info.value.detail == RUNPOD_NEEDS_CONFIG
    assert w.compute.mode == "worker"


def test_start_now_is_refused_in_mode_worker(tmp_path):
    w = own_world(tmp_path)
    with pytest.raises(ServiceError) as info:
        w.compute.start_now()
    assert info.value.status == 409 and info.value.detail == "a pod starts only in mode runpod"


def test_mode_worker_still_reconciles_and_deletes_strays(tmp_path):
    w = own_world(tmp_path)
    w.rp.add("rp-stray")
    w.compute.reconcile_at_start()
    assert w.rp.deleted == ["rp-stray"]
    w.rp.add("rp-later")
    w.tick(30)
    assert w.rp.deleted == ["rp-stray", "rp-later"]


def test_mode_worker_closes_a_vanished_pod(world):
    world.ready()
    world.compute.set_mode("worker")
    world.rp.pods.clear()
    world.tick(30)
    assert world.pod is None and world.last().reason == "vanished" and world.rp.deleted == []


# Migration of the stored mode


def old_schema(w: World, mode: str) -> None:
    """A database from before mode runpod: the mode stored, no marker."""
    w.svc.db.set_setting("mode", mode)
    w.svc.db.set_setting("mode_schema", None)


def test_migrate_mode_turns_a_stored_worker_into_runpod_when_configured(world):
    old_schema(world, "worker")
    world.compute.migrate_mode()
    assert world.compute.mode == "runpod" and world.svc.db.get_setting("mode_schema") == "2"
    assert world.compute.status()["last_event"] == f"2026-10-15T12:00:00.000000Z {MIGRATED}"
    world.compute.set_mode("worker")  # the owner chose own workers deliberately
    world.compute.migrate_mode()
    assert world.compute.mode == "worker"


def test_migrate_mode_leaves_other_cases_alone(tmp_path):
    w = World(tmp_path / "bare", runpod_api_key=None)
    old_schema(w, "worker")
    w.compute.migrate_mode()
    assert w.compute.mode == "worker" and w.svc.db.get_setting("mode_schema") == "2"
    assert w.svc.db.get_setting("last_event") is None

    w = World(tmp_path / "modal")
    old_schema(w, "modal")
    w.compute.migrate_mode()
    assert w.compute.mode == "modal" and w.svc.db.get_setting("mode_schema") == "2"

    w = World(tmp_path / "unseeded", gpu_backend="worker")
    w.svc.db.set_setting("mode", None)
    w.svc.db.set_setting("mode_schema", None)
    w.compute.migrate_mode()
    assert w.svc.db.get_setting("mode") is None and w.svc.db.get_setting("mode_schema") == "2"
    assert w.compute.mode == "worker"  # the first read seeds from GPU_BACKEND


def test_a_fresh_gpu_backend_worker_stays_worker_with_runpod_configured(tmp_path):
    """The services read the mode when built, so the seed already carries the marker."""
    w = own_world(tmp_path)
    assert w.svc.db.get_setting("mode") == "worker" and w.svc.db.get_setting("mode_schema") == "2"
    w.compute.migrate_mode()
    assert w.compute.mode == "worker" and w.svc.db.get_setting("last_event") is None


def test_migrate_mode_writes_nothing_on_the_fake_backend(make_services):
    fake = make_services()
    fake.compute.migrate_mode()
    assert fake.db.get_setting("mode_schema") is None and fake.db.get_setting("mode") is None


@pytest.mark.parametrize("overrides", [{"runpod_api_key": None}, {"worker_image": None}, {}])
def test_mode_availability_agrees_with_mode_refusal(tmp_path, overrides):
    w = World(tmp_path, **overrides)
    modes = mode_availability(w.settings)
    assert list(modes) == ["modal", "worker", "runpod", "serverless"]
    for mode in ("modal", "runpod", "serverless"):
        assert modes[mode]["available"] == (mode_refusal(w.settings, mode) is None)
    assert modes["worker"]["available"]
