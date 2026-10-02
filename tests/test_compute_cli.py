"""The compute switch end to end with a fake app, a fake RunPod, a fake cloudflared and a fake probe."""

from __future__ import annotations

import json
import os

import httpx
import pytest

from radar_desk.compute import state as state_mod
from radar_desk.compute import tunnel
from radar_desk.compute.cli import Deps, main
from test_compute_tunnel import FAKE_URL, fake_cloudflared

SECRETS = {"OWNER_TOKEN": "owner-secret-1", "RUNPOD_API_KEY": "rp-secret-2", "RUNPOD_VOLUME_ID": "vol-secret-3",
           "RUNPOD_REGISTRY_AUTH_ID": "reg-secret-4"}
IMAGE = "ghcr.io/example/radar-worker:test"


class FakeApp:
    def __init__(self) -> None:
        self.backend = "worker"
        self.flip_after: int | None = None
        self.health_calls = 0
        self.tokens: dict[str, str] = {}
        self.revoked: list[str] = []
        self.workers: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/health":
            self.health_calls += 1
            if self.flip_after is not None and self.health_calls > self.flip_after:
                self.backend = "worker"
            return httpx.Response(200, json={"ok": True, "backend": self.backend, "version": "1.2.3"})
        assert request.headers["authorization"] == f"Bearer {SECRETS['OWNER_TOKEN']}"
        if request.method == "POST" and path == "/workers/tokens":
            token_id = f"wtok_{len(self.tokens) + 1}"
            self.tokens[token_id] = f"rdw_plaintext{len(self.tokens) + 1}"
            return httpx.Response(201, json={"id": token_id, "name": json.loads(request.content)["name"],
                                             "created_at": "x", "token": self.tokens[token_id]})
        if request.method == "DELETE" and path.startswith("/workers/tokens/"):
            token_id = path.rsplit("/", 1)[1]
            if token_id not in self.tokens:
                return httpx.Response(404)
            self.revoked.append(token_id)
            return httpx.Response(204)
        if request.method == "GET" and path == "/workers":
            return httpx.Response(200, json={"workers": self.workers, "app_url": "x", "image": "x", "lease_s": 120})
        return httpx.Response(404)


class FakeRunPod:
    def __init__(self) -> None:
        self.pods: list[dict] = []
        self.deployed: list[dict] = []
        self.deleted: list[str] = []
        self.fail_deploy = False

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["user-agent"].startswith("radar-desk-compute/")
        if request.method == "DELETE":
            pod_id = request.url.path.rsplit("/", 1)[1]
            self.deleted.append(pod_id)
            before = len(self.pods)
            self.pods = [p for p in self.pods if p["id"] != pod_id]
            return httpx.Response(200 if len(self.pods) < before else 404)
        query = json.loads(request.content)["query"]
        if "podFindAndDeployOnDemand" in query:
            inp = json.loads(request.content)["variables"]["input"]
            if self.fail_deploy:
                return httpx.Response(200, json={"errors": [{"message": f"bad input {json.dumps(inp)}"}]})
            self.deployed.append(inp)
            pod = {"id": f"pod{len(self.deployed)}", "name": inp["name"], "imageName": inp["imageName"],
                   "costPerHr": 0.39,
                   "desiredStatus": "RUNNING", "runtime": None, "machine": {"gpuDisplayName": "L4"}}
            self.pods.append(pod)
            return httpx.Response(200, json={"data": {"podFindAndDeployOnDemand": pod}})
        spend = sum(p["costPerHr"] for p in self.pods)
        return httpx.Response(200, json={"data": {"myself": {"currentSpendPerHr": spend, "pods": self.pods}}})


class Clock:
    def __init__(self) -> None:
        self.t = 1_790_000_000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


ENV = ("# radar desk\nOWNER_TOKEN={OWNER_TOKEN}\n\n#GPU_BACKEND=modal\nGPU_BACKEND=fake\n"
       "RUNPOD_API_KEY={RUNPOD_API_KEY}\nRUNPOD_VOLUME_ID={RUNPOD_VOLUME_ID}\n"
       "RUNPOD_REGISTRY_AUTH_ID={RUNPOD_REGISTRY_AUTH_ID}\nWORKER_IMAGE={IMAGE}\nDATA_DIR=data\nLAST=1")


class World:
    def __init__(self, tmp_path) -> None:
        self.tmp = tmp_path
        self.env = tmp_path / ".env"
        self.env.write_text(ENV.format(**SECRETS, IMAGE=IMAGE))
        os.chmod(self.env, 0o600)
        self.state_path = tmp_path / "data" / "compute.json"
        self.app = FakeApp()
        self.runpod = FakeRunPod()
        self.clock = Clock()
        self.probes: list[str] = []
        self.on_probe = lambda: {"ok": True, "backend": "worker", "version": "1.2.3"}
        self.deps = Deps(env_path=self.env, environ={},
                         app_client=httpx.Client(transport=httpx.MockTransport(self.app)),
                         runpod_client=httpx.Client(transport=httpx.MockTransport(self.runpod)),
                         cloudflared=fake_cloudflared(tmp_path), probe=self.probe,
                         sleep=self.clock.sleep, clock=self.clock)

    def probe(self, url: str) -> dict:
        self.probes.append(url)
        return self.on_probe()

    def run(self, capsys, *argv: str) -> tuple[int, str, str]:
        code = main(list(argv), self.deps)
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    def state(self) -> dict:
        return state_mod.load(self.state_path)

    def stop_tunnel(self) -> None:
        t = self.state()["tunnel"]
        if t:
            tunnel.stop(t["pid"], t["binary"])


@pytest.fixture
def world(tmp_path):
    w = World(tmp_path)
    yield w
    w.stop_tunnel()


def assert_no_secret(text: str, *extra: str) -> None:
    for value in [*SECRETS.values(), *extra]:
        assert value not in text


def test_app_on_the_wrong_backend_exits_2(world, capsys):
    world.app.backend = "fake"
    code, _out, err = world.run(capsys, "runpod")
    assert code == 2
    assert "uv run python -m radar_desk\n" in err and "--wait" in err and "is on backend fake" in err
    assert world.state()["tunnel"] is None and world.runpod.deployed == []


def test_wait_until_the_app_is_on_worker(world, capsys):
    world.app.backend = "fake"
    world.app.flip_after = 3
    code, _out, err = world.run(capsys, "runpod", "--wait")
    assert code == 0, err
    assert world.clock.sleeps[:3] == [2, 2, 2]
    assert len(world.runpod.deployed) == 1


def test_full_runpod_run_then_a_second_run_deploys_nothing(world, capsys):
    before = world.env.read_bytes()
    code, out, err = world.run(capsys, "runpod")
    assert code == 0, err
    assert world.env.read_bytes() == before.replace(b"GPU_BACKEND=fake\n", b"GPU_BACKEND=worker\n")
    assert world.env.stat().st_mode & 0o777 == 0o600

    [inp] = world.runpod.deployed
    env = {e["key"]: e["value"] for e in inp["env"]}
    plaintext = world.app.tokens["wtok_1"]
    assert env["RADAR_WORKER_TOKEN"] == plaintext
    assert env["RADAR_DESK_URL"] == FAKE_URL and world.probes[0] == FAKE_URL
    assert env["RADAR_WORKER_ID"].startswith("runpod-") and env["RADAR_IMAGE"] == IMAGE == inp["imageName"]
    assert inp["networkVolumeId"] == SECRETS["RUNPOD_VOLUME_ID"]
    assert inp["containerRegistryAuthId"] == SECRETS["RUNPOD_REGISTRY_AUTH_ID"]
    assert inp["dataCenterId"] == "EU-RO-1" and inp["gpuTypeId"] == "NVIDIA L4"

    s = world.state()
    assert s["token_id"] == "wtok_1" and s["worker_id"] == env["RADAR_WORKER_ID"]
    assert s["pod"]["id"] == "pod1" and s["pod"]["cost_per_hr"] == 0.39 and s["pod"]["started_at"]
    assert s["tunnel"]["url"] == FAKE_URL and tunnel.alive(s["tunnel"]["pid"], s["tunnel"]["binary"])
    raw = world.state_path.read_text()
    assert plaintext not in raw and world.state_path.stat().st_mode & 0o777 == 0o600
    assert_no_secret(out + err + raw, plaintext)
    assert f"pod: deployed pod1 on L4 at $0.390/h, image {IMAGE}" in out and FAKE_URL in out

    code, out, err = world.run(capsys, "runpod")
    assert code == 0 and "already running" in out
    assert len(world.runpod.deployed) == 1 and list(world.app.tokens) == ["wtok_1"]


def test_status_prints_pod_tunnel_and_workers(world, capsys):
    assert world.run(capsys, "runpod")[0] == 0
    world.app.workers = [{"id": "runpod-abc123", "online": True, "job_id": "job_7"}]
    code, out, err = world.run(capsys, "status")
    assert code == 0
    assert f".env: GPU_BACKEND=worker, WORKER_IMAGE={IMAGE}" in out
    assert "app: backend worker, version 1.2.3" in out
    assert f"tunnel: {FAKE_URL}" in out and "alive, reachable, backend worker" in out
    assert "pod: pod1" in out and f"image {IMAGE}" in out and "other pods: none" in out
    assert "worker: runpod-abc123, online, job job_7" in out
    assert "runpod spend: $0.390/h" in out
    assert_no_secret(out + err)


def test_status_without_a_runpod_key(world, capsys):
    world.env.write_text("OWNER_TOKEN=owner-secret-1\nGPU_BACKEND=modal\n")
    code, out, _err = world.run(capsys, "status")
    assert code == 0
    assert "pod: unknown, RUNPOD_API_KEY is not set" in out and "app: backend worker" in out
    assert "workers: none" in out


def test_stop_deletes_kills_revokes_and_empties_the_state(world, capsys):
    assert world.run(capsys, "runpod")[0] == 0
    t = world.state()["tunnel"]
    code, out, err = world.run(capsys, "stop")
    assert code == 0, err
    assert world.runpod.deleted == ["pod1"] and world.runpod.pods == []
    assert not tunnel.alive(t["pid"], t["binary"])
    assert world.app.revoked == ["wtok_1"]
    assert world.state() == state_mod.empty()
    assert b"GPU_BACKEND=worker\n" in world.env.read_bytes()
    assert "runpod spend: $0.000/h" in out

    code, out, err = world.run(capsys, "stop", "--modal")
    assert code == 0 and "nothing to stop" in out
    assert b"GPU_BACKEND=modal\n" in world.env.read_bytes()


def test_modal_stops_everything_and_sets_the_mode(world, capsys):
    assert world.run(capsys, "runpod")[0] == 0
    code, out, err = world.run(capsys, "modal")
    assert code == 0, err
    assert b"GPU_BACKEND=modal\n" in world.env.read_bytes()
    assert world.runpod.pods == [] and world.app.revoked == ["wtok_1"]
    assert world.state() == state_mod.empty()
    assert "uv run python -m radar_desk" in out


def test_missing_runpod_key_is_named_and_nothing_is_written(world, capsys):
    world.env.write_text(ENV.format(**SECRETS, IMAGE=IMAGE).replace(f"RUNPOD_API_KEY={SECRETS['RUNPOD_API_KEY']}\n", ""))
    before = world.env.read_bytes()
    code, _out, err = world.run(capsys, "runpod")
    assert code == 1 and "RUNPOD_API_KEY" in err
    assert world.env.read_bytes() == before and not world.state_path.exists()


def test_missing_worker_image_is_named_and_nothing_is_written(world, capsys):
    world.env.write_text(ENV.format(**SECRETS, IMAGE=IMAGE).replace(f"WORKER_IMAGE={IMAGE}\n", ""))
    before = world.env.read_bytes()
    code, _out, err = world.run(capsys, "runpod")
    assert code == 1 and "WORKER_IMAGE" in err
    assert world.env.read_bytes() == before and not world.state_path.exists()


def test_other_pods_block_without_the_flag(world, capsys):
    world.runpod.pods.append({"id": "foreign", "name": "someone-else", "costPerHr": 1.0,
                              "desiredStatus": "RUNNING", "runtime": None, "machine": None})
    code, out, err = world.run(capsys, "runpod")
    assert code == 1 and "foreign" in err and "--allow-other-pods" in err
    assert world.runpod.deployed == [] and world.app.tokens == {}

    code, out, err = world.run(capsys, "runpod", "--allow-other-pods")
    assert code == 0, err
    assert len(world.runpod.deployed) == 1 and "tunnel: reusing" in out


def radar_worker_pod(pod_id: str) -> dict:
    return {"id": pod_id, "name": "radar-worker", "imageName": IMAGE, "costPerHr": 0.39, "desiredStatus": "RUNNING",
            "runtime": {"uptimeInSeconds": 60}, "machine": {"gpuDisplayName": "L4"}}


def test_a_listed_radar_worker_pod_is_adopted(world, capsys):
    world.runpod.pods.append(radar_worker_pod("lost1"))
    code, out, _err = world.run(capsys, "runpod")
    assert code == 0 and "pod: adopted lost1" in out
    assert world.runpod.deployed == [] and world.state()["pod"]["id"] == "lost1"
    code, _out, _err = world.run(capsys, "stop")
    assert code == 0 and world.runpod.deleted == ["lost1"] and world.runpod.pods == []


def test_a_pod_that_appears_before_the_deploy_is_adopted(world, capsys):
    def probe():
        world.runpod.pods.append(radar_worker_pod("late1"))
        return {"backend": "worker"}

    world.on_probe = probe
    code, out, err = world.run(capsys, "runpod")
    assert code == 0, err
    assert "pod: adopted late1" in out
    assert world.runpod.deployed == [] and world.state()["pod"]["id"] == "late1"


def test_a_failed_deploy_never_prints_the_token_and_a_rerun_revokes_it(world, capsys):
    world.runpod.fail_deploy = True
    code, out, err = world.run(capsys, "runpod")
    plaintext = world.app.tokens["wtok_1"]
    assert code == 1 and "bad input" in err and "<worker token>" in err
    assert_no_secret(out + err, plaintext)
    assert world.state()["token_id"] == "wtok_1" and world.state()["pod"] is None

    world.runpod.fail_deploy = False
    code, out, err = world.run(capsys, "runpod")
    assert code == 0, err
    assert "revoked wtok_1 from an earlier attempt" in out and world.app.revoked == ["wtok_1"]
    assert world.state()["token_id"] == "wtok_2" and len(world.runpod.deployed) == 1


def test_a_running_pod_without_its_tunnel_fails(world, capsys):
    assert world.run(capsys, "runpod")[0] == 0
    world.stop_tunnel()
    code, out, err = world.run(capsys, "runpod")
    assert code == 1 and "pod already running: pod1" in out and "tunnel it was given is gone" in err
    assert len(world.runpod.deployed) == 1


def test_a_tunnel_that_never_answers_worker(world, capsys):
    world.on_probe = lambda: None
    code, _out, err = world.run(capsys, "runpod")
    assert code == 1 and "did not answer backend worker" in err
    t = world.state()["tunnel"]
    assert t and tunnel.alive(t["pid"], t["binary"])
    assert world.runpod.deployed == [] and world.app.tokens == {}


def test_stop_with_the_app_down(world, capsys):
    assert world.run(capsys, "runpod")[0] == 0
    t = world.state()["tunnel"]

    def down(request):
        raise httpx.ConnectError("connection refused", request=request)

    world.deps.app_client = httpx.Client(transport=httpx.MockTransport(down))
    code, _out, err = world.run(capsys, "stop")
    assert code == 1 and "wtok_1" in err
    assert world.runpod.pods == [] and not tunnel.alive(t["pid"], t["binary"])
    s = world.state()
    assert s["token_id"] == "wtok_1" and s["pod"] is None and s["tunnel"] is None


def test_a_pod_runpod_no_longer_lists_is_replaced(world, capsys):
    assert world.run(capsys, "runpod")[0] == 0
    world.runpod.pods.clear()
    code, _out, err = world.run(capsys, "runpod")
    assert code == 0, err
    assert len(world.runpod.deployed) == 2 and world.state()["pod"]["id"] == "pod2"
