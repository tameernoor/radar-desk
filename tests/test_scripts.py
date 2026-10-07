"""The command-line scripts that run against the local store."""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from radar_desk.config import Settings
from radar_desk.radar import catalog
from test_services import done_job, ready_scan

ROOT = Path(__file__).resolve().parents[1]


def test_export_all_writes_one_row_per_done_job(make_services, tmp_path):
    svc = make_services()
    scan = ready_scan(svc, tmp_path)
    done_job(svc, scan.id)
    done_job(svc, scan.id)
    svc.jobs.create(scan.id)  # queued, not exported

    out = tmp_path / "out" / "scores.csv"
    env = {k: v for k, v in os.environ.items() if k not in ("GPU_BACKEND", "DATA_DIR")}
    env.update(OWNER_TOKEN="t", SESSION_SECRET="s", DATA_DIR=str(tmp_path / "data"), GPU_BACKEND="fake")
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "export_all.py"), str(out)],
                          cwd=tmp_path, env=env, capture_output=True, text=True, check=True)

    assert proc.stdout.startswith("2 rows written")
    rows = list(csv.reader(io.StringIO(out.read_bytes().decode("utf-8-sig"))))
    assert rows[0] == catalog.csv_header()
    assert len(rows) == 3 and all(r[0] == scan.filename for r in rows[1:])


def test_seed_dev_runs_in_a_fresh_interpreter(tmp_path):
    """The seed imports the poller before the services; a circular import there broke it once."""
    env = {k: v for k, v in os.environ.items() if k not in ("GPU_BACKEND", "DATA_DIR")}
    env.update(OWNER_TOKEN="t", SESSION_SECRET="s", DATA_DIR=str(tmp_path / "data"), GPU_BACKEND="fake")
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "seed_dev.py")],
                          cwd=tmp_path, env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("scan_id scan_")


# scripts/runpod_endpoint.py against a fake RunPod (both hosts) behind httpx.MockTransport

KEY = "rp-secret-key-123"
CATALOG = {"gpus": [
    {"id": "NVIDIA L4", "name": "L4", "pool": "AMPERE_24", "memory": 24, "price": {"serverless": 0.69}},
    {"id": "NVIDIA RTX A5000", "name": "RTX A5000", "pool": "AMPERE_24", "memory": 24},
    {"id": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "name": "MIG 24GB", "pool": "AMPERE_24",
     "memory": 24},
    {"id": "NVIDIA GeForce RTX 4090", "name": "RTX 4090", "pool": "ADA_24", "memory": 24},
    {"id": "NVIDIA RTX PRO 4500 Blackwell", "name": "RTX PRO 4500", "pool": "BLACKWELL_ONLY", "memory": 32},
    {"id": "NVIDIA A40", "name": "A40", "pool": "AMPERE_48", "memory": 48},
    {"id": "NVIDIA RTX PRO 5000", "name": "RTX PRO 5000 Blackwell", "pool": "ADA_24", "memory": 24},
    {"id": "NVIDIA H200", "name": "H200", "pool": None, "memory": 141},
]}


class FakeRunPodApi:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, dict | None]] = []
        self.health = {"jobs": {"completed": 3, "failed": 0, "inProgress": 0, "inQueue": 0, "retried": 0},
                       "workers": {"idle": 1, "running": 0}}
        self.status: int | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        host, path, method = request.url.host, request.url.path, request.method
        payload = json.loads(request.content) if request.content else None
        self.calls.append((host, method, path, payload))
        assert request.headers["authorization"] == f"Bearer {KEY}"
        if self.status:
            return httpx.Response(self.status, json={"title": "Unauthorized", "detail": "invalid api key"})
        if host == "api.runpod.ai":
            assert path == "/v2/ep_new1/health"
            return httpx.Response(200, json=self.health)
        assert host == "api.runpod.io"
        if method == "GET" and path == "/v2/catalog/gpus":
            return httpx.Response(200, json=CATALOG)
        if method == "POST" and path == "/v2/serverless":
            return httpx.Response(201, json={"id": "ep_new1", **payload})
        if method == "GET" and path == "/v2/serverless/ep_new1":
            return httpx.Response(200, json={
                "id": "ep_new1", "name": "radar-desk", "image": "ghcr.io/o/radar-worker:0.3",
                "workers": {"min": 0, "max": 1, "idleTimeout": 60}, "timeout": 1800000, "flashboot": "FLASHBOOT",
                "gpu": {"pools": ["AMPERE_24"], "excludedTypes": ["NVIDIA RTX PRO 6000 Blackwell Workstation Edition"],
                        "count": 1, "allowedCudaVersions": [], "minCudaVersion": "12.4"},
                "dataCenterIds": ["EU-RO-1"], "networkVolumes": ["vol123"], "createdAt": "2026-10-02T10:00:00Z"})
        if method == "GET" and path == "/v2/serverless/ep_new1/workers":
            return httpx.Response(200, json={"endpointVersion": 1, "summary": {
                "running": 0, "idle": 1, "initializing": 2, "throttled": 0, "unhealthy": 0, "total": 3}, "workers": []})
        if method == "PATCH" and path == "/v2/serverless/ep_new1":
            return httpx.Response(200, json={"id": "ep_new1", **payload})
        if method == "DELETE" and path == "/v2/serverless/ep_new1":
            return httpx.Response(204)
        return httpx.Response(404, json={"title": "Not Found", "detail": "no such route"})


def _endpoint_script():
    spec = importlib.util.spec_from_file_location("runpod_endpoint", ROOT / "scripts" / "runpod_endpoint.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # the Deps dataclass looks its module up while it is built
    spec.loader.exec_module(module)
    return module


def _settings(**over) -> Settings:
    values = {"owner_token": "o", "session_secret": "s", "runpod_api_key": KEY,
              "worker_image": "ghcr.io/o/radar-worker:0.3", "runpod_registry_auth_id": "reg1",
              "runpod_volume_id": "vol123", "runpod_datacenter": "EU-RO-1", "runpod_endpoint_id": "ep_new1",
              "runpod_serverless_gpus": "AMPERE_24,ADA_24", "runpod_serverless_idle_s": 60, "gpu_timeout_s": 1800}
    values.update(over)
    return Settings(_env_file=None, **values)


@pytest.fixture
def endpoint(capsys):
    script = _endpoint_script()
    fake = FakeRunPodApi()

    def run(*argv: str, **settings) -> tuple[int, str, str]:
        deps = script.Deps(settings=_settings(**settings), client=httpx.Client(transport=httpx.MockTransport(fake)))
        code = script.main(list(argv), deps)
        captured = capsys.readouterr()
        assert KEY not in captured.out + captured.err
        return code, captured.out, captured.err

    return fake, run


def test_endpoint_create_sends_the_exact_body(endpoint):
    fake, run = endpoint
    code, out, err = run("create", runpod_endpoint_id=None)
    assert code == 0, err
    blackwell = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
    by_name = "NVIDIA RTX PRO 5000"  # a bland id whose catalog name says Blackwell
    posts = [c for c in fake.calls if c[1] == "POST"]
    assert posts == [("api.runpod.io", "POST", "/v2/serverless", {
        "name": "radar-desk", "type": "QUEUE",
        "image": "ghcr.io/o/radar-worker:0.3", "registry": "reg1", "disk": 20,
        "cmd": ["python3", "-u", "-m", "radar_worker.serverless"],
        "env": {"RADAR_WEIGHTS_DIR": "/runpod-volume/radar-weights", "RADAR_DATA_ROOT": "/runpod-volume",
                "RADAR_IMAGE": "ghcr.io/o/radar-worker:0.3", "RADAR_DEVICE": "auto"},
        "gpu": {"pools": ["AMPERE_24", "ADA_24"], "excludedTypes": [blackwell, by_name], "count": 1, "minCudaVersion": "12.4"},
        "scaling": {"type": "QUEUE_DELAY", "queueDelay": 4},
        "workers": {"min": 0, "max": 1, "idleTimeout": 60},
        "timeout": 1800000, "flashboot": "FLASHBOOT",
        "dataCenterIds": ["EU-RO-1"], "networkVolumes": ["vol123"]})]
    lines = out.splitlines()
    assert f"pool AMPERE_24: kept NVIDIA L4, NVIDIA RTX A5000; excluded {blackwell}" in lines
    assert f"pool ADA_24: kept NVIDIA GeForce RTX 4090; excluded {by_name}" in lines
    assert "created endpoint ep_new1 (radar-desk)" in lines
    assert lines[-1] == "RUNPOD_ENDPOINT_ID=ep_new1"


def test_endpoint_create_omits_an_empty_exclusion_list(endpoint):
    fake, run = endpoint
    code, _, _ = run("create", runpod_serverless_gpus="AMPERE_48", runpod_endpoint_id=None)
    assert code == 0
    body = next(c for c in fake.calls if c[1] == "POST")[3]
    assert body["gpu"] == {"pools": ["AMPERE_48"], "count": 1, "minCudaVersion": "12.4"}


def test_endpoint_create_refuses_when_an_endpoint_is_set_unless_forced(endpoint):
    fake, run = endpoint
    code, _, err = run("create")
    assert code == 1 and err.strip() == ("refused: RUNPOD_ENDPOINT_ID is already set (ep_new1); delete that endpoint "
                                         "first, or pass --force to create another")
    assert fake.calls == []
    code, out, err = run("create", "--force")
    assert code == 0, err
    assert [c[2] for c in fake.calls if c[1] == "POST"] == ["/v2/serverless"]
    assert out.splitlines()[-1] == "RUNPOD_ENDPOINT_ID=ep_new1"


def test_endpoint_create_refuses_a_pool_left_empty_or_unknown(endpoint):
    fake, run = endpoint
    code, out, err = run("create", runpod_serverless_gpus="ADA_24,BLACKWELL_ONLY", runpod_endpoint_id=None)
    assert code == 1
    assert "pool BLACKWELL_ONLY: kept none; excluded NVIDIA RTX PRO 4500 Blackwell" in out
    assert "pool BLACKWELL_ONLY holds no usable GPU" in err
    code, _, err = run("create", runpod_serverless_gpus="AMPERE_24,HOPPER_9", runpod_endpoint_id=None)
    assert code == 1 and "RUNPOD_SERVERLESS_GPUS: pool HOPPER_9 is not in RunPod's GPU catalog" in err
    assert not [c for c in fake.calls if c[1] == "POST"]


@pytest.mark.parametrize("image", ["ghcr.io/o/radar-worker:latest", "ghcr.io/o/radar-worker",
                                   "localhost:5000/radar-worker"])
def test_endpoint_create_and_update_refuse_an_unpinned_image(endpoint, image):
    fake, run = endpoint
    for command in ("create", "update"):
        code, _, err = run(command, worker_image=image)
        assert code == 1 and "WORKER_IMAGE" in err and "pinned tag" in err
    assert fake.calls == []


def test_endpoint_create_names_an_unset_setting_before_any_call(endpoint):
    fake, run = endpoint
    code, _, err = run("create", worker_image=None)
    assert code == 1 and "WORKER_IMAGE is not set" in err
    code, _, err = run("show", runpod_endpoint_id=None)
    assert code == 1 and "RUNPOD_ENDPOINT_ID is not set" in err
    code, _, err = run("show", runpod_api_key=None)
    assert code == 1 and "RUNPOD_API_KEY is not set" in err
    assert fake.calls == []


def test_endpoint_show_prints_the_endpoint_workers_and_health(endpoint):
    fake, run = endpoint
    code, out, err = run("show")
    assert code == 0, err
    assert "endpoint ep_new1 (radar-desk), image ghcr.io/o/radar-worker:0.3" in out
    assert "workers: min 0, max 1, idleTimeout 60 s; timeout 1800000 ms; flashboot FLASHBOOT" in out
    assert "gpu: pools AMPERE_24; excluded NVIDIA RTX PRO 6000 Blackwell Workstation Edition" in out
    assert "dataCenterIds EU-RO-1; networkVolumes vol123" in out
    assert "worker summary: running 0, idle 1, initializing 2, throttled 0, unhealthy 0, total 3" in out
    assert "health workers: idle 1, running 0" in out
    assert "health jobs: inQueue 0, inProgress 0, completed 3, failed 0, retried 0" in out
    assert ("api.runpod.ai", "GET", "/v2/ep_new1/health", None) in fake.calls


def test_endpoint_update_patches_image_workers_timeout_and_env(endpoint):
    fake, run = endpoint
    code, out, err = run("update", worker_image="ghcr.io/o/radar-worker:0.4", runpod_serverless_idle_s=30)
    assert code == 0, err
    assert fake.calls == [("api.runpod.io", "PATCH", "/v2/serverless/ep_new1", {
        "image": "ghcr.io/o/radar-worker:0.4", "workers": {"min": 0, "max": 1, "idleTimeout": 30},
        "timeout": 1800000,
        "env": {"RADAR_WEIGHTS_DIR": "/runpod-volume/radar-weights", "RADAR_DATA_ROOT": "/runpod-volume",
                "RADAR_IMAGE": "ghcr.io/o/radar-worker:0.4", "RADAR_DEVICE": "auto"}})]
    assert out.startswith("updated endpoint ep_new1")


def test_endpoint_delete_refuses_jobs_in_flight_unless_forced(endpoint):
    fake, run = endpoint
    fake.health["jobs"]["inQueue"] = 1
    code, out, err = run("delete")
    assert code == 1 and "1 job(s) queued or running" in err and "--force" in err
    assert not [c for c in fake.calls if c[1] == "DELETE"]
    code, out, err = run("delete", "--force")
    assert code == 0, err
    assert ("api.runpod.io", "DELETE", "/v2/serverless/ep_new1", None) in fake.calls
    assert out.splitlines()[-1] == "deleted endpoint ep_new1; remove RUNPOD_ENDPOINT_ID from .env"


def test_endpoint_delete_goes_ahead_when_idle(endpoint):
    _, run = endpoint
    code, out, _ = run("delete")
    assert code == 0 and "deleted endpoint ep_new1" in out


def test_endpoint_runpod_error_exits_1_without_the_key(endpoint):
    fake, run = endpoint
    fake.status = 401
    code, _, err = run("show")
    assert code == 1 and "RunPod answered 401: invalid api key" in err


def test_endpoint_bad_settings_exit_2(capsys, monkeypatch, tmp_path):
    script = _endpoint_script()
    monkeypatch.chdir(tmp_path)
    for name in ("OWNER_TOKEN", "SESSION_SECRET"):
        monkeypatch.delenv(name, raising=False)
    assert script.main(["show"], script.Deps(client=httpx.Client(transport=httpx.MockTransport(FakeRunPodApi())))) == 2
    assert "OWNER_TOKEN is not set" in capsys.readouterr().err


def test_endpoint_help_runs_without_settings(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RUNPOD_", "OWNER_", "SESSION_"))}
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "runpod_endpoint.py"), "--help"],
                          cwd=tmp_path, env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert "create" in proc.stdout and "delete" in proc.stdout


def test_endpoint_keeps_root_only_when_results_go_to_the_runpod_volume():
    script = _endpoint_script()
    image = "ghcr.io/o/radar-worker:0.5"
    assert "RADAR_RUN_AS_ROOT" not in script.worker_env(image, _settings())
    on_volume = _settings(storage_backend="runpod_volume")
    assert script.worker_env(image, on_volume)["RADAR_RUN_AS_ROOT"] == "1"
