"""GET and PUT /compute, POST /compute/pod/start and /compute/pod/stop."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from radar_desk.app import create_app
from radar_desk.config import ConfigError, Settings
from radar_desk.services.compute import (
    FIXED,
    MODAL_NEEDS_STORAGE,
    MODAL_ON_LOCAL,
    MODAL_ON_RUNPOD_VOLUME,
    MODAL_ON_VOLUME,
    RUNPOD_FALLBACK,
    RUNPOD_NEEDS_CONFIG,
    SERVERLESS_FALLBACK,
    SERVERLESS_NEEDS_CONFIG,
    SERVERLESS_NEEDS_STORAGE,
    mode_refusal,
)
from test_compute_service import OWNER, World
from test_serverless_backend import ENDPOINT

FIELDS = {"mode", "changeable", "changed_at", "tunnel_mode", "public_url", "runpod", "serverless", "pod",
          "in_flight", "queued", "held", "problem", "last_event", "spend_month_usd", "budget_usd", "month",
          "storage"}
POD_FIELDS = {"id", "runpod_id", "phase", "gpu", "image", "cost_per_hr", "created_at", "started_at",
              "ready_at", "up_s", "idle_s", "idle_delete_s", "worker_id", "tunnel_url", "tunnel_alive", "job_id"}


def clients(svc) -> tuple[TestClient, TestClient]:
    app = create_app(svc.settings, svc, start_poller=False)
    return TestClient(app, headers={"Authorization": f"Bearer {OWNER}"}), TestClient(app)


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def test_compute_routes(world):
    owner, _ = clients(world.svc)
    body = owner.get("/compute").json()
    assert set(body) == FIELDS
    assert body["mode"] == "runpod" and body["changeable"] is True and body["tunnel_mode"] == "managed"
    assert list(body["storage"]["modes"]) == ["modal", "worker", "runpod", "serverless"]
    assert body["runpod"] == {"configured": True, "datacenter": "EU-RO-1", "max_pod_hours": 3,
                              "gpus": ["NVIDIA L4", "NVIDIA GeForce RTX 4090"], "idle_delete_s": 600,
                              "app_lost_delete_s": 600}
    assert body["serverless"] == {"configured": False, "endpoint_id": None, "gpus": ["AMPERE_24", "ADA_24"],
                                  "idle_s": 60, "price_per_s": 0.00031, "health": None, "job": None}
    assert body["pod"] is None and body["queued"] == 0 and body["month"] == "2026-10"

    r = owner.put("/compute", json={"mode": "modal"})
    assert r.status_code == 200 and r.json()["mode"] == "modal"
    assert owner.get("/health").json()["backend"] == "modal"
    assert owner.post("/compute/pod/start").status_code == 409
    assert owner.put("/compute", json={"mode": "worker"}).json()["mode"] == "worker"
    assert owner.get("/health").json()["backend"] == "worker"
    assert owner.post("/compute/pod/start").status_code == 409
    assert owner.put("/compute", json={"mode": "runpod"}).json()["mode"] == "runpod"
    assert owner.get("/health").json()["backend"] == "runpod"
    r = owner.put("/compute", json={"mode": "cloud"})
    assert r.status_code == 409 and "cloud" in r.json()["detail"]

    r = owner.post("/compute/pod/start")
    assert r.status_code == 202 and set(r.json()) == FIELDS
    world.tick()
    body = owner.get("/compute").json()
    assert set(body["pod"]) == POD_FIELDS and body["pod"]["phase"] == "tunnel"
    assert owner.post("/compute/pod/start").status_code == 409
    r = owner.post("/compute/pod/stop")
    assert r.status_code == 200 and r.json()["pod"] is None
    r = owner.post("/compute/pod/stop")
    assert r.status_code == 409 and r.json()["detail"] == "no pod is running"


def test_modal_on_local_storage_is_refused(tmp_path):
    w = World(tmp_path, storage_backend=None, s3_bucket=None)
    owner, _ = clients(w.svc)
    r = owner.put("/compute", json={"mode": "modal"})
    assert r.status_code == 409 and r.json()["detail"] == MODAL_NEEDS_STORAGE


def test_serverless_through_the_routes(tmp_path):
    w = World(tmp_path, runpod_endpoint_id=ENDPOINT)
    owner, _ = clients(w.svc)
    r = owner.put("/compute", json={"mode": "serverless"})
    assert r.status_code == 200 and r.json()["mode"] == "serverless"
    assert r.json()["serverless"]["configured"] is True and r.json()["serverless"]["endpoint_id"] == ENDPOINT
    assert owner.get("/health").json()["backend"] == "serverless"
    assert owner.get("/gpu/status").json()["compute_mode"] == "serverless"


@pytest.mark.parametrize("overrides,mode,detail", [
    ({}, "serverless", SERVERLESS_NEEDS_CONFIG),
    ({"runpod_endpoint_id": ENDPOINT, "storage_backend": None, "s3_bucket": None}, "serverless",
     SERVERLESS_NEEDS_STORAGE),
    ({"runpod_endpoint_id": ENDPOINT, "storage_backend": "modal_volume"}, "serverless", SERVERLESS_NEEDS_STORAGE),
    ({"storage_backend": "runpod_volume"}, "modal", MODAL_ON_RUNPOD_VOLUME),
    ({"gpu_backend": "worker", "runpod_api_key": None}, "runpod", RUNPOD_NEEDS_CONFIG),
    ({"gpu_backend": "worker", "worker_image": None}, "runpod", RUNPOD_NEEDS_CONFIG),
])
def test_put_refusals(tmp_path, overrides, mode, detail):
    w = World(tmp_path, **overrides)
    owner, _ = clients(w.svc)
    r = owner.put("/compute", json={"mode": mode})
    assert r.status_code == 409 and r.json()["detail"] == detail


@pytest.mark.parametrize("overrides,message", [
    ({"gpu_backend": "modal", "storage_backend": "runpod_volume"}, MODAL_ON_RUNPOD_VOLUME),
    ({"gpu_backend": "serverless", "s3_bucket": "b"}, f"GPU_BACKEND=serverless: {SERVERLESS_NEEDS_CONFIG}"),
    ({"gpu_backend": "serverless", "runpod_api_key": "k", "runpod_endpoint_id": ENDPOINT},
     f"GPU_BACKEND=serverless: {SERVERLESS_NEEDS_STORAGE}"),
    ({"gpu_backend": "runpod", "runpod_api_key": "k"}, f"GPU_BACKEND=runpod: {RUNPOD_NEEDS_CONFIG}"),
])
def test_create_app_refuses_at_start(tmp_path, overrides, message):
    settings = Settings(_env_file=None, owner_token=OWNER, session_secret="s", data_dir=tmp_path, **overrides)
    with pytest.raises(ConfigError) as info:
        create_app(settings, start_poller=False)
    assert str(info.value) == message


@pytest.mark.parametrize("overrides,mode,reason", [
    ({"storage_backend": None, "s3_bucket": None}, "serverless", SERVERLESS_FALLBACK),
    ({}, "serverless", SERVERLESS_FALLBACK),
    ({"storage_backend": "runpod_volume"}, "modal", MODAL_ON_VOLUME),
    ({"gpu_backend": "worker", "runpod_api_key": None}, "runpod", RUNPOD_FALLBACK),
])
def test_a_stored_mode_no_longer_allowed_falls_back_to_worker(tmp_path, overrides, mode, reason):
    w = World(tmp_path, **overrides)
    w.svc.db.set_setting("mode", mode)
    create_app(w.svc.settings, w.svc, start_poller=False)
    assert w.compute.mode == "worker"
    assert w.compute.status()["last_event"].endswith(reason)


def test_a_stored_serverless_mode_that_is_allowed_stays(tmp_path):
    w = World(tmp_path, runpod_endpoint_id=ENDPOINT)
    w.svc.db.set_setting("mode", "serverless")
    create_app(w.svc.settings, w.svc, start_poller=False)
    assert w.compute.mode == "serverless"


def test_create_app_migrates_a_stored_worker_mode(tmp_path):
    w = World(tmp_path)
    w.svc.db.set_setting("mode", "worker")  # a database from before mode runpod, no marker
    w.svc.db.set_setting("mode_schema", None)
    owner, _ = clients(w.svc)
    assert owner.get("/compute").json()["mode"] == "runpod"
    assert w.svc.db.get_setting("mode_schema") == "2"


def test_the_fake_backend_is_fixed(make_services):
    owner, _ = clients(make_services())
    body = owner.get("/compute").json()
    assert body["mode"] == "fake" and body["changeable"] is False and body["runpod"]["configured"] is False
    assert body["storage"]["backend"] == "local" and body["storage"]["name"] == "Local folder"
    assert body["serverless"]["configured"] is False and body["serverless"]["job"] is None
    r = owner.put("/compute", json={"mode": "worker"})
    assert r.status_code == 409 and r.json()["detail"] == FIXED
    assert owner.get("/health").json()["backend"] == "fake"
    assert owner.get("/gpu/status").json()["compute_mode"] == "fake"


def test_compute_routes_need_the_owner(world):
    _, anon = clients(world.svc)
    assert anon.get("/compute").status_code == 401
    assert anon.put("/compute", json={"mode": "modal"}).status_code == 401
    assert anon.post("/compute/pod/start").status_code == 401
    assert anon.post("/compute/pod/stop").status_code == 401
    assert world.compute.mode == "runpod"


def test_a_stored_modal_mode_on_local_storage_falls_back_to_worker(tmp_path):
    w = World(tmp_path, storage_backend=None, s3_bucket=None)
    w.svc.db.set_setting("mode", "modal")
    create_app(w.svc.settings, w.svc, start_poller=False)
    assert w.compute.mode == "worker"
    assert w.compute.status()["last_event"].endswith(MODAL_ON_LOCAL)


def test_the_lifespan_reconciles_before_the_poller(world):
    world.rp.add("rp-stray")
    world.rp.add("someone-elses", name="training")
    with TestClient(create_app(world.svc.settings, world.svc)):
        pass
    assert world.rp.deleted == ["rp-stray"]


LOCAL_MODAL = ("not on local storage", "Modal cannot reach a local folder; this app stores scans under DATA_DIR")
VOLUME_MODAL = ("not on the RunPod volume", "Modal cannot reach the RunPod volume; it has no presigned URLs")
NEEDS_BUCKET = "Serverless needs a shared bucket or the RunPod volume; this app stores scans "
LOCAL_SERVERLESS = ("needs a shared bucket", NEEDS_BUCKET + "in a local folder")
MODAL_SERVERLESS = ("needs a shared bucket", NEEDS_BUCKET + "on the Modal volume")
NO_KEY = ("no RunPod key", "RunPod keys missing; set RUNPOD_API_KEY")
NO_ENDPOINT = ("no endpoint", "No endpoint configured; run scripts/runpod_endpoint.py create and set RUNPOD_ENDPOINT_ID")
NO_POD = ("not configured", "A pod needs RUNPOD_VOLUME_ID, RUNPOD_REGISTRY_AUTH_ID and WORKER_IMAGE")

# kind -> (overrides, backend, name, modal row or None, serverless row from the storage or None)
STORAGES = {
    "local": ({"storage_backend": None, "s3_bucket": None}, "local", "Local folder", LOCAL_MODAL,
              LOCAL_SERVERLESS),
    "s3": ({"storage_backend": None, "s3_bucket": "b", "s3_endpoint_url": "https://fly.storage.tigris.dev"},
           "s3", "Tigris bucket b", None, None),
    "modal_volume": ({"storage_backend": "modal_volume"}, "modal_volume", "Modal volume radar-data", None,
                     MODAL_SERVERLESS),
    "runpod_volume": ({"storage_backend": "runpod_volume", "runpod_s3_access_key_id": "a",
                       "runpod_s3_secret_access_key": "b"}, "runpod_volume", "RunPod volume vol-1 (EU-RO-1)",
                      VOLUME_MODAL, None),
}
# config -> (overrides, serverless row, runpod row)
RUNPOD_CONFIGS = {
    "no key": ({"runpod_api_key": None}, NO_KEY, NO_KEY),
    "key only": ({}, NO_ENDPOINT, None),
    "key and endpoint": ({"runpod_endpoint_id": ENDPOINT}, None, None),
    "no image": ({"worker_image": None}, NO_ENDPOINT, NO_POD),
}


@pytest.mark.parametrize("config", RUNPOD_CONFIGS)
@pytest.mark.parametrize("kind", STORAGES)
def test_the_storage_block(tmp_path, kind, config):
    overrides, backend, name, modal_row, storage_row = STORAGES[kind]
    config_overrides, config_row, runpod_row = RUNPOD_CONFIGS[config]
    w = World(tmp_path, gpu_backend="worker", **overrides, **config_overrides)
    owner, _ = clients(w.svc)
    storage = owner.get("/compute").json()["storage"]
    assert storage["backend"] == backend and storage["name"] == name

    def expect(row):
        return ({"available": False, "note": row[0], "reason": row[1]} if row
                else {"available": True, "note": None, "reason": None})

    assert storage["modes"] == {"modal": expect(modal_row), "worker": expect(None),
                                "runpod": expect(runpod_row), "serverless": expect(storage_row or config_row)}
    for mode in ("modal", "runpod", "serverless"):
        assert storage["modes"][mode]["available"] == (mode_refusal(w.settings, mode) is None)


def test_create_app_logs_the_storage(world, caplog):
    with caplog.at_level("INFO", logger="radar_desk"):
        create_app(world.svc.settings, world.svc, start_poller=False)
    assert "storage: S3 bucket b (s3)" in caplog.messages
