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
    SERVERLESS_FALLBACK,
    SERVERLESS_NEEDS_CONFIG,
    SERVERLESS_NEEDS_STORAGE,
)
from test_compute_service import OWNER, World
from test_serverless_backend import ENDPOINT

FIELDS = {"mode", "changeable", "changed_at", "tunnel_mode", "public_url", "runpod", "serverless", "pod",
          "in_flight", "queued", "held", "problem", "last_event", "spend_month_usd", "budget_usd", "month"}
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
    assert body["mode"] == "worker" and body["changeable"] is True and body["tunnel_mode"] == "managed"
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


def test_the_fake_backend_is_fixed(make_services):
    owner, _ = clients(make_services())
    body = owner.get("/compute").json()
    assert body["mode"] == "fake" and body["changeable"] is False and body["runpod"]["configured"] is False
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
    assert world.compute.mode == "worker"


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
