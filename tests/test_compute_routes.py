"""GET and PUT /compute, POST /compute/pod/start and /compute/pod/stop."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from radar_desk.app import create_app
from radar_desk.services.compute import FIXED, MODAL_NEEDS_STORAGE, MODAL_ON_LOCAL
from test_compute_service import OWNER, World

FIELDS = {"mode", "changeable", "changed_at", "tunnel_mode", "public_url", "runpod", "pod", "in_flight",
          "queued", "held", "problem", "last_event", "spend_month_usd", "budget_usd", "month"}
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


def test_the_fake_backend_is_fixed(make_services):
    owner, _ = clients(make_services())
    body = owner.get("/compute").json()
    assert body["mode"] == "fake" and body["changeable"] is False and body["runpod"]["configured"] is False
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
