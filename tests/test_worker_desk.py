"""The job desk for pull workers: tokens, auth, claim, leases, artefacts, completion, the worker backend."""

from __future__ import annotations

import gzip
import json
import threading
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from radar_desk.app import create_app
from radar_desk.config import Settings
from radar_desk.gpu.fake import canned_result
from radar_desk.gpu.poller import Poller
from radar_desk.records import Lease, Scan
from radar_desk.services import Services, build_services
from radar_desk.services.costs import parse_iso
from radar_desk.services.scans import source_key
from synth import make_nifti

OWNER = "test-owner"
INFO = {"id": "pod-1", "hostname": "pod", "gpu_name": "NVIDIA L4", "device": "cuda",
        "versions": {"torch": "2.5.1", "cuda": "12.4", "python": "3.10", "image": "radar-worker"}}


class Clock:
    def __init__(self, iso: str = "2026-10-15T12:00:00.000000Z") -> None:
        self.t = parse_iso(iso)

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@dataclass
class Desk:
    svc: Services
    app: object
    owner: TestClient
    anon: TestClient
    worker: TestClient
    token: dict
    clock: Clock
    tmp: Path

    def scan(self, name: str = "a.nii.gz") -> Scan:
        path = make_nifti(self.tmp / name, shape=(24, 20, 10))
        ticket = self.svc.scans.begin_upload(path.name, path.stat().st_size, True)
        self.svc.storage.put_bytes(source_key(ticket.scan_id), path.read_bytes())
        scan = self.svc.scans.complete_upload(ticket.scan_id)
        assert scan.state == "ready"
        return scan

    def queue(self, name: str = "a.nii.gz") -> str:
        return self.svc.jobs.create(self.scan(name).id).id

    def claim(self, worker: dict | None = None) -> dict:
        r = self.worker.post("/worker/claim", json={"worker": worker or INFO})
        assert r.status_code == 200, r.text
        return r.json()

    def job(self, job_id: str) -> dict:
        return self.owner.get(f"/jobs/{job_id}").json()


@pytest.fixture
def make_desk(tmp_path):
    def build(**overrides) -> Desk:
        clock = Clock()
        settings = Settings(_env_file=None, owner_token=OWNER, session_secret="s", data_dir=tmp_path / "data",
                            public_base_url="http://testserver", gpu_backend="worker", **overrides)
        svc = build_services(settings, clock=clock)
        app = create_app(settings, svc, start_poller=False)
        owner = TestClient(app, headers={"Authorization": f"Bearer {OWNER}"})
        r = owner.post("/workers/tokens", json={"name": "runpod"})
        assert r.status_code == 201, r.text
        token = r.json()
        worker = TestClient(app, headers={"Authorization": f"Bearer {token['token']}"})
        return Desk(svc, app, owner, TestClient(app), worker, token, clock, tmp_path)

    return build


@pytest.fixture
def desk(make_desk) -> Desk:
    return make_desk()


def complete_body(claim: dict, **changes) -> dict:
    return {**canned_result(claim["job_id"], gpu="NVIDIA L4"), "lease": claim["lease"], **changes}


# Tokens and auth


def test_token_create_list_revoke(desk):
    token = desk.token
    assert token["token"].startswith("rdw_") and len(token["token"]) == 4 + 32
    assert token["id"].startswith("wtok_") and token["name"] == "runpod"
    listed = desk.owner.get("/workers/tokens").json()["tokens"]
    assert listed == [{"id": token["id"], "name": "runpod", "created_at": token["created_at"],
                       "revoked_at": None, "last_used_at": None}]
    assert desk.worker.get("/worker/me").json() == {"token": {"id": token["id"], "name": "runpod"}}
    assert desk.owner.get("/workers/tokens").json()["tokens"][0]["last_used_at"] is not None

    assert desk.owner.delete(f"/workers/tokens/{token['id']}").status_code == 204
    assert desk.owner.delete(f"/workers/tokens/{token['id']}").status_code == 204
    assert desk.owner.delete("/workers/tokens/wtok_nope").status_code == 404
    assert desk.owner.get("/workers/tokens").json()["tokens"][0]["revoked_at"] is not None
    assert desk.worker.get("/worker/me").status_code == 401
    assert desk.owner.post("/workers/tokens", json={"name": "  "}).status_code == 422


def test_last_used_is_stamped_at_most_once_a_minute(desk):
    desk.worker.get("/worker/me")
    first = desk.svc.db.get_worker_token(desk.token["id"]).last_used_at
    desk.clock.advance(30)
    desk.worker.get("/worker/me")
    assert desk.svc.db.get_worker_token(desk.token["id"]).last_used_at == first
    desk.clock.advance(31)
    desk.worker.get("/worker/me")
    assert desk.svc.db.get_worker_token(desk.token["id"]).last_used_at != first


WORKER_ROUTES = [
    ("get", "/worker/me", None),
    ("post", "/worker/claim", {"worker": INFO}),
    ("post", "/worker/jobs/job_x/heartbeat", {"lease": "wk_x"}),
    ("get", "/worker/jobs/job_x/source?lease=wk_x", None),
    ("put", "/worker/jobs/job_x/artefacts/scores.json?lease=wk_x", None),
    ("post", "/worker/jobs/job_x/complete", {"lease": "wk_x"}),
    ("post", "/worker/jobs/job_x/fail", {"lease": "wk_x", "error": {"class": "x", "message": ""}}),
    ("post", "/worker/jobs/job_x/release", {"lease": "wk_x"}),
]


def test_worker_routes_need_a_live_worker_token(desk):
    revoked = desk.owner.post("/workers/tokens", json={"name": "old"}).json()
    desk.owner.delete(f"/workers/tokens/{revoked['id']}")
    for header in (None, f"Bearer {revoked['token']}", f"Bearer {OWNER}", "Bearer rdw_" + "0" * 32):
        headers = {"Authorization": header} if header else {}
        for method, path, body in WORKER_ROUTES:
            kw = {"json": body} if body is not None else {}
            r = desk.anon.request(method.upper(), path, headers=headers, **kw)
            assert r.status_code == 401, (header, path)
            assert r.json() == {"detail": "not a valid worker token"}
    # The owner's session cookie does not pass either.
    login = desk.anon.post("/auth/login", json={"token": OWNER})
    assert login.status_code == 204 and desk.anon.get("/auth/me").status_code == 200
    assert desk.anon.get("/worker/me").status_code == 401


def test_owner_routes_refuse_a_worker_token(desk):
    for path in ("/workers", "/workers/tokens", "/jobs", "/gpu/status", "/scans"):
        r = desk.worker.get(path)
        assert r.status_code == 401, path


# Claim


def test_claim_returns_the_contract_and_204_when_empty(desk):
    assert desk.worker.post("/worker/claim", json={"worker": INFO}).status_code == 204
    job_id = desk.queue()
    claim = desk.claim()
    job = desk.svc.db.get_job(job_id)
    scan = desk.svc.db.get_scan(job.scan_id)
    lease = claim["lease"]
    assert lease.startswith("wk_")
    assert claim == {
        "job_id": job_id,
        "lease": lease,
        "scan": {"id": scan.id, "filename": "a.nii.gz", "size_bytes": scan.size_bytes, "sha256": scan.sha256},
        "model_version": job.model_version,
        "source_url": f"/worker/jobs/{job_id}/source?lease={lease}",
        "artefact_urls": {
            "scores_json": f"/worker/jobs/{job_id}/artefacts/scores.json?lease={lease}",
            "scores_csv": f"/worker/jobs/{job_id}/artefacts/scores.csv?lease={lease}",
            "mask": f"/worker/jobs/{job_id}/artefacts/mask.nii.gz?lease={lease}",
            "trace": f"/worker/jobs/{job_id}/artefacts/trace.json?lease={lease}",
            "log": f"/worker/jobs/{job_id}/artefacts/worker.log?lease={lease}",
        },
        "artefact_keys": {
            "scores_json": f"jobs/{job_id}/scores.json",
            "scores_csv": f"jobs/{job_id}/scores.csv",
            "mask": f"jobs/{job_id}/mask.nii.gz",
            "trace": f"jobs/{job_id}/trace.json",
            "log": f"jobs/{job_id}/worker.log",
        },
        "lease_s": 120,
        "heartbeat_s": 40,
        "timeout_s": 1800,
    }
    assert job.state == "submitted" and job.modal_call_id == lease
    assert job.submitted_at == "2026-10-15T12:00:00.000000Z"
    assert job.lease.worker_id == "pod-1" and job.lease.expires_at == "2026-10-15T12:02:00.000000Z"
    assert desk.worker.post("/worker/claim", json={"worker": INFO}).status_code == 204


def test_claim_takes_the_oldest_unheld_job_and_fails_a_missing_scan(desk):
    gone = desk.queue("gone.nii.gz")
    held = desk.queue("held.nii.gz")
    ok = desk.queue("ok.nii.gz")
    desk.svc.db.update_scan(desk.svc.db.get_job(gone).scan_id, state="rejected")
    desk.svc.db.update_job(held, hold_reason="budget")
    assert desk.claim()["job_id"] == ok
    failed = desk.svc.db.get_job(gone)
    assert failed.state == "failed" and failed.error.klass == "input_error"
    assert desk.svc.db.get_job(held).state == "queued"


def test_claim_needs_the_worker_backend(make_desk):
    desk = make_desk()
    desk.svc.settings = desk.svc.settings.model_copy(update={"gpu_backend": "fake"})
    desk.svc.workers.settings = desk.svc.settings
    desk.queue()
    assert desk.worker.post("/worker/claim", json={"worker": INFO}).status_code == 204


def test_two_threads_claiming_one_job_get_it_once(desk):
    job_id = desk.queue()
    barrier = threading.Barrier(2)
    got, errors = [], []

    def race(n: int) -> None:
        lease = Lease(worker_id=f"w{n}", expires_at="2026-10-15T12:02:00.000000Z",
                      heartbeat_at="2026-10-15T12:00:00.000000Z")
        barrier.wait()
        try:
            got.append(desk.svc.db.claim_job(f"wk_{n}", lease, "2026-10-15T12:00:00.000000Z"))
        except Exception as exc:  # noqa: BLE001 - the test asserts there are none
            errors.append(exc)

    threads = [threading.Thread(target=race, args=(n,)) for n in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    winners = [j for j in got if j is not None]
    assert len(got) == 2 and len(winners) == 1 and winners[0].id == job_id
    assert desk.svc.db.get_job(job_id).modal_call_id == winners[0].modal_call_id


# Leases


def test_heartbeat_extends_the_lease(desk):
    desk.queue()
    claim = desk.claim()
    desk.clock.advance(100)
    r = desk.worker.post(f"/worker/jobs/{claim['job_id']}/heartbeat",
                         json={"lease": claim["lease"], "progress": "infer"})
    assert r.status_code == 200
    assert r.json() == {"state": "submitted", "expires_at": "2026-10-15T12:03:40.000000Z"}
    lease = desk.svc.db.get_job(claim["job_id"]).lease
    assert lease.progress == "infer" and lease.heartbeat_at == "2026-10-15T12:01:40.000000Z"
    desk.clock.advance(100)
    desk.svc.workers.tick(desk.clock())
    assert desk.svc.db.get_job(claim["job_id"]).state == "submitted"
    stale = desk.worker.post(f"/worker/jobs/{claim['job_id']}/heartbeat", json={"lease": "wk_other"})
    assert stale.status_code == 409 and stale.json() == {"detail": "the lease is not current"}


def test_expiry_requeues_and_the_third_loss_fails(desk):
    job_id = desk.queue()
    for losses in (1, 2):
        claim = desk.claim()
        desk.clock.advance(121)
        desk.svc.workers.tick(desk.clock())
        job = desk.svc.db.get_job(job_id)
        assert job.state == "queued" and job.lease_losses == losses
        assert job.lease is None and job.modal_call_id is None and job.submitted_at is None
        assert desk.svc.db.get_worker("pod-1").job_id is None
        r = desk.worker.post(f"/worker/jobs/{job_id}/complete", json=complete_body(claim))
        assert r.status_code == 409
    desk.claim()
    desk.clock.advance(121)
    desk.svc.workers.tick(desk.clock())
    job = desk.svc.db.get_job(job_id)
    assert job.state == "failed" and job.error.klass == "lease_expired" and job.lease_losses == 3


def test_timeout_fails_even_with_heartbeats(desk):
    desk.queue()
    claim = desk.claim()
    for _ in range(19):
        desk.clock.advance(100)
        desk.worker.post(f"/worker/jobs/{claim['job_id']}/heartbeat", json={"lease": claim["lease"]})
        desk.svc.workers.tick(desk.clock())
    job = desk.svc.db.get_job(claim["job_id"])
    assert job.state == "failed" and job.error.klass == "timeout"


def test_retry_resets_lease_losses(desk):
    job_id = desk.queue()
    for _ in range(3):
        desk.claim()
        desk.clock.advance(121)
        desk.svc.workers.tick(desk.clock())
    assert desk.job(job_id)["state"] == "failed"
    retried = desk.owner.post(f"/jobs/{job_id}/retry").json()
    assert retried["state"] == "queued" and retried["lease_losses"] == 0 and retried["lease"] is None


def test_cancel_a_claimed_job(desk):
    desk.queue()
    claim = desk.claim()
    r = desk.owner.post(f"/jobs/{claim['job_id']}/cancel")
    assert r.status_code == 200
    assert r.json()["state"] == "cancelled" and r.json()["cost_estimate_usd"] is None
    r = desk.worker.post(f"/worker/jobs/{claim['job_id']}/heartbeat", json={"lease": claim["lease"]})
    assert r.status_code == 409


# Completion


def test_complete_stores_the_result(desk):
    desk.queue()
    claim = desk.claim()
    desk.clock.advance(60)
    r = desk.worker.post(f"/worker/jobs/{claim['job_id']}/complete", json=complete_body(claim))
    assert r.status_code == 200, r.text
    job = r.json()
    assert job["state"] == "done" and job["gpu_used"] == "L4" and job["cost_estimate_usd"] is None
    assert job["timings"]["total_s"] == 55.0
    assert job["finished_at"] == "2026-10-15T12:01:00.000000Z"
    again = desk.worker.post(f"/worker/jobs/{claim['job_id']}/complete", json=complete_body(claim))
    assert again.status_code == 200 and again.json()["state"] == "done"
    assert len(desk.owner.get(f"/jobs/{claim['job_id']}/result").json()["findings"]) == 146
    assert desk.svc.db.get_worker("pod-1").job_id is None


def test_unknown_gpu_name_passes_through(desk):
    desk.queue()
    claim = desk.claim()
    body = complete_body(claim)
    body["versions"]["gpu"] = "Some Future GPU"
    job = desk.worker.post(f"/worker/jobs/{claim['job_id']}/complete", json=body).json()
    assert job["gpu_used"] == "Some Future GPU"


def test_complete_with_a_stale_lease_or_a_bad_result(desk):
    desk.queue()
    claim = desk.claim()
    url = f"/worker/jobs/{claim['job_id']}/complete"
    assert desk.worker.post(url, json=complete_body(claim, lease="wk_stale")).status_code == 409
    body = complete_body(claim)
    body["findings"] = body["findings"][:10]
    r = desk.worker.post(url, json=body)
    assert r.status_code == 422 and "expected 146 findings, got 10" in r.json()["detail"]
    r = desk.worker.post(url, json={"ok": True, "lease": claim["lease"]})
    assert r.status_code == 422 and "expected 146 findings, got 0" in r.json()["detail"]
    assert desk.svc.db.get_job(claim["job_id"]).state == "submitted"
    assert desk.svc.db.get_result(claim["job_id"]) is None


def test_complete_refuses_bad_timings_versions_or_ok_and_keeps_the_job(desk):
    desk.queue()
    claim = desk.claim()
    url = f"/worker/jobs/{claim['job_id']}/complete"
    for changes in ({"timings": {"total_s": "abc"}}, {"timings": [1]}, {"versions": "x"}, {"ok": "yes"},
                    {"organ_stats": "x"}):
        r = desk.worker.post(url, json=complete_body(claim, **changes))
        assert r.status_code == 422, (changes, r.text)
        assert desk.svc.db.get_job(claim["job_id"]).state == "submitted"
        assert desk.svc.db.get_result(claim["job_id"]) is None
    assert desk.worker.post(url, json=complete_body(claim)).json()["state"] == "done"


def test_complete_stores_the_desks_artefact_keys(desk):
    desk.queue()
    claim = desk.claim()
    artefacts = {**claim["artefact_keys"], "mask": "jobs/job_other/mask.nii.gz"}
    r = desk.worker.post(f"/worker/jobs/{claim['job_id']}/complete",
                         json=complete_body(claim, artefacts=artefacts))
    assert r.status_code == 200
    assert desk.svc.db.get_result(claim["job_id"]).artefacts.model_dump() == claim["artefact_keys"]


def test_complete_not_ok_fails_with_the_class(desk):
    desk.queue()
    claim = desk.claim()
    body = complete_body(claim, ok=False, error={"class": "runtime_error", "message": "CUDA OOM"})
    job = desk.worker.post(f"/worker/jobs/{claim['job_id']}/complete", json=body).json()
    assert job["state"] == "failed" and job["error"] == {"class": "runtime_error", "message": "CUDA OOM"}
    assert job["cost_estimate_usd"] is None


def test_fail_route(desk):
    desk.queue()
    claim = desk.claim()
    r = desk.worker.post(f"/worker/jobs/{claim['job_id']}/fail",
                         json={"lease": claim["lease"], "error": {"class": "input_error", "message": "bad"}})
    assert r.status_code == 200
    assert r.json()["state"] == "failed" and r.json()["error"] == {"class": "input_error", "message": "bad"}
    r = desk.worker.post(f"/worker/jobs/{claim['job_id']}/fail",
                         json={"lease": claim["lease"], "error": {"class": "x", "message": ""}})
    assert r.status_code == 409


def test_release_requeues_and_keeps_lease_losses(desk):
    job_id = desk.queue()
    desk.claim()
    desk.clock.advance(121)
    desk.svc.workers.tick(desk.clock())
    claim = desk.claim()
    r = desk.worker.post(f"/worker/jobs/{job_id}/release", json={"lease": claim["lease"]})
    assert r.status_code == 200
    job = r.json()
    assert job["state"] == "queued" and job["lease"] is None and job["lease_losses"] == 1
    assert desk.worker.post(f"/worker/jobs/{job_id}/release", json={"lease": claim["lease"]}).status_code == 409
    assert desk.claim()["job_id"] == job_id


# Bytes


def test_source_streams_the_scan(desk):
    desk.queue()
    claim = desk.claim()
    url = f"/worker/jobs/{claim['job_id']}/source"
    r = desk.worker.get(url, params={"lease": claim["lease"]})
    assert r.status_code == 200
    expected = b"".join(desk.svc.storage.open_stream(source_key(claim["scan"]["id"])))
    assert r.content == expected
    assert r.headers["content-length"] == str(len(expected)) == str(claim["scan"]["size_bytes"])
    assert r.headers["content-type"] == "application/gzip"
    assert desk.worker.get(url, params={"lease": "wk_stale"}).status_code == 409
    assert desk.worker.get(url).status_code == 422


def test_artefact_put_is_write_once(desk):
    desk.queue()
    claim = desk.claim()
    url = claim["artefact_urls"]["trace"]
    assert desk.worker.put(url, content=b'{"a": 1}').status_code == 204
    key = claim["artefact_keys"]["trace"]
    assert b"".join(desk.svc.storage.open_stream(key)) == b'{"a": 1}'
    r = desk.worker.put(url, content=b"again")
    assert r.status_code == 409
    stale = url.replace(claim["lease"], "wk_stale")
    assert desk.worker.put(stale.replace("trace.json", "worker.log"), content=b"x").status_code == 409
    bad = url.replace("trace.json", "other.bin")
    assert desk.worker.put(bad, content=b"x").status_code == 404


def test_a_half_upload_does_not_block_the_next_attempt(desk):
    job_id = desk.queue()
    first = desk.claim()
    assert desk.worker.put(first["artefact_urls"]["mask"], content=b"half").status_code == 204
    desk.clock.advance(121)
    desk.svc.workers.tick(desk.clock())
    second = desk.claim()
    assert second["job_id"] == job_id and second["lease"] != first["lease"]
    assert not desk.svc.storage.exists(second["artefact_keys"]["mask"])
    assert desk.worker.put(second["artefact_urls"]["mask"], content=b"whole").status_code == 204
    r = desk.worker.post(f"/worker/jobs/{job_id}/complete", json=complete_body(second))
    assert r.status_code == 200 and r.json()["state"] == "done"
    assert desk.svc.db.get_result(job_id).artefacts.model_dump() == second["artefact_keys"]
    assert b"".join(desk.svc.storage.open_stream(second["artefact_keys"]["mask"])) == b"whole"


def test_worker_text_is_bounded(desk):
    desk.queue()
    claim = desk.claim()
    r = desk.worker.post(f"/worker/jobs/{claim['job_id']}/heartbeat",
                         json={"lease": claim["lease"], "progress": "x" * 201})
    assert r.status_code == 422


def test_artefact_put_too_large(make_desk):
    desk = make_desk()
    desk.queue()
    claim = desk.claim()
    desk.svc.settings.max_upload_bytes = 10
    assert desk.worker.put(claim["artefact_urls"]["log"], content=b"x" * 11).status_code == 413
    assert not desk.svc.storage.exists(claim["artefact_keys"]["log"])


def test_results_exports_mask_and_logs_after_a_worker_job(desk):
    desk.queue()
    claim = desk.claim()
    job_id = claim["job_id"]
    assert desk.owner.get(f"/jobs/{job_id}/logs").json()["text"].startswith("worker pod-1, last heartbeat")
    mask = gzip.compress(b"not really a nifti")
    body = complete_body(claim)
    assert desk.worker.put(claim["artefact_urls"]["mask"], content=mask).status_code == 204
    scores = {k: body[k] for k in ("job_id", "findings", "versions", "timings")}
    assert desk.worker.put(claim["artefact_urls"]["scores_json"],
                           content=json.dumps(scores).encode()).status_code == 204
    assert desk.worker.put(claim["artefact_urls"]["log"],
                           content=b"line 1\nline 2\nscored in 55 s\n").status_code == 204
    assert desk.worker.post(f"/worker/jobs/{job_id}/complete", json=body).status_code == 200

    assert len(desk.owner.get(f"/jobs/{job_id}/result").json()["findings"]) == 146
    assert desk.owner.get(f"/jobs/{job_id}/scores.csv").status_code == 200
    assert desk.owner.get(f"/jobs/{job_id}/scores.json").json()["job_id"] == job_id
    r = desk.owner.get(f"/jobs/{job_id}/mask.nii.gz", follow_redirects=False)
    assert r.status_code == 307 and f"/_storage/jobs/{job_id}/mask.nii.gz" in r.headers["location"]
    assert desk.anon.get(r.headers["location"]).content == mask
    assert desk.owner.get(f"/jobs/{job_id}/logs").json() == {"text": "line 1\nline 2\nscored in 55 s\n"}
    assert desk.owner.get(f"/jobs/{job_id}/logs", params={"lines": 1}).json() == {"text": "scored in 55 s\n"}


def test_logs_before_a_claim(desk):
    job_id = desk.queue()
    assert desk.owner.get(f"/jobs/{job_id}/logs").json() == {"text": ""}
    assert desk.svc.backend.logs("wk_none") == ""


# Backend, poller and owner views


def test_worker_backend_through_create_app(desk):
    assert desk.svc.backend.name == "worker" and desk.svc.backend.pull and not desk.svc.backend.priced
    job_id = desk.queue()
    poller = Poller(desk.svc, clock=desk.clock)
    poller.tick()
    assert desk.svc.db.get_job(job_id).state == "queued"
    claim = desk.claim()
    desk.clock.advance(121)
    poller.tick()
    job = desk.svc.db.get_job(claim["job_id"])
    assert job.state == "queued" and job.lease_losses == 1
    with pytest.raises(RuntimeError):
        desk.svc.backend.spawn(job, "", {})


def test_create_app_builds_the_worker_backend_from_settings(tmp_path):
    settings = Settings(_env_file=None, owner_token=OWNER, session_secret="s", data_dir=tmp_path / "d",
                        gpu_backend="worker")
    app = create_app(settings, start_poller=False)
    assert app.state.services.backend.name == "worker"
    client = TestClient(app, headers={"Authorization": f"Bearer {OWNER}"})
    assert client.get("/workers").status_code == 200


def test_workers_list_and_gpu_status(desk):
    desk.queue()
    desk.claim()
    body = desk.owner.get("/workers").json()
    assert body["app_url"] == "http://testserver" and body["image"] == "radar-worker" and body["lease_s"] == 120
    [worker] = body["workers"]
    assert worker["id"] == "pod-1" and worker["online"] is True and worker["gpu_name"] == "NVIDIA L4"
    assert worker["job_id"] and worker["token_name"] == "runpod" and worker["versions"]["torch"] == "2.5.1"
    desk.clock.advance(121)
    assert desk.owner.get("/workers").json()["workers"][0]["online"] is False
    status = desk.owner.get("/gpu/status").json()
    assert status["backend"] == "worker" and status["price_per_hour_usd"] is None
