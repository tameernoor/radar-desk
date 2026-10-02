"""The JSON API end to end: TestClient, local storage in a temp dir, the fake GPU backend, poller by hand."""

from __future__ import annotations

import csv
import io
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from radar_desk.app import WEB_DIST, create_app
from radar_desk.gpu.fake import FakeGpuBackend, synthesize_result
from radar_desk.gpu.poller import Poller
from radar_desk.radar import catalog
from radar_desk.services import ServiceError, Services
from radar_desk.services.scans import source_key
from synth import make_nifti

TOKEN = "test-owner"
BLOBS = [((20, 20, 8), (6, 5, 3), 60.0), ((40, 30, 12), (8, 6, 4), 120.0), ((30, 45, 16), (5, 5, 3), 300.0)]


@dataclass
class Api:
    svc: Services
    app: object
    client: TestClient
    anon: TestClient
    poller: Poller
    backend: FakeGpuBackend
    tmp: Path

    def nifti(self, name: str = "scan.nii.gz", **kw) -> Path:
        kw.setdefault("shape", (64, 64, 24))
        kw.setdefault("blobs", BLOBS)
        return make_nifti(self.tmp / name, **kw)

    def put_file(self, path: Path) -> str:
        """Ask for an upload slot and PUT the file to the signed URL, without a login. Returns the scan id."""
        r = self.client.post("/uploads", json={"filename": path.name, "size_bytes": path.stat().st_size,
                                               "research_only_confirmed": True})
        assert r.status_code == 200, r.text
        ticket = r.json()
        put = self.anon.put(ticket["put_url"], content=path.read_bytes())
        assert put.status_code == 204, put.text
        return ticket["scan_id"]

    def upload(self, path: Path | None = None) -> dict:
        scan_id = self.put_file(path or self.nifti())
        r = self.client.post(f"/uploads/{scan_id}/complete")
        assert r.status_code == 200, r.text
        return r.json()

    def run_job(self, scan_id: str) -> dict:
        job = self.client.post(f"/scans/{scan_id}/jobs").json()
        for _ in range(3):
            self.poller.tick()
        job = self.client.get(f"/jobs/{job['id']}").json()
        assert job["state"] == "done", job
        return job


@pytest.fixture
def make_api(make_services, tmp_path):
    def build(**overrides) -> Api:
        overrides.setdefault("public_base_url", "http://testserver")
        holder: dict[str, Services] = {}

        def synthesize(job_id: str) -> dict:
            svc = holder["svc"]
            job = svc.db.get_job(job_id)
            return synthesize_result(job, svc.db.get_scan(job.scan_id), svc.storage, catalog)

        backend = FakeGpuBackend(synthesize=synthesize)
        svc = make_services(backend=backend, **overrides)
        holder["svc"] = svc
        app = create_app(svc.settings, svc, start_poller=False)
        app.state.sse_interval_s = 0.01
        base = f"http://{overrides['app_hostname']}" if overrides.get("app_hostname") else "http://testserver"
        client = TestClient(app, base_url=base, headers={"Authorization": f"Bearer {TOKEN}"})
        anon = TestClient(app, base_url=base)
        return Api(svc, app, client, anon, Poller(svc, backend), backend, tmp_path)

    return build


@pytest.fixture
def api(make_api) -> Api:
    return make_api()


def sse_events(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        events.append((lines["event"], json.loads(lines["data"])))
    return events


# Auth


def test_routes_need_a_token(api):
    for path in ("/scans", "/jobs", "/gpu/status", "/catalog/findings", "/fixtures", "/auth/me"):
        r = api.anon.get(path)
        assert r.status_code == 401, path
        assert r.json() == {"detail": "not logged in"}
    assert api.anon.post("/uploads", json={}).status_code == 401
    assert api.anon.get("/scans", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert api.anon.get("/scans", headers={"Authorization": f"Basic {TOKEN}"}).status_code == 401


def test_bearer_and_me(api):
    assert api.client.get("/scans").status_code == 200
    assert api.client.get("/auth/me").json() == {"owner": True, "via": "bearer"}


def test_wrong_token_is_401_then_rate_limited(api):
    for _ in range(5):
        r = api.anon.post("/auth/login", json={"token": "nope"})
        assert r.status_code == 401
        assert r.json()["detail"] == "wrong token"
    r = api.anon.post("/auth/login", json={"token": TOKEN})
    assert r.status_code == 429
    assert "too many" in r.json()["detail"]


def test_limiter_window_slides():
    from radar_desk.auth import LoginLimiter

    now = [0.0]
    limiter = LoginLimiter(clock=lambda: now[0])
    for _ in range(5):
        limiter.fail("1.2.3.4")
    assert limiter.blocked("1.2.3.4") and not limiter.blocked("5.6.7.8")
    now[0] = 601.0
    assert not limiter.blocked("1.2.3.4")
    assert limiter._failures == {}
    limiter.fail("9.9.9.9")
    now[0] = 1300.0
    limiter.fail("8.8.8.8")
    assert set(limiter._failures) == {"8.8.8.8"}


def test_cookie_login_and_logout(api):
    r = api.anon.post("/auth/login", json={"token": TOKEN})
    assert r.status_code == 204
    cookie = r.headers["set-cookie"]
    assert cookie.startswith("radar_session=")
    assert "HttpOnly" in cookie and "SameSite=lax" in cookie and "Max-Age=2592000" in cookie
    assert "Secure" not in cookie
    assert api.anon.get("/auth/me").json() == {"owner": True, "via": "cookie"}
    assert api.anon.get("/scans").status_code == 200
    assert api.anon.post("/auth/logout").status_code == 204
    assert api.anon.get("/auth/me").status_code == 401


def test_tampered_cookie_and_secure_flag(make_api):
    api = make_api(public_base_url="https://radar.example")
    r = api.anon.post("/auth/login", json={"token": TOKEN})
    assert "Secure" in r.headers["set-cookie"]
    value = api.app.state.auth.session_cookie()
    fresh = TestClient(api.app)
    assert fresh.get("/auth/me", cookies={"radar_session": value}).status_code == 200
    assert fresh.get("/auth/me", cookies={"radar_session": value[:-2] + "xx"}).status_code == 401


def test_login_body_is_validated(api):
    r = api.anon.post("/auth/login", json={})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert isinstance(detail, str) and "token" in detail and "required" in detail.lower()


# Uploads


def test_upload_flow_ready(api):
    path = api.nifti()
    scan = api.upload(path)
    assert scan["state"] == "ready"
    assert scan["filename"] == "scan.nii.gz"
    assert scan["header"]["dims"] == [64, 64, 24]
    assert scan["header"]["spacing_mm"] == [1.0, 1.0, 5.0]
    assert scan["header"]["orientation"] == "LAS"
    assert scan["sha256"] and scan["research_only_confirmed_at"]
    assert scan["latest_job"] is None
    assert api.svc.storage.exists(source_key(scan["id"]))


def test_upload_ticket_shape(api):
    r = api.client.post("/uploads", json={"filename": "a.nii", "size_bytes": 10, "research_only_confirmed": True})
    body = r.json()
    assert set(body) == {"scan_id", "put_url", "expires_at"}
    assert body["put_url"].startswith(f"http://testserver/_storage/scans/{body['scan_id']}/source.nii.gz?exp=")


def test_upload_rejected_file(api):
    scan_id = api.put_file(api.nifti("bad.nii.gz", dtype=np.uint8, blobs=[]))
    r = api.client.post(f"/uploads/{scan_id}/complete")
    assert r.status_code == 422
    assert "uint8" in r.json()["detail"]
    assert not api.svc.storage.exists(source_key(scan_id))
    scan = api.client.get(f"/scans/{scan_id}").json()
    assert scan["state"] == "rejected" and "uint8" in scan["rejected_reason"]


def test_complete_before_put_is_409(api):
    ticket = api.client.post("/uploads", json={"filename": "a.nii.gz", "size_bytes": 100,
                                               "research_only_confirmed": True}).json()
    r = api.client.post(f"/uploads/{ticket['scan_id']}/complete")
    assert r.status_code == 409
    assert r.json() == {"detail": "the file has not been uploaded yet"}


def test_upload_415_413_422(make_api):
    api = make_api(max_upload_bytes=1000)
    body = {"filename": "scan.dcm", "size_bytes": 10, "research_only_confirmed": True}
    r = api.client.post("/uploads", json=body)
    assert r.status_code == 415 and "nii" in r.json()["detail"]
    r = api.client.post("/uploads", json={**body, "filename": "big.nii.gz", "size_bytes": 5000})
    assert r.status_code == 413 and "limit" in r.json()["detail"]
    r = api.client.post("/uploads", json={**body, "filename": "a.nii.gz", "research_only_confirmed": False})
    assert r.status_code == 422 and "research" in r.json()["detail"]
    r = api.client.post("/uploads", json={"filename": "a.nii.gz"})
    assert r.status_code == 422


def test_complete_unknown_scan_is_404(api):
    r = api.client.post("/uploads/scan_nope/complete")
    assert r.status_code == 404 and r.json() == {"detail": "no scan scan_nope"}


# Local storage routes


def test_storage_signature_checks(make_api):
    api = make_api(max_upload_bytes=1000)
    put = api.svc.storage.put_url("scans/x/source.nii.gz", 60)
    assert api.anon.put(put.replace("sig=", "sig=0"), content=b"abc").status_code == 403
    assert api.anon.put(put.split("?")[0], content=b"abc").status_code == 403
    assert api.anon.put(put, content=b"x" * 2000).status_code == 413
    assert not api.svc.storage.exists("scans/x/source.nii.gz")
    assert api.anon.put(put, content=b"abc").status_code == 204
    get = api.svc.storage.get_url("scans/x/source.nii.gz", 60)
    assert api.anon.get(put).status_code == 403  # a PUT signature cannot read
    r = api.anon.get(get)
    assert r.status_code == 200 and r.content == b"abc"
    assert r.headers["content-type"] == "application/gzip" and r.headers["content-length"] == "3"
    r = api.anon.get(api.svc.storage.get_url("jobs/j/trace.json", 60))
    assert r.status_code == 404
    r = api.anon.put(put, content=b"other")
    assert r.status_code == 409 and "never overwritten" in r.json()["detail"]
    assert api.anon.get(get).content == b"abc"


async def test_storage_put_disconnect_is_400(api):
    url = api.svc.storage.put_url("scans/y/source.nii.gz", 60)
    path, query = url.removeprefix("http://testserver").split("?")
    messages = iter([{"type": "http.request", "body": b"ab", "more_body": True}, {"type": "http.disconnect"}])
    sent = []

    async def receive():
        return next(messages)

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "PUT",
             "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": query.encode(),
             "root_path": "", "headers": [(b"host", b"testserver")], "client": ("1.2.3.4", 1),
             "server": ("testserver", 80)}
    await api.app(scope, receive, send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 400
    assert not api.svc.storage.exists("scans/y/source.nii.gz")


# Scans


def test_scans_list_get_delete_and_source(api):
    path = api.nifti()
    scan = api.upload(path)
    listed = api.client.get("/scans").json()["scans"]
    assert [s["id"] for s in listed] == [scan["id"]] and "latest_job" in listed[0]
    assert api.client.get("/scans", params={"state": "rejected"}).json()["scans"] == []

    r = api.client.get(f"/scans/{scan['id']}/source.nii.gz", follow_redirects=False)
    assert r.status_code == 307
    location = r.headers["location"]
    assert location.startswith(f"http://testserver/_storage/scans/{scan['id']}/source.nii.gz?exp=")
    ct = api.anon.get(location)
    assert ct.status_code == 200 and ct.content == path.read_bytes()

    job = api.client.post(f"/scans/{scan['id']}/jobs").json()
    assert api.client.get(f"/scans/{scan['id']}").json()["latest_job"] == {
        "id": job["id"], "state": "queued", "hold_reason": None, "positives_at_50": None}
    r = api.client.delete(f"/scans/{scan['id']}")
    assert r.status_code == 409 and "cancel it first" in r.json()["detail"]
    api.client.post(f"/jobs/{job['id']}/cancel")
    assert api.client.delete(f"/scans/{scan['id']}").status_code == 204
    assert api.client.get(f"/scans/{scan['id']}").status_code == 404
    assert api.client.delete(f"/scans/{scan['id']}").status_code == 404


def test_source_of_rejected_scan_is_409(api):
    scan_id = api.put_file(api.nifti("bad.nii.gz", dtype=np.uint8, blobs=[]))
    api.client.post(f"/uploads/{scan_id}/complete")
    assert api.client.get(f"/scans/{scan_id}/source.nii.gz", follow_redirects=False).status_code == 409


def test_job_create_is_idempotent(api):
    scan = api.upload()
    first = api.client.post(f"/scans/{scan['id']}/jobs").json()
    second = api.client.post(f"/scans/{scan['id']}/jobs").json()
    assert first["id"] == second["id"] and first["state"] == "queued"
    assert api.client.post("/scans/scan_nope/jobs").status_code == 404


# Jobs


def test_jobs_list_get_and_logs(api):
    scan = api.upload()
    job = api.client.post(f"/scans/{scan['id']}/jobs").json()
    assert api.client.get(f"/jobs/{job['id']}/logs").json() == {"text": ""}
    api.poller.tick()
    got = api.client.get(f"/jobs/{job['id']}").json()
    assert got["state"] == "submitted" and got["modal_call_id"]
    assert "fake backend" in api.client.get(f"/jobs/{job['id']}/logs").json()["text"]
    jobs = api.client.get("/jobs").json()["jobs"]
    assert [j["id"] for j in jobs] == [job["id"]]
    assert api.client.get("/jobs", params={"state": "done"}).json()["jobs"] == []
    assert api.client.get("/jobs", params={"scan_id": scan["id"]}).json()["jobs"][0]["id"] == job["id"]
    assert api.client.get("/jobs/job_nope").status_code == 404


def test_job_events_stream_until_done(api):
    scan = api.upload()
    job = api.client.post(f"/scans/{scan['id']}/jobs").json()

    def drive():
        for _ in range(3):
            time.sleep(0.15)
            api.poller.tick()

    driver = threading.Thread(target=drive)
    driver.start()
    r = api.client.get(f"/jobs/{job['id']}/events")
    driver.join()
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = sse_events(r.text)
    assert all(name == "state" for name, _ in events)
    states = [data["state"] for _, data in events]
    assert states[0] == "queued" and states[-1] == "done"
    assert "submitted" in states
    assert states.index("submitted") > states.index("queued")
    assert events[-1][1]["id"] == job["id"]


def test_job_events_of_terminal_job_is_one_event(api):
    scan = api.upload()
    job = api.run_job(scan["id"])
    events = sse_events(api.client.get(f"/jobs/{job['id']}/events").text)
    assert len(events) == 1 and events[0][1]["state"] == "done"
    assert api.client.get("/jobs/job_nope/events").status_code == 404


def test_cancel_and_retry(api):
    scan = api.upload()
    job = api.client.post(f"/scans/{scan['id']}/jobs").json()
    r = api.client.post(f"/jobs/{job['id']}/cancel")
    assert r.status_code == 200 and r.json()["state"] == "cancelled"
    r = api.client.post(f"/jobs/{job['id']}/cancel")
    assert r.status_code == 409 and "cannot be cancelled" in r.json()["detail"]
    assert api.client.post(f"/jobs/{job['id']}/retry").status_code == 409

    failing = api.client.post(f"/scans/{scan['id']}/jobs").json()
    assert failing["id"] != job["id"]
    api.backend.set_result(failing["id"], {"ok": False, "job_id": failing["id"],
                                           "error": {"class": "input_error", "message": "too small"}})
    api.poller.tick()
    api.poller.tick()
    failed = api.client.get(f"/jobs/{failing['id']}").json()
    assert failed["state"] == "failed" and failed["error"] == {"class": "input_error", "message": "too small"}
    r = api.client.post(f"/jobs/{failing['id']}/retry")
    assert r.status_code == 200 and r.json()["state"] == "queued"


# Results


def test_result_organ_and_compare(api):
    scan = api.upload()
    job = api.run_job(scan["id"])
    result = api.client.get(f"/jobs/{job['id']}/result").json()
    assert len(result["findings"]) == len(catalog.FINDINGS) == 146
    assert result["versions"]["gpu"] == "fake"
    assert {"key", "organ", "finding", "prob"} <= set(result["findings"][0])

    organ = api.client.get(f"/jobs/{job['id']}/organs/liver").json()
    assert organ["organ"] == "Liver" and organ["label"] == catalog.label_for_organ("Liver")
    assert all(f["organ"] == "Liver" for f in organ["findings"]) and organ["findings"]
    r = api.client.get(f"/jobs/{job['id']}/organs/Spleenish")
    assert r.status_code == 404 and "not a scored organ" in r.json()["detail"]

    r = api.client.get(f"/jobs/{job['id']}/compare", params={"against": "fixture"})
    assert r.status_code == 409 and r.json() == {"detail": "this scan is not one of the known fixtures"}
    r = api.client.get(f"/jobs/{job['id']}/compare", params={"against": job["id"]})
    assert r.status_code == 422
    assert api.client.get(f"/jobs/{job['id']}/compare").status_code == 422

    other = api.run_job(scan["id"])
    cmp = api.client.get(f"/jobs/{job['id']}/compare", params={"against": other["id"]}).json()
    assert cmp["against"] == other["id"] and cmp["source"] == "job"
    assert len(cmp["deltas"]) == 146 and cmp["max_abs_delta"] == 0.0 and cmp["over_tolerance"] == 0


def test_result_before_done_is_404(api):
    scan = api.upload()
    job = api.client.post(f"/scans/{scan['id']}/jobs").json()
    r = api.client.get(f"/jobs/{job['id']}/result")
    assert r.status_code == 404 and "no result yet" in r.json()["detail"]
    assert api.client.get(f"/jobs/{job['id']}/scores.csv").status_code == 404


# Exports


def test_exports(api):
    scan = api.upload()
    job = api.run_job(scan["id"])

    r = api.client.get(f"/jobs/{job['id']}/scores.csv")
    assert r.status_code == 200
    assert r.headers["content-type"] == "text/csv; charset=utf-8"
    assert r.headers["content-disposition"] == f'attachment; filename="radar-{job["id"]}.csv"'
    assert r.content.startswith(b"\xef\xbb\xbf")
    rows = list(csv.reader(io.StringIO(r.content.decode("utf-8-sig"))))
    assert rows[0] == catalog.csv_header()
    assert len(rows) == 2 and rows[1][0] == "scan.nii.gz" and len(rows[1]) == 147

    body = api.client.get(f"/jobs/{job['id']}/scores.json").json()
    for field in ("findings", "organs_not_found", "organs_scored", "organ_stats", "versions", "timings",
                  "disclaimer", "model"):
        assert field in body, field
    assert len(body["findings"]) == 146

    for name, ctype in (("mask.nii.gz", "application/gzip"), ("trace.json", "application/octet-stream")):
        r = api.client.get(f"/jobs/{job['id']}/{name}", follow_redirects=False)
        assert r.status_code == 307, name
        assert f"/_storage/jobs/{job['id']}/" in r.headers["location"]
        fetched = api.anon.get(r.headers["location"])
        assert fetched.status_code == 200 and fetched.headers["content-type"] == ctype
    trace = json.loads(api.anon.get(api.client.get(f"/jobs/{job['id']}/trace.json",
                                                   follow_redirects=False).headers["location"]).content)
    assert trace["job_id"] == job["id"]

    api.run_job(api.upload(api.nifti("second.nii.gz"))["id"])
    r = api.client.get("/export/scores.csv")
    assert r.status_code == 200 and r.headers["content-type"] == "text/csv; charset=utf-8"
    assert "attachment" in r.headers["content-disposition"]
    rows = list(csv.reader(io.StringIO(r.content.decode("utf-8-sig"))))
    assert rows[0] == catalog.csv_header()
    assert sorted(row[0] for row in rows[1:]) == ["scan.nii.gz", "second.nii.gz"]


# GPU, catalog, fixtures, health


def test_gpu_status_with_queued_and_in_flight(api):
    status = api.client.get("/gpu/status").json()
    assert status["backend"] == "fake" and status["in_flight"] is None and status["queued"] == []
    assert status["price_per_hour_usd"] is None
    modal = api.svc.costs.gpu_status("modal", time.time())
    assert modal["price_per_hour_usd"] == round(api.svc.settings.gpu_prices_usd_per_s["L4"] * 3600, 4)
    for key in ("gpu_requested", "held", "spend_month_usd", "budget_usd", "month"):
        assert key in status
    scan = api.upload()
    job = api.client.post(f"/scans/{scan['id']}/jobs").json()
    status = api.client.get("/gpu/status").json()
    assert [j["id"] for j in status["queued"]] == [job["id"]]
    api.poller.tick()
    status = api.client.get("/gpu/status").json()
    assert status["in_flight"]["id"] == job["id"] and status["queued"] == []


def test_gpu_status_held_by_budget(make_api):
    api = make_api(gpu_monthly_budget_usd=0.0)
    scan = api.upload()
    job = api.client.post(f"/scans/{scan['id']}/jobs").json()
    api.poller.tick()
    status = api.client.get("/gpu/status").json()
    assert [j["id"] for j in status["held"]] == [job["id"]]
    assert status["held"][0]["hold_reason"] == "budget"
    assert [j["id"] for j in status["queued"]] == [job["id"]]


def test_catalog_and_fixtures(api):
    findings = api.client.get("/catalog/findings").json()["findings"]
    assert len(findings) == 146 and findings[0]["key"] == catalog.FINDINGS[0]["key"]
    labels = api.client.get("/catalog/labels").json()
    assert len([lab for lab in labels["labels"] if lab["label"] > 0]) == 36
    assert len(labels["scored_organs"]) == 18
    assert {"organ", "label", "finding_count"} <= set(labels["scored_organs"][0])
    fixtures = api.client.get("/fixtures").json()["fixtures"]
    assert len(fixtures) == 6 and all("references" in f for f in fixtures)


def test_health_is_open(api):
    r = api.anon.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["backend"] == "fake" and body["version"]


def test_docs_served(api):
    assert api.anon.get("/docs").status_code == 200
    paths = api.anon.get("/openapi.json").json()["paths"]
    for path in ("/scans", "/jobs/{job_id}/events", "/gpu/status", "/uploads"):
        assert path in paths


@pytest.mark.skipif(not (WEB_DIST / "index.html").is_file(), reason="web/dist is not built")
def test_static_pages_and_api_wins(api):
    for page in ("/", "/workspace.html", "/jobs.html"):
        r = api.anon.get(page)
        assert r.status_code == 200 and "text/html" in r.headers["content-type"], page
    assert api.anon.get("/health").headers["content-type"] == "application/json"
    assert api.anon.get("/scans").status_code == 401


def test_no_static_mount_without_dist(make_services, tmp_path):
    svc = make_services()
    app = create_app(svc.settings, svc, start_poller=False, web_dist=tmp_path / "missing")
    assert TestClient(app).get("/").status_code == 404


def test_trusted_host(make_api):
    api = make_api(app_hostname="radar.example")
    assert api.anon.get("/health").status_code == 200
    assert api.anon.get("/health", headers={"host": "localhost:8000"}).status_code == 200
    r = api.anon.get("/health", headers={"host": "evil.example"})
    assert r.status_code == 400


def test_modal_without_bucket_fails_fast(make_services):
    from radar_desk.config import ConfigError

    svc = make_services()
    settings = svc.settings.model_copy(update={"gpu_backend": "modal"})
    with pytest.raises(ConfigError, match="GPU_BACKEND=modal needs S3_BUCKET"):
        create_app(settings, svc, start_poller=False)


def test_service_error_mapping(make_services, tmp_path):
    svc = make_services()
    app = create_app(svc.settings, svc, start_poller=False, web_dist=tmp_path / "missing")

    @app.get("/boom")
    def boom():
        raise ServiceError(418, "a plain message")

    r = TestClient(app).get("/boom")
    assert r.status_code == 418 and r.json() == {"detail": "a plain message"}


def test_lifespan_runs_the_poller(make_services):
    svc = make_services(gpu_poll_interval_s=0.01)
    app = create_app(svc.settings, svc, start_poller=True)
    path = make_nifti(Path(svc.settings.data_dir) / "lp.nii.gz", shape=(24, 20, 10))
    ticket = svc.scans.begin_upload(path.name, path.stat().st_size, True)
    svc.storage.put_bytes(source_key(ticket.scan_id), path.read_bytes())
    svc.scans.complete_upload(ticket.scan_id)
    job = svc.jobs.create(ticket.scan_id)
    with TestClient(app):
        deadline = time.time() + 5
        while svc.jobs.get(job.id).state != "done" and time.time() < deadline:
            time.sleep(0.02)
    assert svc.jobs.get(job.id).state == "done"
