"""The pull worker loop against the real app under uvicorn on a free port, with a fake scorer (no torch)."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import uvicorn
from fastapi.testclient import TestClient

from radar_desk.app import create_app
from radar_desk.config import Settings
from radar_desk.gpu.backend import artefact_keys
from radar_desk.radar import catalog
from radar_desk.services import Services, build_services
from radar_desk.services.costs import parse_iso
from radar_desk.services.scans import source_key
from radar_worker import pull
from synth import make_nifti

ROOT = Path(__file__).resolve().parents[1]
OWNER = "test-owner"
INFO = {"id": "pod-1", "hostname": "pod", "gpu_name": "NVIDIA L4", "device": "cuda",
        "versions": {"torch": None, "cuda": None, "python": "3.10", "image": None}}
ORGANS = catalog.SCORED_ORGANS[:2]
BOX = [[0.0, 0.0, 0.0], [4.0, 4.0, 15.0]]


@dataclass
class Env:
    svc: Services
    owner: TestClient
    url: str
    token: str
    seen: list
    tmp: Path
    faults: list

    def fault(self, path_part: str, status: int, body: bytes = b"<html>ngrok: tunnel not found</html>") -> None:
        """Answer the next request whose path holds `path_part` with `status` and an HTML body, once."""
        self.faults.append((path_part, status, body))

    def queue(self, name: str = "a.nii.gz") -> str:
        path = make_nifti(self.tmp / name, shape=(24, 20, 10))
        ticket = self.svc.scans.begin_upload(path.name, path.stat().st_size, True)
        self.svc.storage.put_bytes(source_key(ticket.scan_id), path.read_bytes())
        scan = self.svc.scans.complete_upload(ticket.scan_id)
        r = self.owner.post(f"/scans/{scan.id}/jobs")
        assert r.status_code in (200, 201), r.text
        return r.json()["id"]

    def client(self) -> pull.DeskClient:
        return pull.DeskClient(self.url, self.token, timeout=10)

    def job(self, job_id: str) -> dict:
        return self.owner.get(f"/jobs/{job_id}").json()

    def paths(self, job_id: str) -> list[str]:
        return [f"{m} {p}" for m, p, _ in self.seen if job_id in p]


@pytest.fixture
def make_env(tmp_path):
    servers = []

    def build(worker_lease_s: int | None = None, **overrides) -> Env:
        settings = Settings(_env_file=None, owner_token=OWNER, session_secret="s", data_dir=tmp_path / "data",
                            public_base_url="http://testserver", gpu_backend="worker", **overrides)
        if worker_lease_s is not None:
            settings.worker_lease_s = worker_lease_s  # below the setting's floor, for one-second heartbeats
        svc = build_services(settings)
        app = create_app(settings, svc, start_poller=False)
        seen: list = []
        faults: list = []

        async def recorder(scope, receive, send):
            if scope["type"] == "http":
                headers = {k.decode().lower(): v.decode() for k, v in scope["headers"]}
                seen.append((scope["method"], scope["path"], headers))
                for fault in faults:
                    if fault[0] in scope["path"]:
                        faults.remove(fault)
                        await send({"type": "http.response.start", "status": fault[1],
                                    "headers": [(b"content-type", b"text/html")]})
                        await send({"type": "http.response.body", "body": fault[2]})
                        return
            await app(scope, receive, send)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        server = uvicorn.Server(uvicorn.Config(recorder, lifespan="off", log_level="warning"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        servers.append((server, thread))
        deadline = time.monotonic() + 5
        while not server.started:
            assert time.monotonic() < deadline, "uvicorn did not start"
            time.sleep(0.01)
        owner = TestClient(app, headers={"Authorization": f"Bearer {OWNER}"})
        token = owner.post("/workers/tokens", json={"name": "pod"}).json()["token"]
        return Env(svc, owner, f"http://127.0.0.1:{sock.getsockname()[1]}", token, seen, tmp_path, faults)

    yield build
    for server, thread in servers:
        server.should_exit = True
        thread.join(timeout=5)


class FakeScorer:
    """Scores like `infer.score_file` from the NIfTI header: two organs with one blob each."""

    device = "cuda"

    def __init__(self, refuse: dict | None = None, on_score=None):
        self.refuse = refuse
        self.on_score = on_score
        self.scored = 0

    def gpu_name(self) -> str:
        return "NVIDIA L4"

    def load(self, log):
        log("fake model loaded")
        return self.refuse

    def score(self, path: str, log) -> dict:
        self.scored += 1
        if self.on_score is not None:
            out = self.on_score(path)
            if out is not None:
                return out
        img = nib.load(path)
        dims = tuple(int(d) for d in img.shape[:3])
        mask = np.zeros(dims, dtype=np.uint8)
        mask[2:6, 3:7, 1:4] = ORGANS[0]["label"]
        mask[10:13, 10:12, 5:7] = ORGANS[1]["label"]
        names = {o["organ"] for o in ORGANS}
        findings = [{"key": f["key"], "organ": f["organ"], "finding": f["finding"],
                     "prob": 0.25 if f["organ"] in names else None} for f in catalog.FINDINGS]
        log(f"scored {Path(path).name}")
        return {
            "ok": True,
            "findings": findings,
            "organs_scored": [{"organ": o["organ"], "label": o["label"], "how": "window", "window_index": 0,
                               "box_mm": BOX} for o in ORGANS],
            "organs_not_found": [o["organ"] for o in catalog.SCORED_ORGANS if o["organ"] not in names],
            "organ_stats": {o["organ"]: {"voxels": 10, "ml": 0.05, "centroid_mm": [1.0, 2.0, 3.0], "bbox_mm": BOX}
                            for o in ORGANS},
            "trace": {"fake": True},
            "timings": {"infer_s": 0.01, "postprocess_s": 0.01},
            "file_name": Path(path).name,
            "mask": mask,
            "affine": np.asarray(img.affine, dtype=float),
        }

    def versions(self) -> dict:
        return {"code_commit": None, "checkpoint_sha256": None, "torch": "2.5.1", "cuda": "12.4",
                "gpu": "NVIDIA L4", "image_id": None}


def run(env: Env, scorer, **kw) -> tuple[pull.Worker, int]:
    kw.setdefault("poll_s", 0.2)
    worker = pull.Worker(env.client(), scorer, INFO, **kw)
    return worker, worker.run()


def test_one_job_end_to_end(make_env, tmp_path):
    env = make_env()
    job_id = env.queue()
    _, code = run(env, FakeScorer(), once=True)
    assert code == 0
    job = env.job(job_id)
    assert job["state"] == "done" and job["gpu_used"] == "L4" and job["timings"]["load_s"] >= 0

    result = env.owner.get(f"/jobs/{job_id}/result").json()
    assert len(result["findings"]) == 146
    keys = artefact_keys(job_id)
    assert result["artefacts"] == keys
    for key in keys.values():
        assert env.svc.storage.exists(key), key
    csv = env.owner.get(f"/jobs/{job_id}/scores.csv")
    assert csv.status_code == 200 and "file_name" in csv.text
    assert env.owner.get(f"/jobs/{job_id}/scores.json").json()["job_id"] == job_id

    r = env.owner.get(f"/jobs/{job_id}/mask.nii.gz", follow_redirects=False)
    assert r.status_code == 307
    mask_path = tmp_path / "mask.nii.gz"
    mask_path.write_bytes(env.owner.get(r.headers["location"]).content)
    data = np.asarray(nib.load(mask_path).dataobj)
    assert data.shape == (24, 20, 10) and int((data == ORGANS[0]["label"]).sum()) == 4 * 4 * 3

    r = env.owner.get(f"/jobs/{job_id}/trace.json", follow_redirects=False)
    assert r.status_code == 307
    trace = env.owner.get(r.headers["location"]).json()
    assert trace["job_id"] == job_id and trace["versions"]["gpu"] == "NVIDIA L4"

    text = env.owner.get(f"/jobs/{job_id}/logs").json()["text"]
    assert f"{job_id} fake model loaded" in text and "scored a.nii.gz" in text

    calls = env.paths(job_id)
    assert sum(c.startswith("PUT ") for c in calls) == 5
    assert f"POST /worker/jobs/{job_id}/complete" in calls


def test_every_request_carries_the_worker_headers(make_env):
    env = make_env()
    env.queue()
    run(env, FakeScorer(), once=True)
    assert len(env.seen) >= 8
    for method, path, headers in env.seen:
        assert headers["authorization"] == f"Bearer {env.token}", path
        assert headers["ngrok-skip-browser-warning"] == "1", path
        assert headers["user-agent"] == "radar-worker", path


def test_weights_mismatch_releases_the_job_and_exits_1(make_env, capsys):
    env = make_env()
    job_id = env.queue()
    env.queue("b.nii.gz")
    scorer = FakeScorer(refuse={"class": "weights_mismatch", "message": "checkpoint: missing"})
    assert run(env, scorer)[1] == 1
    job = env.svc.db.get_job(job_id)
    assert job.state == "queued" and job.lease is None and job.lease_losses == 0
    assert scorer.scored == 0
    assert sum(1 for _, p, _ in env.seen if p == "/worker/claim") == 1
    assert "weights_mismatch: checkpoint: missing" in capsys.readouterr().out


def test_load_that_raises_releases_the_job_and_exits_1(make_env):
    env = make_env()
    job_id = env.queue()

    class Broken(FakeScorer):
        def load(self, log):
            raise RuntimeError("no CUDA device")

    assert run(env, Broken(), once=True)[1] == 1
    job = env.svc.db.get_job(job_id)
    assert job.state == "queued" and job.lease is None and job.lease_losses == 0


def test_runtime_error_fails_with_its_class(make_env):
    env = make_env()
    job_id = env.queue()

    def boom(path):
        raise RuntimeError("boom")

    assert run(env, FakeScorer(on_score=boom), once=True)[1] == 0
    job = env.job(job_id)
    assert job["state"] == "failed" and job["error"] == {"class": "RuntimeError", "message": "boom"}


def test_input_error_fails_with_input_error(make_env):
    env = make_env()
    job_id = env.queue()
    rejected = {"ok": False, "error": {"class": "input_error", "message": "expected a 3D volume"}}
    assert run(env, FakeScorer(on_score=lambda path: rejected), once=True)[1] == 0
    job = env.job(job_id)
    assert job["state"] == "failed" and job["error"] == {"class": "input_error",
                                                         "message": "expected a 3D volume"}


def test_heartbeats_land_with_progress(make_env):
    env = make_env(worker_lease_s=3)
    job_id = env.queue()
    seen = {}

    def wait_for_heartbeat(path):
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            job = env.svc.db.get_job(job_id)
            if job.lease.progress == "scoring":
                seen["job"] = job
                return
            time.sleep(0.05)
        raise AssertionError("no heartbeat while scoring")

    assert run(env, FakeScorer(on_score=wait_for_heartbeat), once=True)[1] == 0
    job = seen["job"]
    assert parse_iso(job.lease.heartbeat_at) > parse_iso(job.submitted_at)
    assert env.job(job_id)["state"] == "done"
    assert any(p == f"POST /worker/jobs/{job_id}/heartbeat" for p in env.paths(job_id))


def test_idle_exit_backs_off(make_env):
    env = make_env()
    now = [0.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    _, code = run(env, FakeScorer(), poll_s=4, idle_exit=10, clock=lambda: now[0], sleep=sleep)
    assert code == 0
    assert sleeps == [1, 2, 4, 4]
    assert sum(1 for m, p, _ in env.seen if p == "/worker/claim") == 5


def test_lost_lease_complete_409_is_logged_and_the_loop_continues(make_env, capsys):
    env = make_env()
    first = env.queue("a.nii.gz")
    second = env.queue("b.nii.gz")

    def cancel_first(path):
        if Path(path).name == "a.nii.gz":
            assert env.owner.post(f"/jobs/{first}/cancel").status_code == 200

    _, code = run(env, FakeScorer(on_score=cancel_first), idle_exit=0.3)
    assert code == 0
    assert env.job(first)["state"] == "cancelled"
    assert env.job(second)["state"] == "done"
    out = capsys.readouterr().out
    assert f"{first}: complete refused (HTTP 409" in out


def test_lease_lost_by_heartbeat_skips_upload_and_complete(make_env, capsys):
    env = make_env(worker_lease_s=3)
    job_id = env.queue()
    holder = {}

    def cancel_and_wait(path):
        assert env.owner.post(f"/jobs/{job_id}/cancel").status_code == 200
        deadline = time.monotonic() + 4
        while not holder["worker"].current.lost.is_set():
            assert time.monotonic() < deadline, "the heartbeat never saw the 409"
            time.sleep(0.05)

    worker = pull.Worker(env.client(), FakeScorer(on_score=cancel_and_wait), INFO, poll_s=0.2, once=True)
    holder["worker"] = worker
    assert worker.run() == 0
    calls = env.paths(job_id)
    assert not any(c.startswith("PUT ") for c in calls)
    assert f"POST /worker/jobs/{job_id}/complete" not in calls
    assert env.job(job_id)["state"] == "cancelled"
    assert "lease lost" in capsys.readouterr().out


def test_stop_during_download_releases_the_job(make_env, monkeypatch):
    env = make_env()
    job_id = env.queue()
    monkeypatch.setattr(pull.wio, "CHUNK", 64)
    holder = {"chunks": 0}

    class StopInDownload(pull.DeskClient):
        def download_source(self, url, dest, on_chunk=None):
            def hook(length):
                holder["chunks"] += 1
                holder["worker"].request_stop()
                on_chunk(length)

            return super().download_source(url, dest, on_chunk=hook)

    scorer = FakeScorer()
    worker = pull.Worker(StopInDownload(env.url, env.token), scorer, INFO, poll_s=0.2)
    holder["worker"] = worker
    assert worker.run() == 0
    job = env.svc.db.get_job(job_id)
    assert job.state == "queued" and job.lease is None
    assert scorer.scored == 0 and holder["chunks"] == 1
    assert f"POST /worker/jobs/{job_id}/release" in env.paths(job_id)


def test_stop_during_scoring_finishes_the_job(make_env):
    env = make_env()
    job_id = env.queue()
    env.queue("b.nii.gz")
    holder = {}
    scorer = FakeScorer(on_score=lambda path: holder["worker"].request_stop())
    worker = pull.Worker(env.client(), scorer, INFO, poll_s=0.2)
    holder["worker"] = worker
    assert worker.run() == 0
    assert env.job(job_id)["state"] == "done"
    assert scorer.scored == 1  # the second job is left queued


def test_complete_retries_a_network_error(make_env, capsys):
    env = make_env()
    job_id = env.queue()
    left = [2]

    class Flaky(pull.DeskClient):
        def complete(self, job_id, body):
            if left[0]:
                left[0] -= 1
                raise pull.DeskError(0, "connection reset")
            return super().complete(job_id, body)

    worker = pull.Worker(Flaky(env.url, env.token), FakeScorer(), INFO, poll_s=0.2, once=True)
    assert worker.run() == 0
    assert env.job(job_id)["state"] == "done"
    assert capsys.readouterr().out.count("complete failed (HTTP 0: connection reset); retrying") == 2


def test_complete_409_after_uploads_is_logged_as_refused(make_env, capsys):
    env = make_env()
    job_id = env.queue()

    class CancelFirst(pull.DeskClient):
        def complete(self, job_id, body):
            assert env.owner.post(f"/jobs/{job_id}/cancel").status_code == 200
            return super().complete(job_id, body)

    worker = pull.Worker(CancelFirst(env.url, env.token), FakeScorer(), INFO, poll_s=0.2, once=True)
    assert worker.run() == 0
    assert env.job(job_id)["state"] == "cancelled"
    assert sum(c.startswith("PUT ") for c in env.paths(job_id)) == 5
    assert f"{job_id}: complete refused (HTTP 409" in capsys.readouterr().out


def test_html_404_on_a_heartbeat_does_not_lose_the_lease(make_env):
    env = make_env(worker_lease_s=3)
    job_id = env.queue()
    env.fault("/heartbeat", 404)
    holder = {}

    def wait_for_two_heartbeats(path):
        deadline = time.monotonic() + 5
        while sum(1 for p in env.paths(job_id) if p.endswith("/heartbeat")) < 2:
            assert time.monotonic() < deadline, "no second heartbeat"
            time.sleep(0.05)
        assert not holder["worker"].current.lost.is_set()

    worker = pull.Worker(env.client(), FakeScorer(on_score=wait_for_two_heartbeats), INFO, poll_s=0.2, once=True)
    holder["worker"] = worker
    assert worker.run() == 0
    assert not env.faults
    assert env.job(job_id)["state"] == "done"


def test_html_503_on_a_claim_backs_off_and_claims_again(make_env):
    env = make_env()
    job_id = env.queue()
    env.fault("/worker/claim", 503)
    sleeps = []
    _, code = run(env, FakeScorer(), once=True, sleep=sleeps.append)
    assert code == 0 and sleeps == [0.2]
    assert env.job(job_id)["state"] == "done"


def test_non_json_4xx_is_transient_and_json_4xx_is_not():
    assert pull.DeskError(404, "<html>", from_app=False).transient
    assert not pull.DeskError(409, "lease is not current").transient
    assert pull.DeskError(0, "reset").transient and pull.DeskError(502, "bad gateway").transient


def test_download_retries_a_network_error(make_env, capsys):
    env = make_env()
    job_id = env.queue()
    left = [1]

    class Flaky(pull.DeskClient):
        def download_source(self, url, dest, on_chunk=None):
            if left[0]:
                left[0] -= 1
                raise pull.wio.TransferError(0, "connection reset", url)
            return super().download_source(url, dest, on_chunk=on_chunk)

    sleeps = []
    worker = pull.Worker(Flaky(env.url, env.token), FakeScorer(), INFO, poll_s=0.2, once=True, sleep=sleeps.append)
    assert worker.run() == 0
    assert env.job(job_id)["state"] == "done" and sleeps == [0.2]
    assert "download failed (HTTP 0" in capsys.readouterr().out


def test_url_always_joins_the_base():
    client = pull.DeskClient("http://a", "t")
    assert client.url("http://evil/x").startswith("http://a/")
    assert client.url("/worker/claim") == "http://a/worker/claim"


def test_signal_asks_the_worker_to_stop(make_env):
    worker = pull.Worker(pull.DeskClient("http://127.0.0.1:9", "rdw_x"), FakeScorer(), INFO)
    previous = pull.install_signals(worker)
    try:
        os.kill(os.getpid(), signal.SIGTERM)
        deadline = time.monotonic() + 2
        while not worker.stopping and time.monotonic() < deadline:
            time.sleep(0.01)
        assert worker.stopping
    finally:
        for sig, old in previous.items():
            signal.signal(sig, old)


def test_main_needs_the_env(monkeypatch):
    monkeypatch.delenv("RADAR_DESK_URL", raising=False)
    monkeypatch.delenv("RADAR_WORKER_TOKEN", raising=False)
    assert pull.main([]) == 2


def test_main_with_a_refused_token_exits_2(make_env, monkeypatch):
    env = make_env()
    monkeypatch.setenv("RADAR_DESK_URL", env.url)
    monkeypatch.setenv("RADAR_WORKER_TOKEN", "rdw_" + "0" * 32)
    assert pull.main(["--poll", "0.2"]) == 2
    assert [p for _, p, _ in env.seen] == ["/worker/me"]


def test_cli_help_without_torch():
    env = {**os.environ, "PYTHONPATH": str(ROOT / "worker")}
    out = subprocess.run([sys.executable, "-m", "radar_worker.pull", "--help"], capture_output=True, text=True,
                         env=env, timeout=30, check=False)
    assert out.returncode == 0 and "--idle-exit" in out.stdout
    probe = subprocess.run([sys.executable, "-c", "import sys, radar_worker.pull; print('torch' in sys.modules)"],
                           capture_output=True, text=True, env=env, timeout=30, check=False)
    assert probe.stdout.strip() == "False", probe.stderr


def test_token_in_clear_only_for_http_to_another_machine():
    assert pull.token_in_clear("http://192.168.1.20:8000")
    assert pull.token_in_clear("http://desk.example.org")
    for same_machine in ("http://127.0.0.1:8000", "http://localhost:8000", "http://[::1]:8000",
                         "http://host.docker.internal:8000"):
        assert not pull.token_in_clear(same_machine)
    assert not pull.token_in_clear("https://abc.trycloudflare.com")
