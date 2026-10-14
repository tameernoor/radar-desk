"""NV-Reason-CT jobs: queued next to RADAR's, run by the poller on their own backend, never touching RADAR."""

from __future__ import annotations

import pytest

from api_client import api_client
from radar_desk.gpu.backend import Finished
from radar_desk.gpu.fake import FakeGpuBackend
from radar_desk.gpu.nvreason_fake import FakeNvReasonBackend, synthesize_nv_result
from radar_desk.gpu.poller import Poller
from radar_desk.services import ServiceError
from radar_desk.services.blobs import read_bytes, read_json, replace_bytes
from radar_desk.services.nvreason import (
    NEEDS_MODAL,
    NV_KIND,
    NV_MODEL_VERSION,
    NV_REGION,
    NV_REPORT_PROMPT,
    NV_REVISION,
    NV_SCALEDOWN_S,
    NV_STUCK_S,
    OFF,
    nv_keys,
    qa_key,
)
from test_chat_tools import ready_scan
from test_poller import Clock


def fake_nv(holder: dict, **kw) -> FakeNvReasonBackend:
    """A fake NV backend that synthesizes from the services in holder["svc"]."""

    def synthesize(job_id: str) -> dict:
        db = holder["svc"].db
        job = db.get_job(job_id)
        return synthesize_nv_result(job, db.get_scan(job.scan_id))

    return FakeNvReasonBackend(synthesize=synthesize, **kw)


@pytest.fixture
def nv(make_services, tmp_path):
    """(services, NV backend, poller, scan, clock). NV is on unless enabled=False; gpu_name sets the compute mode."""

    def build(enabled=True, gpu_name="fake", delay_ticks=0, **overrides):
        clock = Clock()
        holder: dict = {}
        nv_backend = fake_nv(holder, delay_ticks=delay_ticks)
        gpu = FakeGpuBackend(clock=clock)
        gpu.name = gpu_name
        svc = make_services(backend=gpu, clock=clock, nv_backend=nv_backend, nvreason_enabled=enabled, **overrides)
        holder["svc"] = svc
        return svc, nv_backend, Poller(svc, gpu, clock=clock), ready_scan(svc, tmp_path), clock

    return build


def test_scoring_queues_one_nvreason_job_next_to_radar(nv):
    svc, _, _, scan, _ = nv()
    radar = svc.jobs.create(scan.id)
    svc.jobs.create(scan.id)
    jobs = svc.db.list_jobs(kind=NV_KIND)
    assert len(jobs) == 1
    assert jobs[0].scan_id == scan.id and jobs[0].state == "queued"
    assert jobs[0].model_version == NV_MODEL_VERSION and jobs[0].gpu_requested == ["L4"]
    assert [j.id for j in svc.db.list_jobs()] == [radar.id]


def test_no_nvreason_job_when_off(nv):
    svc, _, _, scan, _ = nv(enabled=False)
    svc.jobs.create(scan.id)
    assert svc.db.list_jobs(kind=NV_KIND) == []
    assert svc.nvreason.unavailable() == OFF
    with pytest.raises(ServiceError) as err:
        svc.nvreason.start(scan.id)
    assert err.value.status == 409 and err.value.detail == OFF


def test_no_nvreason_job_outside_modal_compute(nv):
    svc, _, _, scan, _ = nv(gpu_name="worker")
    svc.jobs.create(scan.id)
    assert svc.db.list_jobs(kind=NV_KIND) == []
    assert svc.nvreason.unavailable() == NEEDS_MODAL


def test_the_poller_runs_the_job_and_stores_its_artefacts(nv):
    svc, backend, poller, scan, _ = nv()
    radar = svc.jobs.create(scan.id)
    for _ in range(3):
        poller.tick()
    job = svc.nvreason.latest(scan.id)
    assert job.state == "done" and job.backend == "fake" and job.gpu_used == "fake"
    assert job.cost_estimate_usd is None  # the fake backend is not priced
    keys = nv_keys(job.id)
    report = read_json(svc.storage, keys["nv_report"])
    assert report["prompt"] == NV_REPORT_PROMPT and report["region"] == NV_REGION
    assert report["text"].endswith("</answer>") and "<answer>" not in report["answer"]
    assert report["answer_tag"] == "Liver mass or non-cystic lesion, Renal cyst"
    assert svc.nvreason.report(job.id) == report
    crop = svc.nvreason.crop(job.id)
    assert len(crop["box_mm"]) == 2 and crop["fallback"] is False
    assert read_bytes(svc.storage, keys["crop_nii"])[:2] == b"\x1f\x8b"
    assert b"fake NV-Reason-CT" in read_bytes(svc.storage, keys["log"])
    source = backend.calls["fc-nvfake-1"]["source_ref"]
    assert source.split("?")[0].endswith(f"/scans/{scan.id}/source.nii.gz")
    assert svc.jobs.get(radar.id).state == "done"
    assert svc.nvreason.latest_done(scan.id).id == job.id


def test_an_nv_failure_never_touches_radar(nv):
    svc, backend, poller, scan, _ = nv()
    radar = svc.jobs.create(scan.id)
    job = svc.nvreason.latest(scan.id)
    backend.results[job.id] = RuntimeError("CUDA out of memory")
    for _ in range(3):
        poller.tick()
    failed = svc.db.get_job(job.id)
    assert failed.state == "failed" and failed.error.klass == "RuntimeError"
    assert "CUDA out of memory" in failed.error.message
    done = svc.jobs.get(radar.id)
    assert done.state == "done" and done.error is None
    assert svc.db.get_result(radar.id) is not None
    again = svc.nvreason.start(scan.id)
    assert again.id != job.id and again.state == "queued"


def test_an_input_error_fails_the_job(nv):
    svc, backend, poller, scan, _ = nv()
    svc.jobs.create(scan.id)
    job = svc.nvreason.latest(scan.id)
    backend.results[job.id] = {"ok": False, "job_id": job.id, "error": {"class": "input_error", "message": "HTTP 403"}}
    for _ in range(2):
        poller.tick()
    failed = svc.db.get_job(job.id)
    assert failed.state == "failed" and failed.error.klass == "input_error" and failed.error.message == "HTTP 403"


def test_a_spawn_error_holds_the_job_until_it_works(nv):
    svc, backend, poller, scan, _ = nv()
    svc.jobs.create(scan.id)
    backend.spawn_error = OSError("Modal is unreachable")
    poller.tick()
    job = svc.nvreason.latest(scan.id)
    assert job.state == "queued" and job.hold_reason == "spawn_error" and job.error.klass == "OSError"
    backend.spawn_error = None
    poller.tick()
    job = svc.db.get_job(job.id)
    assert job.state == "submitted" and job.hold_reason is None and job.error is None


def test_a_stuck_job_is_cancelled_and_failed(nv):
    svc, backend, poller, scan, clock = nv(delay_ticks=10**6)
    svc.jobs.create(scan.id)
    poller.tick()
    job = svc.nvreason.latest(scan.id)
    assert job.state == "submitted"
    clock.advance(NV_STUCK_S + 1)
    poller.tick()
    job = svc.db.get_job(job.id)
    assert job.state == "failed" and job.error.klass == "stuck"
    assert backend.cancelled == [job.modal_call_id]


def test_done_listeners_get_the_scan_id_and_the_model_is_warm(nv):
    svc, _, poller, scan, clock = nv()
    seen: list[str] = []
    svc.nvreason.on_done.append(seen.append)
    assert svc.nvreason.warm() is False
    svc.jobs.create(scan.id)
    for _ in range(2):
        poller.tick()
    assert seen == [scan.id]
    assert svc.nvreason.warm() is True
    clock.advance(NV_SCALEDOWN_S)
    assert svc.nvreason.warm() is False


def test_nothing_runs_when_off(nv):
    svc, backend, poller, scan, _ = nv(enabled=False)
    svc.jobs.create(scan.id)
    poller.tick()
    assert backend.calls == {}


def test_settling_again_on_the_volume_replaces_written_objects(make_services, tmp_path):
    """Review Focus 5: a first settle wrote some objects, then the process died before the state change."""
    from fake_volume import FakeVolume
    from radar_desk.storage.modal_volume import ModalVolumeStorage

    storage = ModalVolumeStorage("radar-data", "http://127.0.0.1:8000", volume=FakeVolume())
    holder: dict = {}
    svc = make_services(storage=storage, storage_backend="modal_volume", nv_backend=fake_nv(holder),
                        nvreason_enabled=True)
    holder["svc"] = svc
    scan = ready_scan(svc, tmp_path)
    job = svc.nvreason.ensure_job(scan.id)
    job = svc.db.transition(job, "submitted", modal_call_id="fc-x", backend="fake")
    storage.put_bytes(nv_keys(job.id)["log"], b"from the first attempt")
    storage.put_bytes(nv_keys(job.id)["crop"], b"{}")
    done = svc.nvreason.settle(job, Finished(synthesize_nv_result(job, scan)), 1_800_000_000.0)
    assert done.state == "done"
    assert b"fake NV-Reason-CT" in read_bytes(storage, nv_keys(job.id)["log"])
    assert "box_mm" in read_json(storage, nv_keys(job.id)["crop"])


def test_replace_bytes_overwrites_on_every_storage(make_services):
    svc = make_services()
    replace_bytes(svc.storage, "readers/x/a.json", b"one")
    replace_bytes(svc.storage, "readers/x/a.json", b"two")
    assert read_bytes(svc.storage, "readers/x/a.json") == b"two"
    assert read_bytes(svc.storage, "readers/x/missing.json") is None
    assert read_json(svc.storage, "readers/x/a.json") is None  # not JSON


def test_qa_entries_skip_unreadable_lines(nv):
    svc, *_ = nv()
    svc.storage.put_bytes(qa_key("job_x"), b'{"question": "a", "answer": "b"}\nnot json\n[1]\n')
    assert svc.nvreason.qa_entries("job_x") == [{"question": "a", "answer": "b"}]
    assert svc.nvreason.qa_entries("job_none") == []


def test_constants_match_the_worker():
    from radar_worker import nvreason as worker

    assert (NV_REVISION, NV_REGION, NV_REPORT_PROMPT, NV_SCALEDOWN_S) == (
        worker.REVISION, worker.REGION, worker.REPORT_PROMPT, worker.SCALEDOWN_S)


def test_start_route(nv):
    svc, _, _, scan, _ = nv()
    client = api_client(svc)
    r = client.post(f"/scans/{scan.id}/nvreason")
    assert r.status_code == 200, r.text
    assert r.json()["kind"] == "nvreason" and r.json()["state"] == "queued"
    assert client.post(f"/scans/{scan.id}/nvreason").json()["id"] == r.json()["id"]
    assert client.post("/scans/scan_missing/nvreason").status_code == 404


def test_start_route_when_off(nv):
    svc, _, _, scan, _ = nv(enabled=False)
    r = api_client(svc).post(f"/scans/{scan.id}/nvreason")
    assert r.status_code == 409 and r.json() == {"detail": OFF}


def test_deleting_a_scan_is_refused_while_an_nv_job_is_active(nv):
    svc, _, _, scan, _ = nv()
    job = svc.nvreason.ensure_job(scan.id)
    assert job.state == "queued"
    with pytest.raises(ServiceError) as err:
        svc.scans.delete(scan.id)
    assert err.value.status == 409 and job.id in err.value.detail


def test_radar_cancel_and_retry_refuse_an_nv_job(nv):
    svc, _, _, scan, _ = nv()
    job = svc.nvreason.ensure_job(scan.id)
    client = api_client(svc)
    for action in ("cancel", "retry"):
        r = client.post(f"/jobs/{job.id}/{action}")
        assert r.status_code == 404, r.text
    assert svc.db.get_job(job.id).state == "queued"
