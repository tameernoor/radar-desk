from __future__ import annotations

import asyncio

import pytest

from radar_desk.gpu.fake import FakeGpuBackend, canned_result
from radar_desk.gpu.poller import Poller
from radar_desk.records import Job
from radar_desk.services import build_services
from radar_desk.services.costs import parse_iso
from radar_desk.services.scans import source_key
from synth import make_nifti

L4 = 0.000222
WORST_L4 = (2 * 1800 + 120) * L4


class Clock:
    def __init__(self, iso: str = "2026-10-15T12:00:00.000000Z") -> None:
        self.t = parse_iso(iso)

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def env(make_services, tmp_path, clock):
    """(services, backend, poller, scan) with one ready synthetic scan."""

    def build(backend=None, **overrides):
        backend = backend or FakeGpuBackend(clock=clock)
        svc = make_services(backend=backend, clock=clock, **overrides)
        path = make_nifti(tmp_path / "p.nii.gz", shape=(24, 20, 10))
        ticket = svc.scans.begin_upload(path.name, path.stat().st_size, True)
        svc.storage.put_bytes(source_key(ticket.scan_id), path.read_bytes())
        scan = svc.scans.complete_upload(ticket.scan_id)
        assert scan.state == "ready"
        return svc, backend, Poller(svc, backend, clock=clock), scan

    return build


def l4_result(job_id: str, total_s: float = 100.0) -> dict:
    data = canned_result(job_id, gpu="NVIDIA L4")
    data["timings"]["total_s"] = total_s
    return data


def test_queued_job_is_spawned(env):
    svc, backend, poller, scan = env(radar_gpu="L4,A10")
    job = svc.jobs.create(scan.id)
    poller.tick()
    job = svc.jobs.get(job.id)
    assert job.state == "submitted" and job.modal_call_id == "fc-fake-1"
    assert job.gpu_requested == ["L4", "A10"]
    assert job.submitted_at == "2026-10-15T12:00:00.000000Z"
    call = backend.calls["fc-fake-1"]
    assert f"/_storage/scans/{scan.id}/source.nii.gz?" in call["source_url"]
    assert set(call["artefact_urls"]) == {"scores_json", "scores_csv", "mask", "trace", "log"}
    assert call["artefact_keys"]["log"] == f"jobs/{job.id}/worker.log"


def test_worker_refs_are_volume_paths_on_the_volume_backend(env):
    from fake_volume import FakeVolume
    from radar_desk.storage.modal_volume import ModalVolumeStorage

    storage = ModalVolumeStorage("radar-data", "http://127.0.0.1:8000", volume=FakeVolume())
    svc, backend, poller, scan = env(storage_backend="modal_volume", storage=storage)
    job = svc.jobs.create(scan.id)
    poller.tick()
    call = backend.calls[svc.jobs.get(job.id).modal_call_id]
    assert call["source_url"] == f"volume://scans/{scan.id}/source.nii.gz"
    assert call["artefact_urls"] == {name: f"volume://{key}" for name, key in call["artefact_keys"].items()}
    assert call["artefact_urls"]["mask"] == f"volume://jobs/{job.id}/mask.nii.gz"


def test_worker_refs_are_signed_urls_on_the_local_backend(env):
    svc, backend, poller, scan = env()
    job = svc.jobs.create(scan.id)
    poller.tick()
    call = backend.calls[svc.jobs.get(job.id).modal_call_id]
    assert call["source_url"].startswith(f"http://127.0.0.1:8000/_storage/scans/{scan.id}/source.nii.gz?exp=")
    for name, url in call["artefact_urls"].items():
        assert url.startswith(f"http://127.0.0.1:8000/_storage/{call['artefact_keys'][name]}?exp=")
        assert "sig=" in url


def test_pending_stays_submitted_then_done_with_cost(env, clock):
    svc, backend, poller, scan = env()
    backend.delay_ticks = 2
    job = svc.jobs.create(scan.id)
    backend.set_result(job.id, l4_result(job.id, total_s=100.0))
    poller.tick()  # spawn; polling starts on the next tick
    for _ in range(2):
        clock.advance(10)
        poller.tick()
        assert svc.jobs.get(job.id).state == "submitted"
    clock.advance(10)
    poller.tick()
    done = svc.jobs.get(job.id)
    assert done.state == "done"
    assert done.gpu_used == "L4"
    assert done.timings.total_s == 100.0
    assert done.cost_estimate_usd == pytest.approx((100 + 120) * L4)
    assert svc.results.get(job.id).versions.gpu == "NVIDIA L4"


def test_ok_false_fails_with_function_class(env):
    svc, backend, poller, scan = env()
    job = svc.jobs.create(scan.id)
    backend.set_result(job.id, {"ok": False, "job_id": job.id,
                                "error": {"class": "input_error", "message": "no organs found"}})
    poller.tick()
    poller.tick()
    failed = svc.jobs.get(job.id)
    assert failed.state == "failed"
    assert failed.error.klass == "input_error" and failed.error.message == "no organs found"
    assert failed.cost_estimate_usd > 0


def test_exception_fails_with_class_and_message(env):
    svc, backend, poller, scan = env()
    job = svc.jobs.create(scan.id)
    backend.set_result(job.id, RuntimeError("CUDA out of memory"))
    poller.tick()
    poller.tick()
    failed = svc.jobs.get(job.id)
    assert failed.state == "failed"
    assert failed.error.model_dump() == {"class": "RuntimeError", "message": "CUDA out of memory"}


def test_invalid_result_fails(env):
    svc, backend, poller, scan = env()
    job = svc.jobs.create(scan.id)
    bad = canned_result(job.id)
    bad["findings"] = bad["findings"][:10]
    backend.set_result(job.id, bad)
    poller.tick()
    poller.tick()
    assert svc.jobs.get(job.id).error.klass == "invalid_result"


def test_stuck_job_is_cancelled_and_failed(env, clock):
    svc, backend, poller, scan = env()
    backend.delay_ticks = 10**6
    job = svc.jobs.create(scan.id)
    poller.tick()
    clock.advance(2 * 1800 + 300)
    poller.tick()
    assert svc.jobs.get(job.id).state == "submitted"
    clock.advance(1)
    poller.tick()
    stuck = svc.jobs.get(job.id)
    assert stuck.state == "failed" and stuck.error.klass == "stuck"
    assert backend.cancelled == ["fc-fake-1"]


def test_budget_holds_and_releases_when_raised(env):
    svc, backend, poller, scan = env(gpu_monthly_budget_usd=0.5)
    job = svc.jobs.create(scan.id)
    poller.tick()
    held = svc.jobs.get(job.id)
    assert held.state == "queued" and held.hold_reason == "budget"
    assert backend.calls == {}
    assert [j["id"] for j in svc.gpu_status(poller.clock())["held"]] == [job.id]
    svc.settings.gpu_monthly_budget_usd = 10.0
    poller.tick()
    assert svc.jobs.get(job.id).state == "submitted"
    assert svc.jobs.get(job.id).hold_reason is None


def test_budget_counts_month_spend_and_in_flight(env, clock):
    svc, _backend, poller, scan = env(gpu_monthly_budget_usd=10.0)
    past = Job(id="job_past", scan_id=scan.id, gpu_requested=["L4"])
    svc.db.insert_job(past)
    past = svc.db.transition(past, "submitted", modal_call_id="fc-old")
    svc.db.transition(past, "done", finished_at="2026-10-02T00:00:00.000000Z",
                      cost_estimate_usd=10.0 - WORST_L4 + 0.01)
    job = svc.jobs.create(scan.id)
    poller.tick()
    assert svc.jobs.get(job.id).hold_reason == "budget"
    clock.advance(20 * 86400)  # into November
    poller.tick()
    assert svc.jobs.get(job.id).state == "submitted"


def test_one_job_at_a_time(env):
    svc, backend, poller, scan = env()
    backend.delay_ticks = 1
    first = svc.jobs.create(scan.id)
    second = Job(id="job_second", scan_id=scan.id, gpu_requested=["L4"])
    svc.db.insert_job(second)
    poller.tick()
    assert svc.jobs.get(first.id).state == "submitted"
    assert svc.jobs.get(second.id).state == "queued"  # waits, and its stuck clock has not started
    poller.tick()  # first pending
    assert svc.jobs.get(second.id).state == "queued" and svc.jobs.get(second.id).hold_reason is None
    poller.tick()  # first done, second spawned
    assert svc.jobs.get(first.id).state == "done"
    assert svc.jobs.get(second.id).state == "submitted"


def test_failed_submit_after_spawn_cancels_the_call(env, monkeypatch):
    svc, backend, poller, scan = env()
    job = svc.jobs.create(scan.id)

    def broken(*args, **kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(svc.db, "transition", broken)
    poller.tick()
    assert backend.cancelled == ["fc-fake-1"]
    assert svc.db.get_job(job.id).state == "queued"


def test_error_while_handling_outcome_fails_the_job(env, monkeypatch):
    svc, _backend, poller, scan = env()
    job = svc.jobs.create(scan.id)
    poller.tick()

    def broken(job, data):
        raise KeyError("organ_stats")

    monkeypatch.setattr(svc.results, "store", broken)
    poller.tick()
    failed = svc.jobs.get(job.id)
    assert failed.state == "failed" and failed.error.klass == "KeyError"


def test_spawn_error_holds_then_retries(env):
    svc, backend, poller, scan = env()
    backend.spawn_error = ConnectionError("modal unreachable")
    job = svc.jobs.create(scan.id)
    poller.tick()
    held = svc.jobs.get(job.id)
    assert held.state == "queued" and held.hold_reason == "spawn_error"
    assert held.error.message == "modal unreachable"
    backend.spawn_error = None
    poller.tick()
    job = svc.jobs.get(job.id)
    assert job.state == "submitted" and job.hold_reason is None and job.error is None


def test_restart_resumes_submitted_job(env, clock):
    svc, backend, poller, scan = env()
    job = svc.jobs.create(scan.id)
    poller.tick()
    assert svc.jobs.get(job.id).state == "submitted"
    svc.db.close()
    fresh = build_services(svc.settings, backend=backend, clock=clock)
    Poller(fresh, backend, clock=clock).tick()
    assert fresh.jobs.get(job.id).state == "done"
    assert fresh.results.get(job.id)


def test_retry_adds_attempt_cost(env, clock):
    svc, backend, poller, scan = env()
    job = svc.jobs.create(scan.id)
    backend.set_result(job.id, RuntimeError("boom"))
    poller.tick()
    clock.advance(30)
    poller.tick()
    first = svc.jobs.get(job.id)
    assert first.cost_estimate_usd == pytest.approx((30 + 120) * L4)
    svc.jobs.retry(job.id)
    backend.set_result(job.id, l4_result(job.id, total_s=50.0))
    poller.tick()
    poller.tick()
    done = svc.jobs.get(job.id)
    assert done.state == "done"
    assert done.cost_estimate_usd == pytest.approx((30 + 120) * L4 + (50 + 120) * L4)
    assert svc.db.sum_cost_for_month("2026-10") == pytest.approx(done.cost_estimate_usd)


def test_dev_fake_backend_synthesises_artefacts(make_services, tmp_path, clock):
    base = make_services()
    svc = build_services(base.settings, db=base.db, storage=base.storage, clock=clock)
    assert svc.backend.name == "fake"
    path = make_nifti(tmp_path / "dev.nii.gz", shape=(24, 20, 10))
    ticket = svc.scans.begin_upload(path.name, path.stat().st_size, True)
    svc.storage.put_bytes(source_key(ticket.scan_id), path.read_bytes())
    svc.scans.complete_upload(ticket.scan_id)
    job = svc.jobs.create(ticket.scan_id)
    poller = Poller(svc, clock=clock)
    poller.tick()
    poller.tick()
    assert svc.jobs.get(job.id).state == "done"
    assert svc.jobs.get(job.id).gpu_used == "fake"
    assert svc.exports.artefact_url(job.id, "mask")


async def test_run_loops_until_stopped(env):
    svc, _backend, poller, scan = env(gpu_poll_interval_s=0.01)
    job = svc.jobs.create(scan.id)
    stop = asyncio.Event()
    task = asyncio.create_task(poller.run(stop))
    for _ in range(200):
        if svc.jobs.get(job.id).state == "done":
            break
        await asyncio.sleep(0.01)
    stop.set()
    await asyncio.wait_for(task, 2)
    assert svc.jobs.get(job.id).state == "done"
