"""Job kinds: NV-Reason-CT jobs sit in the jobs table next to RADAR jobs and never reach RADAR's paths."""

from __future__ import annotations

from radar_desk.gpu.fake import FakeGpuBackend
from radar_desk.gpu.poller import Poller
from radar_desk.records import Job, Lease
from test_chat_tools import ready_scan

AT = "2026-10-15T12:00:00.000000Z"


def nv_job(svc, scan_id: str, job_id: str = "job_nv", submitted: bool = False) -> Job:
    job = Job(id=job_id, scan_id=scan_id, kind="nvreason", gpu_requested=["L4"])
    svc.db.insert_job(job)
    if submitted:
        job = svc.db.transition(job, "submitted", modal_call_id=f"fc-{job_id}")
    return job


def test_a_record_without_kind_is_a_radar_job():
    assert Job.model_validate_json('{"id": "job_a", "scan_id": "scan_a"}').kind == "radar"


def test_job_queries_default_to_radar(make_services, tmp_path):
    svc = make_services()
    scan = ready_scan(svc, tmp_path)
    radar = svc.jobs.create(scan.id)
    nv = nv_job(svc, scan.id)
    assert [j.id for j in svc.db.list_jobs()] == [radar.id]
    assert [j.id for j in svc.db.list_jobs(kind="nvreason")] == [nv.id]
    assert {j.id for j in svc.db.list_jobs(kind=None)} == {radar.id, nv.id}
    assert [j.id for j in svc.db.jobs_for_scan(scan.id)] == [radar.id]
    assert [j.id for j in svc.db.jobs_for_scan(scan.id, kind="nvreason")] == [nv.id]
    assert svc.db.latest_job_for_scan(scan.id).id == radar.id
    assert svc.db.latest_job_for_scan(scan.id, kind="nvreason").id == nv.id
    assert svc.jobs.latest_summary(scan.id)["id"] == radar.id
    assert [j.id for j in svc.jobs.list(scan_id=scan.id)] == [radar.id]


def test_a_pull_worker_never_claims_an_nvreason_job(make_services, tmp_path):
    svc = make_services()
    scan = ready_scan(svc, tmp_path)
    nv_job(svc, scan.id)
    lease = Lease(worker_id="w1", expires_at="2026-10-15T12:02:00.000000Z", heartbeat_at=AT)
    assert svc.db.claim_job("wk_1", lease, AT) is None
    radar = svc.jobs.create(scan.id)
    assert svc.db.claim_job("wk_2", lease, AT).id == radar.id


def test_the_radar_poller_ignores_nvreason_jobs(make_services, tmp_path):
    backend = FakeGpuBackend()
    svc = make_services(backend=backend)
    scan = ready_scan(svc, tmp_path)
    nv_job(svc, scan.id, "job_nv_running", submitted=True)
    nv_job(svc, scan.id, "job_nv_waiting")
    radar = svc.jobs.create(scan.id)
    Poller(svc, backend).tick()
    assert [c["job_id"] for c in backend.calls.values()] == [radar.id]
    assert svc.db.get_job("job_nv_waiting").state == "queued"
    assert svc.db.get_job("job_nv_running").state == "submitted"
