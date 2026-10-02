"""Database and records: schema, round trips, job state machine, filters, concurrency."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import pytest

from radar_desk.db import Database, IllegalTransition
from radar_desk.records import (
    Artefacts,
    ChatExecution,
    Finding,
    Header,
    Job,
    JobError,
    OrganScored,
    OrganStats,
    Result,
    Scan,
    Timings,
    Versions,
    new_id,
    now_iso,
)


@pytest.fixture
def db(tmp_path: Path) -> Database:
    return Database(tmp_path / "radar.db")


def make_scan(**kw) -> Scan:
    base = {"id": new_id("scan"), "filename": "ct.nii.gz", "size_bytes": 123, "state": "uploading"}
    base.update(kw)
    return Scan(**base)


def make_job(scan_id: str, **kw) -> Job:
    base = {"id": new_id("job"), "scan_id": scan_id, "gpu_requested": ["L4"]}
    base.update(kw)
    return Job(**base)


def test_ids_and_timestamps():
    a, b = new_id("scan"), new_id("scan")
    assert a.startswith("scan_") and len(a) == len("scan_") + 12 and a != b
    assert now_iso().endswith("Z")


def test_schema_and_wal(tmp_path: Path):
    path = tmp_path / "radar.db"
    Database(path)
    con = sqlite3.connect(path)
    assert con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"schema_version", "scans", "jobs", "results", "chat_executions"} <= tables
    for t in ("scans", "jobs", "results", "chat_executions"):
        cols = {r[1] for r in con.execute(f"PRAGMA table_info({t})")}
        assert {"created_at", "payload"} <= cols


def test_migrations_idempotent(tmp_path: Path):
    path = tmp_path / "radar.db"
    d1 = Database(path)
    scan = make_scan()
    d1.insert_scan(scan)
    d1.close()
    d2 = Database(path)
    assert d2.get_scan(scan.id) == scan
    con = sqlite3.connect(path)
    versions = [r[0] for r in con.execute("SELECT version FROM schema_version")]
    assert versions == sorted(set(versions)) and len(versions) >= 1


def test_scan_round_trip(db: Database):
    header = Header(
        dims=[512, 512, 100],
        spacing_mm=[0.7, 0.7, 3.0],
        dtype="int16",
        orientation="RAS",
        affine=[[0.7, 0, 0, 0], [0, 0.7, 0, 0], [0, 0, 3.0, 0], [0, 0, 0, 1]],
        nifti_version=1,
    )
    scan = make_scan(header=header, sha256="ab" * 32, research_only_confirmed_at=now_iso())
    db.insert_scan(scan)
    assert db.get_scan(scan.id) == scan
    updated = db.update_scan(scan.id, state="ready", decompressed_bytes=999, fixture_id="image1")
    assert updated.state == "ready"
    assert db.get_scan(scan.id) == updated
    assert db.get_scan("scan_missing") is None


def test_list_and_delete_scans(db: Database):
    s1 = make_scan()
    s2 = make_scan(state="ready")
    db.insert_scan(s1)
    db.insert_scan(s2)
    assert [s.id for s in db.list_scans()] == [s2.id, s1.id]
    assert [s.id for s in db.list_scans(state="ready")] == [s2.id]
    assert len(db.list_scans(limit=1)) == 1
    db.delete_scan(s1.id)
    assert db.get_scan(s1.id) is None


def test_job_round_trip_with_error_alias(db: Database):
    scan = make_scan()
    db.insert_scan(scan)
    job = make_job(
        scan.id,
        error=JobError(klass="input_error", message="bad"),
        timings=Timings(load_s=1, download_s=2, infer_s=3, postprocess_s=4, upload_s=5, total_s=15),
        cost_estimate_usd=0.02,
    )
    db.insert_job(job)
    assert db.get_job(job.id) == job
    payload = json.loads(job.model_dump_json())
    assert payload["error"] == {"class": "input_error", "message": "bad"}
    assert JobError.model_validate({"class": "x", "message": "y"}).klass == "x"


def test_result_round_trip(db: Database):
    scan = make_scan()
    db.insert_scan(scan)
    job = make_job(scan.id)
    db.insert_job(job)
    result = Result(
        job_id=job.id,
        findings=[
            Finding(key="肝囊肿", organ="Liver", finding="Liver cyst", prob=0.42),
            Finding(key="脾大", organ="Spleen", finding="Splenomegaly", prob=None),
        ],
        organs_scored=[
            OrganScored(
                organ="Liver", label=1, how="window", window_index=2, box_mm=[[0, 0, 0], [10, 10, 10]]
            )
        ],
        organs_not_found=["Spleen"],
        organ_stats={
            "Liver": OrganStats(voxels=10, ml=1.5, centroid_mm=[1, 2, 3], bbox_mm=[[0, 0, 0], [5, 5, 5]])
        },
        versions=Versions(
            code_commit="abc",
            checkpoint_sha256="d" * 64,
            torch="2.5.1",
            cuda="12.4",
            gpu="fake",
            image_id="im-1",
        ),
        artefacts=Artefacts(scores_json=f"jobs/{job.id}/scores.json", mask=f"jobs/{job.id}/mask.nii.gz"),
    )
    db.insert_result(result)
    assert db.get_result(job.id) == result
    assert db.get_result("job_missing") is None


def test_execution_round_trip(db: Database):
    ex = ChatExecution(
        execution_id=new_id("exec"),
        scan_id="scan_x",
        job_id=None,
        messages=[{"role": "user", "content": "hi"}],
        client_tools=[{"name": "get_view_state", "parameters": {"type": "object"}}],
        pending_tool_calls=[{"id": "call_1", "name": "get_view_state", "arguments": "{}"}],
        state="awaiting",
    )
    db.insert_execution(ex)
    assert db.get_execution(ex.execution_id) == ex
    updated = db.update_execution(ex.execution_id, iteration=2, state="done", pending_tool_calls=[])
    assert updated.iteration == 2 and updated.pending_tool_calls == []
    assert updated.updated_at >= ex.updated_at
    assert db.get_execution(ex.execution_id) == updated


def test_transitions_allowed(db: Database):
    scan = make_scan()
    db.insert_scan(scan)

    job = make_job(scan.id)
    db.insert_job(job)
    job = db.transition(job, "submitted", modal_call_id="fc-1")
    assert job.state == "submitted" and job.submitted_at and job.modal_call_id == "fc-1"
    job = db.transition(job, "done", cost_estimate_usd=0.1)
    assert job.finished_at and db.get_job(job.id) == job

    for path in (["failed"], ["cancelled"], ["submitted", "failed"], ["submitted", "cancelled"]):
        j = make_job(scan.id)
        db.insert_job(j)
        for state in path:
            j = db.transition(j, state)
        assert j.state == path[-1] and j.finished_at


def test_transition_retry(db: Database):
    scan = make_scan()
    db.insert_scan(scan)
    job = make_job(scan.id)
    db.insert_job(job)
    job = db.transition(job, "submitted", modal_call_id="fc-1")
    job = db.transition(job, "failed", error=JobError(klass="stuck", message="lost"))
    job = db.transition(job, "queued")
    assert job.state == "queued"
    assert job.modal_call_id is None and job.error is None
    assert job.submitted_at is None and job.finished_at is None
    job = db.transition(job, "cancelled")
    with pytest.raises(IllegalTransition):
        db.transition(job, "queued")


@pytest.mark.parametrize("target", ["queued", "submitted", "done", "failed", "cancelled"])
def test_transition_from_done_illegal(db: Database, target: str):
    scan = make_scan()
    db.insert_scan(scan)
    job = make_job(scan.id)
    db.insert_job(job)
    job = db.transition(job, "submitted")
    job = db.transition(job, "done")
    with pytest.raises(IllegalTransition):
        db.transition(job, target)


def test_transition_other_illegal(db: Database):
    scan = make_scan()
    db.insert_scan(scan)
    job = make_job(scan.id)
    db.insert_job(job)
    with pytest.raises(IllegalTransition):
        db.transition(job, "done")
    job = db.transition(job, "failed")
    with pytest.raises(IllegalTransition):
        db.transition(job, "submitted")


def test_job_filters(db: Database):
    s1, s2 = make_scan(), make_scan()
    db.insert_scan(s1)
    db.insert_scan(s2)
    j1 = make_job(s1.id)
    j2 = make_job(s1.id)
    j3 = make_job(s2.id)
    for j in (j1, j2, j3):
        db.insert_job(j)
    db.transition(j2, "submitted")
    assert {j.id for j in db.list_jobs(state="queued")} == {j1.id, j3.id}
    assert [j.id for j in db.list_jobs(state="submitted")] == [j2.id]
    assert {j.id for j in db.list_jobs(scan_id=s1.id)} == {j1.id, j2.id}
    assert [j.id for j in db.list_jobs(state="queued", scan_id=s2.id)] == [j3.id]
    assert {j.id for j in db.jobs_for_scan(s1.id)} == {j1.id, j2.id}
    assert db.latest_job_for_scan(s1.id).id == j2.id
    assert db.latest_job_for_scan("scan_none") is None
    assert len(db.list_jobs(limit=2)) == 2


def test_sum_cost_for_month(db: Database):
    scan = make_scan()
    db.insert_scan(scan)

    def finished(state: str, cost: float, finished_at: str) -> None:
        j = make_job(scan.id, state=state, cost_estimate_usd=cost, finished_at=finished_at)
        db.insert_job(j)

    finished("done", 0.10, "2026-10-02T10:00:00.000000Z")
    finished("failed", 0.05, "2026-10-30T23:59:59.000000Z")
    finished("cancelled", 1.00, "2026-10-03T10:00:00.000000Z")
    finished("done", 2.00, "2026-09-30T23:59:59.000000Z")
    db.insert_job(make_job(scan.id))  # queued, no cost
    # A retried job keeps its earlier attempt's cost and is dated by queued_at while in flight.
    db.insert_job(make_job(scan.id, state="queued", cost_estimate_usd=0.20, queued_at="2026-10-05T00:00:00.000000Z"))
    assert db.sum_cost_for_month("2026-10") == pytest.approx(1.35)
    assert db.sum_cost_for_month("2026-09") == pytest.approx(2.0)
    assert db.sum_cost_for_month("2026-01") == 0.0


def test_concurrent_writes(db: Database):
    scan = make_scan()
    db.insert_scan(scan)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            for _ in range(50):
                j = make_job(scan.id)
                db.insert_job(j)
                db.transition(j, "submitted")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(db.jobs_for_scan(scan.id)) == 100
