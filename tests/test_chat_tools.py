from __future__ import annotations

from pathlib import Path

import pytest

from radar_desk.chat.tools import TOOLS, is_error, openai_tools, run_tool, validate_args
from radar_desk.gpu.fake import canned_result
from radar_desk.radar import catalog
from radar_desk.records import Job
from radar_desk.services.scans import source_key
from synth import make_nifti

NAMES = [
    "list_scans", "get_scan", "score_scan", "get_job", "cancel_job",
    "get_scores", "get_organ", "compare_scores", "get_artifacts", "gpu_status",
]


def ready_scan(svc, tmp_path: Path, name="a.nii.gz"):
    path = make_nifti(tmp_path / name, shape=(24, 20, 10))
    ticket = svc.scans.begin_upload(path.name, path.stat().st_size, True)
    svc.storage.put_bytes(source_key(ticket.scan_id), path.read_bytes())
    return svc.scans.complete_upload(ticket.scan_id)


def done_job(svc, scan_id: str, job_id: str, result: dict | None = None, artefacts: bool = False) -> Job:
    job = Job(id=job_id, scan_id=scan_id, gpu_requested=["L4"])
    svc.db.insert_job(job)
    job = svc.db.transition(job, "submitted", modal_call_id="fc-x")
    data = result or canned_result(job.id)
    stored = svc.results.store(job, data)
    if artefacts:
        for key in stored.artefacts.model_dump().values():
            if key:
                svc.storage.put_bytes(key, b"x")
    return svc.db.transition(job, "done", cost_estimate_usd=0.1)


@pytest.fixture
def seeded(make_services, tmp_path):
    svc = make_services()
    scan = ready_scan(svc, tmp_path)
    job = done_job(svc, scan.id, "job_one", artefacts=True)
    return svc, scan, job


def test_registry_shape():
    assert [t.name for t in TOOLS] == NAMES
    for t in TOOLS:
        assert t.description
        assert t.parameters["type"] == "object"
        assert callable(t.run)
    converted = openai_tools(TOOLS)
    assert converted[0] == {
        "type": "function",
        "function": {"name": "list_scans", "description": TOOLS[0].description, "parameters": TOOLS[0].parameters},
    }


def test_validate_args():
    schema = TOOLS[NAMES.index("get_scores")].parameters
    assert validate_args(schema, {"job_id": "j", "min_prob": 0.5}) is None
    assert validate_args(schema, {"job_id": 3}) == "argument 'job_id' must be string"
    assert validate_args(schema, {"min_prob": True}) == "argument 'min_prob' must be number"
    assert validate_args(schema, {"bogus": 1}) == "unknown argument 'bogus'"
    assert validate_args(schema, "nope") == "arguments must be a JSON object"
    assert validate_args(schema, {"_raw": "{bad"}) == "arguments were not valid JSON"


def test_list_and_get_scan(seeded):
    svc, scan, job = seeded
    out = run_tool(svc, "list_scans", {"limit": 5})
    assert [s["id"] for s in out["scans"]] == [scan.id]
    row = out["scans"][0]
    assert row["header"]["dims"] == [24, 20, 10]
    assert row["latest_job"]["id"] == job.id and row["latest_job"]["state"] == "done"
    out = run_tool(svc, "get_scan", {"scan_id": scan.id})
    assert out["scan"]["id"] == scan.id
    assert [j["job_id"] for j in out["jobs"]] == [job.id]
    assert "fixture_id" in out["scan"] and "fixture_id" not in out


def test_score_scan_is_idempotent_and_get_cancel(make_services, tmp_path):
    svc = make_services()
    scan = ready_scan(svc, tmp_path)
    first = run_tool(svc, "score_scan", {"scan_id": scan.id})
    assert first["state"] == "queued" and first["job_id"]
    assert run_tool(svc, "score_scan", {"scan_id": scan.id})["job_id"] == first["job_id"]
    job = run_tool(svc, "get_job", {"job_id": first["job_id"]})
    assert job["state"] == "queued" and job["gpu_requested"] == ["L4"]
    assert {"hold_reason", "timings", "cost_estimate_usd", "gpu_used"} <= set(job)
    assert run_tool(svc, "cancel_job", {"job_id": first["job_id"]}) == {
        "job_id": first["job_id"], "state": "cancelled",
    }


def test_get_scores_by_job_and_scan_with_filters(seeded):
    svc, scan, job = seeded
    out = run_tool(svc, "get_scores", {"job_id": job.id})
    assert len(out["findings"]) == len(catalog.FINDINGS)
    assert set(out) == {"job_id", "findings", "organs_not_found", "versions"}
    by_scan = run_tool(svc, "get_scores", {"scan_id": scan.id, "min_prob": 0.5, "organ": "liver"})
    assert by_scan["job_id"] == job.id
    assert by_scan["findings"]
    assert all(f["organ"] == "Liver" and f["prob"] >= 0.5 for f in by_scan["findings"])
    err = run_tool(svc, "get_scores", {})
    assert err["error"]["class"] == "ServiceError" and "job_id or scan_id" in err["error"]["message"]


def test_get_scores_scan_without_done_job(make_services, tmp_path):
    svc = make_services()
    scan = ready_scan(svc, tmp_path)
    out = run_tool(svc, "get_scores", {"scan_id": scan.id})
    assert out["error"]["status"] == 404


def test_get_organ_compare_artifacts_gpu(seeded):
    svc, scan, job = seeded
    organ = run_tool(svc, "get_organ", {"job_id": job.id, "organ": "Liver"})
    assert organ["organ"] == "Liver" and organ["findings"]
    other = done_job(svc, scan.id, "job_two", result=canned_result("job_two", seed=1))
    cmp = run_tool(svc, "compare_scores", {"job_id": job.id, "against": other.id})
    assert {"deltas", "max_abs_delta", "over_tolerance"} <= set(cmp)
    arts = run_tool(svc, "get_artifacts", {"job_id": job.id})
    assert set(arts["urls"]) == {"mask", "scores_json", "scores_csv", "trace", "log"}
    assert all("/_storage/" in u for u in arts["urls"].values())
    arts_two = run_tool(svc, "get_artifacts", {"job_id": other.id})
    assert arts_two["urls"] == {} and set(arts_two["missing"]) == set(arts["urls"])
    gpu = run_tool(svc, "gpu_status", {})
    assert gpu["backend"] == "fake" and "budget_usd" in gpu


def test_errors_come_back_as_data(seeded, monkeypatch):
    svc, _, _ = seeded
    assert run_tool(svc, "get_job", {"job_id": "job_nope"})["error"] == {
        "class": "ServiceError", "message": "no job job_nope", "status": 404,
    }
    assert run_tool(svc, "get_job", {})["error"]["class"] == "invalid_arguments"
    assert run_tool(svc, "nope", {})["error"]["class"] == "unknown_tool"
    assert is_error(run_tool(svc, "nope", {}))
    assert not is_error(run_tool(svc, "get_job", {"job_id": "job_one"}))  # carries "error": None

    def boom(*a, **k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(svc.jobs, "get", boom)
    assert run_tool(svc, "get_job", {"job_id": "x"})["error"] == {
        "class": "RuntimeError", "message": "disk on fire",
    }
