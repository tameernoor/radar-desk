from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from radar_desk.gpu.backend import artefact_keys
from radar_desk.gpu.fake import canned_result, synthesize_result
from radar_desk.radar import catalog
from radar_desk.records import Job, Scan
from radar_desk.services import ServiceError
from radar_desk.services import costs as costs_mod
from radar_desk.services.fixtures import DEFAULT_ROOT
from radar_desk.services.jobs import model_version
from radar_desk.services.scans import source_key
from synth import make_nifti, make_volume, oblique_affine, write_nifti

N = len(catalog.FINDINGS)


def put_source(svc, path: Path) -> str:
    ticket = svc.scans.begin_upload(path.name, path.stat().st_size, True)
    svc.storage.put_bytes(source_key(ticket.scan_id), path.read_bytes())
    return ticket.scan_id


def upload(svc, path: Path):
    return svc.scans.complete_upload(put_source(svc, path))


def ready_scan(svc, tmp_path: Path, name="a.nii.gz"):
    return upload(svc, make_nifti(tmp_path / name, shape=(24, 20, 10)))


def done_job(svc, scan_id: str, result: dict | None = None, cost: float | None = 0.1) -> Job:
    job = Job(id=f"job_{len(svc.db.list_jobs(limit=1000))}x", scan_id=scan_id, gpu_requested=["L4"])
    svc.db.insert_job(job)
    job = svc.db.transition(job, "submitted", modal_call_id="fc-x")
    svc.results.store(job, result or canned_result(job.id))
    return svc.db.transition(job, "done", cost_estimate_usd=cost)


# Scans


def test_begin_upload_rules(make_services):
    svc = make_services()
    with pytest.raises(ServiceError) as e:
        svc.scans.begin_upload("scan.dcm", 100, True)
    assert e.value.status == 415
    with pytest.raises(ServiceError) as e:
        svc.scans.begin_upload("scan.nii.gz", svc.settings.max_upload_bytes + 1, True)
    assert e.value.status == 413
    with pytest.raises(ServiceError) as e:
        svc.scans.begin_upload("scan.nii.gz", 100, False)
    assert e.value.status == 422
    ticket = svc.scans.begin_upload("dir/Scan.NII.GZ", 100, True)
    assert f"/_storage/scans/{ticket.scan_id}/source.nii.gz?" in ticket.put_url
    scan = svc.scans.get(ticket.scan_id)
    assert scan.state == "uploading" and scan.filename == "Scan.NII.GZ"
    assert scan.research_only_confirmed_at


def test_complete_upload_ready_with_fixture_id(make_services, tmp_path):
    path = make_nifti(tmp_path / "s.nii.gz", shape=(24, 20, 10))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    root = tmp_path / "fx"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"scans": {sha: {"id": "FIX1", "filename": "s.nii.gz"}}}))
    svc = make_services(fixtures_root=root)
    scan = upload(svc, path)
    assert scan.state == "ready"
    assert scan.sha256 == sha
    assert scan.fixture_id == "FIX1"
    assert scan.header.dims == [24, 20, 10]
    assert scan.decompressed_bytes == 352 + 24 * 20 * 10 * 2
    assert svc.scans.complete_upload(scan.id).state == "ready"  # idempotent


@pytest.mark.parametrize("kind", ["uint8", "oblique", "garbage"])
def test_complete_upload_rejects_and_deletes(make_services, tmp_path, kind):
    svc = make_services()
    path = tmp_path / "bad.nii.gz"
    if kind == "uint8":
        make_nifti(path, dtype=np.uint8)
    elif kind == "oblique":
        write_nifti(path, make_volume(), affine=oblique_affine())
    else:
        path.write_bytes(b"\x1f\x8bnot really gzip at all")
    scan_id = put_source(svc, path)
    with pytest.raises(ServiceError) as e:
        svc.scans.complete_upload(scan_id)
    assert e.value.status == 422
    scan = svc.db.get_scan(scan_id)
    assert scan.state == "rejected"
    assert scan.rejected_reason == e.value.detail
    assert not svc.storage.exists(source_key(scan_id))


def test_complete_before_object_is_409(make_services):
    svc = make_services()
    ticket = svc.scans.begin_upload("a.nii", 10, True)
    with pytest.raises(ServiceError) as e:
        svc.scans.complete_upload(ticket.scan_id)
    assert e.value.status == 409
    with pytest.raises(ServiceError) as e:
        svc.scans.complete_upload("scan_nope")
    assert e.value.status == 404


def test_delete_scan(make_services, tmp_path):
    svc = make_services()
    scan = ready_scan(svc, tmp_path)
    job = svc.jobs.create(scan.id)
    with pytest.raises(ServiceError) as e:
        svc.scans.delete(scan.id)
    assert e.value.status == 409
    svc.jobs.cancel(job.id)
    svc.scans.delete(scan.id)
    assert svc.db.get_scan(scan.id) is None
    assert not svc.storage.exists(source_key(scan.id))


def test_scan_view_latest_job(make_services, tmp_path):
    svc = make_services()
    scan = ready_scan(svc, tmp_path)
    assert svc.scan_view(scan.id)["latest_job"] is None
    result = canned_result("j")
    for i, f in enumerate(result["findings"]):
        f["prob"] = 0.9 if i < 3 else (0.5 if i == 3 else 0.1)
    job = done_job(svc, scan.id, result)
    latest = svc.scan_view(scan.id)["latest_job"]
    assert latest == {"id": job.id, "state": "done", "hold_reason": None, "positives_at_50": 4}


# Jobs


def test_create_job_is_idempotent_while_active(make_services, tmp_path):
    svc = make_services(radar_gpu="L4,A10")
    scan = ready_scan(svc, tmp_path)
    a = svc.jobs.create(scan.id)
    assert a.state == "queued" and a.gpu_requested == ["L4", "A10"]
    assert a.model_version.startswith("fake-radar-pretrain-")  # fake backend
    assert svc.jobs.create(scan.id).id == a.id
    svc.jobs.cancel(a.id)
    assert svc.jobs.create(scan.id).id != a.id


def test_create_job_needs_ready_scan(make_services):
    svc = make_services()
    ticket = svc.scans.begin_upload("a.nii", 10, True)
    with pytest.raises(ServiceError) as e:
        svc.jobs.create(ticket.scan_id)
    assert e.value.status == 409
    with pytest.raises(ServiceError) as e:
        svc.jobs.create("scan_missing")
    assert e.value.status == 404


def test_cancel_and_retry(make_services, tmp_path):
    svc = make_services()
    scan = ready_scan(svc, tmp_path)
    job = svc.jobs.create(scan.id)
    assert svc.jobs.cancel(job.id).state == "cancelled"
    with pytest.raises(ServiceError) as e:
        svc.jobs.retry(job.id)
    assert e.value.status == 409

    failed = svc.jobs.create(scan.id)
    failed = svc.db.transition(failed, "submitted", modal_call_id="fc-1")
    failed = svc.db.transition(failed, "failed", cost_estimate_usd=0.25)
    again = svc.jobs.retry(failed.id)
    assert again.state == "queued" and again.modal_call_id is None
    assert again.cost_estimate_usd == 0.25  # earlier attempt still counts

    svc.db.update_job(again.id, hold_reason="budget")
    assert svc.jobs.retry(again.id).hold_reason is None

    done = done_job(svc, scan.id)
    for op in (svc.jobs.retry, svc.jobs.cancel):
        with pytest.raises(ServiceError) as e:
            op(done.id)
        assert e.value.status == 409


def test_cancel_submitted_calls_backend(make_services, tmp_path):
    svc = make_services()
    scan = ready_scan(svc, tmp_path)
    job = svc.jobs.create(scan.id)
    svc.db.transition(job, "submitted", modal_call_id="fc-9")
    cancelled = svc.jobs.cancel(job.id)
    assert svc.backend.cancelled == ["fc-9"]
    assert cancelled.state == "cancelled" and cancelled.cost_estimate_usd > 0


def test_real_backend_model_version_has_no_fake_prefix(make_services):
    from radar_desk.gpu.modal_backend import ModalGpuBackend

    svc = make_services(backend=ModalGpuBackend(make_services().settings))
    assert svc.jobs.model_version.startswith("radar-pretrain-")


def test_create_job_is_atomic_across_threads(make_services, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    svc = make_services()
    scan = ready_scan(svc, tmp_path)
    with ThreadPoolExecutor(8) as pool:
        ids = set(pool.map(lambda _: svc.jobs.create(scan.id).id, range(16)))
    assert len(ids) == 1


def test_model_version(tmp_path):
    w = tmp_path / "weights.json"
    w.write_text(json.dumps({"files": [{"path": "checkpoint_radar_pretrain.pth", "sha256": "abcdef0123456789"}]}))
    assert model_version(w) == "radar-pretrain-abcdef012345"
    assert model_version(tmp_path / "missing.json") == "radar-pretrain-5e8b1b50b921"


# Results


def test_store_result_validates_and_fills_not_found(make_services, tmp_path):
    svc = make_services()
    scan = ready_scan(svc, tmp_path)
    job = svc.jobs.create(scan.id)
    short = canned_result(job.id)
    short["findings"] = short["findings"][:-1]
    with pytest.raises(ServiceError):
        svc.results.store(job, short)
    swapped = canned_result(job.id)
    swapped["findings"][0], swapped["findings"][1] = swapped["findings"][1], swapped["findings"][0]
    with pytest.raises(ServiceError):
        svc.results.store(job, swapped)

    data = canned_result(job.id)
    for f in data["findings"]:
        if f["organ"] == "Liver":
            f["prob"] = None
    result = svc.results.store(job, data)
    assert "Liver" in result.organs_not_found
    assert len(result.findings) == N
    assert result.artefacts.mask == f"jobs/{job.id}/mask.nii.gz"
    assert svc.results.get(job.id).job_id == job.id


def test_get_organ(make_services, tmp_path):
    svc = make_services()
    scan = ready_scan(svc, tmp_path)
    job = done_job(svc, scan.id)
    organ = svc.results.get_organ(job.id, "liver")
    assert organ["organ"] == "Liver"
    assert len(organ["findings"]) == len(catalog.findings_for_organ("Liver"))
    with pytest.raises(ServiceError) as e:
        svc.results.get_organ(job.id, "Spleenish")
    assert e.value.status == 404


def test_compare_two_jobs(make_services, tmp_path):
    svc = make_services()
    scan = ready_scan(svc, tmp_path)
    a = done_job(svc, scan.id, canned_result("a", seed=1))
    b_data = canned_result("b", seed=1)
    b_data["findings"][5]["prob"] += 0.01
    b = done_job(svc, scan.id, b_data)
    out = svc.results.compare(a.id, b.id)
    assert out["source"] == "job" and out["pending"] is False
    assert len(out["deltas"]) == N
    assert out["max_abs_delta"] == pytest.approx(0.01)
    assert out["over_tolerance"] == 1


def _fixture_root(tmp_path, refs, files=None):
    root = tmp_path / "fx"
    root.mkdir(exist_ok=True)
    manifest = {"scans": {"sha-1": {"id": "FIX1", "filename": "f.nii.gz", "references": refs}}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    for rel, content in (files or {}).items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(content)
    return root


def _fixture_scan(svc):
    scan = Scan(id="scan_fx", filename="f.nii.gz", size_bytes=1, state="ready", sha256="sha-1", fixture_id="FIX1")
    svc.db.insert_scan(scan)
    return scan


REFS = [
    {"source": "radar-web", "file": "expected/radar-web/FIX1.json", "tolerance": 0.001},
    {"source": "tally", "file": "expected/tally.json", "tolerance": 0.01},
]


def test_compare_fixture_tally_only_listed_keys(make_services, tmp_path):
    tally = {"tolerance": 0.01, "scans": {"FIX1": {"Liver_Periportal edema": 0.5, "Aorta_Aortic dissection": 0.2}}}
    root = _fixture_root(tmp_path, REFS, {"expected/tally.json": json.dumps(tally)})
    svc = make_services(fixtures_root=root)
    data = canned_result("j")
    keys = {catalog.english_to_key(k) for k in tally["scans"]["FIX1"]}
    for f in data["findings"]:
        if f["key"] == catalog.english_to_key("Aorta_Aortic dissection"):
            f["prob"] = 0.25
    job = done_job(svc, _fixture_scan(svc).id, data)
    out = svc.results.compare(job.id, "fixture")
    assert out["source"] == "tally" and out["tolerance"] == 0.01 and out["pending"] is False
    assert {d["key"] for d in out["deltas"]} == keys
    aorta = next(d for d in out["deltas"] if d["key"] == catalog.english_to_key("Aorta_Aortic dissection"))
    assert aorta["delta"] == pytest.approx(0.05)


def test_compare_fixture_prefers_radar_web(make_services, tmp_path):
    findings = [{"key": f["key"], "organ": f["organ"], "finding": f["finding"], "prob": 0.3}
                for f in catalog.FINDINGS]
    root = _fixture_root(tmp_path, REFS, {
        "expected/radar-web/FIX1.json": json.dumps({"findings": findings, "organs_not_found": []}),
        "expected/tally.json": json.dumps({"scans": {"FIX1": {"Liver_Periportal edema": 0.5}}}),
    })
    svc = make_services(fixtures_root=root)
    job = done_job(svc, _fixture_scan(svc).id)
    out = svc.results.compare(job.id, "fixture")
    assert out["source"] == "radar-web" and len(out["deltas"]) == N


def test_compare_fixture_pending_and_not_a_fixture(make_services, tmp_path):
    svc = make_services(fixtures_root=_fixture_root(tmp_path, REFS))
    job = done_job(svc, _fixture_scan(svc).id)
    out = svc.results.compare(job.id, "fixture")
    assert out["pending"] is True and out["deltas"] == [] and out["source"] == "radar-web"
    other = done_job(svc, ready_scan(svc, tmp_path).id)
    with pytest.raises(ServiceError) as e:
        svc.results.compare(other.id, "fixture")
    assert e.value.status == 409


def _damo_demo_result(job_id: str) -> dict:
    with (DEFAULT_ROOT / "expected" / "damo-demo.csv").open(encoding="utf-8-sig", newline="") as fh:
        row = list(csv.reader(fh))[1]
    data = canned_result(job_id)
    for f, value in zip(data["findings"], row[1:], strict=True):
        f["prob"] = float(value) if value else None
    return data


def test_compare_damo_demo_case(make_services):
    svc = make_services()
    sha = next(s for s, e in svc.fixtures._by_sha.items() if e["id"] == "AC423ccbe")
    svc.db.insert_scan(Scan(id="scan_demo", filename="AC423ccbe.nii.gz", size_bytes=1, state="ready",
                            sha256=sha, fixture_id="AC423ccbe"))
    job = done_job(svc, "scan_demo", _damo_demo_result("j"))
    out = svc.results.compare(job.id, "fixture")
    assert out["source"] == "damo-demo" and out["max_abs_delta"] == 0.0 and out["over_tolerance"] == 0


# Exports


def test_scores_csv_matches_upstream_bytes(make_services):
    svc = make_services()
    svc.db.insert_scan(Scan(id="scan_demo", filename="AC423ccbe.nii.gz", size_bytes=1, state="ready"))
    job = done_job(svc, "scan_demo", _damo_demo_result("j"))
    ours = svc.exports.scores_csv(job.id)
    assert ours == (DEFAULT_ROOT / "expected" / "damo-demo.csv").read_bytes()


def test_scores_csv_blanks_and_header(make_services, tmp_path):
    svc = make_services()
    scan = ready_scan(svc, tmp_path, "blank.nii.gz")
    data = canned_result("j")
    data["findings"][0]["prob"] = None
    job = done_job(svc, scan.id, data)
    raw = svc.exports.scores_csv(job.id)
    assert raw.startswith(b"\xef\xbb\xbf")
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"))))
    assert rows[0] == catalog.csv_header()
    assert rows[1][0] == "blank.nii.gz" and rows[1][1] == ""


def test_scores_json_and_export_all(make_services, tmp_path):
    svc = make_services()
    scan = ready_scan(svc, tmp_path)
    a = done_job(svc, scan.id)
    done_job(svc, scan.id)
    svc.jobs.create(scan.id)  # queued, not exported
    body = svc.exports.scores_json(a.id)
    for key in ("findings", "organs_not_found", "organs_scored", "organ_stats", "versions", "timings",
                "disclaimer", "model"):
        assert key in body
    rows = list(csv.reader(io.StringIO(svc.exports.export_all_csv().decode("utf-8-sig"))))
    assert len(rows) == 3 and rows[1][0] == scan.filename


def test_export_all_skips_fake_results_for_a_real_backend(make_services, tmp_path):
    from radar_desk.gpu.modal_backend import ModalGpuBackend

    svc = make_services(backend=ModalGpuBackend(make_services().settings))
    scan = ready_scan(svc, tmp_path)
    done_job(svc, scan.id)  # versions.gpu "fake"
    done_job(svc, scan.id, canned_result("r", gpu="NVIDIA L4"))
    rows = list(csv.reader(io.StringIO(svc.exports.export_all_csv().decode("utf-8-sig"))))
    assert len(rows) == 2


def test_artefact_url(make_services, tmp_path):
    svc = make_services()
    scan = ready_scan(svc, tmp_path)
    job = done_job(svc, scan.id)
    with pytest.raises(ServiceError) as e:
        svc.exports.artefact_url(job.id, "mask")
    assert e.value.status == 404
    svc.storage.put_bytes(artefact_keys(job.id)["mask"], b"x")
    assert "/_storage/jobs/" in svc.exports.artefact_url(job.id, "mask")
    with pytest.raises(ServiceError):
        svc.exports.artefact_url(job.id, "secrets")


def test_synthesize_result_writes_artefacts(make_services, tmp_path):
    svc = make_services()
    scan = ready_scan(svc, tmp_path)
    job = svc.jobs.create(scan.id)
    data = synthesize_result(job, scan, svc.storage, catalog)
    assert data["versions"]["gpu"] == "fake" and data["ok"] is True
    assert data == synthesize_result(job, scan, svc.storage, catalog)  # deterministic
    for key in artefact_keys(job.id).values():
        assert svc.storage.exists(key)
    result = svc.results.build(job, data)
    assert len(result.organs_scored) + len(result.organs_not_found) == len(catalog.SCORED_ORGANS)


# Costs


def test_cost_estimate_and_budget(make_services):
    s = make_services(radar_gpu="L4,A10").settings
    assert costs_mod.estimate({"total_s": 100.0}, "L4", s) == pytest.approx(220 * 0.000222)
    assert costs_mod.estimate({"total_s": 100.0}, "A10", s) == pytest.approx(220 * 0.000306)
    assert costs_mod.estimate(None, "NVIDIA something", s, elapsed_s=30) == pytest.approx(150 * 0.000222)
    # elapsed counts boot and Modal's retry, so the larger of the two wins
    assert costs_mod.estimate({"total_s": 100.0}, "L4", s, elapsed_s=400) == pytest.approx(520 * 0.000222)
    assert costs_mod.estimate({"total_s": 100.0}, "L4", s, elapsed_s=40) == pytest.approx(220 * 0.000222)
    worst = (2 * 1800 + 120) * 0.000306  # dearest of L4 and A10
    assert costs_mod.worst_case(s) == pytest.approx(worst)
    assert costs_mod.budget_allows(10.0 - worst, s)
    assert not costs_mod.budget_allows(10.0 - worst + 0.01, s)
    assert not costs_mod.budget_allows(0.0, s, in_flight=10.0)


@pytest.mark.parametrize(("device", "gpu"), [
    ("NVIDIA L4", "L4"), ("NVIDIA A10G", "A10"), ("NVIDIA L40S", "L40S"),
    ("NVIDIA A100-SXM4-40GB", "A100-40GB"), ("NVIDIA A100-SXM4-80GB", "A100-80GB"),
    ("NVIDIA H100 80GB HBM3", "H100"), ("fake", None),
])
def test_gpu_type_from_device(device, gpu):
    assert costs_mod.gpu_type_from_device(device) == gpu
