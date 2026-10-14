"""The Readers tab's rows: report and NV reads, RADAR's best score per row, NV's coverage, asked answers."""

from __future__ import annotations

import gzip

import nibabel as nib
import numpy as np
import pytest

from api_client import api_client
from radar_desk.gpu.fake import canned_result
from radar_desk.radar import catalog
from radar_desk.readers.labels import LABELS
from radar_desk.records import Result
from radar_desk.services.nvreason import OFF
from radar_desk.services.readers import build_rows, differs, organ_coverage
from readers_seed import NV_READ, REPORT_READ, SYNTHETIC_REPORT
from synth import affine_for
from test_chat_tools import done_job


def test_coverage_follows_the_mask_orientation():
    """Review Focus 3: a LAS mask (x runs from 0 down to -38 mm) against RAS boxes."""
    labels = np.zeros((20, 20, 20), dtype=np.uint8)
    liver, lung = catalog.label_for_organ("Liver"), catalog.label_for_organ("Lung")
    labels[:, :, :10] = liver
    labels[:, :, 10:] = lung
    gz = gzip.compress(nib.Nifti1Image(labels, affine_for((2.0, 2.0, 2.0), "LAS")).to_bytes())
    assert organ_coverage(gz, [[-40.0, -1.0, -1.0], [1.0, 39.0, 19.0]]) == {"Liver": 1.0, "Lung": 0.0}
    assert organ_coverage(gz, [[-19.0, -1.0, -1.0], [1.0, 39.0, 19.0]])["Liver"] == 0.5
    assert organ_coverage(gz, [[-1.0, -1.0, -1.0], [40.0, 39.0, 19.0]])["Liver"] == pytest.approx(0.05)
    plain = nib.Nifti1Image(labels, affine_for((2.0, 2.0, 2.0), "LAS")).to_bytes()
    assert organ_coverage(plain, [[-40.0, -1.0, -1.0], [1.0, 39.0, 19.0]])["Liver"] == 1.0


def test_differs_only_on_an_opposite_yes_or_no():
    assert differs("Yes.", "absent") is True
    assert differs("No", "present") is True
    assert differs("**Yes**", "absent") is True
    assert differs("Yes", "present") is False
    assert differs("Yes.", "not_mentioned") is False
    assert differs("The liver shows a lesion.", "absent") is False
    assert differs("", None) is False


def _result() -> Result:
    data = canned_result("job_r")
    for f in data["findings"]:
        f["prob"] = None if f["organ"] == "Esophagus" else 0.1
    by_key = {f["key"]: f for f in data["findings"]}
    by_key[catalog.english_to_key("Liver_Cyst")]["prob"] = 0.8
    return Result(job_id="job_r", findings=data["findings"], organs_not_found=["Esophagus"])


def test_rows_from_reads_radar_coverage_and_asked_answers():
    import json

    report = json.loads(REPORT_READ)["labels"]
    nv = json.loads(NV_READ)["labels"]
    asked = [{"label": "ascites", "question": "Is there free fluid? Answer only Yes or No.", "answer": "Yes.",
              "seconds": 12.1, "warm": "warm", "asked_at": "2026-10-15T12:00:00.000000Z"},
             {"label": None, "question": "Describe the scan.", "answer": "A long answer."}]
    rows = {r["id"]: r for r in build_rows(LABELS, {"labels": report}, {"labels": nv}, _result(),
                                           {"Liver": 1.0, "Kidney": 0.3}, asked)}
    assert len(rows) == 22
    liver = rows["liver_lesion"]
    assert liver["radar"]["scored"] is True and liver["radar"]["max_prob"] == 0.8
    assert len(liver["radar"]["findings"]) == 8
    assert liver["nv"] == {"state": "present", "quote": "A small hypodense lesion in the right lobe.",
                           "coverage": 1.0, "in_view": True}
    assert rows["renal_cyst"]["nv"]["in_view"] is False and rows["renal_cyst"]["nv"]["coverage"] == 0.3
    assert rows["hiatal_hernia"]["radar"] == {"scored": False, "max_prob": None,
                                              "findings": rows["hiatal_hernia"]["radar"]["findings"]}
    assert rows["ascites"]["radar"]["scored"] is False and rows["ascites"]["radar"]["findings"] == []
    assert rows["lymphadenopathy"]["radar"]["scored"] is True
    assert rows["ascites"]["nv"]["in_view"] is None
    assert rows["ascites"]["report"] == {"state": "present", "quote": "Small volume ascites in the pelvis."}
    assert rows["ascites"]["asked"] == [{"question": "Is there free fluid? Answer only Yes or No.", "answer": "Yes.",
                                        "seconds": 12.1, "warm": "warm", "asked_at": "2026-10-15T12:00:00.000000Z",
                                        "differs": True}]
    assert rows["liver_lesion"]["asked"] == []


def test_rows_without_reads_or_radar():
    rows = build_rows(LABELS, None, None, None, None, [])
    assert all(r["report"] is None and r["nv"] is None and r["radar"]["scored"] is False for r in rows)


def test_view_of_an_old_scan_without_nv_or_report(readers_world):
    """Review Focus 1: scored before this feature, NV off, no report."""
    w = readers_world(llm=False, enabled=False)
    done_job(w.svc, w.scan.id, "job_old")
    view = w.svc.readers.view(w.scan.id)
    assert view["has_report"] is False and view["report"]["status"] == "none"
    assert view["nv"] == {"unavailable": OFF, "job": None, "read": {"status": "none", "error": None}, "crop": None,
                          "fake": False, "warm": False, "asking": [], "ask_errors": {}}
    assert view["radar"] == {"job_id": "job_old", "coverage": False}
    assert len(view["rows"]) == 22
    assert all(r["report"] is None and r["nv"] is None for r in view["rows"])
    assert any(r["radar"]["scored"] for r in view["rows"])
    assert view["other_mentions"] == []


def test_view_with_everything(readers_world):
    w = readers_world([REPORT_READ, NV_READ])
    w.svc.scans.put_report(w.scan.id, SYNTHETIC_REPORT)
    w.svc.jobs.create(w.scan.id)
    for _ in range(3):
        w.poller.tick()
    view = w.svc.readers.view(w.scan.id)
    assert view["has_report"] is True
    assert view["report"]["status"] == "ready" and view["nv"]["read"]["status"] == "ready"
    assert view["nv"]["job"]["state"] == "done" and view["nv"]["fake"] is True
    assert view["nv"]["crop"]["box_mm"] and view["radar"]["coverage"] is True
    rows = {r["id"]: r for r in view["rows"]}
    assert rows["ascites"]["report"]["state"] == "present" and rows["ascites"]["nv"]["state"] == "absent"
    assert any(isinstance(r["nv"]["coverage"], float) for r in view["rows"] if r["nv"])
    assert {"reader": "report", "quote": "Degenerative change of the lumbar spine."} in view["other_mentions"]


def test_readers_route(readers_world):
    w = readers_world(llm=False)
    client = api_client(w.svc)
    r = client.get(f"/scans/{w.scan.id}/readers")
    assert r.status_code == 200, r.text
    assert r.json()["scan_id"] == w.scan.id and len(r.json()["rows"]) == 22
    assert client.get("/scans/scan_missing/readers").status_code == 404


def test_slabbed_coverage_equals_a_one_shot_computation():
    rng = np.random.default_rng(3)
    labels = rng.integers(0, 4, size=(9, 7, 37)).astype(np.uint8)  # spans five slabs of 8
    for organ in ("Liver", "Lung", "Kidney"):
        labels[labels == 3] = catalog.label_for_organ(organ)
    labels[labels == 1] = catalog.label_for_organ("Liver")
    labels[labels == 2] = catalog.label_for_organ("Lung")
    aff = affine_for((2.0, 3.0, 1.5), "LAS")
    box = [[-10.0, 2.0, 5.0], [3.0, 15.0, 40.0]]
    idx = np.nonzero(labels)
    lab = labels[idx]
    world = aff[:3, :3] @ np.vstack(idx).astype(float) + aff[:3, 3:4]
    inside = np.all((world >= np.array(box[0])[:, None]) & (world <= np.array(box[1])[:, None]), axis=0)
    expected = {o["organ"]: round(float((inside & (lab == o["label"])).sum() / (lab == o["label"]).sum()), 3)
                for o in catalog.SCORED_ORGANS if (lab == o["label"]).any()}
    got = organ_coverage(nib.Nifti1Image(labels, aff).to_bytes(), box)
    assert got == expected and len(got) >= 2


def test_a_row_scores_from_its_other_organ_when_its_primary_organ_is_not_found():
    data = canned_result("job_r")
    for f in data["findings"]:
        f["prob"] = None
    by_key = {f["key"]: f for f in data["findings"]}
    by_key["小肠_梗阻"]["prob"] = None
    by_key["十二指肠_梗阻"]["prob"] = 0.6
    result = Result(job_id="job_r", findings=data["findings"], organs_not_found=["Small bowel"])
    row = {r["id"]: r for r in build_rows(LABELS, None, None, result, None, [])}["sb_obstruction"]
    assert row["radar"]["scored"] is True and row["radar"]["max_prob"] == 0.6


def test_a_corrupt_mask_leaves_coverage_empty_instead_of_failing(readers_world):
    w = readers_world([REPORT_READ, NV_READ])
    w.svc.scans.put_report(w.scan.id, SYNTHETIC_REPORT)
    w.svc.jobs.create(w.scan.id)
    for _ in range(3):
        w.poller.tick()
    result = w.svc.readers.context(w.scan.id, w.svc.readers.plan(w.scan.id))["result"]
    w.svc.storage.put_bytes(result.artefacts.mask, b"not a nifti")
    w.svc.readers._coverage.clear()
    view = w.svc.readers.view(w.scan.id)
    assert view["radar"]["coverage"] is False
    assert all(r["nv"]["coverage"] is None for r in view["rows"] if r["nv"])
