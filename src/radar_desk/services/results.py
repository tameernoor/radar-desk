"""Stored results, per-organ views and comparisons against another job or a fixture reference."""

from __future__ import annotations

from typing import Any

from radar_desk.gpu.backend import artefact_keys
from radar_desk.radar import catalog
from radar_desk.records import Finding, Job, Result
from radar_desk.services.errors import ServiceError
from radar_desk.services.fixtures import FixtureService

JOB_COMPARE_TOLERANCE = 0.001


class ResultService:
    def __init__(self, db: Any, fixtures: FixtureService) -> None:
        self.db = db
        self.fixtures = fixtures

    def build(self, job: Job, data: dict) -> Result:
        """Check a worker result dict against the catalog and turn it into a Result."""
        raw = data.get("findings")
        if not isinstance(raw, list) or len(raw) != len(catalog.FINDINGS):
            count = len(raw) if isinstance(raw, list) else 0
            raise ServiceError(422, f"expected {len(catalog.FINDINGS)} findings, got {count}")
        findings = []
        for expected, item in zip(catalog.FINDINGS, raw, strict=True):
            if item.get("key") != expected["key"]:
                raise ServiceError(422, f"finding {expected['index']} is {item.get('key')!r}, "
                                        f"expected {expected['key']!r}")
            prob = item.get("prob")
            if prob is not None and not 0.0 <= float(prob) <= 1.0:
                raise ServiceError(422, f"probability out of range for {expected['key']}: {prob}")
            findings.append(Finding(key=expected["key"], organ=expected["organ"],
                                    finding=expected["finding"], prob=prob))
        blank = [o for o in catalog.scored_organ_names()
                 if all(f.prob is None for f in findings if f.organ == o)]
        given = list(data.get("organs_not_found") or [])
        not_found = given + [o for o in blank if o not in given]
        artefacts = {**artefact_keys(job.id), **{k: v for k, v in (data.get("artefacts") or {}).items() if v}}
        return Result(
            job_id=job.id,
            findings=findings,
            organs_scored=data.get("organs_scored") or [],
            organs_not_found=not_found,
            organ_stats=data.get("organ_stats") or {},
            versions=data.get("versions") or {},
            artefacts=artefacts,
        )

    def store(self, job: Job, data: dict) -> Result:
        """Validate and insert, or return the stored result when the job already has one."""
        existing = self.db.get_result(job.id)
        if existing is not None:
            return existing
        result = self.build(job, data)
        self.db.insert_result(result)
        return result

    def get(self, job_id: str) -> Result:
        if self.db.get_job(job_id) is None:
            raise ServiceError(404, f"no job {job_id}")
        result = self.db.get_result(job_id)
        if result is None:
            raise ServiceError(404, f"job {job_id} has no result yet")
        return result

    def get_organ(self, job_id: str, organ: str) -> dict:
        result = self.get(job_id)
        names = {o.lower(): o for o in catalog.scored_organ_names()}
        name = names.get(organ.strip().lower())
        if name is None:
            raise ServiceError(404, f"{organ!r} is not a scored organ")
        scored = next((s for s in result.organs_scored if s.organ == name), None)
        stats = result.organ_stats.get(name)
        return {
            "job_id": job_id,
            "organ": name,
            "label": catalog.label_for_organ(name),
            "found": name not in result.organs_not_found,
            "findings": [f.model_dump() for f in result.findings if f.organ == name],
            "stats": stats.model_dump() if stats else None,
            "scored": scored.model_dump() if scored else None,
        }

    def compare(self, job_id: str, against: str) -> dict:
        """Per-finding deltas against another job or the scan's fixture reference."""
        ours = {f.key: f.prob for f in self.get(job_id).findings}
        if against == "fixture":
            job = self.db.get_job(job_id)
            scan = self.db.get_scan(job.scan_id)
            fixture = self.fixtures.get(scan.fixture_id) if scan and scan.fixture_id else None
            if fixture is None:
                raise ServiceError(409, "this scan is not one of the known fixtures")
            for ref in fixture.get("references", []):
                reference = self.fixtures.load_reference(fixture["id"], ref)
                if reference is not None:
                    return _deltas(ours, reference, "fixture", ref["source"], ref["tolerance"])
            first = (fixture.get("references") or [{}])[0]
            return {"against": "fixture", "source": first.get("source"), "tolerance": first.get("tolerance"),
                    "pending": True, "deltas": [], "max_abs_delta": None, "over_tolerance": 0}
        if against == job_id:
            raise ServiceError(422, "compare a job against a different job")
        other = {f.key: f.prob for f in self.get(against).findings}
        return _deltas(ours, other, against, "job", JOB_COMPARE_TOLERANCE)


def _deltas(ours: dict, reference: dict, against: str, source: str, tolerance: float) -> dict:
    rows, worst, over = [], None, 0
    for f in catalog.FINDINGS:
        key = f["key"]
        if key not in reference:
            continue
        a, b = ours.get(key), reference[key]
        delta = None if a is None or b is None else a - b
        if delta is not None:
            worst = abs(delta) if worst is None else max(worst, abs(delta))
            over += abs(delta) > tolerance
        rows.append({"key": key, "ours": a, "reference": b, "delta": delta})
    return {"against": against, "source": source, "tolerance": tolerance, "pending": False,
            "deltas": rows, "max_abs_delta": worst, "over_tolerance": over}
