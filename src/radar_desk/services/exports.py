"""Exports from the stored result, so they work without the bucket.

The CSV matches upstream's `RADAR_infer_results_*.csv`: utf-8-sig, `file_name` then the 146
`key (English)` columns, probabilities as Python float repr, blanks for unscored findings.
"""

from __future__ import annotations

import csv
import io
from typing import Any

from radar_desk.radar import catalog
from radar_desk.records import Result
from radar_desk.services.errors import ServiceError

LICENCE = ("Model: RADAR by Alibaba DAMO Academy, CC BY-NC-SA 4.0. Paper: An expert-level generalist AI for abdominal "
           "CT diagnosis, Science 393(6817), eaec6129, doi 10.1126/science.aec6129. "
           "Weights: huggingface.co/radar-generalist/RADAR.")
DISCLAIMER = (
    "Research use only. RADAR output is not a diagnosis. 50% is a display threshold, not a "
    "clinical cut-off, and a quiet organ is not a normal organ."
)
ARTEFACT_URL_S = 60 * 60


def csv_row(filename: str, probs: list[float | None]) -> list[str]:
    return [filename] + ["" if p is None else repr(float(p)) for p in probs]


def csv_bytes(rows: list[list[str]]) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(catalog.csv_header())
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")


def result_row(filename: str, result: Result) -> list[str]:
    by_key = {f.key: f.prob for f in result.findings}
    return csv_row(filename, [by_key.get(f["key"]) for f in catalog.FINDINGS])


class ExportService:
    def __init__(self, db: Any, storage: Any, include_fake: bool = False) -> None:
        self.db = db
        self.storage = storage
        self.include_fake = include_fake

    def _job_result(self, job_id: str):
        job = self.db.get_job(job_id)
        if job is None:
            raise ServiceError(404, f"no job {job_id}")
        result = self.db.get_result(job_id)
        if result is None:
            raise ServiceError(404, f"job {job_id} has no result yet")
        return job, result

    def _filename(self, scan_id: str) -> str:
        scan = self.db.get_scan(scan_id)
        return scan.filename if scan else scan_id

    def scores_csv(self, job_id: str) -> bytes:
        job, result = self._job_result(job_id)
        return csv_bytes([result_row(self._filename(job.scan_id), result)])

    def scores_json(self, job_id: str) -> dict:
        job, result = self._job_result(job_id)
        return {
            "job_id": job.id,
            "scan_id": job.scan_id,
            "file_name": self._filename(job.scan_id),
            "model": job.model_version,
            "findings": [f.model_dump() for f in result.findings],
            "organs_not_found": result.organs_not_found,
            "organs_scored": [o.model_dump() for o in result.organs_scored],
            "organ_stats": {k: v.model_dump() for k, v in result.organ_stats.items()},
            "versions": result.versions.model_dump(),
            "timings": job.timings.model_dump() if job.timings else None,
            "disclaimer": DISCLAIMER,
        "licence": LICENCE,
        }

    def export_all_csv(self) -> bytes:
        """One row per done job with a result, oldest first. Fake results are left out unless the
        backend is the fake one."""
        rows = []
        for job in reversed(self.db.list_jobs(state="done", limit=1_000_000)):
            result = self.db.get_result(job.id)
            if result is not None and (self.include_fake or result.versions.gpu != "fake"):
                rows.append(result_row(self._filename(job.scan_id), result))
        return csv_bytes(rows)

    def artefact_url(self, job_id: str, name: str) -> str:
        """A one-hour GET URL for one artefact (mask, trace, log, scores_json, scores_csv)."""
        _, result = self._job_result(job_id)
        key = getattr(result.artefacts, name, None) if name in type(result.artefacts).model_fields else None
        if not key:
            raise ServiceError(404, f"job {job_id} has no {name} artefact")
        if not self.storage.exists(key):
            raise ServiceError(404, f"the {name} artefact of job {job_id} is not in the bucket")
        return self.storage.get_url(key, ARTEFACT_URL_S)
