"""Scoring jobs: create, cancel, retry, logs (design.md, Data flow 2 and Errors)."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from radar_desk.records import Job, new_id
from radar_desk.services.costs import CostService, iso_at
from radar_desk.services.errors import ServiceError

log = logging.getLogger(__name__)

WEIGHTS_JSON = Path(__file__).resolve().parents[3] / "worker" / "weights.json"
FALLBACK_CHECKPOINT_PREFIX = "5e8b1b50b921"
CHECKPOINT_NAME = "checkpoint_radar_pretrain.pth"
ACTIVE = ("queued", "submitted")


def model_version(weights_json: Path = WEIGHTS_JSON) -> str:
    """ "radar-pretrain-" plus the first 12 hex digits of the checkpoint sha256."""
    prefix = FALLBACK_CHECKPOINT_PREFIX
    try:
        files = json.loads(weights_json.read_text("utf-8"))["files"]
        sha = next(f["sha256"] for f in files if f["path"].endswith(CHECKPOINT_NAME))
        if sha:
            prefix = sha[:12]
    except (OSError, ValueError, KeyError, StopIteration, TypeError):
        pass
    return f"radar-pretrain-{prefix}"


class JobService:
    def __init__(
        self,
        db: Any,
        settings: Any,
        backend: Any,
        costs: CostService,
        clock: Callable[[], float] = time.time,
        version: str | None = None,
    ) -> None:
        self.db = db
        self.settings = settings
        self.backend = backend
        self.costs = costs
        self.clock = clock
        version = version or model_version()
        # Fake results must never pass for real ones, so they get their own model version.
        self.model_version = f"fake-{version}" if getattr(backend, "name", None) == "fake" else version
        self._create_lock = threading.Lock()

    def get(self, job_id: str) -> Job:
        job = self.db.get_job(job_id)
        if job is None:
            raise ServiceError(404, f"no job {job_id}")
        return job

    def list(self, state: str | None = None, scan_id: str | None = None, limit: int = 100) -> list[Job]:
        return self.db.list_jobs(state=state, scan_id=scan_id, limit=limit)

    def create(self, scan_id: str) -> Job:
        """Queue a job, or return the queued or submitted one for this scan and model version."""
        scan = self.db.get_scan(scan_id)
        if scan is None:
            raise ServiceError(404, f"no scan {scan_id}")
        if scan.state != "ready":
            raise ServiceError(409, f"scan {scan_id} is {scan.state}, not ready")
        with self._create_lock:  # the check and the insert as one step
            for job in self.db.jobs_for_scan(scan_id):
                if job.state in ACTIVE and job.model_version == self.model_version:
                    return job
            job = Job(
                id=new_id("job"),
                scan_id=scan_id,
                gpu_requested=self.settings.gpu_list,
                model_version=self.model_version,
            )
            self.db.insert_job(job)
            return job

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.state == "queued":
            return self.db.transition(job, "cancelled", finished_at=iso_at(self.clock()))
        if job.state != "submitted":
            raise ServiceError(409, f"job {job_id} is {job.state} and cannot be cancelled")
        if job.modal_call_id:
            try:
                self.backend.cancel(job.modal_call_id)
            except Exception as exc:
                raise ServiceError(502, f"could not cancel the GPU call: {exc}") from exc
        now = self.clock()
        cost = (job.cost_estimate_usd or 0.0) + self.costs.attempt_cost(job, None, None, now)
        return self.db.transition(job, "cancelled", finished_at=iso_at(now), cost_estimate_usd=cost)

    def retry(self, job_id: str) -> Job:
        """Re-queue a failed job, or release a held one. Earlier attempts' cost stays."""
        job = self.get(job_id)
        if job.state == "failed":
            return self.db.transition(job, "queued", queued_at=iso_at(self.clock()))
        if job.state == "queued":
            return self.db.update_job(job_id, hold_reason=None) if job.hold_reason else job
        raise ServiceError(409, f"job {job_id} is {job.state} and cannot be retried")

    def logs(self, job_id: str, lines: int = 200) -> str:
        job = self.get(job_id)
        if not job.modal_call_id:
            return ""
        try:
            return self.backend.logs(job.modal_call_id, lines)
        except Exception as exc:
            raise ServiceError(502, f"could not read the logs: {exc}") from exc

    def latest_summary(self, scan_id: str) -> dict | None:
        """{id, state, hold_reason, positives_at_50} of the scan's newest job, or None."""
        job = self.db.latest_job_for_scan(scan_id)
        if job is None:
            return None
        result = self.db.get_result(job.id)
        positives = None
        if result is not None:
            positives = sum(1 for f in result.findings if f.prob is not None and f.prob >= 0.5)
        return {"id": job.id, "state": job.state, "hold_reason": job.hold_reason, "positives_at_50": positives}
