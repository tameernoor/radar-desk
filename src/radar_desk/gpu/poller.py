"""The poller: spawns queued jobs and collects submitted ones (design.md, Job states and the poller).

Every decision is in `tick()`, one synchronous pass over the job rows, so a restart resumes from the
database and tests drive it with a fake clock. `run()` calls it in a thread every poll interval.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

from radar_desk.gpu.backend import Errored, Finished, Pending, artefact_keys
from radar_desk.records import Job, JobError, Timings
from radar_desk.services.costs import gpu_type_from_device, iso_at, parse_iso
from radar_desk.services.errors import ServiceError
from radar_desk.services.scans import source_key

log = logging.getLogger(__name__)

URL_TTL_S = 12 * 60 * 60
STUCK_MARGIN_S = 300
ALL = 100_000


class Poller:
    def __init__(self, services: Any, backend: Any = None, clock: Callable[[], float] = time.time) -> None:
        self.services = services
        self.backend = backend if backend is not None else services.backend
        self.clock = clock

    @property
    def db(self) -> Any:
        return self.services.db

    @property
    def settings(self) -> Any:
        return self.services.settings

    def tick(self) -> None:
        """One pass: collect submitted jobs first, then spawn the oldest queued job if none is running.

        Jobs run one at a time. The function has max_containers=1 anyway, and a job queued behind
        another must not start its stuck clock. On a pull backend the workers claim jobs themselves,
        so the poller only runs the job desk's lease and timeout checks.
        """
        if getattr(self.backend, "pull", False):
            self.services.workers.tick(self.clock())
            return
        for job in reversed(self.db.list_jobs(state="submitted", limit=ALL)):
            try:
                self._check_submitted(job)
            except Exception:
                log.exception("poller: checking job %s failed", job.id)
        if self.db.list_jobs(state="submitted", limit=1):
            return
        for job in reversed(self.db.list_jobs(state="queued", limit=ALL)):
            try:
                if self._try_spawn(job):
                    return
            except Exception:
                log.exception("poller: spawning job %s failed", job.id)

    async def run(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            try:
                await asyncio.to_thread(self.tick)
            except Exception:
                log.exception("poller tick failed")
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.settings.gpu_poll_interval_s)
            except TimeoutError:
                pass

    # Queued

    def _hold(self, job: Job, reason: str, error: JobError | None = None) -> None:
        if job.hold_reason != reason or job.error != error:
            self.db.update_job(job.id, hold_reason=reason, error=error)

    def _try_spawn(self, job: Job) -> bool:
        """Spawn one queued job. True when it is now submitted."""
        now = self.clock()
        scan = self.db.get_scan(job.scan_id)
        if scan is None or scan.state != "ready":
            self.db.transition(job, "failed", finished_at=iso_at(now),
                               error=JobError(klass="input_error", message="the scan is gone or not ready"))
            return False
        if not self.services.costs.budget_allows_now(now):
            self._hold(job, "budget")
            return False
        keys = artefact_keys(job.id)
        storage = self.services.storage
        # Presigned URLs for the URL adapters, volume:// paths for the Modal Volume.
        source_url = storage.worker_ref(source_key(scan.id), "GET", URL_TTL_S)
        put_urls = {name: storage.worker_ref(key, "PUT", URL_TTL_S) for name, key in keys.items()}
        # A spawn that raises after Modal accepted the call (a response timeout, say) is held as
        # spawn_error and spawned again next tick, which can duplicate a run.
        try:
            call_id = self.backend.spawn(job, source_url, put_urls, keys)
        except Exception as exc:  # noqa: BLE001 - held and retried next tick
            log.warning("poller: spawn of job %s failed: %s", job.id, exc)
            self._hold(job, "spawn_error", JobError(klass=type(exc).__name__, message=str(exc)))
            return False
        try:
            self.db.transition(
                job, "submitted",
                modal_call_id=call_id,
                gpu_requested=self.settings.gpu_list,
                submitted_at=iso_at(now),
                hold_reason=None,
                error=None,
            )
        except Exception:
            # The call runs but the row did not record it (a concurrent cancel, a database error).
            # Cancel it so it is not orphaned, and let tick() log the failure.
            log.error("poller: job %s spawned as call %s but could not be marked submitted", job.id, call_id)
            try:
                self.backend.cancel(call_id)
            except Exception as exc:  # noqa: BLE001 - best effort, already logged above
                log.warning("poller: cancel of orphaned call %s failed: %s", call_id, exc)
            raise
        return True

    # Submitted

    def _finish(self, job: Job, state: str, now: float, timings: dict | None = None,
                gpu_used: str | None = None, error: JobError | None = None) -> Job:
        return _finish(self.services, job, state, now, timings, gpu_used, error)

    def _check_submitted(self, job: Job) -> None:
        now = self.clock()
        if not job.modal_call_id:
            self._finish(job, "failed", now, error=JobError(klass="lost", message="no call id was stored"))
            return
        try:
            outcome = self.backend.poll(job.modal_call_id)
        except Exception as exc:  # noqa: BLE001 - the backend itself is unreachable; try again next tick
            log.warning("poller: poll of job %s failed: %s", job.id, exc)
            outcome = Pending()
        if isinstance(outcome, Pending):
            started = parse_iso(job.submitted_at) if job.submitted_at else now
            if now - started > 2 * self.settings.gpu_timeout_s + STUCK_MARGIN_S:
                try:
                    self.backend.cancel(job.modal_call_id)
                except Exception as exc:  # noqa: BLE001 - the job fails as stuck anyway
                    log.warning("poller: cancel of stuck job %s failed: %s", job.id, exc)
                self._finish(job, "failed", now, error=JobError(
                    klass="stuck", message="no result after twice the function timeout; the call was cancelled"))
            return
        try:
            self._handle_outcome(job, outcome, now)
        except Exception as exc:
            log.exception("poller: handling the outcome of job %s failed", job.id)
            current = self.db.get_job(job.id)
            if current is not None and current.state == "submitted":
                self._finish(current, "failed", now,
                             error=JobError(klass=type(exc).__name__, message=str(exc)))

    def _handle_outcome(self, job: Job, outcome: Finished | Errored, now: float) -> None:
        settle(self.services, job, outcome, now)


def _finish(services: Any, job: Job, state: str, now: float, timings: dict | None = None,
            gpu_used: str | None = None, error: JobError | None = None) -> Job:
    """Move a submitted job to its final state. The attempt's cost is added only on a priced backend."""
    fields: dict[str, Any] = {}
    if getattr(services.backend, "priced", True):
        attempt = services.costs.attempt_cost(job, timings, gpu_used, now)
        fields["cost_estimate_usd"] = (job.cost_estimate_usd or 0.0) + attempt
    return services.db.transition(
        job, state,
        finished_at=iso_at(now),
        timings=Timings(**timings) if timings else None,
        gpu_used=gpu_used,
        error=error,
        **fields,
    )


def settle(services: Any, job: Job, outcome: Finished | Errored, now: float) -> Job:
    """Store a finished call's result and finish the job, or fail it with the call's error class.

    Shared by the poller and the pull worker desk, so cost, timings and `gpu_used` are handled in one place.
    """
    if isinstance(outcome, Errored):
        return _finish(services, job, "failed", now, error=JobError(klass=outcome.klass, message=outcome.message))
    data = outcome.result
    timings = data.get("timings") if isinstance(data.get("timings"), dict) else None
    device = (data.get("versions") or {}).get("gpu")
    gpu_used = gpu_type_from_device(device) or device
    if data.get("ok") is not True:
        err = data.get("error") or {}
        return _finish(services, job, "failed", now, timings, gpu_used, JobError(
            klass=str(err.get("class") or "input_error"), message=str(err.get("message") or "")))
    try:
        services.results.store(job, data)
    except ServiceError as exc:
        return _finish(services, job, "failed", now, timings, gpu_used,
                       JobError(klass="invalid_result", message=exc.detail))
    return _finish(services, job, "done", now, timings, gpu_used)
