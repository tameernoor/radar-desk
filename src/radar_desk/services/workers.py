"""The job desk for pull workers: tokens, claims, leases, artefacts and completion (plan.md, Pull worker).

A claim moves the oldest queued job to submitted under a lease id that is also the job's `modal_call_id`.
Every lease-bound call checks that the job is still submitted under that lease, so a worker that lost
its lease gets 409 and stops. The desk's own checks and state changes run under one lock, so a tick that
re-queues a job cannot interleave with a worker's complete. An owner's cancel is outside that lock; the
worker's next lease-bound call then answers 409.
"""

from __future__ import annotations

import hashlib
import secrets
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pydantic import ValidationError

from radar_desk.db import IllegalTransition
from radar_desk.gpu.backend import ARTEFACT_FILES, Errored, Finished, artefact_keys
from radar_desk.records import Job, JobError, Lease, Timings, Worker, WorkerToken, new_id
from radar_desk.services.costs import iso_at, parse_iso
from radar_desk.services.errors import ServiceError
from radar_desk.services.scans import source_key
from radar_desk.storage import ObjectExists, ObjectMissing

TOKEN_PREFIX = "rdw_"
LAST_USED_EVERY_S = 60
MAX_LEASE_LOSSES = 3
STALE = "the lease is not current"


def hash_token(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode()).hexdigest()


class WorkerService:
    def __init__(self, db: Any, settings: Any, storage: Any, results: Any,
                 clock: Callable[[], float] = time.time) -> None:
        self.db = db
        self.settings = settings
        self.storage = storage
        self.results = results
        self.clock = clock
        self._lock = threading.RLock()
        # What `settle` needs. Worker jobs are never priced, whatever the configured backend.
        self._settle_on = SimpleNamespace(db=db, results=results, costs=None,
                                          backend=SimpleNamespace(priced=False))

    # Tokens

    def create_token(self, name: str) -> tuple[WorkerToken, str]:
        """A new token and its plaintext, which is shown once and never stored."""
        plaintext = TOKEN_PREFIX + secrets.token_hex(16)
        token = WorkerToken(id=new_id("wtok"), name=name, token_hash=hash_token(plaintext),
                            created_at=iso_at(self.clock()))
        self.db.insert_worker_token(token)
        return token, plaintext

    def list_tokens(self) -> list[WorkerToken]:
        return self.db.list_worker_tokens()

    def revoke_token(self, token_id: str) -> WorkerToken:
        token = self.db.get_worker_token(token_id)
        if token is None:
            raise ServiceError(404, f"no worker token {token_id}")
        if token.revoked_at:
            return token
        return self.db.update_worker_token(token_id, revoked_at=iso_at(self.clock()))

    def authenticate(self, plaintext: str | None) -> WorkerToken | None:
        """The token for this plaintext, or None when it is unknown or revoked. Stamps `last_used_at`
        at most once a minute."""
        if not plaintext or not plaintext.startswith(TOKEN_PREFIX):
            return None
        token = self.db.get_worker_token_by_hash(hash_token(plaintext))
        if token is None or token.revoked_at:
            return None
        now = self.clock()
        if token.last_used_at is None or now - parse_iso(token.last_used_at) >= LAST_USED_EVERY_S:
            token = self.db.update_worker_token(token.id, last_used_at=iso_at(now))
        return token

    # Workers

    def _seen(self, worker_id: str, now: float, job_id: str | None, token_id: str | None = None,
              info: dict | None = None) -> None:
        """Record that a worker was heard from, and which job it holds."""
        current = self.db.get_worker(worker_id)
        if current is None and token_id is None:
            return
        fields = {"id": worker_id, "token_id": token_id or current.token_id, "last_seen_at": iso_at(now),
                  "job_id": job_id, "first_seen_at": current.first_seen_at if current else iso_at(now)}
        for name in ("hostname", "gpu_name", "device", "versions"):
            if info is not None and info.get(name) is not None:
                fields[name] = info[name]
            elif current is not None:
                fields[name] = getattr(current, name)
        self.db.upsert_worker(Worker(**fields))

    def list_workers(self, now: float | None = None) -> list[dict]:
        """Every worker with `token_name` and `online` (heard from within the lease)."""
        now = self.clock() if now is None else now
        names = {t.id: t.name for t in self.db.list_worker_tokens()}
        return [
            {**w.model_dump(), "token_name": names.get(w.token_id),
             "online": now - parse_iso(w.last_seen_at) <= self.settings.worker_lease_s}
            for w in self.db.list_workers()
        ]

    # Claim and lease

    def claim(self, token: WorkerToken, worker_info: dict) -> dict | None:
        """Claim the oldest queued job for this worker, or None when there is nothing to do.

        Only on GPU_BACKEND=worker, so a worker never takes a job another backend would spawn. A job whose
        scan is gone or not ready is failed as input_error and the next one is tried.
        """
        worker_id = worker_info["id"]
        with self._lock:
            now = self.clock()
            job = None
            while self.settings.gpu_backend == "worker":
                lease_id = f"wk_{secrets.token_hex(8)}"
                lease = Lease(worker_id=worker_id, expires_at=iso_at(now + self.settings.worker_lease_s),
                              heartbeat_at=iso_at(now))
                job = self.db.claim_job(lease_id, lease, iso_at(now))
                if job is None:
                    break
                scan = self.db.get_scan(job.scan_id)
                if scan is not None and scan.state == "ready":
                    # A queued job owns no live artefacts; drop what an earlier attempt left half uploaded.
                    for key in artefact_keys(job.id).values():
                        self.storage.delete(key)
                    break
                self.db.transition(job, "failed", finished_at=iso_at(now), error=JobError(
                    klass="input_error", message="the scan is gone or not ready"))
                job = None
            self._seen(worker_id, now, job.id if job else None, token.id, worker_info)
        if job is None:
            return None
        lease_s = self.settings.worker_lease_s
        base = f"/worker/jobs/{job.id}"
        return {
            "job_id": job.id,
            "lease": job.modal_call_id,
            "scan": {"id": scan.id, "filename": scan.filename, "size_bytes": scan.size_bytes,
                     "sha256": scan.sha256},
            "model_version": job.model_version,
            "source_url": f"{base}/source?lease={job.modal_call_id}",
            "artefact_urls": {name: f"{base}/artefacts/{fname}?lease={job.modal_call_id}"
                              for name, fname in ARTEFACT_FILES.items()},
            "artefact_keys": artefact_keys(job.id),
            "lease_s": lease_s,
            "heartbeat_s": lease_s // 3,
            "timeout_s": self.settings.gpu_timeout_s,
        }

    def _leased(self, job_id: str, lease: str) -> Job:
        """The job, when it is still submitted under this lease. 409 otherwise."""
        job = self.db.get_job(job_id)
        if job is None:
            raise ServiceError(404, f"no job {job_id}")
        if job.state != "submitted" or job.lease is None or job.modal_call_id != lease:
            raise ServiceError(409, STALE)
        return job

    def heartbeat(self, job_id: str, lease: str, progress: str | None = None) -> dict:
        with self._lock:
            job = self._leased(job_id, lease)
            now = self.clock()
            renewed = job.lease.model_copy(update={
                "expires_at": iso_at(now + self.settings.worker_lease_s),
                "heartbeat_at": iso_at(now),
                "progress": progress,
            })
            job = self.db.update_job(job_id, lease=renewed)
            self._seen(renewed.worker_id, now, job_id)
        return {"state": job.state, "expires_at": renewed.expires_at}

    def _transition(self, job: Job, settle_with: Finished | Errored | None, now: float) -> Job:
        """Settle the job, or re-queue it when `settle_with` is None; a concurrent change is a 409."""
        from radar_desk.gpu.poller import settle  # here, not at the top: the poller imports services

        try:
            if settle_with is None:
                done = self.db.transition(job, "queued", queued_at=iso_at(now))
            else:
                done = settle(self._settle_on, job, settle_with, now)
        except IllegalTransition:
            raise ServiceError(409, STALE) from None
        self._seen(job.lease.worker_id, now, None)
        return done

    def complete(self, job_id: str, lease: str, data: dict) -> Job:
        """Store the worker's result and finish the job. A repeat for a job already done under this lease
        returns it; `ok: false` fails the job with the given class; a result the catalog rejects is a 422
        and the job stays submitted."""
        with self._lock:
            job = self.db.get_job(job_id)
            if job is not None and job.state == "done" and job.modal_call_id == lease:
                return job
            job = self._leased(job_id, lease)
            if data.get("job_id") not in (None, job_id):
                raise ServiceError(422, f"the result is for job {data.get('job_id')}, not {job_id}")
            if not isinstance(data.get("ok"), bool):
                raise ServiceError(422, "ok: must be true or false")
            if not isinstance(data.get("versions", {}), dict):
                raise ServiceError(422, "versions: must be an object")
            try:
                if "timings" in data:
                    if not isinstance(data["timings"], dict):
                        raise ServiceError(422, "timings: must be an object")
                    Timings(**data["timings"])
                # The keys are the desk's, whatever the worker sent.
                data = {**data, "artefacts": artefact_keys(job_id)}
                if data["ok"]:
                    self.results.build(job, data)
            except ValidationError as exc:
                err = exc.errors()[0]
                where = ".".join(str(part) for part in err["loc"])
                raise ServiceError(422, f"invalid result: {where}: {err['msg']}") from None
            return self._transition(job, Finished(data), self.clock())

    def fail(self, job_id: str, lease: str, error: JobError) -> Job:
        with self._lock:
            job = self._leased(job_id, lease)
            return self._transition(job, Errored(error.klass, error.message), self.clock())

    def release(self, job_id: str, lease: str) -> Job:
        """Give the job back to the queue; `lease_losses` is unchanged."""
        with self._lock:
            job = self._leased(job_id, lease)
            return self._transition(job, None, self.clock())

    # Bytes

    def source_stream(self, job_id: str, lease: str) -> tuple[int, Iterator[bytes]]:
        """The size and a byte stream of the job's scan."""
        job = self._leased(job_id, lease)
        key = source_key(job.scan_id)
        try:
            return self.storage.size(key), self.storage.open_stream(key)
        except ObjectMissing:
            raise ServiceError(404, f"the scan of job {job_id} is not in storage") from None

    def artefact_key(self, job_id: str, lease: str, name: str) -> str:
        """The storage key for artefact file `name`, checked for the lease and for being still free."""
        self._leased(job_id, lease)
        if name not in ARTEFACT_FILES.values():
            raise ServiceError(404, f"{name!r} is not an artefact; use one of {sorted(ARTEFACT_FILES.values())}")
        key = f"jobs/{job_id}/{name}"
        if self.storage.exists(key):
            raise ServiceError(409, "the object already exists; objects are never overwritten")
        return key

    def put_artefact(self, job_id: str, lease: str, name: str, path: Path,
                     content_type: str | None = None) -> str:
        """Store the file at `path` as the job's artefact `name`, once. Returns the key."""
        key = self.artefact_key(job_id, lease, name)
        try:
            self.storage.put_file(key, Path(path), content_type)
        except ObjectExists:
            raise ServiceError(409, "the object already exists; objects are never overwritten") from None
        return key

    # Desk tick

    def tick(self, now: float | None = None) -> None:
        """Fail submitted jobs past GPU_TIMEOUT_S as `timeout`; re-queue jobs whose lease expired, failing
        them as `lease_expired` on the third loss."""
        now = self.clock() if now is None else now
        with self._lock:
            for job in self.db.list_jobs(state="submitted", limit=100_000):
                if job.submitted_at and now - parse_iso(job.submitted_at) > self.settings.gpu_timeout_s:
                    self.db.transition(job, "failed", finished_at=iso_at(now), error=JobError(
                        klass="timeout", message=f"no result within {self.settings.gpu_timeout_s} s"))
                elif job.lease is not None and parse_iso(job.lease.expires_at) < now:
                    losses = job.lease_losses + 1
                    if losses >= MAX_LEASE_LOSSES:
                        self.db.transition(job, "failed", finished_at=iso_at(now), lease_losses=losses,
                                           error=JobError(klass="lease_expired",
                                                          message=f"the worker's lease expired {losses} times"))
                    else:
                        self.db.transition(job, "queued", queued_at=iso_at(now), lease_losses=losses)
                else:
                    continue
                if job.lease is not None:
                    worker = self.db.get_worker(job.lease.worker_id)
                    if worker is not None and worker.job_id == job.id:
                        self.db.upsert_worker(worker.model_copy(update={"job_id": None}))
