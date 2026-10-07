"""The pull modes worker and runpod: pull workers claim jobs from the app's job desk (services/workers.py).

Nothing is spawned or polled here; the poller sees `pull` and runs the desk's tick instead. Worker jobs are
not priced. Cancel is a no-op: the job is cancelled in the database and the worker's next lease-bound call
gets 409.
"""

from __future__ import annotations

from typing import Any

from radar_desk.gpu.backend import Pending, PollOutcome, artefact_keys
from radar_desk.records import Job


class WorkerGpuBackend:
    name = "worker"
    pull = True
    priced = False

    def __init__(self, db: Any, storage: Any) -> None:
        self.db = db
        self.storage = storage

    def spawn(
        self,
        job: Job,
        source_url: str,
        artefact_urls: dict[str, str],
        artefact_keys: dict[str, str] | None = None,
    ) -> str:
        raise RuntimeError("the worker backend does not spawn; workers claim jobs")

    def poll(self, call_id: str) -> PollOutcome:
        return Pending()

    def cancel(self, call_id: str) -> None:
        return None

    def logs(self, call_id: str, lines: int = 200) -> str:
        """The tail of the job's worker.log when it is in storage, else a line about the lease."""
        job = self.db.get_job_by_call_id(call_id)
        if job is None:
            return ""
        result = self.db.get_result(job.id)
        key = (result.artefacts.log if result else None) or artefact_keys(job.id)["log"]
        if self.storage.exists(key):
            text = b"".join(self.storage.open_stream(key)).decode("utf-8", errors="replace")
            return "".join(text.splitlines(keepends=True)[-lines:])
        if job.lease is None:
            return "no worker has claimed this job\n"
        lease = job.lease
        return (f"worker {lease.worker_id}, last heartbeat {lease.heartbeat_at}, "
                f"progress {lease.progress or 'none'}\n")
