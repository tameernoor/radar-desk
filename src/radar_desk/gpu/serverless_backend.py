"""GPU_BACKEND=serverless: one job per request on a RunPod Serverless endpoint (plan.md, RunPod Serverless).

It has the Modal shape. `spawn` is `POST /run` with the job's references and returns RunPod's job id, which
the poller stores as the call id; `poll` is `GET /status/{id}` mapped to Pending, Finished or Errored, plus one
`GET /health` per poll for the worker and queue counts that GET /compute shows. The handler on the worker
writes the artefacts and `result.json` to the storage the app reads, so two rules cover the gaps in RunPod's
answers. A COMPLETED job waits up to RUNPOD_SERVERLESS_VISIBILITY_S for its mask to be visible through the
app's storage before it settles (a write on the volume mount can take a moment to show through the S3 API),
and a 404 from `/status` (RunPod keeps a result 30 minutes, the TTL bounds a lost job) settles from
`result.json` when it is there. The handler's answer arrives wrapped as `{"result": ...}`, because the RunPod SDK
marks any output with a truthy top-level `error` as FAILED and would turn our `{ok: false, error}` into a string.

A 5xx, a 429 or a transport error raises, so the poller keeps the job submitted under its stuck rule. The
API key is only ever in the request headers; RunPodApi keeps it out of its messages.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

import httpx

from radar_desk.compute.runpod import RunPodApi, RunPodError
from radar_desk.gpu.backend import Errored, Finished, Pending, PollOutcome
from radar_desk.gpu.backend import artefact_keys as default_artefact_keys
from radar_desk.gpu.poller import URL_TTL_S
from radar_desk.records import Job
from radar_desk.services.costs import iso_at

log = logging.getLogger(__name__)

JOBS_URL = "https://api.runpod.ai/v2"
PENDING = ("IN_QUEUE", "IN_PROGRESS", "RUNNING")


def result_key(job_id: str) -> str:
    return f"jobs/{job_id}/result.json"


class ServerlessGpuBackend:
    name = "serverless"
    priced = True

    def __init__(self, settings: Any, storage: Any, db: Any, client: httpx.Client | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.settings = settings
        self.storage = storage
        self.db = db
        self.client = client
        self.clock = clock
        self._api: RunPodApi | None = None
        self.health: dict | None = None  # the last /health answer, {workers, jobs, at}
        self.health_failed_since: float | None = None
        self.statuses: dict[str, tuple[str, str]] = {}  # call id -> (RunPod status, when it was read)
        self.completed_at: dict[str, float] = {}  # call id -> first COMPLETED sighting, for the visibility wait
        self._unknown: set[str] = set()

    @property
    def api(self) -> RunPodApi:
        if self._api is None:
            s = self.settings
            if s.runpod_api_key is None or not s.runpod_endpoint_id:
                raise RunPodError("serverless needs RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID")
            self._api = RunPodApi(s.runpod_api_key.get_secret_value(), self.client,
                                  base_url=f"{JOBS_URL}/{s.runpod_endpoint_id}")
        return self._api

    @property
    def gpu_requested(self) -> list[str]:
        """The pools the endpoint draws from, stored on the job row in place of RADAR_GPU's types."""
        return self.settings.runpod_serverless_gpu_list

    def spawn(
        self,
        job: Job,
        source_url: str,
        artefact_urls: dict[str, str],
        artefact_keys: dict[str, str] | None = None,
    ) -> str:
        keys = artefact_keys or default_artefact_keys(job.id)
        rkey = result_key(job.id)
        # A queued job owns no live artefacts; an earlier attempt's leftovers would pass the visibility check.
        for key in [*keys.values(), rkey]:
            self.storage.delete(key)
        scan = self.db.get_scan(job.scan_id)
        timeout_s = int(self.settings.gpu_timeout_s)
        body = {
            "input": {
                "job_id": job.id,
                "source": source_url,
                "artefacts": artefact_urls,
                "artefact_keys": keys,
                "result": self.storage.worker_ref(rkey, "PUT", URL_TTL_S),
                "expected_size": scan.size_bytes if scan else None,
            },
            "policy": {"executionTimeout": timeout_s * 1000, "ttl": (2 * timeout_s + 600) * 1000},
        }
        try:
            answer = self.api.call("POST", "/run", json=body)
        except RunPodError as exc:
            if exc.status == 404:
                raise RunPodError(f"{exc}; check RUNPOD_ENDPOINT_ID", exc.status, exc.detail) from exc
            raise
        call_id = answer.get("id") if isinstance(answer, dict) else None
        if not call_id:
            raise RunPodError("RunPod accepted the job but returned no id")
        return str(call_id)

    def _read_health(self, now: float) -> None:
        """One GET /health. A failure keeps the last answer and starts the clock HEALTH_DOWN reads."""
        try:
            body = self.api.call("GET", "/health")
        except (RunPodError, httpx.HTTPError) as exc:
            log.warning("serverless: /health failed: %s", exc)
            if self.health_failed_since is None:
                self.health_failed_since = now
            return
        body = body if isinstance(body, dict) else {}
        self.health = {"workers": body.get("workers"), "jobs": body.get("jobs"), "at": iso_at(now)}
        self.health_failed_since = None

    def poll(self, call_id: str) -> PollOutcome:
        outcome = self._poll(call_id)
        if not isinstance(outcome, Pending):  # the call is over; nothing reads its entries again
            self.statuses.pop(call_id, None)
            self.completed_at.pop(call_id, None)
        return outcome

    def _poll(self, call_id: str) -> PollOutcome:
        now = self.clock()
        self._read_health(now)
        try:
            body = self.api.call("GET", f"/status/{call_id}")
        except RunPodError as exc:
            if exc.status == 404:
                return self._from_storage(call_id)
            raise
        body = body if isinstance(body, dict) else {}
        status = str(body.get("status") or "")
        self.statuses[call_id] = (status, iso_at(now))
        if status == "COMPLETED":
            return self._completed(call_id, body, now)
        if status == "FAILED":
            return Errored("runpod_failed", str(body.get("error") or ""))
        if status == "TIMED_OUT":
            return Errored("timeout", str(body.get("error") or "")
                           or f"RunPod timed the job out; GPU_TIMEOUT_S is {self.settings.gpu_timeout_s} s")
        if status == "CANCELLED":
            return Errored("cancelled", "cancelled outside the app")
        if status not in PENDING and status not in self._unknown:
            self._unknown.add(status)
            log.warning("serverless: RunPod answered status %r, waiting as if queued", status)
        return Pending()

    def _completed(self, call_id: str, body: dict, now: float) -> PollOutcome:
        wrapped = body.get("output")
        output = wrapped.get("result") if isinstance(wrapped, dict) else None
        if not isinstance(output, dict):
            shape = type(wrapped).__name__ if not isinstance(wrapped, dict) else "a dict without a dict result"
            return Errored("bad_result", f"RunPod returned {shape}, not {{\"result\": dict}}")
        if output.get("ok") is True:
            job = self.db.get_job_by_call_id(call_id)
            first = self.completed_at.setdefault(call_id, now)
            if job is not None and not self.storage.exists(default_artefact_keys(job.id)["mask"]):
                limit = self.settings.runpod_serverless_visibility_s
                if now - first < limit:
                    return Pending()
                log.warning("serverless: job %s completed but its mask is not visible after %s s; settling",
                            job.id, limit)
        output = dict(output)
        execution = body.get("executionTime")
        output["runpod"] = {"id": body.get("id"), "delayTime": body.get("delayTime"), "executionTime": execution}
        timings = output.get("timings")
        if isinstance(timings, dict) and isinstance(execution, int | float) and not isinstance(execution, bool):
            output["timings"] = {**timings, "runpod_execution_s": execution / 1000}
        return Finished(output)

    def _from_storage(self, call_id: str) -> PollOutcome:
        """RunPod forgot the job (its 30 minutes are over, or the TTL ran out): the handler's result.json."""
        job = self.db.get_job_by_call_id(call_id)
        if job is not None and self.storage.exists(result_key(job.id)):
            try:
                data = json.loads(b"".join(self.storage.open_stream(result_key(job.id))))
            except ValueError as exc:
                return Errored("bad_result", f"result.json does not parse: {exc}")
            if isinstance(data, dict):
                return Finished(data)
            return Errored("bad_result", f"result.json holds {type(data).__name__}, not a dict")
        ttl = 2 * int(self.settings.gpu_timeout_s) + 600
        return Errored("lost", "RunPod no longer knows the job; results expire 30 min after completion "
                               f"and the TTL was {ttl} s")

    def cancel(self, call_id: str) -> None:
        self.api.call("POST", f"/cancel/{call_id}")

    def last_status(self, call_id: str) -> tuple[str, str] | None:
        return self.statuses.get(call_id)

    def logs(self, call_id: str, lines: int = 200) -> str:
        """The tail of the job's worker.log when it is in storage, else a line from the last status and /health."""
        job = self.db.get_job_by_call_id(call_id)
        if job is not None:
            key = default_artefact_keys(job.id)["log"]
            if self.storage.exists(key):
                text = b"".join(self.storage.open_stream(key)).decode("utf-8", errors="replace")
                return "".join(text.splitlines(keepends=True)[-lines:])
        found = self.statuses.get(call_id)
        status = f"{found[0] or '?'} since {found[1]}" if found else "status unknown"
        if self.health is None:
            counts = "no /health yet"
        else:
            workers, jobs = self.health.get("workers") or {}, self.health.get("jobs") or {}
            counts = (f"workers idle {workers.get('idle', 0)} running {workers.get('running', 0)}, "
                      f"jobs in queue {jobs.get('inQueue', 0)} in progress {jobs.get('inProgress', 0)}")
        return f"RunPod job {call_id}: {status}; {counts}\n"
