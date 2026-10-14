"""NV-Reason-CT jobs for the Readers lens: one job per scan, next to RADAR's and independent of it.

A job of kind nvreason asks the deployed Modal class radar-desk-nvreason / NvReason (worker/nvreason_app.py) for the
crop it reads and one structured report. It runs only with NVREASON_ENABLED and in compute mode modal; mode fake
uses the in-process stand-in. The poller calls tick(). RADAR's jobs never pass through here, and an NV failure
never touches them. Artefacts go under jobs/{job_id}/: nv_report.json, crop.json, crop.nii.gz, worker.log, and
qa.jsonl for questions asked later (one JSON object per line).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from radar_desk.gpu.backend import Errored, Finished, Pending
from radar_desk.gpu.nvreason_backend import NvAskTimeout
from radar_desk.records import Job, JobError, Timings, new_id
from radar_desk.services.blobs import read_bytes, read_json, replace_bytes, write_json
from radar_desk.services.costs import gpu_type_from_device, iso_at, parse_iso, price_per_s
from radar_desk.services.errors import ServiceError
from radar_desk.services.scans import source_key

log = logging.getLogger(__name__)

NV_KIND = "nvreason"
NV_REVISION = "386b93e034983f6c1fc841a43833a1b6a0cd9c13"  # mirrors worker/radar_worker/nvreason.py
NV_MODEL_VERSION = f"nv-reason-ct-{NV_REVISION[:12]}"
NV_REGION = "abdomen"
NV_REPORT_PROMPT = "Write a structured abdominal CT report."
NV_YES_NO_SUFFIX = "Answer only Yes or No."
NV_SCALEDOWN_S = 300  # the class's scaledown_window
NV_CLASS_TIMEOUT_S = 1200  # the class's timeout
NV_STUCK_S = 2 * NV_CLASS_TIMEOUT_S + NV_SCALEDOWN_S
NV_ASK_TIMEOUT_S = 240  # a cold question: model load about 70 s plus the answer
NV_GPU = "L4"
NV_YES_NO_TOKENS = 16  # a Yes/No answer, thinking off, as in the smoke runs
NV_ANSWER_TOKENS = 2048
COLD_NOTICE = "Waking NV-Reason-CT, about 2 minutes."
NV_MODES = ("modal", "fake")
URL_TTL_S = 12 * 60 * 60
ACTIVE = ("queued", "submitted")
NV_ARTEFACTS = {"nv_report": "nv_report.json", "crop": "crop.json", "crop_nii": "crop.nii.gz", "log": "worker.log"}
OFF = "NV-Reason-CT is off; set NVREASON_ENABLED=1"
NEEDS_MODAL = "needs Modal compute"


def nv_keys(job_id: str) -> dict[str, str]:
    return {name: f"jobs/{job_id}/{fname}" for name, fname in NV_ARTEFACTS.items()}


def qa_key(job_id: str) -> str:
    return f"jobs/{job_id}/qa.jsonl"


def conversation_messages(report_text: str, history: list[dict], question: str) -> list[dict]:
    """NVIDIA's conversation shape: the image with the report prompt, NV-Reason-CT's own report, each earlier question
    and answer of this conversation, then the new question. One scan and one region per conversation."""
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": NV_REPORT_PROMPT}]},
        {"role": "assistant", "content": [{"type": "text", "text": report_text}]},
    ]
    for entry in history:
        messages.append({"role": "user", "content": [{"type": "text", "text": entry["question"]}]})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": entry["answer"]}]})
    messages.append({"role": "user", "content": [{"type": "text", "text": question}]})
    return messages


class NvReasonService:
    def __init__(self, db: Any, settings: Any, storage: Any, compute: Any, costs: Any,
                 backend_factory: Callable[[str], Any], clock: Callable[[], float] = time.time) -> None:
        self.db = db
        self.settings = settings
        self.storage = storage
        self.compute = compute
        self.costs = costs
        self.backend_factory = backend_factory
        self.clock = clock
        self.on_done: list[Callable[[str], None]] = []
        self._backends: dict[str, Any] = {}
        self._create_lock = threading.Lock()
        self._qa_lock = threading.Lock()
        self._last_gpu_at: float | None = None

    # Whether NV can run, and on what

    def unavailable(self) -> str | None:
        """None when NV can run now, else the reason the NV column shows."""
        if not self.settings.nvreason_enabled:
            return OFF
        if self.compute.mode not in NV_MODES:
            return NEEDS_MODAL
        return None

    def backend(self, name: str | None = None) -> Any:
        name = name or self.compute.mode
        if name not in self._backends:
            self._backends[name] = self.backend_factory(name)
        return self._backends[name]

    def warm(self, now: float | None = None) -> bool:
        """Whether the model's container is probably still up (it idles NV_SCALEDOWN_S after the last call)."""
        now = self.clock() if now is None else now
        return self._last_gpu_at is not None and now - self._last_gpu_at < NV_SCALEDOWN_S - 30

    # Jobs

    def latest(self, scan_id: str) -> Job | None:
        return self.db.latest_job_for_scan(scan_id, kind=NV_KIND)

    def latest_done(self, scan_id: str) -> Job | None:
        return next((j for j in self.db.jobs_for_scan(scan_id, kind=NV_KIND) if j.state == "done"), None)

    def ensure_job(self, scan_id: str) -> Job | None:
        """The scan's NV job: an active or done one is returned, else a new one is queued. None when NV cannot
        run or the scan is not ready. Called after every RADAR job create."""
        if self.unavailable():
            return None
        scan = self.db.get_scan(scan_id)
        if scan is None or scan.state != "ready":
            return None
        with self._create_lock:
            for job in self.db.jobs_for_scan(scan_id, kind=NV_KIND):
                if job.model_version == NV_MODEL_VERSION and job.state in (*ACTIVE, "done"):
                    return job
            job = Job(id=new_id("job"), scan_id=scan_id, kind=NV_KIND, gpu_requested=[NV_GPU],
                      model_version=NV_MODEL_VERSION)
            self.db.insert_job(job)
            return job

    def start(self, scan_id: str) -> Job:
        """POST /scans/{id}/nvreason: ensure_job, with the reason as an error when it cannot run."""
        why = self.unavailable()
        if why:
            raise ServiceError(409, why)
        scan = self.db.get_scan(scan_id)
        if scan is None:
            raise ServiceError(404, f"no scan {scan_id}")
        job = self.ensure_job(scan_id)
        if job is None:
            raise ServiceError(409, f"scan {scan_id} is {scan.state}, not ready")
        return job

    # The poller's pass

    def tick(self, now: float | None = None) -> None:
        """Collect submitted NV jobs (always, so none is stranded), then spawn at most one queued job, and only
        while none is submitted: the class runs one container. When NV cannot run, queued jobs are cancelled."""
        now = self.clock() if now is None else now
        for job in reversed(self.db.list_jobs(state="submitted", kind=NV_KIND, limit=1000)):
            try:
                self._check(job, now)
            except Exception:
                log.exception("nvreason: checking job %s failed", job.id)
        queued = list(reversed(self.db.list_jobs(state="queued", kind=NV_KIND, limit=1000)))
        why = self.unavailable()
        if why:
            for job in queued:
                self.db.transition(job, "cancelled", finished_at=iso_at(now),
                                   error=JobError(klass="unavailable", message=f"{why}; start it again when it is on"))
            return
        if not queued or self.db.list_jobs(state="submitted", kind=NV_KIND, limit=1):
            return
        try:
            self._spawn(queued[0], now)
        except Exception:
            log.exception("nvreason: spawning job %s failed", queued[0].id)

    def _hold(self, job: Job, reason: str, error: JobError | None = None) -> None:
        if job.hold_reason != reason or job.error != error:
            self.db.update_job(job.id, hold_reason=reason, error=error)

    def _spawn(self, job: Job, now: float) -> None:
        scan = self.db.get_scan(job.scan_id)
        if scan is None or scan.state != "ready":
            self.db.transition(job, "failed", finished_at=iso_at(now),
                               error=JobError(klass="input_error", message="the scan is gone or not ready"))
            return
        backend = self.backend()
        if getattr(backend, "priced", True) and not self.costs.budget_allows_now(now, "modal"):
            self._hold(job, "budget")
            return
        source = self.storage.worker_ref(source_key(scan.id), "GET", URL_TTL_S)
        try:
            call_id = backend.spawn_report(job.id, source)
        except Exception as exc:  # noqa: BLE001 - held and retried next tick
            log.warning("nvreason: spawn of job %s failed: %s", job.id, exc)
            self._hold(job, "spawn_error", JobError(klass=type(exc).__name__, message=str(exc)))
            return
        self.db.transition(job, "submitted", modal_call_id=call_id, backend=backend.name,
                           submitted_at=iso_at(now), hold_reason=None, error=None)

    def _check(self, job: Job, now: float) -> None:
        backend = self.backend(job.backend or "modal")
        if not job.modal_call_id:
            self._finish(job, "failed", now, error=JobError(klass="lost", message="no call id was stored"))
            return
        try:
            outcome = backend.poll(job.modal_call_id)
        except Exception as exc:  # noqa: BLE001 - Modal unreachable; read as pending so the stuck rule applies
            log.warning("nvreason: poll of job %s failed: %s", job.id, exc)
            outcome = Pending()
        if isinstance(outcome, Pending):
            started = parse_iso(job.submitted_at) if job.submitted_at else now
            if now - started > NV_STUCK_S:
                try:
                    backend.cancel(job.modal_call_id)
                except Exception as exc:  # noqa: BLE001 - the job fails as stuck anyway
                    log.warning("nvreason: cancel of stuck job %s failed: %s", job.id, exc)
                self._finish(job, "failed", now, error=JobError(
                    klass="stuck", message="no result after twice the class timeout; the call was cancelled"))
            return
        try:
            self.settle(job, outcome, now)
        except Exception as exc:
            log.exception("nvreason: settling job %s failed", job.id)
            current = self.db.get_job(job.id)
            if current is not None and current.state == "submitted":
                self._finish(current, "failed", now, error=JobError(klass=type(exc).__name__, message=str(exc)))

    def settle(self, job: Job, outcome: Finished | Errored, now: float) -> Job:
        """Write a finished call's artefacts and finish the job, or fail it. Writes replace, so settling the same
        call again after a crash works on the write-once Volume too."""
        current = self.db.get_job(job.id)
        if current is None or current.state != "submitted":
            return current or job
        if isinstance(outcome, Errored):
            return self._finish(current, "failed", now, error=JobError(klass=outcome.klass, message=outcome.message))
        data = outcome.result
        timings = data.get("timings") if isinstance(data.get("timings"), dict) else None
        versions = data.get("versions") if isinstance(data.get("versions"), dict) else {}
        device = versions.get("gpu")
        gpu = gpu_type_from_device(device) or device
        if data.get("ok") is not True:
            err = data.get("error") or {}
            return self._finish(current, "failed", now, timings, gpu, JobError(
                klass=str(err.get("class") or "input_error"), message=str(err.get("message") or "")))
        report, crop = data.get("nv_report"), data.get("crop")
        if (not isinstance(report, dict) or not isinstance(report.get("text"), str)
                or not isinstance(crop, dict) or "box_mm" not in crop):
            return self._finish(current, "failed", now, timings, gpu, JobError(
                klass="invalid_result", message="the result has no report or no crop box"))
        keys = nv_keys(job.id)
        write_json(self.storage, keys["nv_report"], {**report, "versions": versions, "timings": timings or {}})
        write_json(self.storage, keys["crop"], crop)
        crop_gz = data.get("crop_gz")
        if isinstance(crop_gz, bytes | bytearray):
            replace_bytes(self.storage, keys["crop_nii"], bytes(crop_gz), "application/gzip")
        replace_bytes(self.storage, keys["log"], str(data.get("log") or "").encode("utf-8"),
                      "text/plain; charset=utf-8")
        done = self._finish(current, "done", now, timings, gpu)
        self._last_gpu_at = now
        for listener in self.on_done:
            try:
                listener(job.scan_id)
            except Exception:
                log.exception("nvreason: a done listener failed for scan %s", job.scan_id)
        return done

    def _finish(self, job: Job, state: str, now: float, timings: dict | None = None,
                gpu_used: str | None = None, error: JobError | None = None) -> Job:
        fields: dict[str, Any] = {}
        if getattr(self.backend(job.backend or "modal"), "priced", True):
            fields["cost_estimate_usd"] = (job.cost_estimate_usd or 0.0) + self._cost(job, timings, now)
        known = {k: v for k, v in (timings or {}).items() if k in Timings.model_fields}
        return self.db.transition(job, state, finished_at=iso_at(now), timings=Timings(**known) if known else None,
                                  gpu_used=gpu_used, error=error, **fields)

    def _cost(self, job: Job, timings: dict | None, now: float) -> float:
        """Run seconds (the larger of total_s and the time since submit) plus the idle window, at the L4 price."""
        elapsed = now - parse_iso(job.submitted_at) if job.submitted_at else 0.0
        total = float((timings or {}).get("total_s") or 0.0)
        return (max(total, elapsed) + NV_SCALEDOWN_S) * price_per_s(NV_GPU, self.settings)

    # Artefacts

    def report(self, job_id: str) -> dict | None:
        return read_json(self.storage, nv_keys(job_id)["nv_report"])

    def crop(self, job_id: str) -> dict | None:
        return read_json(self.storage, nv_keys(job_id)["crop"])

    def qa_entries(self, job_id: str) -> list[dict]:
        """The questions asked about this job's report, oldest first; unreadable lines are skipped."""
        data = read_bytes(self.storage, qa_key(job_id)) or b""
        out = []
        for line in data.decode("utf-8", "replace").splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if isinstance(entry, dict):
                out.append(entry)
        return out

    # Questions

    def cold_start_notice(self, scan_id: str) -> str | None:
        """The chat line shown before a question that has to wake the model first, or None."""
        if self.unavailable() or self.latest_done(scan_id) is None or self.warm():
            return None
        return COLD_NOTICE

    def ask(self, scan_id: str, question: str, label: str | None = None) -> dict:
        """Ask NV-Reason-CT one question about the scan, as a follow-up to its own report, and log it to qa.jsonl.
        A question ending "Answer only Yes or No." runs with thinking off and 16 new tokens, others with thinking."""
        question = question.strip()
        if not question:
            raise ServiceError(422, "the question is empty")
        why = self.unavailable()
        if why:
            raise ServiceError(409, why)
        job = self.latest_done(scan_id)
        if job is None:
            raise ServiceError(409, "NV-Reason-CT has not read this scan yet")
        report = self.report(job.id) or {}
        if not isinstance(report.get("text"), str):
            raise ServiceError(409, "NV-Reason-CT's report for this scan is missing")
        backend = self.backend()
        started = self.clock()
        if getattr(backend, "priced", True) and not self.costs.budget_allows_now(started, "modal"):
            raise ServiceError(409, "the monthly GPU budget is used up")
        conversation = f"{job.id}:{NV_REGION}"
        history = [e for e in self.qa_entries(job.id) if e.get("conversation") == conversation]
        yes_no = question.endswith(NV_YES_NO_SUFFIX)
        warm = self.warm(started)
        source = self.storage.worker_ref(source_key(scan_id), "GET", URL_TTL_S)
        try:
            out = backend.ask(source, NV_REGION, conversation_messages(report["text"], history, question),
                              not yes_no, NV_YES_NO_TOKENS if yes_no else NV_ANSWER_TOKENS, NV_ASK_TIMEOUT_S)
        except NvAskTimeout as exc:
            raise ServiceError(504, str(exc)) from None
        except Exception as exc:  # noqa: BLE001 - shown to the user, nothing is logged
            raise ServiceError(502, f"NV-Reason-CT failed: {type(exc).__name__}: {exc}") from None
        finished = self.clock()
        self._last_gpu_at = finished
        entry = {
            "conversation": conversation,
            "question": question,
            "answer": str(out.get("answer") or "").strip(),
            "thinking": str(out.get("thinking") or ""),
            "seconds": round(finished - started, 2),
            "warm": "warm" if warm else "cold",
            "label": label,
            "asked_at": iso_at(finished),
        }
        self._append_qa(job.id, entry)
        return {**entry, "job_id": job.id}

    def _append_qa(self, job_id: str, entry: dict) -> None:
        """One line per question. Under a lock, so two questions at once both land."""
        line = (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8")
        with self._qa_lock:
            data = read_bytes(self.storage, qa_key(job_id)) or b""
            replace_bytes(self.storage, qa_key(job_id), data + line, "application/x-ndjson")
