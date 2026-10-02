"""The compute choice and the RunPod pod the app starts for pull workers (plan.md, Compute switch B).

The mode, `modal` or `worker`, lives in the database and is seeded once from GPU_BACKEND; the fake backend
is fixed at start-up. In mode worker with RunPod configured, `tick` starts one pod when work is queued and
moves its row through `tunnel`, `starting` and `ready`, and stops it when idle, at the hour cap, or when
something under it fails. A pod is deleted on RunPod before its row is closed.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from radar_desk.compute.runpod import PodSpec, RunPodError, StockError
from radar_desk.compute.tunnel import TunnelError
from radar_desk.records import Job, PodRecord
from radar_desk.services.costs import iso_at, month_of, parse_iso, price_per_s, worst_case
from radar_desk.services.errors import ServiceError
from radar_desk.storage import storage_backend

log = logging.getLogger(__name__)

MODES = ("modal", "worker")
POD_NAME = "radar-worker"
MODAL_NEEDS_STORAGE = ("GPU_BACKEND=modal needs S3_BUCKET or STORAGE_BACKEND=modal_volume: "
                       "Modal cannot reach the local storage URLs")
FIXED = "The fake backend is chosen at start-up."
START_HOLDS = ("no_gpu", "tunnel_unreachable", "pod_failed", "budget", "owner")
RECONCILE_EVERY_S = 30
TUNNEL_TIMEOUT_S = 120
RETRY_S = 60
MAX_START_FAILURES = 2
RUNPOD_DOWN = "RunPod did not answer"
TUNNEL_DIED = "Tunnel died, pod stopped; a new one starts when work is queued"
UNREACHABLE = "Worker URL not reachable from outside"
POD_FAILED = "Pod started twice without a worker reporting in; check the image and the volume"
WORKER_LOST = "Worker went silent, pod stopped; a new one starts when work is queued"
UNCONFIGURED = "RunPod is not fully configured, the pod is managed but no new one starts"
MODAL_ON_LOCAL = "Compute mode was modal but Modal cannot reach local storage; set to worker"
RUNPOD_ERRORS = (RunPodError, httpx.HTTPError)
ALL = 100_000


def job_backend(job: Job) -> str:
    """The backend a job ran on. An older row without one is read from its call id."""
    if job.backend:
        return job.backend
    return "worker" if (job.modal_call_id or "").startswith("wk_") else "modal"


def _ts(value: str | None) -> float | None:
    return parse_iso(value) if value else None


class ComputeService:
    def __init__(self, db: Any, settings: Any, workers: Any, costs: Any,
                 backends: dict[str, Callable[[], Any]], fixed: bool, runpod: Any = None, tunnel: Any = None,
                 probe: Callable[[str], dict | None] | None = None, port: int = 8000,
                 clock: Callable[[], float] = time.time) -> None:
        self.db = db
        self.settings = settings
        self.workers = workers
        self.costs = costs
        self.backends = backends  # name -> factory, built on first use
        self.fixed = fixed
        self.runpod = runpod
        self.tunnel = tunnel
        self.probe = probe
        self.port = port
        self.clock = clock
        self._built: dict[str, Any] = {}
        self._reconciled_at: float | None = None
        self._lock = threading.RLock()

    # Mode and backends

    @property
    def changeable(self) -> bool:
        return not self.fixed

    @property
    def mode(self) -> str:
        """The fixed backend's name, else the stored mode, seeded from GPU_BACKEND on first read.

        Takes no lock: /health reads it while a tick holding the lock probes /health through the tunnel.
        """
        if self.fixed:
            return next(iter(self.backends))
        mode = self.db.get_setting("mode")
        if mode in MODES:
            return mode
        seed = self.settings.gpu_backend if self.settings.gpu_backend in MODES else "modal"
        return self.db.seed_setting("mode", seed)

    def _backend(self, name: str) -> Any:
        with self._lock:
            if name not in self._built:
                self._built[name] = self.backends[name]()
            return self._built[name]

    @property
    def backend(self) -> Any:
        return self._backend(self.mode)

    def backend_for(self, job: Job) -> Any:
        """The backend the job ran on, or the current one when that is not available."""
        name = job_backend(job)
        return self._backend(name) if name in self.backends else self.backend

    def set_mode(self, mode: str, now: float | None = None) -> dict:
        now = self.clock() if now is None else now
        with self._lock:
            if not self.changeable:
                raise ServiceError(409, FIXED)
            if mode not in MODES:
                raise ServiceError(409, f"unknown mode {mode!r}; use modal or worker")
            if mode == "modal" and storage_backend(self.settings) == "local":
                raise ServiceError(409, MODAL_NEEDS_STORAGE)
            if mode != self.mode:
                self.db.set_setting("mode", mode)
                self.db.set_setting("mode_changed_at", iso_at(now))
            return self.status(now)

    # Small stored values

    def _get_float(self, key: str) -> float:
        value = self.db.get_setting(key)
        return float(value) if value else 0.0

    def _set(self, key: str, value: Any) -> None:
        self.db.set_setting(key, None if value is None else str(value))

    def _event(self, text: str, now: float) -> None:
        self._set("last_event", f"{iso_at(now)} {text}")

    def fall_back_from_modal(self) -> None:
        """At start-up: a stored modal mode on local storage becomes worker, so the app still starts."""
        log.warning("compute: %s", MODAL_ON_LOCAL)
        self.db.set_setting("mode", "worker")
        self._event(MODAL_ON_LOCAL, self.clock())

    # Status

    def configured(self) -> bool:
        return bool(self.settings.runpod_configured and self.runpod is not None)

    def _last_finished(self) -> float:
        return _ts(self.db.last_finished_at()) or 0.0

    def _waiting(self, mode: str) -> bool:
        """A worker job is submitted, or, in mode worker, a queued job waits without a hold."""
        if any(job_backend(j) == "worker" for j in self.db.list_jobs(state="submitted", limit=ALL)):
            return True
        return mode == "worker" and any(
            j.hold_reason is None for j in self.db.list_jobs(state="queued", limit=ALL))

    def _idle_limit_s(self, mode: str) -> float:
        return self.settings.runpod_idle_min * 60 if mode == "worker" else 0.0

    def _tunnel_alive(self, pod: PodRecord) -> bool:
        return bool(self.tunnel.alive(pod.tunnel.get("pid"), pod.tunnel.get("binary")))

    def pod_view(self, now: float | None = None) -> dict | None:
        """The open pod as GET /compute shows it, or None."""
        now = self.clock() if now is None else now
        with self._lock:
            pod = self.db.open_pod()
            if pod is None:
                return None
            mode = self.mode
            started, ready = _ts(pod.started_at), _ts(pod.ready_at)
            idle_s = max(0, int(now - max(ready, self._last_finished()))) if ready is not None else None
            stops_in_s = None
            if idle_s is not None and not self._waiting(mode):
                stops_in_s = max(0, int(self._idle_limit_s(mode) - idle_s))
            tunnel_alive = None
            if self.settings.tunnel_mode == "managed":
                tunnel_alive = bool(pod.tunnel) and self._tunnel_alive(pod)
            job = next((j for j in self.db.list_jobs(state="submitted", limit=ALL)
                        if job_backend(j) == "worker"), None)
            return {
                "id": pod.id, "runpod_id": pod.runpod_id, "phase": pod.phase, "gpu": pod.gpu,
                "image": pod.image,
                "cost_per_hr": pod.cost_per_hr, "created_at": pod.created_at, "started_at": pod.started_at,
                "ready_at": pod.ready_at, "up_s": max(0, int(now - started)) if started is not None else None,
                "idle_s": idle_s, "stops_in_s": stops_in_s, "worker_id": pod.worker_id,
                "tunnel_url": pod.tunnel_url, "tunnel_alive": tunnel_alive, "job_id": job.id if job else None,
            }

    def status(self, now: float | None = None) -> dict:
        """The body of GET /compute."""
        now = self.clock() if now is None else now
        s = self.settings
        with self._lock:
            submitted = self.db.list_jobs(state="submitted", limit=ALL)
            queued = self.db.list_jobs(state="queued", limit=ALL)
            in_flight = submitted[-1] if submitted else None
            month = month_of(now)
            return {
                "mode": self.mode,
                "changeable": self.changeable,
                "changed_at": self.db.get_setting("mode_changed_at"),
                "tunnel_mode": s.tunnel_mode,
                "public_url": s.worker_public_url,
                "runpod": {"configured": self.configured(), "datacenter": s.runpod_datacenter,
                           "idle_min": s.runpod_idle_min, "max_pod_hours": s.runpod_max_pod_hours},
                "pod": self.pod_view(now),
                "in_flight": ({"job_id": in_flight.id, "backend": job_backend(in_flight)}
                              if in_flight else None),
                "queued": len(queued),
                "held": [{"job_id": j.id, "hold_reason": j.hold_reason}
                         for j in reversed(queued) if j.hold_reason],
                "problem": self.db.get_setting("problem"),
                "last_event": self.db.get_setting("last_event"),
                "spend_month_usd": round(self.costs.spend_for_month(month, now), 6),
                "budget_usd": s.gpu_monthly_budget_usd,
                "month": month,
            }

    # Holds

    def hold_queued(self, reason: str) -> None:
        for job in self.db.list_jobs(state="queued", limit=ALL):
            if job.hold_reason is None:
                self.db.update_job(job.id, hold_reason=reason)

    def release_holds(self, reasons: tuple[str, ...] | list[str]) -> None:
        for job in self.db.list_jobs(state="queued", limit=ALL):
            if job.hold_reason in reasons:
                self.db.update_job(job.id, hold_reason=None)

    # Owner actions

    def start_now(self, now: float | None = None) -> dict:
        now = self.clock() if now is None else now
        with self._lock:
            if self.mode != "worker":
                raise ServiceError(409, "a pod starts only in the worker mode")
            if not self.configured():
                raise ServiceError(409, "RunPod is not configured: set RUNPOD_API_KEY, RUNPOD_VOLUME_ID, "
                                        "RUNPOD_REGISTRY_AUTH_ID and WORKER_IMAGE")
            if self.db.open_pod() is not None:
                raise ServiceError(409, "a pod is already running")
            self._set("pod_failures", None)
            self._set("next_start_at", None)
            self._set("problem", None)
            self.release_holds(START_HOLDS)
            self._set("start_requested", "1")
            return self.status(now)

    def stop_now(self, now: float | None = None) -> dict:
        now = self.clock() if now is None else now
        with self._lock:
            pod = self.db.open_pod()
            if pod is None:
                raise ServiceError(409, "no pod is running")
            if self._stop(pod, "owner", now):
                self.hold_queued("owner")  # Start now or the owner's retry releases them
            return self.status(now)

    # Stopping

    def _stop(self, pod: PodRecord, reason: str, now: float, problem: str | None = None,
              delete: bool = True) -> bool:
        """Delete the pod, revoke its token, stop its managed tunnel and close the row. False when RunPod
        refused the delete; the row then stays open and the next tick tries again."""
        if delete and pod.runpod_id:
            try:
                if self.runpod is None:
                    raise RunPodError("RUNPOD_API_KEY is not set")
                self.runpod.delete(pod.runpod_id)
            except RUNPOD_ERRORS as exc:
                log.warning("compute: deleting pod %s failed: %s", pod.runpod_id, exc)
                self.db.update_pod(pod.id, error=str(exc))
                self._set("problem", f"Could not delete pod {pod.runpod_id}: {exc}")
                return False
        if pod.token_id:
            try:
                self.workers.revoke_token(pod.token_id)
            except Exception:
                log.exception("compute: revoking token %s failed", pod.token_id)
        if pod.tunnel:
            try:
                self.tunnel.stop(pod.tunnel.get("pid"), pod.tunnel.get("binary"))
            except Exception:
                log.exception("compute: stopping the tunnel of pod %s failed", pod.id)
        started = _ts(pod.started_at)
        cost = (pod.cost_per_hr or 0.0) * max(0.0, now - started) / 3600 if started is not None else 0.0
        self.db.update_pod(pod.id, phase="stopped", stopped_at=iso_at(now), reason=reason,
                           cost_usd=round(cost, 6))
        self._set("problem", problem)
        self._set("start_requested", None)
        self._event(f"pod {pod.runpod_id or pod.id} stopped: {reason}", now)
        return True

    def _delete_strays(self, listed: list, keep: str | None, now: float) -> None:
        """Delete listed radar-worker pods that no open row names."""
        for found in listed:
            if found.name != POD_NAME or found.id == keep:
                continue
            try:
                self.runpod.delete(found.id)
                self._event(f"deleted unknown {POD_NAME} pod {found.id}", now)
            except RUNPOD_ERRORS as exc:
                log.warning("compute: deleting unknown pod %s failed: %s", found.id, exc)

    # Start-up

    def reconcile_at_start(self, now: float | None = None) -> None:
        """Keep the open row when its pod and tunnel are still there, else stop it as `orphan`; delete
        radar-worker pods no row names."""
        now = self.clock() if now is None else now
        with self._lock:
            if self.runpod is None:
                return
            try:
                listed = self.runpod.pods()
            except RUNPOD_ERRORS as exc:
                log.warning("compute: listing pods at start failed: %s", exc)
                return
            ids = {p.id for p in listed}
            pod = self.db.open_pod()
            keep = None
            if pod is not None:
                keep = pod.runpod_id
                tunnel_live = bool(pod.tunnel) and self._tunnel_alive(pod)
                tunnel_ok = not pod.tunnel or tunnel_live
                if not ((pod.runpod_id in ids and tunnel_ok) or (pod.phase == "tunnel" and tunnel_live)):
                    self._stop(pod, "orphan", now, delete=pod.runpod_id in ids)
            self._delete_strays(listed, keep, now)

    # Tick

    def tick(self, now: float | None = None) -> None:
        """One pass of the pod rules, in order: reconcile, tunnel health, tunnel, starting, stop, start.

        Reconciling runs in every mode, so a lost pod is deleted even after a switch to modal; only new
        starts need RunPod fully configured.
        """
        now = self.clock() if now is None else now
        with self._lock:
            if self.runpod is None:
                return
            pod = self.db.open_pod()
            mode = self.mode

            # 1. Reconcile with RunPod's list.
            if self._reconciled_at is None or now - self._reconciled_at >= RECONCILE_EVERY_S:
                try:
                    listed = self.runpod.pods()
                except RUNPOD_ERRORS as exc:
                    log.warning("compute: listing pods failed: %s", exc)
                    self._set("problem", RUNPOD_DOWN)
                    return
                self._reconciled_at = now
                if self.db.get_setting("problem") == RUNPOD_DOWN:
                    self._set("problem", None)
                found = next((p for p in listed if pod is not None and p.id == pod.runpod_id), None)
                if pod is not None and pod.runpod_id and found is None:
                    self._stop(pod, "vanished", now, delete=False)
                    pod = None
                elif found is not None and not pod.cost_per_hr:  # a deploy answer without a price
                    price = found.cost_per_hr or price_per_s("L4", self.settings) * 3600
                    pod = self.db.update_pod(pod.id, cost_per_hr=price)
                self._delete_strays(listed, pod.runpod_id if pod else None, now)

            configured = self.configured()
            if pod is None and (mode != "worker" or not configured):
                return
            if pod is not None and not configured:
                self._set("problem", UNCONFIGURED)

            # 2. A pod never outlives its managed tunnel.
            if pod is not None and pod.tunnel and not self._tunnel_alive(pod):
                if not self._stop(pod, "tunnel_died", now, problem=TUNNEL_DIED):
                    return
                pod = None

            # 3. Waiting for the tunnel, then the deploy.
            if pod is not None and pod.phase == "tunnel":
                if mode != "worker" or not self._wanted(no_gpu_waits=True):
                    if not self._stop(pod, "mode" if mode != "worker" else "idle", now):
                        return
                    pod = None
                elif configured and now >= self._get_float("next_start_at"):
                    pod = self._tunnel_phase(pod, now)

            # 4. Waiting for the worker to report in.
            if pod is not None and pod.phase == "starting":
                worker = self.db.get_worker(pod.worker_id)
                if worker is not None and parse_iso(worker.last_seen_at) >= parse_iso(pod.started_at):
                    pod = self.db.update_pod(pod.id, phase="ready", ready_at=iso_at(now))
                    self._set("pod_failures", None)
                    self._event(f"pod {pod.runpod_id} ready", now)
                elif now - parse_iso(pod.started_at) >= self.settings.runpod_start_timeout_s:
                    failures = int(self._get_float("pod_failures")) + 1
                    failed = failures >= MAX_START_FAILURES
                    if not self._stop(pod, "start_timeout", now, problem=POD_FAILED if failed else None):
                        return
                    self._set("pod_failures", failures)
                    if failed:
                        self.hold_queued("pod_failed")
                    pod = None

            # A ready pod whose worker went silent is of no use, even with work queued.
            if pod is not None and pod.phase == "ready" and not self._worker_job_submitted():
                worker = self.db.get_worker(pod.worker_id)
                seen = parse_iso(worker.last_seen_at) if worker else parse_iso(pod.ready_at)
                if now - seen > self.settings.worker_lease_s:
                    if not self._stop(pod, "worker_lost", now, problem=WORKER_LOST):
                        return
                    pod = None

            # 5. The hour cap, then idle (at once when the mode is no longer worker).
            if pod is not None and pod.phase in ("starting", "ready"):
                started = parse_iso(pod.started_at)
                if now - started >= self.settings.runpod_max_pod_hours * 3600:
                    if not self._stop(pod, "cap", now):
                        return
                    pod = None
                elif not self._waiting(mode):
                    since = max(_ts(pod.ready_at) or started, self._last_finished())
                    if now - since >= self._idle_limit_s(mode):
                        if not self._stop(pod, "idle" if mode == "worker" else "mode", now):
                            return
                        pod = None

            # 6. Start a pod when work waits or the owner asked.
            if (pod is None and mode == "worker" and configured and now >= self._get_float("next_start_at")
                    and self._wanted()):
                self._start(now)

    def _worker_job_submitted(self) -> bool:
        return any(job_backend(j) == "worker" for j in self.db.list_jobs(state="submitted", limit=ALL))

    def _wanted(self, no_gpu_waits: bool = False) -> bool:
        """The owner asked for a pod, or a queued job waits without a hold (or, while a pod waits for
        stock, held as no_gpu)."""
        if self.db.get_setting("start_requested"):
            return True
        free = (None, "no_gpu") if no_gpu_waits else (None,)
        return any(j.hold_reason in free for j in self.db.list_jobs(state="queued", limit=ALL))

    def _start(self, now: float) -> None:
        s = self.settings
        one_pod = s.runpod_max_pod_hours * price_per_s("L4", s) * 3600
        modal_jobs = sum(1 for j in self.db.list_jobs(state="submitted", limit=ALL) if job_backend(j) != "worker")
        in_flight = worst_case(s) * modal_jobs
        if self.costs.spend_for_month(month_of(now), now) + in_flight + one_pod > s.gpu_monthly_budget_usd:
            self.hold_queued("budget")
            self._set("problem", "Budget reached")
            self._set("start_requested", None)
            return
        pod = PodRecord(created_at=iso_at(now))
        if s.tunnel_mode == "managed":
            try:
                t = self.tunnel.start(self.port, Path(s.data_dir) / "cloudflared.log")
            except (TunnelError, OSError) as exc:
                self._set("problem", str(exc))
                self._set("next_start_at", now + RETRY_S)
                return
            pod.tunnel = {"pid": t.pid, "url": t.url, "log": str(t.log), "binary": t.binary}
            pod.tunnel_url = t.url
        else:
            pod.tunnel_url = s.worker_public_url
        self.db.insert_pod(pod)  # start_requested stays set until the deploy, so a tunnel row is not idle
        self._set("problem", None)
        self._event(f"starting a pod, waiting for {pod.tunnel_url}", now)

    def _tunnel_phase(self, pod: PodRecord, now: float) -> PodRecord | None:
        """Deploy once the tunnel answers as the worker backend; give up after TUNNEL_TIMEOUT_S."""
        try:
            body = self.probe(pod.tunnel_url)
        except Exception:  # noqa: BLE001 - an unreachable URL, the same as no answer
            body = None
        if not (isinstance(body, dict) and body.get("backend") == "worker"):
            if now - parse_iso(pod.created_at) < TUNNEL_TIMEOUT_S:
                return pod
            if self._stop(pod, "tunnel_timeout", now, problem=UNREACHABLE):
                self.hold_queued("tunnel_unreachable")
                self._set("next_start_at", now + RETRY_S)
                return None
            return pod
        s = self.settings
        token, plaintext = self.workers.create_token(f"pod {pod.id}")
        spec = PodSpec(image=s.worker_image, registry_auth_id=s.runpod_registry_auth_id,
                       datacenter=s.runpod_datacenter, volume_id=s.runpod_volume_id,
                       env={"RADAR_DESK_URL": pod.tunnel_url, "RADAR_WORKER_TOKEN": plaintext,
                            "RADAR_WORKER_ID": pod.worker_id, "RADAR_IMAGE": s.worker_image})
        try:
            deployed = self.runpod.deploy(spec, attempts=1)
        except Exception as exc:  # noqa: BLE001 - a malformed answer too; the token must not outlive it
            self.workers.revoke_token(token.id)
            if isinstance(exc, StockError):
                problem = f"No L4 or RTX 4090 in {s.runpod_datacenter}, retrying in {RETRY_S} s"
            else:  # a GraphQL server can echo the whole input, token included
                problem = str(exc).replace(plaintext, "[token]")
            self.hold_queued("no_gpu")
            self._set("problem", problem)
            self._set("next_start_at", now + RETRY_S)
            return self.db.update_pod(pod.id, error=problem)
        pod = self.db.update_pod(pod.id, runpod_id=deployed.id, gpu=deployed.gpu,
                                 image=deployed.image or s.worker_image, cost_per_hr=deployed.cost_per_hr,
                                 started_at=iso_at(now), token_id=token.id, phase="starting", error=None)
        self.release_holds(START_HOLDS)
        self._set("problem", None)
        self._set("start_requested", None)
        self._event(f"pod {deployed.id} deployed, {deployed.gpu} at ${deployed.cost_per_hr}/h", now)
        return pod
