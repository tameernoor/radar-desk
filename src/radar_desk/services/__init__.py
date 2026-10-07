"""Every rule of the app lives here; routes and chat tools are thin calls into these services."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import urlparse

from radar_desk.db import Database
from radar_desk.radar import catalog as catalog_module
from radar_desk.records import Job, Scan
from radar_desk.services.compute import ComputeService
from radar_desk.services.costs import CostService
from radar_desk.services.errors import ServiceError
from radar_desk.services.exports import ExportService
from radar_desk.services.fixtures import DEFAULT_ROOT, FixtureService
from radar_desk.services.jobs import JobService
from radar_desk.services.results import ResultService
from radar_desk.services.scans import ScanService
from radar_desk.services.workers import WorkerService
from radar_desk.storage import Storage, make_storage

__all__ = ["ServiceError", "Services", "build_services", "make_backend"]


@dataclass
class Services:
    settings: Any
    db: Database
    storage: Storage
    compute: ComputeService
    catalog: ModuleType
    fixtures: FixtureService
    costs: CostService
    scans: ScanService
    jobs: JobService
    results: ResultService
    exports: ExportService
    workers: WorkerService

    @property
    def backend(self) -> Any:
        """The backend of the current compute mode."""
        return self.compute.backend

    def backend_for(self, job: Job) -> Any:
        return self.compute.backend_for(job)

    def scan_view(self, scan: str | Scan) -> dict:
        """A scan (record or id) as the API shows it: the record plus `latest_job`."""
        if not isinstance(scan, Scan):
            scan = self.scans.get(scan)
        return {**scan.model_dump(), "latest_job": self.jobs.latest_summary(scan.id)}

    def gpu_status(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        body = self.costs.gpu_status(self.backend.name, now, self.compute.mode, self.compute.pod_view(now))
        return {**body, "runpod_configured": self.compute.configured(),
                "serverless": self.compute.serverless_view(), "problem": self.compute.problem(now)}


def make_backend(settings: Any, db: Database, storage: Storage, name: str | None = None) -> Any:
    """The GPU backend `name`, GPU_BACKEND by default. The fake one synthesises dev results through
    storage."""
    name = name or settings.gpu_backend
    if name == "modal":
        from radar_desk.gpu.modal_backend import ModalGpuBackend

        return ModalGpuBackend(settings)
    if name in ("worker", "runpod"):
        from radar_desk.gpu.worker_backend import WorkerGpuBackend

        return WorkerGpuBackend(db, storage)
    if name == "serverless":
        from radar_desk.gpu.serverless_backend import ServerlessGpuBackend

        return ServerlessGpuBackend(settings, storage, db)
    from radar_desk.gpu.fake import FakeGpuBackend, synthesize_result

    def synthesize(job_id: str) -> dict:
        job = db.get_job(job_id)
        return synthesize_result(job, db.get_scan(job.scan_id), storage, catalog_module)

    return FakeGpuBackend(synthesize=synthesize)


def build_services(
    settings: Any,
    *,
    db: Database | None = None,
    storage: Storage | None = None,
    backend: Any = None,
    runpod: Any = None,
    tunnel: Any = None,
    probe: Callable[[str], dict | None] | None = None,
    fixtures_root: Path = DEFAULT_ROOT,
    clock: Callable[[], float] = time.time,
) -> Services:
    """Wire the services. A backend given here, or GPU_BACKEND=fake, is fixed; modal, worker, runpod and
    serverless give a choice the owner can change, each backend built on first use."""
    db = db or Database(settings.db_path)
    storage = storage or make_storage(settings)
    if backend is None and settings.gpu_backend == "fake":
        backend = make_backend(settings, db, storage)
    if backend is not None:
        backends = {getattr(backend, "name", None) or "fake": lambda: backend}
    else:
        backends = {name: (lambda name=name: make_backend(settings, db, storage, name))
                    for name in ("modal", "worker", "runpod", "serverless")}
    if runpod is None and settings.runpod_configured:
        from radar_desk.compute.runpod import RunPod

        runpod = RunPod(settings.runpod_api_key.get_secret_value())
    if tunnel is None:
        from radar_desk.compute import tunnel
    if probe is None:
        from radar_desk.compute.tunnel import public_health as probe
    fixtures = FixtureService(fixtures_root)
    costs = CostService(db, settings)
    results = ResultService(db, fixtures)
    workers = WorkerService(db, settings, storage, results, clock=clock)
    compute = ComputeService(db, settings, workers, costs, backends, fixed=backend is not None, runpod=runpod,
                             tunnel=tunnel, probe=probe, port=urlparse(settings.public_base_url).port or 8000,
                             clock=clock)
    workers.mode = lambda: compute.mode
    return Services(
        settings=settings,
        db=db,
        storage=storage,
        compute=compute,
        catalog=catalog_module,
        fixtures=fixtures,
        costs=costs,
        scans=ScanService(db, storage, settings, fixtures),
        jobs=JobService(db, settings, compute, costs, clock=clock),
        results=results,
        exports=ExportService(db, storage, include_fake=compute.mode == "fake"),
        workers=workers,
    )
