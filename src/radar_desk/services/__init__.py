"""Every rule of the app lives here; routes and chat tools are thin calls into these services."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from radar_desk.db import Database
from radar_desk.radar import catalog as catalog_module
from radar_desk.records import Scan
from radar_desk.services.costs import CostService
from radar_desk.services.errors import ServiceError
from radar_desk.services.exports import ExportService
from radar_desk.services.fixtures import DEFAULT_ROOT, FixtureService
from radar_desk.services.jobs import JobService
from radar_desk.services.results import ResultService
from radar_desk.services.scans import ScanService
from radar_desk.storage import Storage, make_storage

__all__ = ["ServiceError", "Services", "build_services", "make_backend"]


@dataclass
class Services:
    settings: Any
    db: Database
    storage: Storage
    backend: Any
    catalog: ModuleType
    fixtures: FixtureService
    costs: CostService
    scans: ScanService
    jobs: JobService
    results: ResultService
    exports: ExportService

    def scan_view(self, scan: str | Scan) -> dict:
        """A scan (record or id) as the API shows it: the record plus `latest_job`."""
        if not isinstance(scan, Scan):
            scan = self.scans.get(scan)
        return {**scan.model_dump(), "latest_job": self.jobs.latest_summary(scan.id)}

    def gpu_status(self, now: float | None = None) -> dict:
        return self.costs.gpu_status(self.backend.name, time.time() if now is None else now)


def make_backend(settings: Any, db: Database, storage: Storage) -> Any:
    """The configured GPU backend. The fake one synthesises dev results through storage."""
    if settings.gpu_backend == "modal":
        from radar_desk.gpu.modal_backend import ModalGpuBackend

        return ModalGpuBackend(settings)
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
    fixtures_root: Path = DEFAULT_ROOT,
    clock: Callable[[], float] = time.time,
) -> Services:
    db = db or Database(settings.db_path)
    storage = storage or make_storage(settings)
    backend = backend if backend is not None else make_backend(settings, db, storage)
    fixtures = FixtureService(fixtures_root)
    costs = CostService(db, settings)
    return Services(
        settings=settings,
        db=db,
        storage=storage,
        backend=backend,
        catalog=catalog_module,
        fixtures=fixtures,
        costs=costs,
        scans=ScanService(db, storage, settings, fixtures),
        jobs=JobService(db, settings, backend, costs, clock=clock),
        results=ResultService(db, fixtures),
        exports=ExportService(db, storage, include_fake=getattr(backend, "name", None) == "fake"),
    )
