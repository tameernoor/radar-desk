"""Seed the dev store with one synthetic scan and one finished fake job.

Run with the same environment as the app (or a .env in the working directory), with GPU_BACKEND=fake:

    .venv/bin/python scripts/seed_dev.py

Prints the scan id and the job id. The workspace is then at /workspace.html?scan=<scan id>.
With --readers (and NVREASON_ENABLED=1) the scan also gets a synthetic report, a finished fake NV-Reason-CT job
and scripted reads.
With --if-empty it does nothing when a finished job already exists (the Playwright web server uses this).
Run it before the server starts: the fake backend keeps its calls in memory, so a running server's poller
would fail a job this script submitted.
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import nibabel as nib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from radar_desk.config import ConfigError, load_settings
from radar_desk.gpu.poller import Poller
from radar_desk.records import TERMINAL_JOB_STATES
from radar_desk.services import ServiceError, build_services
from radar_desk.services.scans import source_key
from synth import affine_for, make_volume

SHAPE = (64, 64, 24)
SPACING = (1.0, 1.0, 5.0)
BLOBS = [
    ((22, 26, 10), (9, 7, 4), 60.0),
    ((42, 30, 12), (6, 6, 3), 30.0),
    ((32, 44, 14), (5, 4, 3), 300.0),
    ((30, 20, 6), (4, 4, 2), 150.0),
]
MAX_TICKS = 20


def synthetic_scan() -> bytes:
    affine = affine_for(SPACING, "LAS")
    img = nib.Nifti1Image(make_volume(SHAPE, BLOBS), affine)
    img.set_sform(affine, code=1)
    img.set_qform(affine, code=1)
    return gzip.compress(img.to_bytes())


def seed_readers(services, scan_id: str) -> None:
    """A synthetic report, the finished fake NV job and the reads, from a scripted model. No chat model is called,
    here or later: the running app (without CHAT_MODEL) finds every read up to date."""
    from fake_llm import ScriptedLLM
    from readers_seed import SYNTHETIC_REPORT, scripted_replies

    if not services.settings.nvreason_enabled:
        sys.exit("seed_dev: --readers needs NVREASON_ENABLED=1")
    services.scans.put_report(scan_id, SYNTHETIC_REPORT)
    services.nvreason.ensure_job(scan_id)
    poller = Poller(services)
    for _ in range(MAX_TICKS):
        if services.nvreason.latest_done(scan_id) is not None:
            break
        poller.tick()
    if services.nvreason.latest_done(scan_id) is None:
        sys.exit(f"seed_dev: the fake NV-Reason-CT job did not finish: {services.nvreason.latest(scan_id)}")
    services.readers.llm = ScriptedLLM(scripted_replies())
    services.readers.refresh(scan_id)
    status = services.readers.status(scan_id)
    if any(s["status"] != "ready" for s in status.values()):
        sys.exit(f"seed_dev: the reads are not ready: {status}")
    print(f"readers seeded for {scan_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed one synthetic scan with a finished fake job.")
    parser.add_argument("--if-empty", action="store_true", help="skip when a finished job already exists")
    parser.add_argument("--readers", action="store_true",
                        help="also attach a synthetic report, finish the fake NV-Reason-CT job and store scripted "
                             "reads for the Readers tab (needs NVREASON_ENABLED=1)")
    args = parser.parse_args()
    try:
        settings = load_settings()
    except ConfigError as exc:
        sys.exit(f"seed_dev: {exc}")
    if settings.gpu_backend != "fake":
        sys.exit("seed_dev: set GPU_BACKEND=fake; this script never starts a real GPU job")
    services = build_services(settings)
    if args.if_empty:
        done = services.db.list_jobs(state="done", limit=1)
        if done:
            print(f"seed_dev: a finished job exists ({done[0].id}), nothing to do")
            return
    data = synthetic_scan()

    ticket = services.scans.begin_upload("synthetic-dev.nii.gz", len(data), True)
    services.storage.put_bytes(source_key(ticket.scan_id), data, "application/gzip")
    try:
        scan = services.scans.complete_upload(ticket.scan_id)
    except ServiceError as exc:
        sys.exit(f"seed_dev: the synthetic scan was rejected: {exc.detail}")

    job = services.jobs.create(scan.id)
    poller = Poller(services)
    for _ in range(MAX_TICKS):
        poller.tick()
        job = services.jobs.get(job.id)
        if job.state in TERMINAL_JOB_STATES:
            break
    if job.state != "done":
        sys.exit(f"seed_dev: job {job.id} ended {job.state}: {job.error}")

    if args.readers:
        seed_readers(services, scan.id)
    print(f"scan_id {scan.id}")
    print(f"job_id {job.id}")


if __name__ == "__main__":
    main()
