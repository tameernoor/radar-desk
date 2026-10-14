"""Shared pytest fixtures. Add fixtures here only when more than one test module needs them."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

FIXTURE_DIR = Path(os.environ.get("RADAR_FIXTURE_DIR") or ROOT / "fixtures")
FIXTURE_FILES = {
    "AC4214dbd": "merlin-AC4214dbd.nii.gz",
    "AC4240fff": "merlin-AC4240fff.nii.gz",
    "AC4242a2f": "merlin-AC4242a2f.nii.gz",
    "AC4242a55": "merlin-AC4242a55.nii.gz",
    "image1": "merlin-image1.nii.gz",
}


LLM_ENV = ("LLM_PROVIDER", "LLM_BASE_URL", "LLM_API_KEY", "CHAT_MODEL")
COMPUTE_ENV = ("WORKER_IMAGE", "WORKER_PUBLIC_URL", "WORKER_TUNNEL", "RUNPOD_API_KEY", "RUNPOD_VOLUME_ID",
               "RUNPOD_REGISTRY_AUTH_ID", "RUNPOD_DATACENTER", "RUNPOD_MAX_POD_HOURS", "RUNPOD_START_TIMEOUT_S",
               "RADAR_POD_IDLE_DELETE_S", "RADAR_POD_APP_LOST_DELETE_S", "RUNPOD_ENDPOINT_ID",
               "RUNPOD_S3_ACCESS_KEY_ID", "RUNPOD_S3_SECRET_ACCESS_KEY", "RUNPOD_SERVERLESS_GPUS",
               "RUNPOD_SERVERLESS_PRICE_USD_PER_S", "RUNPOD_SERVERLESS_IDLE_S", "RUNPOD_SERVERLESS_VISIBILITY_S",
               "RUNPOD_SERVERLESS_QUEUE_WARN_S", "NVREASON_ENABLED")


@pytest.fixture(autouse=True)
def no_llm_env(monkeypatch):
    """A developer's shell settings must not make a test configure chat or RunPod and call a real service."""
    for name in LLM_ENV + COMPUTE_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    """The read-only folder with the five real scans; skips when absent."""
    if not FIXTURE_DIR.is_dir():
        pytest.skip(f"real scans not present at {FIXTURE_DIR} (set RADAR_FIXTURE_DIR)")
    return FIXTURE_DIR


@pytest.fixture(scope="session")
def fixture_paths(fixture_dir: Path) -> dict[str, Path]:
    paths = {k: fixture_dir / v for k, v in FIXTURE_FILES.items()}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        pytest.skip(f"missing real scans: {missing}")
    return paths


@pytest.fixture
def make_services(tmp_path: Path):
    """Build Services on a temp data dir with the local storage adapter, never reading .env."""
    from radar_desk.config import Settings
    from radar_desk.gpu.fake import FakeGpuBackend
    from radar_desk.services import build_services

    def build(backend=None, clock=None, fixtures_root=None, storage=None, nv_backend=None,
              readers_llm=None, readers_runner=None, **overrides):
        settings = Settings(
            _env_file=None,
            owner_token="test-owner",
            session_secret="test-secret",
            data_dir=tmp_path / "data",
            **overrides,
        )
        kw = {"clock": clock} if clock is not None else {}
        if storage is not None:
            kw["storage"] = storage
        if fixtures_root is not None:
            kw["fixtures_root"] = fixtures_root
        if nv_backend is not None:
            kw["nv_backend"] = nv_backend
        if readers_llm is not None:
            kw["readers_llm"] = readers_llm
        if readers_runner is not None:
            kw["readers_runner"] = readers_runner
        return build_services(settings, backend=backend if backend is not None else FakeGpuBackend(), **kw)

    return build


@pytest.fixture
def readers_world(make_services, tmp_path):
    """Services with NV on (fake backends that write a RADAR mask and an NV report), a scripted chat model for the
    reading radiologist, refreshes run inline, and one ready synthetic scan. Returns a namespace with svc, llm, nv,
    poller, scan and clock."""
    from types import SimpleNamespace

    from fake_llm import ScriptedLLM
    from radar_desk.gpu.fake import FakeGpuBackend, synthesize_result
    from radar_desk.gpu.nvreason_fake import FakeNvReasonBackend, synthesize_nv_result
    from radar_desk.gpu.poller import Poller
    from radar_desk.radar import catalog
    from test_chat_tools import ready_scan
    from test_poller import Clock

    def build(replies=(), llm=True, enabled=True, **overrides):
        clock = Clock()
        holder: dict = {}

        def job_and_scan(job_id):
            db = holder["svc"].db
            job = db.get_job(job_id)
            return job, db.get_scan(job.scan_id)

        gpu = FakeGpuBackend(clock=clock, synthesize=lambda job_id: synthesize_result(
            *job_and_scan(job_id), holder["svc"].storage, catalog))
        nv = FakeNvReasonBackend(synthesize=lambda job_id: synthesize_nv_result(*job_and_scan(job_id)))
        model = ScriptedLLM(list(replies)) if llm else None
        svc = make_services(backend=gpu, clock=clock, nv_backend=nv, readers_llm=model,
                            readers_runner=lambda work: work(), nvreason_enabled=enabled, **overrides)
        holder["svc"] = svc
        return SimpleNamespace(svc=svc, llm=model, nv=nv, poller=Poller(svc, gpu, clock=clock),
                               scan=ready_scan(svc, tmp_path), clock=clock)

    return build
