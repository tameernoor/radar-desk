"""Shared pytest fixtures. Add fixtures here only when more than one test module needs them."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

FIXTURE_DIR = Path(os.environ.get("RADAR_FIXTURE_DIR", "/Users/lars/Downloads/merlin-radar"))
FIXTURE_FILES = {
    "AC4214dbd": "merlin-AC4214dbd.nii.gz",
    "AC4240fff": "merlin-AC4240fff.nii.gz",
    "AC4242a2f": "merlin-AC4242a2f.nii.gz",
    "AC4242a55": "merlin-AC4242a55.nii.gz",
    "image1": "merlin-image1.nii.gz",
}


LLM_ENV = ("LLM_PROVIDER", "LLM_BASE_URL", "LLM_API_KEY", "CHAT_MODEL")


@pytest.fixture(autouse=True)
def no_llm_env(monkeypatch):
    """A developer's shell settings must not make a test configure chat and run a real provider check."""
    for name in LLM_ENV:
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

    def build(backend=None, clock=None, fixtures_root=None, storage=None, **overrides):
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
        return build_services(settings, backend=backend if backend is not None else FakeGpuBackend(), **kw)

    return build
