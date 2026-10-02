from __future__ import annotations

import logging
from pathlib import Path

import pytest

from radar_desk.config import ConfigError, Settings, load_settings

ENV_NAMES = [
    "OWNER_TOKEN", "SESSION_SECRET", "GPU_BACKEND", "MAX_UPLOAD_BYTES", "GPU_MONTHLY_BUDGET_USD",
    "RADAR_GPU", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "LLM_PROVIDER", "LLM_API_KEY", "LLM_BASE_URL", "DATA_DIR", "CHAT_MODEL",
    "STORAGE_BACKEND", "MODAL_DATA_VOLUME", "S3_BUCKET", "WORKER_LEASE_S",
    "WORKER_IMAGE", "WORKER_PUBLIC_URL", "WORKER_TUNNEL", "RUNPOD_API_KEY", "RUNPOD_VOLUME_ID",
    "RUNPOD_REGISTRY_AUTH_ID", "RUNPOD_DATACENTER", "RUNPOD_MAX_POD_HOURS",
    "RUNPOD_START_TIMEOUT_S", "RUNPOD_GPUS", "RADAR_POD_IDLE_DELETE_S", "RADAR_POD_APP_LOST_DELETE_S",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_loads_from_env_with_defaults(monkeypatch):
    monkeypatch.setenv("OWNER_TOKEN", "owner-secret-value")
    monkeypatch.setenv("SESSION_SECRET", "session-secret-value")
    s = load_settings(_env_file=None)
    assert s.owner_token.get_secret_value() == "owner-secret-value"
    assert s.gpu_backend == "fake"
    assert s.max_upload_bytes == 300 * 1024 * 1024
    assert s.gpu_monthly_budget_usd == 10.0
    assert s.gpu_timeout_s == 1800 and s.gpu_scaledown_window_s == 120
    assert s.gpu_prices_usd_per_s["L4"] == 0.000222
    assert s.gpu_prices_usd_per_s["H100"] == 0.001097
    assert s.gpu_list == ["L4"]
    assert s.llm_api_key is None
    assert s.llm_provider is None and s.llm_base_url is None
    assert s.resolved_llm_provider == "openrouter"
    assert s.resolved_llm_base_url == "https://openrouter.ai/api/v1"


def test_loads_from_env_file(tmp_path: Path):
    env = tmp_path / "test.env"
    env.write_text(
        "OWNER_TOKEN=a\nSESSION_SECRET=b\nRADAR_GPU=L4,A10\nGPU_MONTHLY_BUDGET_USD=25\n"
        "MODAL_PROFILE=someone\nCHAT_MODEL=\n"
    )
    s = load_settings(_env_file=env)
    assert s.gpu_list == ["L4", "A10"]
    assert s.gpu_monthly_budget_usd == 25.0
    assert s.chat_model is None  # empty means unset


@pytest.mark.parametrize("missing", ["OWNER_TOKEN", "SESSION_SECRET"])
def test_missing_secret_fails_fast(monkeypatch, missing):
    monkeypatch.setenv("OWNER_TOKEN", "x")
    monkeypatch.setenv("SESSION_SECRET", "y")
    monkeypatch.delenv(missing)
    with pytest.raises(ConfigError, match=missing):
        load_settings(_env_file=None)


def test_secrets_never_shown(monkeypatch, caplog):
    values = {
        "OWNER_TOKEN": "tok-AAAA1111",
        "SESSION_SECRET": "sess-BBBB2222",
        "MODAL_TOKEN_ID": "ak-CCCC3333",
        "MODAL_TOKEN_SECRET": "as-DDDD4444",
        "LLM_API_KEY": "or-EEEE5555",
    }
    for k, v in values.items():
        monkeypatch.setenv(k, v)
    s = Settings(_env_file=None)
    shown = repr(s) + str(s) + str(s.model_dump())
    with caplog.at_level(logging.INFO):
        logging.getLogger("test").info("settings %s", s)
    shown += caplog.text
    for v in values.values():
        assert v not in shown


def test_invalid_value_error_does_not_echo_input(monkeypatch):
    monkeypatch.setenv("OWNER_TOKEN", "tok-secret-9999")
    monkeypatch.setenv("SESSION_SECRET", "s")
    monkeypatch.setenv("GPU_BACKEND", "cloud")
    with pytest.raises(ConfigError) as info:
        load_settings(_env_file=None)
    assert "GPU_BACKEND" in str(info.value)
    assert "tok-secret-9999" not in str(info.value)


def test_worker_lease_has_a_lower_bound(monkeypatch):
    monkeypatch.setenv("OWNER_TOKEN", "o")
    monkeypatch.setenv("SESSION_SECRET", "s")
    monkeypatch.setenv("WORKER_LEASE_S", "5")
    with pytest.raises(ConfigError, match="WORKER_LEASE_S"):
        load_settings(_env_file=None)


# Storage backend


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, owner_token="o", session_secret="s", **kw)


def test_storage_backend_selection(tmp_path: Path):
    from radar_desk.storage import LocalStorage, ModalVolumeStorage, S3Storage, make_storage, storage_backend

    base = {"data_dir": tmp_path}
    assert _settings(**base).storage_backend is None
    assert _settings(**base).modal_data_volume == "radar-data"
    cases = [
        ({}, "local", LocalStorage),
        ({"s3_bucket": "b"}, "s3", S3Storage),
        ({"storage_backend": "local", "s3_bucket": "b"}, "local", LocalStorage),
        ({"storage_backend": "s3", "s3_bucket": "b"}, "s3", S3Storage),
        ({"storage_backend": "modal_volume", "s3_bucket": "b"}, "modal_volume", ModalVolumeStorage),
    ]
    for kw, name, cls in cases:
        s = _settings(**base, **kw)
        assert storage_backend(s) == name, kw
        assert isinstance(make_storage(s), cls), kw


def test_storage_backend_from_env(monkeypatch, tmp_path: Path):
    from radar_desk.storage import make_storage

    monkeypatch.setenv("OWNER_TOKEN", "o")
    monkeypatch.setenv("SESSION_SECRET", "s")
    monkeypatch.setenv("STORAGE_BACKEND", "modal_volume")
    monkeypatch.setenv("MODAL_DATA_VOLUME", "radar-data-dev")
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-x")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-y")
    s = load_settings(_env_file=None, data_dir=tmp_path)
    storage = make_storage(s)
    assert storage.volume_name == "radar-data-dev" and storage._volume is None  # nothing looked up yet
    assert storage._credentials == ("ak-x", "as-y")


def test_storage_backend_rejects_unknown_and_s3_without_bucket(monkeypatch, tmp_path: Path):
    from radar_desk.storage import make_storage

    monkeypatch.setenv("OWNER_TOKEN", "o")
    monkeypatch.setenv("SESSION_SECRET", "s")
    monkeypatch.setenv("STORAGE_BACKEND", "gcs")
    with pytest.raises(ConfigError, match="STORAGE_BACKEND"):
        load_settings(_env_file=None)
    with pytest.raises(ConfigError, match="STORAGE_BACKEND=s3 needs S3_BUCKET"):
        make_storage(_settings(data_dir=tmp_path, storage_backend="s3"))


def test_modal_gpu_needs_a_bucket_or_the_volume(make_services):
    from radar_desk.app import create_app

    svc = make_services()
    modal = {"gpu_backend": "modal"}
    with pytest.raises(ConfigError, match="needs S3_BUCKET or STORAGE_BACKEND=modal_volume"):
        create_app(svc.settings.model_copy(update=modal), svc, start_poller=False)
    with pytest.raises(ConfigError):
        create_app(svc.settings.model_copy(update={**modal, "storage_backend": "local", "s3_bucket": "b"}),
                   svc, start_poller=False)
    for ok in ({"storage_backend": "modal_volume"}, {"s3_bucket": "b"}):
        create_app(svc.settings.model_copy(update={**modal, **ok}), svc, start_poller=False)


# Compute switch


def test_tunnel_mode_is_inferred_from_the_public_url():
    assert _settings().tunnel_mode == "managed"
    assert _settings(worker_public_url="https://desk.example").tunnel_mode == "external"
    assert _settings(worker_tunnel="managed").tunnel_mode == "managed"
    external = _settings(worker_tunnel="external", worker_public_url="https://desk.example")
    assert external.tunnel_mode == "external"


def test_tunnel_mode_contradictions_are_config_errors(monkeypatch):
    monkeypatch.setenv("OWNER_TOKEN", "o")
    monkeypatch.setenv("SESSION_SECRET", "s")
    monkeypatch.setenv("WORKER_TUNNEL", "external")
    with pytest.raises(ConfigError, match="WORKER_TUNNEL=external needs WORKER_PUBLIC_URL"):
        load_settings(_env_file=None)
    monkeypatch.setenv("WORKER_TUNNEL", "managed")
    monkeypatch.setenv("WORKER_PUBLIC_URL", "https://desk.example")
    with pytest.raises(ConfigError, match="WORKER_TUNNEL=managed starts its own tunnel"):
        load_settings(_env_file=None)


def test_runpod_settings_and_defaults(monkeypatch):
    s = _settings()
    assert s.worker_image is None and not s.runpod_configured
    assert (s.runpod_datacenter, s.runpod_max_pod_hours, s.runpod_start_timeout_s) == ("EU-RO-1", 3, 600)
    assert (s.radar_pod_idle_delete_s, s.radar_pod_app_lost_delete_s) == (600, 600)
    for bad in (0, 0.5):
        with pytest.raises(ValueError):
            _settings(radar_pod_idle_delete_s=bad)
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-secret-7777")
    full = _settings(runpod_volume_id="v", runpod_registry_auth_id="r", worker_image="img")
    assert full.runpod_configured and "rp-secret-7777" not in repr(full)
    assert not _settings(runpod_volume_id="v", runpod_registry_auth_id="r").runpod_configured


def test_runpod_gpus_default_and_custom_order(monkeypatch):
    monkeypatch.setenv("OWNER_TOKEN", "x")
    monkeypatch.setenv("SESSION_SECRET", "y")
    assert load_settings(_env_file=None).runpod_gpu_list == ["NVIDIA L4", "NVIDIA GeForce RTX 4090"]
    monkeypatch.setenv("RUNPOD_GPUS", "NVIDIA GeForce RTX 4090, NVIDIA L4")
    assert load_settings(_env_file=None).runpod_gpu_list == ["NVIDIA GeForce RTX 4090", "NVIDIA L4"]


@pytest.mark.parametrize("value", ["NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "B200", " , "])
def test_runpod_gpus_refuses_blackwell_and_empty(monkeypatch, value):
    monkeypatch.setenv("OWNER_TOKEN", "x")
    monkeypatch.setenv("SESSION_SECRET", "y")
    monkeypatch.setenv("RUNPOD_GPUS", value)
    with pytest.raises(ConfigError, match="RUNPOD_GPUS") as info:
        load_settings(_env_file=None)
    assert "Value error" not in str(info.value)


def test_runpod_gpus_unknown_id_warns(monkeypatch, caplog):
    monkeypatch.setenv("OWNER_TOKEN", "x")
    monkeypatch.setenv("SESSION_SECRET", "y")
    monkeypatch.setenv("RUNPOD_GPUS", "NVIDIA RTX A2000")
    with caplog.at_level(logging.WARNING):
        s = load_settings(_env_file=None)
    assert s.runpod_gpu_list == ["NVIDIA RTX A2000"]
    assert "NVIDIA RTX A2000 is not a GPU type this app knows" in caplog.text
