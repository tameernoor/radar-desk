"""Settings from the environment and `.env` in the working directory.

Secrets are SecretStr, so `repr` and logs show `**********`. `GPU_BACKEND` defaults to `fake`;
a Modal token in `.env` never starts GPU spend by itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from radar_desk.compute.runpod import DEFAULT_GPUS, parse_gpus

DEFAULT_GPU_PRICES_USD_PER_S: dict[str, float] = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10": 0.000306,
    "L40S": 0.000542,
    "A100-40GB": 0.000583,
    "A100": 0.000583,
    "A100-80GB": 0.000694,
    "H100": 0.001097,
}


class ConfigError(RuntimeError):
    """A required setting is missing or invalid. The message names the variable."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )

    owner_token: SecretStr
    session_secret: SecretStr
    public_base_url: str = "http://127.0.0.1:8000"
    app_hostname: str | None = None
    data_dir: Path = Path("./data")

    # None means s3 when S3_BUCKET is set, else local; see storage.storage_backend
    storage_backend: Literal["local", "s3", "modal_volume"] | None = None
    modal_data_volume: str = "radar-data"

    s3_bucket: str | None = None
    s3_endpoint_url: str | None = None
    s3_region: str | None = None
    aws_access_key_id: SecretStr | None = None
    aws_secret_access_key: SecretStr | None = None

    gpu_backend: Literal["fake", "modal", "worker"] = "fake"
    modal_token_id: SecretStr | None = None
    modal_token_secret: SecretStr | None = None
    modal_app_name: str = "radar-desk"
    modal_function_name: str = "score"
    radar_gpu: str = "L4"
    gpu_timeout_s: int = 1800
    gpu_scaledown_window_s: int = 120
    gpu_monthly_budget_usd: float = 10.0
    gpu_poll_interval_s: float = 10.0
    gpu_prices_usd_per_s: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_GPU_PRICES_USD_PER_S))
    worker_lease_s: int = Field(default=120, ge=10)
    worker_image: str | None = None

    # RunPod pods started by the app (plan.md, Compute switch B)
    worker_public_url: str | None = None
    worker_tunnel: Literal["managed", "external"] | None = None  # None means inferred from WORKER_PUBLIC_URL
    runpod_api_key: SecretStr | None = None
    runpod_volume_id: str | None = None
    runpod_registry_auth_id: str | None = None
    runpod_datacenter: str = "EU-RO-1"
    runpod_gpus: str = DEFAULT_GPUS  # RunPod gpuTypeIds in order; compute/runpod.py owns the check
    runpod_max_pod_hours: float = 3
    runpod_start_timeout_s: float = 600
    radar_pod_idle_delete_s: float = Field(default=600, ge=1)  # passed to the pod, which deletes itself
    radar_pod_app_lost_delete_s: float = Field(default=600, ge=1)  # passed to the pod, likewise

    llm_api_key: SecretStr | None = None
    llm_provider: Literal["openrouter", "ollama", "openai"] | None = None  # None means inferred from the URL
    llm_base_url: str | None = None  # None means the provider's default
    chat_model: str | None = None

    max_upload_bytes: int = 314_572_800

    @field_validator("runpod_gpus")
    @classmethod
    def _gpus_usable(cls, value: str) -> str:
        parse_gpus(value, warn=True)  # the one place an unknown id is logged
        return value

    @model_validator(mode="after")
    def _tunnel_matches_url(self) -> Settings:
        if self.worker_tunnel == "external" and not self.worker_public_url:
            raise ValueError("WORKER_TUNNEL=external needs WORKER_PUBLIC_URL")
        if self.worker_tunnel == "managed" and self.worker_public_url:
            raise ValueError("WORKER_TUNNEL=managed starts its own tunnel; unset WORKER_PUBLIC_URL")
        return self

    @property
    def tunnel_mode(self) -> str:
        """WORKER_TUNNEL when set, else external with WORKER_PUBLIC_URL and managed without."""
        return self.worker_tunnel or ("external" if self.worker_public_url else "managed")

    @property
    def runpod_configured(self) -> bool:
        """Whether the app can start pods: the API key, volume, registry auth and image are all set."""
        return bool(self.runpod_api_key and self.runpod_volume_id and self.runpod_registry_auth_id
                    and self.worker_image)

    @property
    def gpu_list(self) -> list[str]:
        """RADAR_GPU as an ordered list, so "L4,A10" gives ["L4", "A10"]."""
        return [g.strip() for g in self.radar_gpu.split(",") if g.strip()]

    @property
    def runpod_gpu_list(self) -> list[str]:
        """RUNPOD_GPUS as an ordered list of RunPod gpuTypeIds."""
        return parse_gpus(self.runpod_gpus)

    @property
    def resolved_llm_provider(self) -> str:
        """LLM_PROVIDER when set, else inferred from LLM_BASE_URL (see chat/providers.py)."""
        from radar_desk.chat.providers import resolved_provider_name

        return resolved_provider_name(self)

    @property
    def resolved_llm_base_url(self) -> str:
        """LLM_BASE_URL when set, else the default base URL of LLM_PROVIDER."""
        from radar_desk.chat.providers import resolved_base_url

        return resolved_base_url(self)

    @property
    def db_path(self) -> Path:
        return Path(self.data_dir) / "radar-desk.sqlite3"


def load_settings(**overrides: Any) -> Settings:
    """Build Settings, turning a missing required value into a ConfigError that names it."""
    try:
        return Settings(**overrides)
    except ValidationError as exc:
        problems = []
        for err in exc.errors():
            name = ".".join(str(p) for p in err["loc"]).upper()
            if err["type"] == "missing":
                problems.append(f"{name} is not set")
            elif not name:  # a rule across settings; its message names them
                problems.append(err["msg"].removeprefix("Value error, "))
            else:
                msg = err["msg"].removeprefix("Value error, ")
                problems.append(msg if msg.startswith(f"{name}") else f"{name}: {msg}")
        raise ConfigError("invalid settings: " + "; ".join(problems)) from None
