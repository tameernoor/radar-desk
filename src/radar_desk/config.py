"""Settings from the environment and `.env` in the working directory.

Secrets are SecretStr, so `repr` and logs show `**********`. `GPU_BACKEND` defaults to `fake`;
a Modal token in `.env` never starts GPU spend by itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

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

    gpu_backend: Literal["fake", "modal"] = "fake"
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

    llm_api_key: SecretStr | None = None
    llm_base_url: str = "https://openrouter.ai/api/v1"
    chat_model: str | None = None

    max_upload_bytes: int = 314_572_800

    @property
    def gpu_list(self) -> list[str]:
        """RADAR_GPU as an ordered list, so "L4,A10" gives ["L4", "A10"]."""
        return [g.strip() for g in self.radar_gpu.split(",") if g.strip()]

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
            else:
                problems.append(f"{name}: {err['msg']}")
        raise ConfigError("invalid settings: " + "; ".join(problems)) from None
