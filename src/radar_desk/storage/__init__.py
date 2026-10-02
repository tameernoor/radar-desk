"""Object storage: local files, an S3 bucket, or a Modal Volume (STORAGE_BACKEND, design.md Storage and records)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from radar_desk.config import ConfigError
from radar_desk.storage.base import ObjectExists, ObjectMissing, Storage, StorageError, validate_key
from radar_desk.storage.local import LocalStorage
from radar_desk.storage.modal_volume import ModalVolumeStorage
from radar_desk.storage.s3 import S3Storage

__all__ = [
    "LocalStorage",
    "ModalVolumeStorage",
    "ObjectExists",
    "ObjectMissing",
    "S3Storage",
    "Storage",
    "StorageError",
    "make_storage",
    "storage_backend",
    "validate_key",
]


def _plain(value: Any) -> Any:
    """Unwrap a pydantic SecretStr, pass anything else through."""
    return value.get_secret_value() if hasattr(value, "get_secret_value") else value


def storage_backend(settings: Any) -> str:
    """STORAGE_BACKEND when set, else s3 with a bucket and local without one."""
    chosen = getattr(settings, "storage_backend", None)
    if chosen:
        return chosen
    return "s3" if getattr(settings, "s3_bucket", None) else "local"


def make_storage(settings: Any) -> Storage:
    """Build the adapter from settings-like attributes (see the Settings class in config.py)."""
    backend = storage_backend(settings)
    if backend == "modal_volume":
        tid, tsecret = getattr(settings, "modal_token_id", None), getattr(settings, "modal_token_secret", None)
        creds = (_plain(tid), _plain(tsecret)) if tid is not None and tsecret is not None else None
        return ModalVolumeStorage(settings.modal_data_volume, settings.public_base_url, credentials=creds)
    if backend == "s3":
        if not getattr(settings, "s3_bucket", None):
            raise ConfigError("STORAGE_BACKEND=s3 needs S3_BUCKET")
        return S3Storage(
            bucket=settings.s3_bucket,
            endpoint_url=settings.s3_endpoint_url,
            region=settings.s3_region,
            access_key=_plain(settings.aws_access_key_id),
            secret_key=_plain(settings.aws_secret_access_key),
        )
    return LocalStorage(Path(settings.data_dir), settings.public_base_url, _plain(settings.session_secret))
