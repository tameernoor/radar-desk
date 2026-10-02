"""Objects on a RunPod network volume, through the volume's S3-compatible API.

The serverless worker mounts the same volume at `/runpod-volume`, so key `a/b` is the file
`/runpod-volume/a/b` there and the worker gets `volume://{key}` references. The S3 API signs no
presigned URLs, so the browser goes through the API like on a Modal Volume: `put_url` and `get_url`
are the relative `/_volume/{key}` routes (routes/volume_storage.py). Reads, writes and deletes are the
S3 adapter's HeadObject, GetObject, PutObject and DeleteObject, all supported by RunPod.

Path-style addressing is set explicitly because the docs never name a style; every documented example
amounts to path style. A single PutObject is capped at 500 MB, so the adapter refuses an upload limit
at or above that at construction rather than at the first big scan (the fix would be multipart).
"""

from __future__ import annotations

from typing import Any

from botocore.config import Config

from radar_desk.config import ConfigError
from radar_desk.storage.base import validate_key
from radar_desk.storage.modal_volume import VOLUME_SCHEME
from radar_desk.storage.s3 import S3Storage

PUT_OBJECT_CAP = 500 * 1024 * 1024


class RunPodVolumeStorage(S3Storage):
    name = "runpod_volume"
    browser_via_api = True  # app.py mounts routes/volume_storage.py

    def __init__(self, volume_id: str, datacenter: str, access_key: str | None, secret_key: str | None,
                 max_upload_bytes: int, client: Any = None) -> None:
        if max_upload_bytes >= PUT_OBJECT_CAP:
            raise ConfigError(f"MAX_UPLOAD_BYTES must be under {PUT_OBJECT_CAP} on the RunPod volume "
                              f"(one PutObject); it is {max_upload_bytes}")
        super().__init__(
            bucket=volume_id,
            endpoint_url=f"https://s3api-{datacenter.lower()}.runpod.io",
            region=datacenter,
            access_key=access_key,
            secret_key=secret_key,
            client=client,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )

    @staticmethod
    def _url(key: str) -> str:
        return f"/_volume/{validate_key(key)}"

    def put_url(self, key: str, expires_s: int) -> str:
        return self._url(key)

    def get_url(self, key: str, expires_s: int) -> str:
        return self._url(key)

    def worker_ref(self, key: str, method: str, expires_s: int) -> str:
        if method not in ("GET", "PUT"):
            raise ValueError(f"method must be GET or PUT, not {method!r}")
        return VOLUME_SCHEME + validate_key(key)
