"""S3-compatible bucket (Tigris in production) through boto3. Presigning needs no network."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from radar_desk.storage.base import CHUNK_BYTES, ObjectMissing, StorageError, UrlWorkerRefs, validate_key

_MISSING_CODES = {"404", "NoSuchKey", "NotFound"}


class S3Storage(UrlWorkerRefs):
    def __init__(
        self,
        bucket: str,
        endpoint_url: str | None,
        region: str | None,
        access_key: str | None,
        secret_key: str | None,
        client: Any = None,
        config: Config | None = None,
    ) -> None:
        self.bucket = bucket
        self.client = client or boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=config or Config(signature_version="s3v4"),
        )

    def _params(self, key: str) -> dict[str, str]:
        return {"Bucket": self.bucket, "Key": validate_key(key)}

    def put_url(self, key: str, expires_s: int) -> str:
        return self.client.generate_presigned_url("put_object", Params=self._params(key), ExpiresIn=expires_s)

    def get_url(self, key: str, expires_s: int) -> str:
        return self.client.generate_presigned_url("get_object", Params=self._params(key), ExpiresIn=expires_s)

    def _head(self, key: str) -> dict | None:
        try:
            return self.client.head_object(**self._params(key))
        except ClientError as exc:
            if str(exc.response.get("Error", {}).get("Code")) in _MISSING_CODES:
                return None
            raise StorageError(f"head_object {key}: {exc}") from exc
        except BotoCoreError as exc:
            raise StorageError(f"head_object {key}: {exc}") from exc

    def exists(self, key: str) -> bool:
        return self._head(key) is not None

    def size(self, key: str) -> int:
        head = self._head(key)
        if head is None:
            raise ObjectMissing(key)
        return int(head["ContentLength"])

    def open_stream(self, key: str) -> Iterator[bytes]:
        try:
            body = self.client.get_object(**self._params(key))["Body"]
        except ClientError as exc:
            if str(exc.response.get("Error", {}).get("Code")) in _MISSING_CODES:
                raise ObjectMissing(key) from exc
            raise StorageError(f"get_object {key}: {exc}") from exc
        except BotoCoreError as exc:
            raise StorageError(f"get_object {key}: {exc}") from exc
        return body.iter_chunks(CHUNK_BYTES)

    def put_bytes(self, key: str, data: bytes, content_type: str | None = None) -> None:
        extra = {"ContentType": content_type} if content_type else {}
        try:
            self.client.put_object(**self._params(key), Body=data, **extra)
        except (ClientError, BotoCoreError) as exc:
            raise StorageError(f"put_object {key}: {exc}") from exc

    def put_file(self, key: str, path: Path, content_type: str | None = None) -> None:
        """put_object with the open file as the body, so boto3 reads it from disk."""
        extra = {"ContentType": content_type} if content_type else {}
        try:
            with open(path, "rb") as fh:
                self.client.put_object(**self._params(key), Body=fh, **extra)
        except (ClientError, BotoCoreError) as exc:
            raise StorageError(f"put_object {key}: {exc}") from exc

    def delete(self, key: str) -> None:
        """Deleting a missing object is fine, as in the other adapters; S3 answers 204 but others may 404."""
        try:
            self.client.delete_object(**self._params(key))
        except ClientError as exc:
            if str(exc.response.get("Error", {}).get("Code")) in _MISSING_CODES:
                return
            raise StorageError(f"delete_object {key}: {exc}") from exc
        except BotoCoreError as exc:
            raise StorageError(f"delete_object {key}: {exc}") from exc
