"""The storage interface shared by the local, S3 and Modal Volume adapters."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Literal, Protocol

CHUNK_BYTES = 1024 * 1024


class StorageError(Exception):
    pass


class ObjectMissing(StorageError):
    pass


class ObjectExists(StorageError):
    """A write to a key that is already taken; objects are never overwritten."""


def validate_key(key: str) -> str:
    """Reject keys that could escape the bucket or the objects folder. Returns the key."""
    if not key:
        raise StorageError("empty key")
    if key.startswith("/"):
        raise StorageError(f"key must be relative: {key!r}")
    if "\\" in key:
        raise StorageError(f"backslash in key: {key!r}")
    if any(ord(c) < 32 or ord(c) == 127 for c in key):
        raise StorageError(f"control character in key: {key!r}")
    if any(part in ("", ".", "..") for part in key.split("/")):
        raise StorageError(f"bad path segment in key: {key!r}")
    return key


class Storage(Protocol):
    def put_url(self, key: str, expires_s: int) -> str: ...

    def get_url(self, key: str, expires_s: int) -> str: ...

    def worker_ref(self, key: str, method: Literal["GET", "PUT"], expires_s: int) -> str:
        """The reference the GPU worker gets for this key: a presigned URL, or a Volume path."""
        ...

    def exists(self, key: str) -> bool: ...

    def size(self, key: str) -> int: ...

    def open_stream(self, key: str) -> Iterator[bytes]: ...

    def put_bytes(self, key: str, data: bytes, content_type: str | None = None) -> None: ...

    def put_file(self, key: str, path: Path, content_type: str | None = None) -> None:
        """Store a local file under key, streaming it from disk. The file is left in place."""
        ...

    def delete(self, key: str) -> None: ...


class UrlWorkerRefs:
    """worker_ref for adapters whose worker reaches objects over presigned URLs (put_url, get_url)."""

    def worker_ref(self, key: str, method: str, expires_s: int) -> str:
        if method == "GET":
            return self.get_url(key, expires_s)
        if method == "PUT":
            return self.put_url(key, expires_s)
        raise ValueError(f"method must be GET or PUT, not {method!r}")
