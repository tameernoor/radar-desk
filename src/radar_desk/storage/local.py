"""Local stand-in for the bucket: files under root/objects and HMAC-signed URLs.

The API serves the URLs on PUT and GET /_storage/{key}, checking them with `verify`. The method is
part of the signature, so a GET URL cannot be used to upload.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import tempfile
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from urllib.parse import quote

from radar_desk.storage.base import CHUNK_BYTES, ObjectMissing, StorageError, UrlWorkerRefs, validate_key


class LocalStorage(UrlWorkerRefs):
    def __init__(self, root: Path, public_base_url: str, secret: str) -> None:
        self.objects = (Path(root) / "objects").resolve()
        self.objects.mkdir(parents=True, exist_ok=True)
        self.public_base_url = public_base_url.rstrip("/")
        self._secret = secret.encode()

    def path_for(self, key: str) -> Path:
        path = (self.objects / validate_key(key)).resolve()
        if not path.is_relative_to(self.objects):
            raise StorageError(f"key escapes the storage root: {key!r}")
        return path

    def _sign(self, method: str, key: str, exp: int) -> str:
        msg = f"{method}\n{key}\n{exp}".encode()
        return hmac.new(self._secret, msg, hashlib.sha256).hexdigest()

    def _url(self, method: str, key: str, expires_s: int) -> str:
        validate_key(key)
        exp = int(time.time()) + int(expires_s)
        sig = self._sign(method, key, exp)
        return f"{self.public_base_url}/_storage/{quote(key, safe='/')}?exp={exp}&sig={sig}"

    def put_url(self, key: str, expires_s: int) -> str:
        return self._url("PUT", key, expires_s)

    def get_url(self, key: str, expires_s: int) -> str:
        return self._url("GET", key, expires_s)

    def verify(self, method: str, key: str, exp: int | str, sig: str, now: int | None = None) -> bool:
        try:
            validate_key(key)
            exp_int = int(exp)
        except (StorageError, ValueError, TypeError):
            return False
        if exp_int < (int(time.time()) if now is None else now):
            return False
        expected = self._sign(method.upper(), key, exp_int)
        return hmac.compare_digest(expected, str(sig))

    def exists(self, key: str) -> bool:
        return self.path_for(key).is_file()

    def size(self, key: str) -> int:
        path = self.path_for(key)
        if not path.is_file():
            raise ObjectMissing(key)
        return path.stat().st_size

    def open_stream(self, key: str) -> Iterator[bytes]:
        path = self.path_for(key)
        if not path.is_file():
            raise ObjectMissing(key)
        return self._read(path)

    @staticmethod
    def _read(path: Path) -> Iterator[bytes]:
        with path.open("rb") as fh:
            while chunk := fh.read(CHUNK_BYTES):
                yield chunk

    def write_stream(self, key: str, chunks: Iterable[bytes]) -> int:
        """Write chunks to a temp file and move it into place. Returns the byte count."""
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".upload-")
        written = 0
        try:
            with os.fdopen(fd, "wb") as fh:
                for chunk in chunks:
                    fh.write(chunk)
                    written += len(chunk)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return written

    def put_bytes(self, key: str, data: bytes, content_type: str | None = None) -> None:
        self.write_stream(key, [data])

    def put_file(self, key: str, path: Path, content_type: str | None = None) -> None:
        """Copy the file in chunks; the caller still owns the original."""
        self.write_stream(key, self._read(Path(path)))

    def delete(self, key: str) -> None:
        self.path_for(key).unlink(missing_ok=True)
