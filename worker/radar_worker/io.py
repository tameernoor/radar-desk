"""Plain HTTP transfers against presigned URLs, standard library only.

`download` streams a GET to disk; `upload` and `upload_bytes` PUT with an explicit
Content-Length, which S3-style presigned PUTs require. Any non-2xx answer or a network
failure raises `TransferError`; network failures carry status 0.
"""

from __future__ import annotations

import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path

CHUNK = 1 << 20


class TransferError(Exception):
    def __init__(self, status: int, body_snippet: str, url: str = ""):
        self.status = int(status)
        self.body_snippet = body_snippet
        self.url = url
        where = url.split("?", 1)[0]
        super().__init__(f"HTTP {status} for {where}: {body_snippet}")


def _snippet(data: bytes, limit: int = 300) -> str:
    return data[:limit].decode("utf-8", "replace")


def _open(req: urllib.request.Request, timeout: float):
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as err:
        try:
            body = err.read()
        except OSError:
            body = b""
        raise TransferError(err.code, _snippet(body), req.full_url) from None
    except (urllib.error.URLError, OSError) as err:
        raise TransferError(0, str(getattr(err, "reason", err)), req.full_url) from None


def download(url: str, path, timeout: float = 600) -> int:
    """GET `url` into `path` through a temporary file; returns the byte count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    req = urllib.request.Request(url, method="GET")
    with _open(req, timeout) as resp:
        status = getattr(resp, "status", 200)
        if not 200 <= status < 300:
            raise TransferError(status, _snippet(resp.read(300)), url)
        try:
            with open(part, "wb") as fh:
                shutil.copyfileobj(resp, fh, CHUNK)
        except OSError as err:
            part.unlink(missing_ok=True)
            raise TransferError(0, f"read failed: {err}", url) from None
    os.replace(part, path)
    return path.stat().st_size


def _put(url: str, body, length: int, content_type: str, timeout: float) -> None:
    req = urllib.request.Request(
        url,
        data=body,
        method="PUT",
        headers={"Content-Length": str(length), "Content-Type": content_type},
    )
    with _open(req, timeout) as resp:
        status = getattr(resp, "status", 200)
        tail = resp.read()
        if not 200 <= status < 300:
            raise TransferError(status, _snippet(tail), url)


def upload(path, url: str, content_type: str, timeout: float = 600) -> None:
    """PUT the file at `path` to `url` with Content-Length set."""
    path = Path(path)
    size = path.stat().st_size
    with open(path, "rb") as fh:
        _put(url, fh, size, content_type, timeout)


def upload_bytes(data: bytes, url: str, content_type: str, timeout: float = 600) -> None:
    """PUT `data` to `url` with Content-Length set."""
    _put(url, bytes(data), len(data), content_type, timeout)
