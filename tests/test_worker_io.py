"""Tests for worker/radar_worker/io.py against a local http.server thread."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import pytest

from radar_worker import io as wio

PAYLOAD = bytes(range(256)) * 4096  # 1 MiB


class _Store:
    objects: ClassVar[dict[str, bytes]] = {}
    headers: ClassVar[dict[str, dict[str, str]]] = {}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep test output quiet
        pass

    def _deny(self):
        body = b"<Error><Code>AccessDenied</Code><Message>Request has expired</Message></Error>"
        self.send_response(403)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/deny"):
            return self._deny()
        data = _Store.objects.get(self.path)
        if data is None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_PUT(self):
        length = int(self.headers.get("Content-Length", "-1"))
        data = self.rfile.read(length) if length >= 0 else b""
        if self.path.startswith("/deny"):
            return self._deny()
        _Store.objects[self.path] = data
        _Store.headers[self.path] = dict(self.headers.items())
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()


@pytest.fixture()
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    _Store.objects = {"/src/scan.nii.gz": PAYLOAD}
    _Store.headers = {}
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_download_streams_to_disk(server, tmp_path):
    dest = tmp_path / "in" / "scan.nii.gz"
    n = wio.download(f"{server}/src/scan.nii.gz", dest)
    assert n == len(PAYLOAD)
    assert dest.read_bytes() == PAYLOAD
    assert not list(dest.parent.glob("*.part"))


def test_download_4xx_raises_with_status(server, tmp_path):
    with pytest.raises(wio.TransferError) as exc:
        wio.download(f"{server}/deny/scan.nii.gz", tmp_path / "x")
    assert exc.value.status == 403
    assert "AccessDenied" in exc.value.body_snippet
    assert not (tmp_path / "x").exists()
    with pytest.raises(wio.TransferError) as exc404:
        wio.download(f"{server}/missing", tmp_path / "y")
    assert exc404.value.status == 404


def test_download_unreachable_host_raises(tmp_path):
    with pytest.raises(wio.TransferError) as exc:
        wio.download("http://127.0.0.1:9/nothing", tmp_path / "z", timeout=5)
    assert exc.value.status == 0


def test_upload_puts_file_with_length_and_type(server, tmp_path):
    src = tmp_path / "mask.nii.gz"
    src.write_bytes(PAYLOAD)
    wio.upload(src, f"{server}/out/mask.nii.gz?sig=abc", "application/gzip")
    assert _Store.objects["/out/mask.nii.gz?sig=abc"] == PAYLOAD
    hdrs = {k.lower(): v for k, v in _Store.headers["/out/mask.nii.gz?sig=abc"].items()}
    assert hdrs["content-length"] == str(len(PAYLOAD))
    assert hdrs["content-type"] == "application/gzip"


def test_upload_bytes(server):
    wio.upload_bytes(b'{"ok": true}', f"{server}/out/scores.json", "application/json")
    assert _Store.objects["/out/scores.json"] == b'{"ok": true}'


def test_upload_4xx_raises_with_status(server, tmp_path):
    src = tmp_path / "f"
    src.write_bytes(b"abc")
    with pytest.raises(wio.TransferError) as exc:
        wio.upload(src, f"{server}/deny/f", "text/plain")
    assert exc.value.status == 403
    assert "expired" in str(exc.value)
