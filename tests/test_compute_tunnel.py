"""The cloudflared quick tunnel, with a fake cloudflared written by the test."""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from radar_desk.compute import tunnel

FAKE_URL = "https://quiet-otter-lamp-test.trycloudflare.com"
BANNER = [
    "2026-10-02T16:31:36Z INF Requesting new quick Tunnel on trycloudflare.com...",
    "2026-10-02T16:31:38Z INF +--------------------------------------------------------------------------------------------+",
    "2026-10-02T16:31:38Z INF |  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  |",
    f"2026-10-02T16:31:38Z INF |  {FAKE_URL}                                           |",
    "2026-10-02T16:31:38Z INF +--------------------------------------------------------------------------------------------+",
]


API_ERROR = ('2026-10-02T16:31:37Z ERR Error unmarshaling QuickTunnel response: error="failed to unmarshal" '
             "url=https://api.trycloudflare.com/tunnel")


def fake_cloudflared(tmp_path, lines=BANNER, name: str = "fake-cloudflared", stay: bool = True) -> str:
    """A fake that prints `lines` to stderr, then sleeps until SIGTERM (or exits when not `stay`).

    A bash wrapper runs Python under the fake's own path as argv[0], so `ps` shows
    `<path>/fake-cloudflared <body> tunnel ...` the way it shows the real binary.
    """
    body = tmp_path / f"{name}.py"
    body.write_text(f"import sys, time\nfor line in {list(lines)!r}:\n    print(line, file=sys.stderr, flush=True)\n"
                    + ("while True:\n    time.sleep(1)\n" if stay else "sys.exit(1)\n"))
    path = tmp_path / name
    path.write_text(f'#!/bin/bash\nexec -a "$0" {sys.executable} {body} "$@"\n')
    path.chmod(0o755)
    return str(path)


def gone(pid: int, binary: str, within_s: float = 5) -> bool:
    deadline = time.monotonic() + within_s
    while time.monotonic() < deadline:
        if not tunnel.alive(pid, binary):
            return True
        time.sleep(0.05)
    return False


def test_start_alive_stop(tmp_path):
    binary = fake_cloudflared(tmp_path)
    t = tunnel.start(8123, tmp_path / "data" / "cloudflared.log", binary)
    try:
        assert t.url == FAKE_URL and t.pid > 0 and t.binary == binary
        assert tunnel.alive(t.pid, binary)
        assert not tunnel.alive(t.pid, "cloudflared-other")
    finally:
        tunnel.stop(t.pid, binary)
    assert gone(t.pid, binary)


def test_tunnel_survives_the_caller(tmp_path):
    binary = fake_cloudflared(tmp_path)
    code = (f"from radar_desk.compute import tunnel\n"
            f"t = tunnel.start(8123, {str(tmp_path / 'cf.log')!r}, {binary!r})\nprint(t.pid, t.url)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30, check=True)
    pid, url = out.stdout.split()
    pid = int(pid)
    try:
        assert url == FAKE_URL
        assert tunnel.alive(pid, binary)
    finally:
        tunnel.stop(pid, binary)
    assert gone(pid, binary)


def test_no_url_times_out_and_kills(tmp_path):
    binary = fake_cloudflared(tmp_path, ["2026-10-02T16:31:36Z INF Starting tunnel"], "mute-cloudflared")
    procs = []

    def popen(*args, **kwargs):
        procs.append(subprocess.Popen(*args, **kwargs))
        return procs[-1]

    with pytest.raises(tunnel.TunnelError, match="no URL"):
        tunnel.start(8123, tmp_path / "cf.log", binary, timeout_s=1, popen=popen)
    assert gone(procs[0].pid, binary)


def test_missing_binary(tmp_path):
    with pytest.raises(tunnel.TunnelError, match="not installed"):
        tunnel.start(8123, tmp_path / "cf.log", str(tmp_path / "no-such-cloudflared"))


def test_alive_needs_the_binary_name():
    assert not tunnel.alive(os.getpid(), "cloudflared")
    assert not tunnel.alive(None, "cloudflared")


def test_public_health_without_an_address():
    assert tunnel.public_health("https://x.trycloudflare.com", resolver=lambda host: None) is None


def test_api_error_line_is_not_the_url(tmp_path):
    binary = fake_cloudflared(tmp_path, [API_ERROR, *BANNER])
    t = tunnel.start(8123, tmp_path / "cf.log", binary)
    try:
        assert t.url == FAKE_URL
    finally:
        tunnel.stop(t.pid, binary)


def test_only_an_api_error_line_fails(tmp_path):
    binary = fake_cloudflared(tmp_path, [API_ERROR], "broken-cloudflared", stay=False)
    with pytest.raises(tunnel.TunnelError, match="exited"):
        tunnel.start(8123, tmp_path / "cf.log", binary, timeout_s=5)


def test_alive_needs_the_binary_as_the_command(tmp_path):
    binary = str(tmp_path / "fake-cloudflared")
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", "fake-cloudflared", "tunnel"])
    try:
        assert not tunnel.alive(proc.pid, binary)
    finally:
        proc.kill()
        proc.wait()
