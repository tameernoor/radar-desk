"""A cloudflared quick tunnel to the app, and a reachability probe that does not trust the Mac's resolver.

The tunnel runs in its own session with its log as stdout and stderr, so it outlives the command that
started it. A fresh trycloudflare.com name reaches 1.1.1.1 before the local resolver, so the probe
resolves there and connects to that address with the name as SNI.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import shutil
import signal
import socket
import ssl
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

URL_RE = re.compile(r"\|\s+(https://[a-z0-9-]+\.trycloudflare\.com)")  # the banner line only
IPV4_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


class TunnelError(RuntimeError):
    """cloudflared did not start or did not print a URL in time."""


@dataclass
class Tunnel:
    pid: int
    url: str
    log: str
    binary: str


def start(port: int, log: Path | str, binary: str = "cloudflared", timeout_s: float = 30,
          popen: Callable = subprocess.Popen) -> Tunnel:
    """Start a quick tunnel to 127.0.0.1:`port` and return it once its URL is in the log."""
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [binary, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"]
    with open(log, "wb") as out:
        try:
            # A bare environment, so the long-lived tunnel never holds secrets from a shell that sourced .env.
            env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "")}
            proc = popen(cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, env=env,
                         start_new_session=True)
        except FileNotFoundError:
            raise TunnelError(f"{binary} is not installed or not on PATH") from None
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        m = URL_RE.search(log.read_text(encoding="utf-8", errors="replace"))
        if m:
            return Tunnel(pid=proc.pid, url=m.group(1), log=str(log), binary=binary)
        if proc.poll() is not None:
            raise TunnelError(f"{binary} exited with {proc.returncode}, see {log}")
        time.sleep(0.2)
    stop(proc.pid, binary)
    raise TunnelError(f"{binary} printed no URL in {timeout_s:g} s, see {log}")


def alive(pid: int | None, binary: str) -> bool:
    """Whether `pid` runs as `binary tunnel ...`: the first word's file name is the binary's."""
    if not pid or pid <= 0:
        return False
    try:
        if os.waitpid(pid, os.WNOHANG)[0] == pid:  # our own child that already exited
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    # -ww: no width limit; Linux procps cuts the line at 80 columns when not on a terminal.
    ps = subprocess.run(["ps", "-ww", "-o", "command=", "-p", str(pid)], capture_output=True, text=True,
                        check=False)
    command = ps.stdout.strip()
    return bool(command) and Path(command.split()[0]).name == Path(binary).name and " tunnel " in command


def stop(pid: int, binary: str, grace_s: float = 5) -> None:
    """SIGTERM, then SIGKILL after `grace_s`. Does nothing unless the pid is the tunnel."""
    if not alive(pid, binary):
        return
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if not alive(pid, binary):
            return
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def resolve_at_cloudflare(host: str) -> str | None:
    """An IPv4 address for `host` from 1.1.1.1, else from the system resolver, else None."""
    if shutil.which("dig"):
        try:
            out = subprocess.run(["dig", "+short", "@1.1.1.1", host, "A"], capture_output=True, text=True,
                                 timeout=10, check=False).stdout
            for line in out.split():
                if IPV4_RE.match(line):
                    return line
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        return socket.gethostbyname(host)
    except OSError:
        return None


def public_health(url: str, resolver: Callable[[str], str | None] = resolve_at_cloudflare,
                  timeout_s: float = 10) -> dict | None:
    """GET /health through the public URL, or None when it does not answer."""
    host = urlparse(url).hostname
    ip = resolver(host) if host else None
    if not ip:
        return None
    try:
        ctx = ssl.create_default_context()
        sock = ctx.wrap_socket(socket.create_connection((ip, 443), timeout=timeout_s), server_hostname=host)
        conn = http.client.HTTPSConnection(host, 443, timeout=timeout_s, context=ctx)
        conn.sock = sock
        try:
            conn.request("GET", "/health", headers={"Host": host})
            resp = conn.getresponse()
            return json.loads(resp.read()) if resp.status == 200 else None
        finally:
            conn.close()
    except (OSError, ValueError, http.client.HTTPException):
        return None
