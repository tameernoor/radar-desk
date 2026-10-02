"""Single-owner auth: the raw OWNER_TOKEN as a bearer, or a signed session cookie from /auth/login.
Pull workers authenticate separately, with a worker token as a bearer (`require_worker`).

The cookie carries a fingerprint of the owner token, so rotating OWNER_TOKEN or SESSION_SECRET ends
every session. Login failures are rate limited per IP in memory.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException, Request
from itsdangerous import BadSignature, TimestampSigner

from radar_desk.records import WorkerToken

COOKIE_NAME = "radar_session"
COOKIE_MAX_AGE_S = 30 * 24 * 60 * 60
LOGIN_MAX_FAILURES = 5
LOGIN_WINDOW_S = 10 * 60


def _secret(value: Any) -> str:
    return value.get_secret_value() if hasattr(value, "get_secret_value") else str(value)


class LoginLimiter:
    """Counts failed logins per IP over a sliding window."""

    def __init__(self, max_failures: int = LOGIN_MAX_FAILURES, window_s: float = LOGIN_WINDOW_S,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.max_failures = max_failures
        self.window_s = window_s
        self.clock = clock
        self._failures: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def _recent(self, ip: str, now: float) -> deque[float] | None:
        """The IP's failures inside the window. An entry left empty is dropped."""
        times = self._failures.get(ip)
        if times is None:
            return None
        while times and now - times[0] > self.window_s:
            times.popleft()
        if not times:
            del self._failures[ip]
            return None
        return times

    def blocked(self, ip: str) -> bool:
        with self._lock:
            times = self._recent(ip, self.clock())
            return times is not None and len(times) >= self.max_failures

    def fail(self, ip: str) -> None:
        with self._lock:
            now = self.clock()
            for other in list(self._failures):
                self._recent(other, now)
            self._failures.setdefault(ip, deque()).append(now)

    def reset(self, ip: str) -> None:
        with self._lock:
            self._failures.pop(ip, None)


class Auth:
    def __init__(self, settings: Any, limiter: LoginLimiter | None = None) -> None:
        self._token = _secret(settings.owner_token).encode()
        fingerprint = hashlib.sha256(self._token).hexdigest()[:32]
        self._subject = f"owner:{fingerprint}".encode()
        self._signer = TimestampSigner(_secret(settings.session_secret), salt="radar-desk-session")
        self.secure_cookie = str(settings.public_base_url).lower().startswith("https://")
        self.limiter = limiter or LoginLimiter()

    def token_ok(self, token: str | None) -> bool:
        return hmac.compare_digest(self._token, (token or "").encode())

    def session_cookie(self) -> str:
        return self._signer.sign(self._subject).decode()

    def cookie_ok(self, value: str | None) -> bool:
        if not value:
            return False
        try:
            subject = self._signer.unsign(value, max_age=COOKIE_MAX_AGE_S)
        except BadSignature:
            return False
        return hmac.compare_digest(subject, self._subject)

    def set_cookie(self, response: Any) -> None:
        response.set_cookie(
            COOKIE_NAME,
            self.session_cookie(),
            max_age=COOKIE_MAX_AGE_S,
            httponly=True,
            samesite="lax",
            secure=self.secure_cookie,
            path="/",
        )

    def clear_cookie(self, response: Any) -> None:
        response.delete_cookie(COOKIE_NAME, path="/", httponly=True, samesite="lax", secure=self.secure_cookie)

    def identify(self, request: Request) -> str | None:
        """"bearer" or "cookie" when the request carries valid credentials, else None."""
        header = request.headers.get("authorization", "")
        scheme, _, value = header.partition(" ")
        if scheme.lower() == "bearer" and value and self.token_ok(value.strip()):
            return "bearer"
        if self.cookie_ok(request.cookies.get(COOKIE_NAME)):
            return "cookie"
        return None


def client_ip(request: Request) -> str:
    """The caller's IP. On Fly (FLY_APP_NAME set) the proxy's Fly-Client-IP, else the socket peer.
    The header is ignored elsewhere, since a caller could rotate it to dodge the login limit."""
    forwarded = request.headers.get("fly-client-ip") if os.environ.get("FLY_APP_NAME") else None
    if forwarded:
        return forwarded.strip()
    return request.client.host if request.client else "unknown"


def require_owner(request: Request) -> str:
    """FastAPI dependency: 401 unless the request is the owner. Returns how it authenticated."""
    via = request.app.state.auth.identify(request)
    if via is None:
        raise HTTPException(401, "not logged in", headers={"WWW-Authenticate": "Bearer"})
    return via


def require_worker(request: Request) -> WorkerToken:
    """FastAPI dependency: 401 unless the request carries a live worker token (`rdw_...`) as a bearer.
    Returns the WorkerToken. The owner's token and cookie do not pass."""
    scheme, _, value = request.headers.get("authorization", "").partition(" ")
    token = None
    if scheme.lower() == "bearer":
        token = request.app.state.services.workers.authenticate(value.strip())
    if token is None:
        raise HTTPException(401, "not a valid worker token", headers={"WWW-Authenticate": "Bearer"})
    return token
