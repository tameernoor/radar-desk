"""The app's own API as the compute command needs it: health, the Compute choice and the workers."""

from __future__ import annotations

import httpx


class DeskError(RuntimeError):
    """The app answered an owner call with a status other than 2xx; the message is the app's detail."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status


class Desk:
    def __init__(self, base_url: str, owner_token: str, client: httpx.Client | None = None) -> None:
        self.base = base_url.rstrip("/")
        self.owner_token = owner_token
        self.client = client or httpx.Client(timeout=10)

    def _owner(self, method: str, path: str, **kwargs) -> dict:
        headers = {"Authorization": f"Bearer {self.owner_token}"}
        r = self.client.request(method, self.base + path, headers=headers, **kwargs)
        if not r.is_success:
            try:
                detail = r.json()["detail"]
            except (ValueError, KeyError, TypeError):
                detail = None
            raise DeskError(r.status_code, str(detail) if detail else f"{method} {path} answered {r.status_code}")
        return r.json()

    def health(self) -> dict | None:
        """GET /health, or None when the app does not answer it."""
        try:
            r = self.client.get(self.base + "/health")
            return r.json() if r.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            return None

    def workers(self) -> list[dict]:
        return self._owner("GET", "/workers")["workers"]

    def compute(self) -> dict:
        return self._owner("GET", "/compute")

    def set_mode(self, mode: str) -> dict:
        return self._owner("PUT", "/compute", json={"mode": mode})

    def pod_start(self) -> dict:
        return self._owner("POST", "/compute/pod/start")

    def pod_stop(self) -> dict:
        return self._owner("POST", "/compute/pod/stop")
