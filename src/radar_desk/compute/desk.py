"""The app's own API as the compute switch needs it: health and the owner's worker routes."""

from __future__ import annotations

import httpx


class DeskError(RuntimeError):
    """The app answered an owner call with an unexpected status."""


class Desk:
    def __init__(self, base_url: str, owner_token: str, client: httpx.Client | None = None) -> None:
        self.base = base_url.rstrip("/")
        self.owner_token = owner_token
        self.client = client or httpx.Client(timeout=10)

    def _owner(self, method: str, path: str, **kwargs) -> httpx.Response:
        headers = {"Authorization": f"Bearer {self.owner_token}"}
        return self.client.request(method, self.base + path, headers=headers, **kwargs)

    def health(self) -> dict | None:
        """GET /health, or None when the app does not answer it."""
        try:
            r = self.client.get(self.base + "/health")
            return r.json() if r.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            return None

    def create_token(self, name: str) -> tuple[str, str]:
        """A new worker token's id and plaintext."""
        r = self._owner("POST", "/workers/tokens", json={"name": name})
        if r.status_code != 201:
            raise DeskError(f"POST /workers/tokens answered {r.status_code}")
        body = r.json()
        return body["id"], body["token"]

    def revoke_token(self, token_id: str) -> bool:
        """Revoke a token; False when the app does not know it."""
        r = self._owner("DELETE", f"/workers/tokens/{token_id}")
        if r.status_code == 404:
            return False
        if r.status_code != 204:
            raise DeskError(f"DELETE /workers/tokens/{token_id} answered {r.status_code}")
        return True

    def workers(self) -> list[dict]:
        r = self._owner("GET", "/workers")
        if r.status_code != 200:
            raise DeskError(f"GET /workers answered {r.status_code}")
        return r.json()["workers"]
