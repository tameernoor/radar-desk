"""A logged-in TestClient for a Services built by the make_services fixture."""

from __future__ import annotations

from fastapi.testclient import TestClient

from radar_desk.app import create_app


def api_client(svc) -> TestClient:
    app = create_app(svc.settings, svc, start_poller=False)
    return TestClient(app, headers={"Authorization": "Bearer test-owner"})
