"""The upload, viewer and mask flow with STORAGE_BACKEND=modal_volume, against the FakeVolume."""

from __future__ import annotations

from pathlib import Path

import pytest

from fake_volume import FakeVolume
from radar_desk.services.scans import source_key
from radar_desk.storage.modal_volume import ModalVolumeStorage
from test_api import api, make_api  # noqa: F401  (fixtures)


@pytest.fixture
def make_vapi(make_api):  # noqa: F811
    def build(**overrides):
        volume = FakeVolume()
        storage = ModalVolumeStorage("radar-data", "http://testserver", volume=volume)
        vapi = make_api(storage_backend="modal_volume", storage=storage, **overrides)
        vapi.volume = volume
        return vapi

    return build


@pytest.fixture
def vapi(make_vapi):
    return make_vapi()


def begin(vapi, path: Path) -> dict:
    r = vapi.client.post("/uploads", json={"filename": path.name, "size_bytes": path.stat().st_size,
                                           "research_only_confirmed": True})
    assert r.status_code == 200, r.text
    return r.json()


def temp_files(vapi) -> list[Path]:
    return list((Path(vapi.svc.settings.data_dir) / "volume-uploads").glob("*"))


def test_upload_view_and_mask_end_to_end(vapi):
    path = vapi.nifti()
    ticket = begin(vapi, path)
    key = source_key(ticket["scan_id"])
    assert ticket["put_url"] == f"/_volume/{key}"  # relative, so the browser keeps its origin and cookie

    assert vapi.anon.put(ticket["put_url"], content=path.read_bytes()).status_code == 401
    r = vapi.client.put(ticket["put_url"], content=path.read_bytes())
    assert r.status_code == 204, r.text
    assert vapi.volume.files[key] == path.read_bytes()
    assert temp_files(vapi) == []

    scan = vapi.client.post(f"/uploads/{ticket['scan_id']}/complete").json()
    assert scan["state"] == "ready" and scan["header"]["dims"] == [64, 64, 24]

    r = vapi.client.get(f"/scans/{scan['id']}/source.nii.gz", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == f"/_volume/{key}"
    followed = vapi.client.get(f"/scans/{scan['id']}/source.nii.gz")  # the client resolves the relative Location
    assert followed.status_code == 200 and followed.content == path.read_bytes()
    assert vapi.anon.get(r.headers["location"]).status_code == 401
    ct = vapi.client.get(r.headers["location"])
    assert ct.status_code == 200 and ct.content == path.read_bytes()
    assert ct.headers["content-length"] == str(path.stat().st_size)
    assert ct.headers["content-type"] == "application/gzip"

    job = vapi.run_job(scan["id"])
    r = vapi.client.get(f"/jobs/{job['id']}/mask.nii.gz", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == f"/_volume/jobs/{job['id']}/mask.nii.gz"
    mask = vapi.client.get(r.headers["location"])
    assert mask.status_code == 200 and mask.content == vapi.volume.files[f"jobs/{job['id']}/mask.nii.gz"]
    trace = vapi.client.get(f"/_volume/jobs/{job['id']}/trace.json")
    assert trace.status_code == 200 and trace.headers["content-type"] == "application/octet-stream"


def test_second_put_is_409(vapi):
    path = vapi.nifti()
    url = begin(vapi, path)["put_url"]
    assert vapi.client.put(url, content=b"first").status_code == 204
    r = vapi.client.put(url, content=b"second")
    assert r.status_code == 409 and "never overwritten" in r.json()["detail"]
    assert vapi.volume.files[url.split("/_volume/", 1)[1]] == b"first"


def test_put_over_the_cap_is_413(make_vapi):
    vapi = make_vapi(max_upload_bytes=1000)
    url = "/_volume/scans/x/source.nii.gz"
    assert vapi.client.put(url, content=b"x" * 2000).status_code == 413  # declared length
    chunked = vapi.client.put(url, content=iter([b"x" * 600, b"x" * 600]))  # no length, counted
    assert chunked.status_code == 413
    assert not vapi.svc.storage.exists("scans/x/source.nii.gz")
    assert temp_files(vapi) == []


def test_put_race_is_409_and_cleans_up(vapi, monkeypatch):
    """exists() says no, then another writer lands first and the batch raises FileExistsError."""
    key = "scans/r/source.nii.gz"
    real_exists = vapi.svc.storage.exists

    def exists_then_race(k):
        found = real_exists(k)
        vapi.volume.files[k] = b"other"
        vapi.volume.sizes[k] = 5
        return found

    monkeypatch.setattr(vapi.svc.storage, "exists", exists_then_race)
    r = vapi.client.put(f"/_volume/{key}", content=b"mine")
    assert r.status_code == 409 and "never overwritten" in r.json()["detail"]
    assert vapi.volume.files[key] == b"other"
    assert temp_files(vapi) == []


def test_missing_object_is_404_and_bad_key_is_400(vapi):
    assert vapi.client.get("/_volume/jobs/nope/trace.json").status_code == 404
    assert vapi.client.get("/_volume/a//b").status_code == 400


async def test_put_disconnect_is_400(vapi):
    path = "/_volume/scans/y/source.nii.gz"
    messages = iter([{"type": "http.request", "body": b"ab", "more_body": True}, {"type": "http.disconnect"}])
    sent = []

    async def receive():
        return next(messages)

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "PUT",
             "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"",
             "root_path": "", "headers": [(b"host", b"testserver"), (b"authorization", b"Bearer test-owner")],
             "client": ("1.2.3.4", 1), "server": ("testserver", 80)}
    await vapi.app(scope, receive, send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 400
    assert not vapi.svc.storage.exists("scans/y/source.nii.gz")
    assert temp_files(vapi) == []


def test_volume_route_only_on_volume_storage(api):  # noqa: F811
    assert api.client.get("/_volume/scans/x/source.nii.gz").status_code == 404


def test_signed_local_route_not_on_volume_storage(vapi):
    assert vapi.client.get("/_storage/scans/x/source.nii.gz").status_code == 404
