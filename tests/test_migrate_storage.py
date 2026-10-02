"""scripts/migrate_storage.py: from a FakeVolume-backed Modal volume to a local folder or a stubbed S3 bucket."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest
from botocore.stub import ANY, Stubber

from fake_volume import FakeVolume
from radar_desk.gpu.backend import artefact_keys
from radar_desk.gpu.fake import canned_result
from radar_desk.records import Job
from radar_desk.services.scans import source_key
from radar_desk.storage import LocalStorage, ModalVolumeStorage, ObjectExists, S3Storage, StorageError
from test_services import done_job, ready_scan

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location("migrate_storage", ROOT / "scripts" / "migrate_storage.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # the dataclasses look their module up while they are built
    spec.loader.exec_module(module)
    return module


ms = _script()


@pytest.fixture
def world(make_services, tmp_path):
    """A ready scan and a done job with five artefacts, all on the Modal volume; an empty local destination."""
    vol = FakeVolume()
    src = ModalVolumeStorage("radar-data", "http://x", volume=vol)
    svc = make_services(storage=src)
    scan = ready_scan(svc, tmp_path)
    job = done_job(svc, scan.id)
    for i, key in enumerate(artefact_keys(job.id).values()):
        src.put_bytes(key, f"artefact {i} ".encode() * (i + 1))
    dst = LocalStorage(tmp_path / "dst", "http://x", "s")
    keys = [source_key(scan.id), *artefact_keys(job.id).values()]
    return svc, vol, src, dst, scan, keys


def run(svc, src, dst, dry_run=False):
    lines: list[str] = []
    summary = ms.migrate(svc.db, src, dst, svc.settings.data_dir, dry_run, lines.append)
    return summary, lines


def test_copies_every_object_once_and_leaves_the_source(world):
    svc, vol, src, dst, scan, keys = world
    before = dict(vol.files)
    summary, lines = run(svc, src, dst)
    assert (summary.copied, summary.existing, summary.missing, summary.problems) == (6, 0, 0, 0)
    for key in keys:
        assert dst.size(key) == src.size(key)
        assert f"copied {key} {src.size(key)} bytes" in lines
    assert hashlib.sha256(dst.path_for(keys[0]).read_bytes()).hexdigest() == scan.sha256
    assert summary.bytes == sum(src.size(k) for k in keys)
    assert summary.line() == f"copied 6 ({summary.bytes} bytes), skipped 0 existing, missing 0, problems 0"
    assert not (svc.settings.data_dir / "migrate-tmp").exists()  # removed once empty

    summary, lines = run(svc, src, dst)
    assert (summary.copied, summary.existing) == (0, 6)
    assert set(lines) == {f"exists {k}" for k in keys}
    assert vol.files == before


def test_dry_run_writes_nothing(world):
    svc, _, src, dst, _, keys = world
    summary, lines = run(svc, src, dst, dry_run=True)
    assert set(lines) == {f"would copy {k} {src.size(k)} bytes" for k in keys}
    assert summary.copied == 6 and summary.problems == 0
    assert summary.bytes == sum(src.size(k) for k in keys)
    assert summary.line() == f"would copy 6 ({summary.bytes} bytes), skipped 0 existing, missing 0, problems 0"
    assert not any(dst.exists(k) for k in keys)
    assert not (svc.settings.data_dir / "migrate-tmp").exists()


def test_a_missing_source_is_reported(world):
    svc, _, src, dst, _, keys = world
    src.delete(keys[-1])
    summary, lines = run(svc, src, dst)
    assert f"missing {keys[-1]}" in lines
    assert (summary.copied, summary.missing, summary.problems) == (5, 1, 0)


def test_a_sha256_mismatch_is_not_copied(world):
    svc, _, src, dst, scan, keys = world
    svc.db.update_scan(scan.id, sha256="0" * 64)
    summary, lines = run(svc, src, dst)
    assert f"sha256 mismatch {keys[0]}" in lines
    assert not dst.exists(keys[0])
    assert (summary.copied, summary.problems) == (5, 1)


def test_a_different_size_at_the_destination_is_not_overwritten(world):
    svc, _, src, dst, _, keys = world
    dst.put_bytes(keys[1], b"other")
    summary, lines = run(svc, src, dst)
    assert f"differs {keys[1]} ({src.size(keys[1])} vs 5 bytes)" in lines
    assert dst.path_for(keys[1]).read_bytes() == b"other"
    assert (summary.copied, summary.problems) == (5, 1)


def test_an_object_is_never_held_whole(make_services, tmp_path):
    import tracemalloc

    svc = make_services()
    job = done_job(svc, ready_scan(svc, tmp_path).id)
    key = artefact_keys(job.id)["mask"]
    path = svc.storage.path_for(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    size = 64 * 1024 * 1024
    with path.open("wb") as fh:
        fh.truncate(size)  # sparse on the source side
    vol = FakeVolume(keep=False)
    tracemalloc.start()
    try:
        summary, _ = run(svc, svc.storage, ModalVolumeStorage("radar-data", "http://x", volume=vol))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert summary.copied == 2 and vol.sizes[key] == size  # the mask and the scan source
    assert peak < 16 * 1024 * 1024


def test_to_an_s3_bucket(make_services, tmp_path):
    import boto3

    svc = make_services()
    scan = ready_scan(svc, tmp_path)
    key, size = source_key(scan.id), svc.storage.size(source_key(scan.id))
    client = boto3.client("s3", endpoint_url="https://example.invalid", region_name="auto",
                          aws_access_key_id="AKIDUMMY", aws_secret_access_key="dummysecret")
    dst = S3Storage("radar-bucket", None, None, None, None, client=client)
    params = {"Bucket": "radar-bucket", "Key": key}
    with Stubber(client) as stub:
        stub.add_client_error("head_object", "404", "Not Found", 404, expected_params=params)
        stub.add_response("put_object", {}, {**params, "Body": ANY})
        stub.add_response("head_object", {"ContentLength": size}, params)
        summary, lines = run(svc, svc.storage, dst)
        stub.assert_no_pending_responses()
    assert lines == [f"copied {key} {size} bytes"] and summary.copied == 1


def test_main(world, monkeypatch):
    svc, _, src, dst, scan, _ = world
    with pytest.raises(SystemExit) as info:
        ms.main(["--from", "s3", "--to", "s3"], ms.Deps(settings=svc.settings))
    assert info.value.code == 2

    err: list[str] = []
    assert ms.main(["--from", "local", "--to", "s3"], ms.Deps(settings=svc.settings, err=err.append)) == 2
    assert err == ["STORAGE_BACKEND=s3 needs S3_BUCKET"]

    monkeypatch.setattr(ms, "make_storage", lambda settings, backend: {"modal_volume": src, "local": dst}[backend])
    out: list[str] = []
    assert ms.main(["--from", "modal_volume", "--to", "local"], ms.Deps(settings=svc.settings, out=out.append)) == 0
    assert out[-1].startswith("copied 6 (") and out[-1].endswith("skipped 0 existing, missing 0, problems 0")

    svc.db.update_scan(scan.id, sha256="0" * 64)
    dst.delete(source_key(scan.id))
    out.clear()
    assert ms.main(["--from", "modal_volume", "--to", "local"], ms.Deps(settings=svc.settings, out=out.append)) == 1
    assert out[-1].endswith("skipped 5 existing, missing 0, problems 1")

    src.delete(source_key(scan.id))
    out.clear()
    assert ms.main(["--from", "modal_volume", "--to", "local"], ms.Deps(settings=svc.settings, out=out.append)) == 1
    assert out[-1].endswith("skipped 5 existing, missing 1, problems 0")


def test_only_ready_scans_and_done_jobs_with_a_result_count(world):
    svc, _, _, _, scan, keys = world
    uploading = svc.scans.begin_upload("u.nii.gz", 100, True).scan_id
    rejected = svc.scans.begin_upload("r.nii.gz", 100, True).scan_id
    svc.db.update_scan(rejected, state="rejected")
    failed = Job(id="job_failed", scan_id=scan.id, gpu_requested=["L4"])
    svc.db.insert_job(failed)
    failed = svc.db.transition(failed, "submitted", modal_call_id="fc-f")
    svc.results.store(failed, canned_result(failed.id))
    svc.db.transition(failed, "failed")
    bare = Job(id="job_bare", scan_id=scan.id, gpu_requested=["L4"])
    svc.db.insert_job(bare)
    svc.db.transition(svc.db.transition(bare, "submitted", modal_call_id="fc-b"), "done")
    found = [k for k, _ in ms.objects(svc.db)]
    assert sorted(found) == sorted(keys)
    assert not any(x in k for k in found for x in (uploading, rejected, "job_failed", "job_bare"))


class Refusing:
    """A destination whose put_file raises for one key."""

    def __init__(self, inner, key, exc):
        self.inner, self.key, self.exc = inner, key, exc

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def put_file(self, key, path, content_type=None):
        if key == self.key:
            raise self.exc
        self.inner.put_file(key, path, content_type)


def test_a_modal_volume_destination_never_overwrites(world, monkeypatch):
    svc, _, src, _, _, keys = world
    vol = FakeVolume()
    dst = ModalVolumeStorage("radar-data-dst", "http://x", volume=vol)
    dst.put_bytes(keys[1], b"other")
    summary, lines = run(svc, src, Refusing(dst, keys[2], ObjectExists(f"{keys[2]} already exists")))
    assert f"differs {keys[1]} ({src.size(keys[1])} vs 5 bytes)" in lines
    assert f"error {keys[2]}: {keys[2]} already exists" in lines
    assert vol.files[keys[1]] == b"other" and keys[2] not in vol.files
    assert all(f"copied {k} {src.size(k)} bytes" in lines for k in keys[3:])
    assert (summary.copied, summary.problems) == (4, 2)


def test_a_failed_size_check_removes_the_copy(world, monkeypatch):
    svc, _, src, dst, _, keys = world
    real = dst.size
    monkeypatch.setattr(dst, "size", lambda key: real(key) + 1 if key == keys[3] else real(key))
    summary, lines = run(svc, src, dst)
    assert f"size check failed {keys[3]}, removed" in lines
    assert not dst.path_for(keys[3]).exists()
    assert (summary.copied, summary.problems) == (5, 1)


def test_a_storage_error_part_way_does_not_stop_the_run(world, monkeypatch):
    svc, _, src, dst, _, keys = world
    real = src.open_stream

    def stream(key):
        if key == keys[1]:
            raise StorageError("the network went away")
        return real(key)

    monkeypatch.setattr(src, "open_stream", stream)
    summary, lines = run(svc, src, dst)
    assert f"error {keys[1]}: the network went away" in lines
    assert lines.index(f"error {keys[1]}: the network went away") < len(lines) - 1
    assert all(dst.exists(k) for k in keys if k != keys[1]) and not dst.exists(keys[1])
    assert (summary.copied, summary.problems) == (5, 1)
    assert summary.line().endswith("problems 1")
