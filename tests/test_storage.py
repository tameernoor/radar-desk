"""Storage adapters: local signed URLs and files, S3 presigning and head_object via Stubber."""

from __future__ import annotations

import io
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import boto3
import pytest
from botocore.response import StreamingBody
from botocore.stub import Stubber
from pydantic import SecretStr

from radar_desk.storage import make_storage
from radar_desk.storage.base import ObjectMissing, StorageError, validate_key
from radar_desk.storage.local import LocalStorage
from radar_desk.storage.s3 import S3Storage

KEY = "scans/scan_abc/source.nii.gz"


@pytest.fixture
def local(tmp_path: Path) -> LocalStorage:
    return LocalStorage(tmp_path, "http://localhost:8000/", "s3cret")


def _params(url: str) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}


@pytest.mark.parametrize("bad", ["", "/abs", "a/../b", "..", "a\\b", "a\x00b", "a\nb", "a//b", "a/"])
def test_validate_key_rejects(bad: str):
    with pytest.raises(StorageError):
        validate_key(bad)


def test_validate_key_accepts():
    validate_key(KEY)
    validate_key("jobs/job_1/scores.json")


def test_local_put_bytes_and_read(local: LocalStorage, tmp_path: Path):
    local.put_bytes(KEY, b"hello")
    assert (tmp_path / "objects" / KEY).read_bytes() == b"hello"
    assert local.path_for(KEY) == tmp_path / "objects" / KEY
    assert local.exists(KEY)
    assert local.size(KEY) == 5
    assert b"".join(local.open_stream(KEY)) == b"hello"


def test_local_stream_chunks(local: LocalStorage):
    data = bytes(range(256)) * 9000  # just over 2 MiB
    n = local.write_stream(KEY, iter([data[:1000], data[1000:]]))
    assert n == len(data)
    chunks = list(local.open_stream(KEY))
    assert len(chunks) == 3 and max(len(c) for c in chunks) == 1024 * 1024
    assert b"".join(chunks) == data


def test_local_delete_and_missing(local: LocalStorage):
    local.put_bytes(KEY, b"x")
    local.delete(KEY)
    assert not local.exists(KEY)
    local.delete(KEY)  # deleting a missing object is fine
    with pytest.raises(ObjectMissing):
        local.size(KEY)
    with pytest.raises(ObjectMissing):
        list(local.open_stream(KEY))


def test_local_rejects_traversal(local: LocalStorage):
    with pytest.raises(StorageError):
        local.put_bytes("../escape", b"x")
    with pytest.raises(StorageError):
        local.put_url("a/../../b", 60)


def test_local_urls_and_verify(local: LocalStorage):
    put = local.put_url(KEY, 900)
    get = local.get_url(KEY, 3600)
    for url in (put, get):
        assert url.startswith(f"http://localhost:8000/_storage/{KEY}?")
        assert {"exp", "sig"} <= _params(url).keys()
    p, g = _params(put), _params(get)
    assert int(p["exp"]) == pytest.approx(time.time() + 900, abs=5)
    assert local.verify("PUT", KEY, p["exp"], p["sig"])
    assert local.verify("GET", KEY, g["exp"], g["sig"])
    # A GET URL cannot be used to PUT, and the other way round.
    assert not local.verify("PUT", KEY, g["exp"], g["sig"])
    assert not local.verify("GET", KEY, p["exp"], p["sig"])
    # Tampered key, tampered exp, garbage values.
    assert not local.verify("GET", "scans/other/source.nii.gz", g["exp"], g["sig"])
    assert not local.verify("GET", KEY, str(int(g["exp"]) + 1), g["sig"])
    assert not local.verify("GET", KEY, "nope", g["sig"])
    assert not local.verify("GET", KEY, g["exp"], "00")


def test_local_verify_expired_and_wrong_secret(local: LocalStorage, tmp_path: Path):
    g = _params(local.get_url(KEY, 60))
    assert not local.verify("GET", KEY, g["exp"], g["sig"], now=int(g["exp"]) + 1)
    assert local.verify("GET", KEY, g["exp"], g["sig"], now=int(g["exp"]) - 1)
    other = LocalStorage(tmp_path, "http://localhost:8000", "different")
    assert not other.verify("GET", KEY, g["exp"], g["sig"])


def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url="https://example.invalid",
        region_name="auto",
        aws_access_key_id="AKIDUMMY",
        aws_secret_access_key="dummysecret",
    )


@pytest.fixture
def s3() -> tuple[S3Storage, Stubber]:
    client = _s3_client()
    storage = S3Storage(
        "radar-bucket", "https://example.invalid", "auto", "AKIDUMMY", "dummysecret", client=client
    )
    with Stubber(client) as stub:
        yield storage, stub
        stub.assert_no_pending_responses()


def test_s3_presigned_urls(s3):
    storage, _ = s3
    for url in (storage.put_url(KEY, 900), storage.get_url(KEY, 3600)):
        assert "radar-bucket" in url
        assert KEY in url
        assert "X-Amz-Signature" in url
    assert "X-Amz-Expires=900" in storage.put_url(KEY, 900)


def test_s3_presigned_url_without_injected_client():
    storage = S3Storage("radar-bucket", "https://example.invalid", "auto", "AKIDUMMY", "dummysecret")
    url = storage.get_url(KEY, 60)
    assert "radar-bucket" in url and "X-Amz-Signature" in url


def test_s3_exists_and_size_via_head(s3):
    storage, stub = s3
    params = {"Bucket": "radar-bucket", "Key": KEY}
    stub.add_response("head_object", {"ContentLength": 42}, params)
    assert storage.exists(KEY)
    stub.add_client_error("head_object", "404", "Not Found", 404, expected_params=params)
    assert not storage.exists(KEY)
    stub.add_response("head_object", {"ContentLength": 42}, params)
    assert storage.size(KEY) == 42
    stub.add_client_error("head_object", "404", "Not Found", 404, expected_params=params)
    with pytest.raises(ObjectMissing):
        storage.size(KEY)
    stub.add_client_error("head_object", "403", "Forbidden", 403, expected_params=params)
    with pytest.raises(StorageError):
        storage.exists(KEY)


def test_s3_stream_put_delete(s3):
    storage, stub = s3
    params = {"Bucket": "radar-bucket", "Key": KEY}
    body = StreamingBody(io.BytesIO(b"abc" * 10), 30)
    stub.add_response("get_object", {"Body": body, "ContentLength": 30}, params)
    assert b"".join(storage.open_stream(KEY)) == b"abc" * 10
    stub.add_response("put_object", {}, {**params, "Body": b"data", "ContentType": "application/json"})
    storage.put_bytes(KEY, b"data", content_type="application/json")
    stub.add_response("delete_object", {}, params)
    storage.delete(KEY)


def test_s3_delete_missing_is_fine_other_errors_raise(s3):
    storage, stub = s3
    params = {"Bucket": "radar-bucket", "Key": KEY}
    stub.add_client_error("delete_object", "404", "Not Found", 404, expected_params=params)
    storage.delete(KEY)
    stub.add_client_error("delete_object", "NoSuchKey", "Not Found", 404, expected_params=params)
    storage.delete(KEY)
    stub.add_client_error("delete_object", "403", "Forbidden", 403, expected_params=params)
    with pytest.raises(StorageError):
        storage.delete(KEY)


def test_make_storage_picks_adapter(tmp_path: Path):
    base = {
        "s3_endpoint_url": "https://example.invalid",
        "s3_region": "auto",
        "aws_access_key_id": "AKIDUMMY",
        "aws_secret_access_key": "dummysecret",
        "data_dir": tmp_path,
        "public_base_url": "http://localhost:8000",
        "session_secret": "s",
    }
    assert isinstance(make_storage(SimpleNamespace(s3_bucket=None, **base)), LocalStorage)
    assert isinstance(make_storage(SimpleNamespace(s3_bucket="", **base)), LocalStorage)
    assert isinstance(make_storage(SimpleNamespace(s3_bucket="radar-bucket", **base)), S3Storage)
    secret = SimpleNamespace(s3_bucket=None, **{**base, "session_secret": SecretStr("s")})
    url = make_storage(secret).get_url(KEY, 60)
    assert LocalStorage(tmp_path, "http://x", "s").verify(
        "GET", KEY, _params(url)["exp"], _params(url)["sig"]
    )


# put_file and worker_ref on the URL adapters


def test_local_put_file_copies_and_worker_ref_is_the_signed_url(local: LocalStorage, tmp_path: Path):
    src = tmp_path / "in.bin"
    src.write_bytes(b"z" * 3000)
    local.put_file(KEY, src)
    assert local.path_for(KEY).read_bytes() == b"z" * 3000 and src.is_file()
    get, put = local.worker_ref(KEY, "GET", 60), local.worker_ref(KEY, "PUT", 60)
    assert local.verify("GET", KEY, _params(get)["exp"], _params(get)["sig"])
    assert local.verify("PUT", KEY, _params(put)["exp"], _params(put)["sig"])
    with pytest.raises(ValueError):
        local.worker_ref(KEY, "DELETE", 60)


def test_s3_put_file_and_worker_refs(s3, tmp_path: Path, monkeypatch):
    from botocore.stub import ANY

    storage, stub = s3
    src = tmp_path / "in.bin"
    src.write_bytes(b"data")
    stub.add_response("put_object", {},
                      {"Bucket": "radar-bucket", "Key": KEY, "Body": ANY, "ContentType": "application/gzip"})
    storage.put_file(KEY, src, content_type="application/gzip")
    # with the signing clock frozen, presigning is deterministic and the refs equal the URLs exactly
    import datetime

    import botocore.auth

    frozen = datetime.datetime(2026, 10, 1, 12, 0, 0, tzinfo=datetime.UTC)
    monkeypatch.setattr(botocore.auth, "get_current_datetime", lambda *a, **k: frozen)
    assert storage.worker_ref(KEY, "GET", 60) == storage.get_url(KEY, 60)
    assert storage.worker_ref(KEY, "PUT", 60) == storage.put_url(KEY, 60)
    assert storage.get_url(KEY, 60) != storage.put_url(KEY, 60)
    assert _params(storage.get_url(KEY, 60))["X-Amz-Date"] == "20261001T120000Z"


# Modal Volume adapter, against the FakeVolume in tests/fake_volume.py


@pytest.fixture
def vol():
    from fake_volume import FakeVolume

    return FakeVolume()


@pytest.fixture
def mv(vol):
    from radar_desk.storage.modal_volume import ModalVolumeStorage

    return ModalVolumeStorage("radar-data", "http://localhost:8000/", volume=vol)


def test_volume_put_bytes_maps_keys_to_absolute_paths(mv, vol):
    assert mv.name == "modal_volume"
    mv.put_bytes(KEY, b"hello", "application/gzip")
    assert vol.remote_paths == ["/" + KEY]
    assert vol.files[KEY] == b"hello"
    assert mv.exists(KEY) and mv.size(KEY) == 5


def test_volume_exists_and_size_when_missing(mv, vol):
    assert not mv.exists(KEY)
    with pytest.raises(ObjectMissing):
        mv.size(KEY)
    mv.put_bytes("scans/scan_abc/other.bin", b"x")
    # The folder exists but the key does not: listdir lists the folder's children, not the key.
    assert not mv.exists(KEY)
    assert not mv.exists("scans/scan_abc")


def test_volume_open_stream_chunks_and_missing(mv, vol):
    data = bytes(range(256)) * 9000  # just over 2 MiB
    mv.put_bytes(KEY, data)
    assert b"".join(mv.open_stream(KEY)) == data
    with pytest.raises(ObjectMissing):  # raised on the call, before any iteration
        mv.open_stream("scans/nope/source.nii.gz")


def test_volume_put_never_overwrites(mv, vol):
    from radar_desk.storage.base import ObjectExists

    mv.put_bytes(KEY, b"first")
    with pytest.raises(ObjectExists):
        mv.put_bytes(KEY, b"second")
    assert vol.files[KEY] == b"first"


def test_volume_put_file_and_delete(mv, vol, tmp_path: Path):
    src = tmp_path / "in.bin"
    src.write_bytes(b"abc" * 1000)
    mv.put_file(KEY, src)
    assert vol.files[KEY] == b"abc" * 1000 and src.is_file()
    mv.delete(KEY)
    assert not mv.exists(KEY)
    mv.delete(KEY)  # deleting a missing object is fine


def test_volume_delete_reraises_other_invalid_errors(mv, vol, monkeypatch):
    from modal.exception import InvalidError

    def boom(path, recursive=False):
        raise InvalidError("Volume is read-only")

    monkeypatch.setattr(vol, "remove_file", boom)
    with pytest.raises(InvalidError):
        mv.delete(KEY)


def test_volume_put_file_streams_from_disk(tmp_path: Path):
    import tracemalloc

    from fake_volume import FakeVolume
    from radar_desk.storage.modal_volume import ModalVolumeStorage

    vol = FakeVolume(keep=False)
    mv = ModalVolumeStorage("radar-data", "http://x", volume=vol)
    big = tmp_path / "sparse.bin"
    size = 1024**3
    with big.open("wb") as fh:
        fh.truncate(size)  # sparse: a GiB on paper, nothing on disk
    tracemalloc.start()
    try:
        mv.put_file(KEY, big)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert vol.sizes[KEY] == size
    assert vol.max_read <= 1024 * 1024
    assert peak < 32 * 1024 * 1024


def test_volume_urls_and_worker_refs(mv):
    assert mv.put_url(KEY, 900) == f"/_volume/{KEY}"  # relative: the browser's own origin and cookie
    assert mv.get_url(KEY, 3600) == f"/_volume/{KEY}"
    assert mv.worker_ref(KEY, "GET", 60) == f"volume://{KEY}"
    assert mv.worker_ref("jobs/j1/mask.nii.gz", "PUT", 60) == "volume://jobs/j1/mask.nii.gz"
    with pytest.raises(ValueError):
        mv.worker_ref(KEY, "POST", 60)


@pytest.mark.parametrize("bad", ["", "/abs", "a/../b", "a//b"])
def test_volume_rejects_bad_keys(mv, bad):
    for call in (lambda: mv.put_bytes(bad, b"x"), lambda: mv.exists(bad), lambda: mv.put_url(bad, 60),
                 lambda: mv.worker_ref(bad, "GET", 60)):
        with pytest.raises(StorageError):
            call()


def test_volume_is_looked_up_lazily_once(monkeypatch):
    import modal

    from fake_volume import FakeVolume
    from radar_desk.storage.modal_volume import ModalVolumeStorage

    looked_up = []

    def from_name(name, **kw):
        looked_up.append((name, kw))
        return FakeVolume()

    monkeypatch.setattr(modal.Volume, "from_name", from_name)
    monkeypatch.setattr(modal.Client, "from_credentials", lambda tid, ts: ("client", tid, ts))
    mv = ModalVolumeStorage("radar-data", "http://x", credentials=("ak-1", "as-2"))
    assert looked_up == []  # nothing touches Modal until the first call
    assert not mv.exists(KEY)
    mv.put_bytes(KEY, b"x")
    assert looked_up == [("radar-data", {"create_if_missing": True, "client": ("client", "ak-1", "as-2")})]


def test_make_storage_picks_the_volume(tmp_path: Path):
    from radar_desk.storage.modal_volume import ModalVolumeStorage

    s = SimpleNamespace(storage_backend="modal_volume", modal_data_volume="radar-data-x", s3_bucket="ignored",
                        public_base_url="http://localhost:8000", modal_token_id=None, modal_token_secret=None,
                        data_dir=tmp_path, session_secret="s")
    storage = make_storage(s)
    assert isinstance(storage, ModalVolumeStorage)
    assert storage.volume_name == "radar-data-x" and storage._volume is None


# RunPod volume adapter: the S3 adapter on the volume's S3 API, with /_volume routes and volume:// refs


RUNPOD_BUCKET = "vol0exampl"


def _runpod(**kw):
    from radar_desk.storage.runpod_volume import RunPodVolumeStorage

    args = {"volume_id": RUNPOD_BUCKET, "datacenter": "EU-RO-1", "access_key": "user_DUMMY",
            "secret_key": "rps_dummy", "max_upload_bytes": 300 * 1024 * 1024}
    return RunPodVolumeStorage(**{**args, **kw})


def test_runpod_volume_builds_its_client_from_the_settings():
    storage = _runpod()
    meta = storage.client.meta
    assert storage.name == "runpod_volume" and storage.bucket == RUNPOD_BUCKET
    assert meta.endpoint_url == "https://s3api-eu-ro-1.runpod.io"
    assert meta.region_name == "EU-RO-1"
    assert meta.config.signature_version == "s3v4"
    assert meta.config.s3["addressing_style"] == "path"


def test_runpod_volume_urls_and_worker_refs():
    storage = _runpod()
    assert storage.put_url(KEY, 900) == f"/_volume/{KEY}"
    assert storage.get_url(KEY, 3600) == f"/_volume/{KEY}"
    assert storage.worker_ref(KEY, "GET", 60) == f"volume://{KEY}"
    assert storage.worker_ref("jobs/j1/result.json", "PUT", 60) == "volume://jobs/j1/result.json"
    with pytest.raises(ValueError):
        storage.worker_ref(KEY, "DELETE", 60)
    for call in (lambda: storage.put_url("a/../b", 60), lambda: storage.worker_ref("/abs", "GET", 60)):
        with pytest.raises(StorageError):
            call()


def test_runpod_volume_upload_limit_must_fit_one_put_object():
    from radar_desk.config import ConfigError
    from radar_desk.storage.runpod_volume import PUT_OBJECT_CAP

    assert PUT_OBJECT_CAP == 500 * 1024 * 1024
    _runpod(max_upload_bytes=PUT_OBJECT_CAP - 1)
    for n in (PUT_OBJECT_CAP, PUT_OBJECT_CAP + 1):
        with pytest.raises(ConfigError, match=f"MAX_UPLOAD_BYTES must be under {PUT_OBJECT_CAP}.*it is {n}"):
            _runpod(max_upload_bytes=n)


@pytest.fixture
def rp() -> tuple:
    client = _s3_client()
    storage = _runpod(client=client)
    with Stubber(client) as stub:
        yield storage, stub
        stub.assert_no_pending_responses()


def test_runpod_volume_inherits_head_get_put_delete(rp, tmp_path: Path):
    from botocore.stub import ANY

    storage, stub = rp
    params = {"Bucket": RUNPOD_BUCKET, "Key": KEY}
    stub.add_response("head_object", {"ContentLength": 42}, params)
    assert storage.exists(KEY)
    stub.add_client_error("head_object", "404", "Not Found", 404, expected_params=params)
    assert not storage.exists(KEY)
    stub.add_response("head_object", {"ContentLength": 42}, params)
    assert storage.size(KEY) == 42
    stub.add_client_error("head_object", "404", "Not Found", 404, expected_params=params)
    with pytest.raises(ObjectMissing):
        storage.size(KEY)
    body = StreamingBody(io.BytesIO(b"abc" * 10), 30)
    stub.add_response("get_object", {"Body": body, "ContentLength": 30}, params)
    assert b"".join(storage.open_stream(KEY)) == b"abc" * 10
    stub.add_response("put_object", {}, {**params, "Body": b"data", "ContentType": "application/json"})
    storage.put_bytes(KEY, b"data", content_type="application/json")
    src = tmp_path / "in.bin"
    src.write_bytes(b"data")
    stub.add_response("put_object", {}, {**params, "Body": ANY, "ContentType": "application/gzip"})
    storage.put_file(KEY, src, content_type="application/gzip")
    stub.add_response("delete_object", {}, params)
    storage.delete(KEY)


def _runpod_settings(tmp_path: Path, **kw) -> SimpleNamespace:
    base = {"storage_backend": "runpod_volume", "runpod_volume_id": RUNPOD_BUCKET, "runpod_datacenter": "EU-RO-1",
            "runpod_s3_access_key_id": SecretStr("user_DUMMY"), "runpod_s3_secret_access_key": SecretStr("rps_x"),
            "max_upload_bytes": 300 * 1024 * 1024, "s3_bucket": "ignored", "data_dir": tmp_path,
            "public_base_url": "http://localhost:8000", "session_secret": "s"}
    return SimpleNamespace(**{**base, **kw})


def test_make_storage_picks_the_runpod_volume(tmp_path: Path):
    from radar_desk.storage import RunPodVolumeStorage

    storage = make_storage(_runpod_settings(tmp_path))
    assert isinstance(storage, RunPodVolumeStorage) and storage.bucket == RUNPOD_BUCKET
    assert storage.client.meta.endpoint_url == "https://s3api-eu-ro-1.runpod.io"
    creds = storage.client._request_signer._credentials
    assert (creds.access_key, creds.secret_key) == ("user_DUMMY", "rps_x")  # unwrapped SecretStr


@pytest.mark.parametrize("missing", ["runpod_volume_id", "runpod_s3_access_key_id", "runpod_s3_secret_access_key"])
def test_make_storage_runpod_volume_names_the_missing_setting(tmp_path: Path, missing):
    from radar_desk.config import ConfigError

    with pytest.raises(ConfigError, match=f"STORAGE_BACKEND=runpod_volume needs {missing.upper()}$"):
        make_storage(_runpod_settings(tmp_path, **{missing: None}))


def test_make_storage_runpod_volume_refuses_a_large_upload_limit(tmp_path: Path):
    from radar_desk.config import ConfigError

    with pytest.raises(ConfigError, match="MAX_UPLOAD_BYTES"):
        make_storage(_runpod_settings(tmp_path, max_upload_bytes=600 * 1024 * 1024))


def test_browser_via_api_only_on_the_volume_adapters(local, mv, s3):
    assert _runpod().browser_via_api is True
    assert mv.browser_via_api is True
    assert getattr(local, "browser_via_api", False) is False
    assert getattr(s3[0], "browser_via_api", False) is False


# describe_storage and the backend override


@pytest.mark.parametrize("over,name", [
    ({"s3_bucket": "my-radar-bucket", "s3_endpoint_url": "https://fly.storage.tigris.dev"},
     "Tigris bucket my-radar-bucket"),
    ({"s3_bucket": "b", "s3_endpoint_url": "https://s3.example.com:9000/x"}, "S3 bucket b (s3.example.com)"),
    ({"s3_bucket": "b", "s3_endpoint_url": "https://nottigris.dev"}, "S3 bucket b (nottigris.dev)"),
    ({"s3_bucket": "b"}, "S3 bucket b"),
    ({"storage_backend": "modal_volume"}, "Modal volume radar-data-x"),
    ({"storage_backend": "runpod_volume"}, f"RunPod volume {RUNPOD_BUCKET} (EU-RO-1)"),
    ({}, "Local folder"),
])
def test_describe_storage(over, name):
    from radar_desk.storage import describe_storage

    base = {"storage_backend": None, "s3_bucket": None, "s3_endpoint_url": None, "modal_data_volume": "radar-data-x",
            "runpod_volume_id": RUNPOD_BUCKET, "runpod_datacenter": "EU-RO-1"}
    assert describe_storage(SimpleNamespace(**{**base, **over})) == name


def test_make_storage_backend_overrides_the_setting(tmp_path: Path):
    from radar_desk.storage import ModalVolumeStorage, RunPodVolumeStorage

    s = _runpod_settings(tmp_path, storage_backend=None, s3_endpoint_url="https://example.invalid", s3_region="auto",
                         aws_access_key_id="AKIDUMMY", aws_secret_access_key="dummysecret",
                         modal_data_volume="radar-data", modal_token_id=None, modal_token_secret=None)
    assert type(make_storage(s, backend="local")) is LocalStorage
    assert type(make_storage(s, backend="s3")) is S3Storage
    assert isinstance(make_storage(s, backend="runpod_volume"), RunPodVolumeStorage)
    volume = make_storage(s, backend="modal_volume")
    assert isinstance(volume, ModalVolumeStorage) and volume._volume is None
    assert type(make_storage(s)) is S3Storage  # s3_bucket is set and STORAGE_BACKEND is not
