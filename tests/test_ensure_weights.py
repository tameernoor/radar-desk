"""ensure_weights against a small fake manifest and a fake downloader; no network."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import threading
import time
from pathlib import Path

import pytest

from radar_worker import weights as w

ROOT = Path(__file__).resolve().parents[1]

CONTENT = {
    "checkpoint_radar_pretrain.pth": b"checkpoint bytes" * 10,
    "infer_text_embedding_radar.pt": b"embedding",
    "bert-base-chinese/config.json": b'{"x": 1}',
    "bert-base-chinese/vocab.txt": b"a\nb\nc\n",
}


def _manifest() -> dict:
    return {
        "hf_repo": "example/repo",
        "hf_revision": "0" * 40,
        "files": [
            {"path": p, "size": len(b), "sha256": hashlib.sha256(b).hexdigest()} for p, b in CONTENT.items()
        ],
    }


class FakeDownload:
    def __init__(self, content=CONTENT):
        self.content = content
        self.calls = []

    def __call__(self, rel_path, dest, on_chunk=None):
        self.calls.append(rel_path)
        assert f".partial.{w.WRITER}" in str(dest)
        Path(dest).write_bytes(self.content[rel_path])
        if on_chunk:
            on_chunk(len(self.content[rel_path]))


@pytest.fixture(autouse=True)
def vendored(tmp_path, monkeypatch):
    src = tmp_path / "vendor" / "infer_text_embedding_radar.pt"
    src.parent.mkdir()
    src.write_bytes(CONTENT["infer_text_embedding_radar.pt"])
    monkeypatch.setattr(w, "VENDORED", {"infer_text_embedding_radar.pt": src})
    return src


def _fill(d: Path, skip=()):
    for p, b in CONTENT.items():
        if p in skip:
            continue
        (d / p).parent.mkdir(parents=True, exist_ok=True)
        (d / p).write_bytes(b)


def _run(d, **kw):
    return w.ensure_weights(d, _manifest(), log=lambda m: None, **kw)


def test_all_present_means_no_download(tmp_path):
    vol = tmp_path / "vol"
    _fill(vol)
    dl = FakeDownload()
    rep = _run(vol, download=dl)
    assert rep["source"] == "volume" and rep["dir"] == str(vol)
    assert dl.calls == [] and rep["downloaded"] == [] and rep["bytes"] == 0


def test_one_missing_downloads_only_that_one(tmp_path):
    vol = tmp_path / "vol"
    _fill(vol, skip=("bert-base-chinese/vocab.txt",))
    dl = FakeDownload()
    rep = _run(vol, download=dl)
    assert dl.calls == ["bert-base-chinese/vocab.txt"]
    assert rep["source"] == "downloaded-to-volume"
    assert rep["downloaded"] == ["bert-base-chinese/vocab.txt"] and rep["bytes"] == len(CONTENT["bert-base-chinese/vocab.txt"])
    assert not (vol / w.LOCK_NAME).exists()


def test_corrupt_files_are_replaced(tmp_path):
    # the start-up check sees every size and the checkpoint's sha256 (as Modal does)
    vol = tmp_path / "vol"
    _fill(vol)
    (vol / "bert-base-chinese/config.json").write_bytes(b"trunc")
    ckpt = CONTENT["checkpoint_radar_pretrain.pth"]
    (vol / "checkpoint_radar_pretrain.pth").write_bytes(b"X" * len(ckpt))
    dl = FakeDownload()
    rep = _run(vol, download=dl)
    assert sorted(dl.calls) == ["bert-base-chinese/config.json", "checkpoint_radar_pretrain.pth"]
    assert (vol / "bert-base-chinese/config.json").read_bytes() == CONTENT["bert-base-chinese/config.json"]
    assert (vol / "checkpoint_radar_pretrain.pth").read_bytes() == ckpt
    assert rep["source"] == "downloaded-to-volume"


def test_wrong_bytes_raise_and_leave_no_file_under_the_real_name(tmp_path):
    vol = tmp_path / "vol"
    vol.mkdir()
    bad = dict(CONTENT, **{"checkpoint_radar_pretrain.pth": b"Z" * len(CONTENT["checkpoint_radar_pretrain.pth"])})
    with pytest.raises(RuntimeError, match="sha256"):
        _run(vol, download=FakeDownload(bad))
    assert not (vol / "checkpoint_radar_pretrain.pth").exists()
    assert not list(vol.rglob("*.partial*"))
    assert not (vol / w.LOCK_NAME).exists()


def test_missing_mount_uses_fallback(tmp_path):
    local = tmp_path / "local"
    target = tmp_path / "no-mount" / "radar-weights"
    dl = FakeDownload()
    rep = _run(target, fallback_dir=local, download=dl)
    assert rep["source"] == "downloaded-to-local" and rep["dir"] == str(local)
    assert not (tmp_path / "no-mount").exists()
    assert sorted(rep["downloaded"]) == sorted(p for p in CONTENT if p != "infer_text_embedding_radar.pt")
    rep2 = _run(target, fallback_dir=local, download=dl)
    assert rep2["source"] == "local" and rep2["downloaded"] == []


def test_mount_present_target_missing_is_created_and_filled(tmp_path):
    mount = tmp_path / "workspace"
    mount.mkdir()
    rep = _run(mount / "radar-weights", fallback_dir=tmp_path / "local", download=FakeDownload())
    assert rep["source"] == "downloaded-to-volume" and rep["dir"] == str(mount / "radar-weights")
    assert not (tmp_path / "local").exists()
    assert w.check_weights(mount / "radar-weights", _manifest(), hash_all=True)["ok"]


def test_failed_volume_fill_falls_back_to_local_with_a_note(tmp_path):
    vol = tmp_path / "vol"
    vol.mkdir()

    def broken(rel_path, dest, on_chunk=None):
        raise OSError("No space left on device")

    calls = []

    def dl(rel_path, dest, on_chunk=None):
        if dest.parent.parts[: len(vol.parts)] == vol.parts:
            return broken(rel_path, dest)
        calls.append(rel_path)
        Path(dest).write_bytes(CONTENT[rel_path])

    rep = _run(vol, fallback_dir=tmp_path / "local", download=dl)
    assert rep["source"] == "downloaded-to-local"
    assert any("No space left" in n for n in rep["notes"])
    assert not (vol / w.LOCK_NAME).exists() and not list(vol.rglob("*.partial*"))
    with pytest.raises(OSError, match="No space"):
        _run(vol, download=broken)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_read_only_target_uses_fallback(tmp_path):
    vol = tmp_path / "vol"
    _fill(vol, skip=("bert-base-chinese/vocab.txt",))
    vol.chmod(0o555)
    try:
        rep = _run(vol, fallback_dir=tmp_path / "local", download=FakeDownload())
    finally:
        vol.chmod(0o755)
    assert rep["source"] == "downloaded-to-local"
    assert not (vol / "bert-base-chinese/vocab.txt").exists()


def test_no_fallback_and_unusable_target_raises(tmp_path):
    with pytest.raises(RuntimeError, match="no fallback"):
        _run(tmp_path / "no-mount" / "missing", download=FakeDownload())


def test_vendored_embedding_is_copied_not_downloaded(tmp_path):
    vol = tmp_path / "vol"
    vol.mkdir()
    dl = FakeDownload()
    rep = _run(vol, download=dl)
    assert "infer_text_embedding_radar.pt" not in dl.calls
    assert rep["copied"] == ["infer_text_embedding_radar.pt"]
    assert (vol / "infer_text_embedding_radar.pt").read_bytes() == CONTENT["infer_text_embedding_radar.pt"]


def test_second_caller_waits_for_the_lock_then_sees_the_files(tmp_path):
    vol = tmp_path / "vol"
    vol.mkdir()
    lock = vol / w.LOCK_NAME
    lock.write_text(json.dumps({"pid": 1, "host": "other", "time": time.time()}))
    dl = FakeDownload()
    out = {}
    t = threading.Thread(target=lambda: out.update(rep=_run(vol, download=dl, poll=0.02)))
    t.start()
    time.sleep(0.2)
    assert t.is_alive()  # waiting on the lock
    _fill(vol)  # the other worker finishes
    lock.unlink()
    t.join(5)
    assert dl.calls == []
    assert out["rep"]["source"] == "volume" and out["rep"]["downloaded"] == []


def test_two_writers_do_not_clobber_each_others_partial(tmp_path):
    entry = next(e for e in _manifest()["files"] if e["path"] == "bert-base-chinese/vocab.txt")
    body = CONTENT[entry["path"]]
    seen = []

    def inner(rel_path, dest, on_chunk=None):
        seen.append(Path(dest).name)
        Path(dest).write_bytes(body)

    def outer(rel_path, dest, on_chunk=None):
        seen.append(Path(dest).name)
        Path(dest).write_bytes(body[:2])  # half written when the other writer runs
        w._fetch(entry, tmp_path, inner, lambda m: None, writer="hostB.2")
        with open(dest, "ab") as fh:
            fh.write(body[2:])

    w._fetch(entry, tmp_path, outer, lambda m: None, writer="hostA.1")
    assert seen == ["vocab.txt.partial.hostA.1", "vocab.txt.partial.hostB.2"]
    assert (tmp_path / entry["path"]).read_bytes() == body
    assert not list(tmp_path.rglob("*.partial*"))


def test_stale_lock_taken_by_two_waiters_yields_one_owner(tmp_path):
    lock = tmp_path / w.LOCK_NAME
    lock.write_text(json.dumps({"pid": 1, "host": "dead", "token": "old", "time": time.time() - 7200}))
    tokens, errors = [], []
    barrier = threading.Barrier(2)

    def waiter():
        barrier.wait()
        try:
            tokens.append(w._acquire_lock(lock, 60, 0.5, 0.01, lambda m: None))
        except TimeoutError as exc:
            errors.append(exc)

    threads = [threading.Thread(target=waiter) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert len(tokens) == 1 and len(errors) == 1
    assert w._owns(lock, tokens[0])
    assert not list(tmp_path.glob("*.stale.*"))


def test_holder_never_deletes_a_lock_it_no_longer_owns(tmp_path):
    lock = tmp_path / w.LOCK_NAME
    mine = w._acquire_lock(lock, 60, 1, 0.01, lambda m: None)
    w._write_lock(lock, "someone-else")  # taken over meanwhile
    w._release_lock(lock, mine, lambda m: None)
    assert lock.exists() and w._owns(lock, "someone-else")


def test_lock_is_refreshed_during_a_download(tmp_path):
    vol = tmp_path / "vol"
    _fill(vol, skip=("checkpoint_radar_pretrain.pth",))
    stamps = []

    def slow(rel_path, dest, on_chunk=None):
        body = CONTENT[rel_path]
        with open(dest, "wb") as fh:
            for i in range(3):
                stamps.append(json.loads((vol / w.LOCK_NAME).read_text())["time"])
                time.sleep(0.02)
                fh.write(body[i * len(body) // 3:(i + 1) * len(body) // 3])
                on_chunk(1)

    _run(vol, download=slow, refresh_every=0)
    assert stamps == sorted(stamps) and len(set(stamps)) == 3


def test_token_follows_same_host_redirects_only():
    handler = w._SameHostAuth()

    def redirected(newurl):
        req = w.urllib.request.Request("https://huggingface.co/r/resolve/x/f")
        req.add_unredirected_header("Authorization", "Bearer t")
        return handler.redirect_request(req, None, 302, "Found", {}, newurl)

    assert redirected("https://huggingface.co/api/resolve-cache/f").unredirected_hdrs.get("Authorization") == "Bearer t"
    other = redirected("https://cas-bridge.xethub.hf.co/f")
    assert "Authorization" not in other.unredirected_hdrs and "Authorization" not in other.headers


def test_stale_lock_is_taken_over(tmp_path):
    vol = tmp_path / "vol"
    vol.mkdir()
    (vol / w.LOCK_NAME).write_text(json.dumps({"pid": 1, "host": socket.gethostname(), "time": time.time() - 7200}))
    dl = FakeDownload()
    rep = _run(vol, download=dl, stale_after=60)
    assert rep["source"] == "downloaded-to-volume" and dl.calls
    assert not (vol / w.LOCK_NAME).exists()


def test_cli_prints_json_report(tmp_path, capsys):
    import importlib.util

    spec = importlib.util.spec_from_file_location("ensure_weights_cli", ROOT / "scripts" / "ensure_weights.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    vol = tmp_path / "vol"
    _fill(vol)
    man = tmp_path / "m.json"
    man.write_text(json.dumps(_manifest()))
    assert cli.main(["--dir", str(vol), "--manifest", str(man)]) == 0
    assert json.loads(capsys.readouterr().out)["source"] == "volume"
    not_a_dir = tmp_path / "file"
    not_a_dir.write_text("")
    assert cli.main(["--dir", str(tmp_path / "no-mount" / "missing"), "--fallback", str(not_a_dir), "--manifest", str(man)]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_cli_defaults_fit_a_runpod_pod(monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("ensure_weights_cli", ROOT / "scripts" / "ensure_weights.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    for k in ("RADAR_WEIGHTS_DIR", "RADAR_WEIGHTS_FALLBACK", "RADAR_WEIGHTS_MANIFEST"):
        monkeypatch.delenv(k, raising=False)
    args = cli.parse_args([])
    assert (args.dir, args.fallback) == ("/workspace/radar-weights", "/root/radar-weights")
    monkeypatch.setenv("RADAR_WEIGHTS_DIR", "/runpod-volume/w")
    monkeypatch.setenv("RADAR_WEIGHTS_FALLBACK", "/tmp/w")
    args = cli.parse_args([])
    assert (args.dir, args.fallback) == ("/runpod-volume/w", "/tmp/w")


def test_entrypoint_exports_the_resolved_dir(tmp_path):
    import subprocess
    import sys

    vol = tmp_path / "vol"
    _fill(vol)
    man = tmp_path / "m.json"
    man.write_text(json.dumps(_manifest()))
    env = dict(os.environ, RADAR_WEIGHTS_DIR=str(vol), RADAR_WEIGHTS_FALLBACK=str(tmp_path / "local"),
               RADAR_WEIGHTS_MANIFEST=str(man), PYTHON=sys.executable)
    script = ROOT / "worker" / "docker" / "entrypoint.sh"
    out = subprocess.run(["sh", str(script), "sh", "-c", 'echo "resolved=$RADAR_WEIGHTS_RESOLVED"'],
                         env=env, capture_output=True, text=True, timeout=60, check=False)
    assert out.returncode == 0, out.stderr
    assert f"resolved={vol}" in out.stdout
    assert subprocess.run(["sh", str(script)], env=env, capture_output=True, timeout=60, check=False).returncode == 0


def test_real_manifest_loads_and_pins_a_revision():
    data = json.loads((ROOT / "worker" / "weights.json").read_text())
    assert data["hf_repo"] == "radar-generalist/RADAR"
    assert len(data["hf_revision"]) == 40 and int(data["hf_revision"], 16) >= 0
    assert w.VENDORED  # the module default points at the vendored embedding
