"""Weights check against worker/weights.json, standard library only.

Used by the Modal function at container start and by scripts/score_local.py, so both
refuse the same mismatches: every file by size, the checkpoint (or every file with
hash_all) by sha256.

ensure_weights (run by scripts/ensure_weights.py when a non-Modal worker starts) uses the
weights already on a volume, or fills the volume (or a local fallback directory) from
Hugging Face at the revision pinned in weights.json. Modal does not use it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

CHECKPOINT = "checkpoint_radar_pretrain.pth"
VENDORED = {
    "infer_text_embedding_radar.pt": Path(__file__).resolve().parents[1]
    / "vendor" / "damo-radar" / "ckpt" / "infer_text_embedding_radar.pt",
}
LOCK_NAME = ".ensure_weights.lock"


def sha256_file(path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def check_weights(weights_dir, manifest: dict, hash_all: bool = False) -> dict:
    """Compare files on disk with the manifest. Sizes always; sha256 of the checkpoint, or of all with hash_all."""
    rows, problems = [], []
    for entry in manifest["files"]:
        path = Path(weights_dir) / entry["path"]
        row = {"path": entry["path"], "expected_size": entry["size"], "expected_sha256": entry.get("sha256")}
        if not path.is_file():
            row.update(size=None, sha256=None, ok=False)
            problems.append(f"{entry['path']}: missing")
            rows.append(row)
            continue
        row["size"] = path.stat().st_size
        ok = row["size"] == entry["size"]
        if not ok:
            problems.append(f"{entry['path']}: size {row['size']} != {entry['size']}")
        if hash_all or entry["path"] == CHECKPOINT:
            row["sha256"] = sha256_file(path)
            if entry.get("sha256") and row["sha256"] != entry["sha256"]:
                ok = False
                problems.append(f"{entry['path']}: sha256 {row['sha256']} != {entry['sha256']}")
        row["ok"] = ok
        rows.append(row)
    ckpt = next((r for r in rows if r["path"] == CHECKPOINT), {})
    return {"ok": not problems, "problems": problems, "rows": rows, "checkpoint_sha256": ckpt.get("sha256")}


def code_commit(vendored_md) -> str | None:
    """The upstream commit named in the vendored tree's VENDORED.md, or None."""
    path = Path(vendored_md)
    if not path.is_file():
        return None
    m = re.search(r"\b([0-9a-f]{40})\b", path.read_text())
    return m.group(1) if m else None


# ---------------------------------------------------------------- ensure (use or download)

HF_HOST = "huggingface.co"
WRITER = f"{socket.gethostname()}.{os.getpid()}"  # names this process's partial files


class _SameHostAuth(urllib.request.HTTPRedirectHandler):
    """Carry the Authorization header across redirects that stay on the same host, never to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        auth = req.unredirected_hdrs.get("Authorization")
        if new is not None and auth and urllib.parse.urlsplit(newurl).hostname == urllib.parse.urlsplit(req.full_url).hostname:
            new.add_unredirected_header("Authorization", auth)
        return new


def hf_downloader(manifest: dict):
    """download(rel_path, dest, on_chunk) fetching rel_path from the manifest's pinned Hugging Face revision."""
    repo, rev = manifest["hf_repo"], manifest["hf_revision"]
    token = os.environ.get("HF_TOKEN")
    opener = urllib.request.build_opener(_SameHostAuth)

    def download(rel_path: str, dest, on_chunk=None) -> None:
        req = urllib.request.Request(f"https://{HF_HOST}/{repo}/resolve/{rev}/{rel_path}")
        if token:
            req.add_unredirected_header("Authorization", f"Bearer {token}")
        with opener.open(req, timeout=60) as resp, open(dest, "wb") as fh:
            while True:
                block = resp.read(8 << 20)
                if not block:
                    break
                fh.write(block)
                if on_chunk:
                    on_chunk(len(block))

    return download


def _writable(d: Path) -> bool:
    if not d.is_dir():
        return False
    probe = d / f".write_probe.{WRITER}"
    try:
        probe.write_bytes(b"")
        probe.unlink()
        return True
    except OSError:
        return False


def _usable(d: Path) -> bool:
    """A writable directory, created when only its parent (the mount) exists."""
    if d.is_dir():
        return _writable(d)
    if not _writable(d.parent):
        return False
    try:
        d.mkdir(exist_ok=True)
    except OSError:
        return False
    return _writable(d)


def _read_lock(path: Path) -> dict:
    try:
        info = json.loads(path.read_text() or "{}")
        if not info.get("time"):
            info["time"] = path.stat().st_mtime
        return info
    except (OSError, ValueError):
        return {}


def _write_lock(path: Path, token: str) -> None:
    tmp = path.with_name(f"{path.name}.{token}.tmp")
    tmp.write_text(json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "token": token, "time": time.time()}))
    os.replace(tmp, path)


def _owns(path: Path, token: str) -> bool:
    return _read_lock(path).get("token") == token


def _take_stale(path: Path, stale_after: float, log) -> None:
    """Move a stale lock aside; only the waiter whose rename succeeds gets to remove it."""
    aside = path.with_name(f"{path.name}.stale.{WRITER}.{uuid.uuid4().hex}")
    try:
        os.rename(path, aside)
    except FileNotFoundError:
        return  # another waiter moved it first
    info = _read_lock(aside)
    if time.time() - float(info.get("time") or 0) <= stale_after:
        # a fresh lock was created between our read and our rename; put it back unless replaced again
        try:
            os.link(aside, path)
        except FileExistsError:
            pass
        except OSError:
            os.rename(aside, path)
            return
        aside.unlink(missing_ok=True)
        return
    log(f"taking over stale lock {path} ({info})")
    aside.unlink(missing_ok=True)


def _acquire_lock(path: Path, stale_after: float, wait_timeout: float, poll: float, log) -> str:
    deadline = time.monotonic() + wait_timeout
    token = f"{WRITER}.{uuid.uuid4().hex}"
    announced = False
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            _write_lock(path, token)
            return token
        except FileExistsError:
            pass
        info = _read_lock(path)
        if info and time.time() - float(info["time"]) > stale_after:
            _take_stale(path, stale_after, log)
            continue
        if not announced:
            log(f"waiting for lock {path} held by {info}")
            announced = True
        if time.monotonic() > deadline:
            raise TimeoutError(f"lock {path} still held by {info} after {wait_timeout}s")
        time.sleep(poll)


def _release_lock(path: Path, token: str, log) -> None:
    if _owns(path, token):
        path.unlink(missing_ok=True)
    else:
        log(f"lock {path} is no longer ours; leaving it")


def _fetch(entry: dict, d: Path, download, log, on_chunk=None, writer: str = WRITER) -> str:
    """Put one manifest file in place via this writer's own partial file, verified before the rename."""
    final = d / entry["path"]
    final.parent.mkdir(parents=True, exist_ok=True)
    partial = final.with_name(f"{final.name}.partial.{writer}")
    vendored = VENDORED.get(entry["path"])
    how = "copied" if vendored is not None and vendored.is_file() else "downloaded"
    try:
        if how == "copied":
            shutil.copyfile(vendored, partial)
        else:
            download(entry["path"], partial, on_chunk)
        size = partial.stat().st_size
        if size != entry["size"]:
            raise RuntimeError(f"{entry['path']}: {how} size {size} != {entry['size']}")
        digest = sha256_file(partial)
        if entry.get("sha256") and digest != entry["sha256"]:
            raise RuntimeError(f"{entry['path']}: {how} sha256 {digest} != {entry['sha256']}")
        os.replace(partial, final)
    finally:
        partial.unlink(missing_ok=True)
    log(f"{how} {entry['path']} ({entry['size']} bytes)")
    return how


def _fill(d: Path, manifest: dict, download, log, stale_after, wait_timeout, poll, refresh_every) -> dict:
    lock = d / LOCK_NAME
    token = _acquire_lock(lock, stale_after, wait_timeout, poll, log)
    last = [time.monotonic()]

    def refresh(_n=0):  # keep the lock fresh during long downloads so it never looks stale
        if time.monotonic() - last[0] >= refresh_every and _owns(lock, token):
            _write_lock(lock, token)
            last[0] = time.monotonic()

    try:
        for old in d.rglob("*.partial.*"):  # left by a writer that died; we hold the lock
            try:
                if time.time() - old.stat().st_mtime > stale_after:
                    old.unlink(missing_ok=True)
            except FileNotFoundError:  # its writer finished or cleaned up meanwhile
                pass
        state = check_weights(d, manifest, hash_all=True)  # someone may have filled it while we waited
        bad = {r["path"] for r in state["rows"] if not r["ok"]}
        done = {"downloaded": [], "copied": [], "bytes": 0}
        for entry in manifest["files"]:
            if entry["path"] not in bad:
                continue
            how = _fetch(entry, d, download, log, on_chunk=refresh)
            done[how].append(entry["path"])
            done["bytes"] += entry["size"]
            refresh()
        if done["downloaded"] or done["copied"]:
            state = check_weights(d, manifest, hash_all=True)
        if not state["ok"]:
            raise RuntimeError(f"weights in {d} still wrong after filling: {'; '.join(state['problems'])}")
        return done
    finally:
        _release_lock(lock, token, log)


def ensure_weights(target_dir, manifest: dict, *, fallback_dir=None, download=None, log=print,
                   stale_after: float = 1800, wait_timeout: float = 7200, poll: float = 2.0,
                   refresh_every: float = 30) -> dict:
    """Use the weights in target_dir if they check out; otherwise fill target_dir, or fallback_dir
    when target_dir cannot be used or filled, and report where they came from."""
    t0 = time.perf_counter()
    download = download or hf_downloader(manifest)
    fill_args = (download, log, stale_after, wait_timeout, poll, refresh_every)
    target = Path(target_dir)
    notes, done = [], None

    if target.is_dir() and check_weights(target, manifest)["ok"]:
        d, source = target, "volume"
    else:
        d = source = None
        if _usable(target):
            try:
                done = _fill(target, manifest, *fill_args)
                d, source = target, "downloaded-to-volume"
            except (OSError, RuntimeError) as exc:
                if fallback_dir is None:
                    raise
                notes.append(f"filling {target} failed: {type(exc).__name__}: {exc}")
                log(notes[-1])
        else:
            notes.append(f"{target} (or its mount) is missing or not writable")
        if d is None:
            if fallback_dir is None:
                raise RuntimeError(f"weights in {target} are not usable and no fallback directory was given")
            d = Path(fallback_dir)
            d.mkdir(parents=True, exist_ok=True)
            if check_weights(d, manifest)["ok"]:
                source = "local"
            else:
                source = "downloaded-to-local"
                done = _fill(d, manifest, *fill_args)

    done = done or {"downloaded": [], "copied": [], "bytes": 0}
    if source.startswith("downloaded") and not (done["downloaded"] or done["copied"]):
        source = "volume" if d == target else "local"  # another worker filled it while we waited
    return {"source": source, "dir": str(d), "downloaded": done["downloaded"], "copied": done["copied"],
            "bytes": done["bytes"], "seconds": round(time.perf_counter() - t0, 2), "notes": notes}
