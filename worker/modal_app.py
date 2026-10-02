"""Modal app `radar-desk`: one GPU function, `score`, that scores a CT scan with RADAR.

Deploy with `modal deploy worker/modal_app.py` (RADAR_GPU picks the GPU, a name or a
comma-separated ordered list, default L4; MODAL_DATA_VOLUME names the data Volume, default
radar-data). The function downloads the scan from a presigned GET, runs
`radar_worker.infer`, writes five artefacts, PUTs each to its presigned URL and returns the
result dict. With `volume://<key>` references instead of URLs it reads the scan from and
writes the artefacts to the data Volume mounted at /data.

First manual run without the API: `modal run worker/modal_app.py --path scan.nii.gz`
stages the scan on a scratch Volume, scores it on the same image and options, and
saves the artefacts under `smoke-out/<job id>/`.

Importing this module defines objects only; nothing talks to Modal until deploy or run.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from radar_worker.weights import check_weights  # shared with scripts/score_local.py

APP_NAME = "radar-desk"
WEIGHTS_VOLUME = "radar-weights"
WEIGHTS_DIR = "/weights"
VENDOR_REMOTE = "/root/damo-radar"
MANIFEST_REMOTE = "/root/weights.json"
SMOKE_VOLUME = "radar-smoke"
SMOKE_DIR = "/smoke"
DATA_VOLUME = os.environ.get("MODAL_DATA_VOLUME", "radar-data")  # read at deploy time
DATA_DIR = "/data"
VOLUME_SCHEME = "volume://"

# name -> (file name, content type); the names match Result.artefacts
ARTEFACTS = {
    "scores_json": ("scores.json", "application/json"),
    "scores_csv": ("scores.csv", "text/csv; charset=utf-8"),
    "mask": ("mask.nii.gz", "application/gzip"),
    "trace": ("trace.json", "application/json"),
    "log": ("worker.log", "text/plain; charset=utf-8"),
}


def parse_gpu(value: str | None):
    """RADAR_GPU as Modal wants it: one name as a string, several as an ordered list."""
    parts = [p.strip() for p in (value or "").split(",") if p.strip()]
    if not parts:
        return "L4"
    return parts[0] if len(parts) == 1 else parts


RADAR_GPU = parse_gpu(os.environ.get("RADAR_GPU"))

image = (
    modal.Image.debian_slim(python_version="3.10")
    # torch alone first, from the CUDA 12.4 index; 2.5.1 because torch.load defaults change in 2.6
    .uv_pip_install("torch==2.5.1", extra_index_url="https://download.pytorch.org/whl/cu124")
    .uv_pip_install(
        "numpy<2",
        "monai==1.4.0",
        "nibabel",
        "SimpleITK",
        "transformers==4.25.1",
        "huggingface_hub==0.16.4",
        "pandas",
        "tqdm",
        "scipy",
        "requests",
    )
    .env({"RADAR_VENDOR_DIR": f"{VENDOR_REMOTE}/RADAR_inference", "PYTHONUNBUFFERED": "1"})
    # local files last: with copy=False they are attached at container start, after the build steps
    .add_local_dir(HERE / "vendor" / "damo-radar", VENDOR_REMOTE)
    .add_local_file(HERE / "weights.json", MANIFEST_REMOTE)
    .add_local_python_source("radar_worker")
)

weights = modal.Volume.from_name(WEIGHTS_VOLUME)
smoke_volume = modal.Volume.from_name(SMOKE_VOLUME, create_if_missing=True)
data_volume = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)

app = modal.App(APP_NAME)

FUNCTION_OPTIONS = {
    "image": image,
    "gpu": RADAR_GPU,
    "timeout": 1800,
    "retries": modal.Retries(max_retries=1, initial_delay=5.0),
    "max_containers": 1,
    "scaledown_window": 120,
    "cpu": 4.0,
    "memory": 16384,
}


# ---------------------------------------------------------------- weights


_WEIGHTS_STATE: dict | None = None


def weights_state(log) -> dict:
    """Verify the Volume once per container and cache the answer."""
    global _WEIGHTS_STATE
    if _WEIGHTS_STATE is None:
        manifest = json.loads(Path(MANIFEST_REMOTE).read_text())
        t = time.perf_counter()
        _WEIGHTS_STATE = check_weights(WEIGHTS_DIR, manifest)
        log(f"weights check {'ok' if _WEIGHTS_STATE['ok'] else 'FAILED'} in {time.perf_counter() - t:.1f}s"
            + ("" if _WEIGHTS_STATE["ok"] else ": " + "; ".join(_WEIGHTS_STATE["problems"])))
    return _WEIGHTS_STATE


def versions(checkpoint_sha256: str | None) -> dict:
    import torch

    commit = None
    vendored = Path(VENDOR_REMOTE) / "VENDORED.md"
    if vendored.is_file():
        m = re.search(r"\b([0-9a-f]{40})\b", vendored.read_text())
        commit = m.group(1) if m else None
    return {
        "code_commit": commit,
        "checkpoint_sha256": checkpoint_sha256,
        "torch": str(torch.__version__),  # TorchVersion is not a plain str and cannot be unpickled without torch
        "cuda": str(torch.version.cuda) if torch.version.cuda else None,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "image_id": os.environ.get("MODAL_IMAGE_ID"),
    }


# ---------------------------------------------------------------- one job


class JobLog:
    """Collects log lines for worker.log and prints them for Modal's own log."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.lines: list[str] = []

    def __call__(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime())}Z {self.job_id} {msg}"
        self.lines.append(line)
        print(line, flush=True)

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def _plain(result: dict) -> dict:
    """Round-trip through JSON so the result holds only plain types.

    Modal pickles return values; the API process has no torch or monai, so anything
    else would fail there instead of here.
    """
    return json.loads(json.dumps(result, ensure_ascii=False))


def _error(job_id: str, klass: str, message: str) -> dict:
    return {"ok": False, "job_id": job_id, "error": {"class": klass, "message": message}}


def scores_csv_text(loaded, file_name: str, findings: list) -> str:
    """One row in upstream's CSV format (header `key (English)`, blank for unscored, utf-8-sig on write)."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(loaded.csv_header())
    by_key = {f["key"]: f["prob"] for f in findings}
    writer.writerow([file_name] + ["" if by_key[k] is None else repr(by_key[k]) for k in loaded.test_items])
    return buf.getvalue()


def nifti_name(path: Path) -> str:
    """The .nii or .nii.gz name the file's bytes call for (its own name when that already fits)."""
    with open(path, "rb") as fh:
        gz = fh.read(2) == b"\x1f\x8b"
    want = ".nii.gz" if gz else ".nii"
    name = path.name.lower()
    if name.endswith(want) and not (want == ".nii" and name.endswith(".nii.gz")):
        return path.name
    stem = re.sub(r"(\.nii(\.gz)?|\.gz)$", "", path.name, flags=re.IGNORECASE)
    return stem + want


def ensure_nifti_name(path: Path) -> Path:
    """Give the file a .nii or .nii.gz name so MONAI and nibabel pick the NIfTI reader."""
    name = nifti_name(path)
    if name == path.name:
        return path
    new = path.with_name(name)
    path.rename(new)
    return new


def run_job(job_id: str, fetch_source, publish, artefact_keys: dict | None = None) -> dict:
    """Shared body of `score` and `smoke_score`.

    fetch_source(workdir, log) -> local path of the scan; raises radar_worker.io.TransferError.
    publish(name, path, content_type) stores one artefact; raises TransferError.
    """
    from radar_worker import geometry, infer
    from radar_worker.io import TransferError

    t_start = time.perf_counter()
    log = JobLog(job_id)
    timings = {}

    t = time.perf_counter()
    state = weights_state(log)
    if not state["ok"]:
        return _error(job_id, "weights_mismatch", "; ".join(state["problems"]))
    loaded = infer.load_model(WEIGHTS_DIR, device="cuda")
    timings["load_s"] = round(time.perf_counter() - t, 3)

    work = Path(tempfile.mkdtemp(prefix=f"radar-{job_id}-"))
    try:
        t = time.perf_counter()
        try:
            src = ensure_nifti_name(Path(fetch_source(work, log)))
        except TransferError as err:
            # a refused or expired URL is the caller's problem; network and 5xx raise so Modal retries once
            if 400 <= err.status < 500:
                return _error(job_id, "input_error", f"download failed: {err}")
            raise
        timings["download_s"] = round(time.perf_counter() - t, 3)
        log(f"source {src.name}, {src.stat().st_size} bytes")

        out = infer.score_file(str(src), loaded, log)
        if not out.get("ok"):
            err = out.get("error", {})
            log(f"input error: {err.get('message')}")
            return _error(job_id, err.get("class", "input_error"), err.get("message", ""))
        timings["infer_s"] = out["timings"]["infer_s"]

        t = time.perf_counter()
        vers = versions(state["checkpoint_sha256"])
        files = {
            "mask": geometry.write_mask(out["mask"], out["affine"], work / ARTEFACTS["mask"][0]),
            "scores_json": work / ARTEFACTS["scores_json"][0],
            "scores_csv": work / ARTEFACTS["scores_csv"][0],
            "trace": work / ARTEFACTS["trace"][0],
            "log": work / ARTEFACTS["log"][0],
        }
        files["scores_json"].write_text(json.dumps(
            {"job_id": job_id, "file_name": out["file_name"], "findings": out["findings"]},
            ensure_ascii=False, indent=1), encoding="utf-8")
        files["scores_csv"].write_text(scores_csv_text(loaded, out["file_name"], out["findings"]),
                                       encoding="utf-8-sig")
        trace = dict(out["trace"], job_id=job_id, versions=vers)
        files["trace"].write_text(json.dumps(trace, ensure_ascii=False), encoding="utf-8")
        timings["postprocess_s"] = round(out["timings"]["postprocess_s"] + time.perf_counter() - t, 3)

        t = time.perf_counter()
        try:
            for name in ("mask", "scores_json", "scores_csv", "trace"):
                publish(name, files[name], ARTEFACTS[name][1])
            timings["upload_s"] = round(time.perf_counter() - t, 3)
            timings["total_s"] = round(time.perf_counter() - t_start, 3)
            log(f"timings {timings}")
            files["log"].write_text(log.text(), encoding="utf-8")
            publish("log", files["log"], ARTEFACTS["log"][1])
        except TransferError as err:
            if err.status == 409:
                # a retry after a crash mid-write, or a duplicate spawn; the scan is not at fault
                where = err.url.split("?", 1)[0]
                return _error(job_id, "artefact_exists", f"artefact already exists, not overwritten: {where}")
            if 400 <= err.status < 500:
                return _error(job_id, "input_error", f"artefact upload refused: {err}")
            raise
        timings["upload_s"] = round(time.perf_counter() - t, 3)
        timings["total_s"] = round(time.perf_counter() - t_start, 3)

        keys = artefact_keys or {}
        return {
            "ok": True,
            "job_id": job_id,
            "findings": out["findings"],
            "organs_scored": out["organs_scored"],
            "organs_not_found": out["organs_not_found"],
            "organ_stats": out["organ_stats"],
            "timings": timings,
            "versions": vers,
            "artefacts": {name: keys.get(name, name) for name in ARTEFACTS},
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _source_name(url: str) -> str:
    path = url.split("?", 1)[0].rstrip("/")
    name = path.rsplit("/", 1)[-1] if "/" in path else ""
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name or "source.nii.gz"


# ---------------------------------------------------------------- data Volume


def ref_kind(source_ref: str, artefact_refs: dict):
    """"volume" when every reference is volume://, "url" when none is, None when they are mixed."""
    on_volume = [r.startswith(VOLUME_SCHEME) for r in [source_ref, *artefact_refs.values()]]
    if all(on_volume):
        return "volume"
    return "url" if not any(on_volume) else None


def volume_path(ref: str, root) -> Path:
    """Where volume://<key> sits under the mount; the key rules match the API's validate_key."""
    if not ref.startswith(VOLUME_SCHEME):
        raise ValueError(f"not a volume reference: {ref!r}")
    key = ref[len(VOLUME_SCHEME):]
    if (not key or key.startswith("/") or "\\" in key or any(ord(c) < 32 or ord(c) == 127 for c in key)
            or any(part in ("", ".", "..") for part in key.split("/"))):
        raise ValueError(f"bad volume key: {key!r}")
    return Path(root) / key


def volume_io(source_ref: str, artefact_refs: dict, root, commit):
    """fetch and publish for run_job when the scan and the artefacts live on the data Volume.

    The caller reloads the Volume before anything on it is opened (a reload fails while files
    are open, ref_Volume.md lines 23-24 and 371). Keys are written once; an existing artefact is
    refused, never overwritten (concurrent writes to one file are last write wins, lines 16-18).
    `commit` runs once after the fifth artefact and before the call returns (lines 355-358).
    Background commits and the shutdown commit can persist some artefacts earlier (guide_volumes.md
    lines 258-264), so this is no all-or-nothing write. The guarantee is that the poller reads the
    artefacts only after FunctionCall.get returns, and by then commit() has run.
    """
    from radar_worker.io import TransferError

    written = []

    def fetch(work: Path, log) -> Path:
        src = volume_path(source_ref, root)
        if not src.is_file():
            raise TransferError(404, "no such object on the data Volume", source_ref)
        name = nifti_name(src)
        if name == src.name:
            return src  # read in place; run_job's ensure_nifti_name then leaves it alone
        dest = Path(work) / name  # never rename on the Volume
        shutil.copyfile(src, dest)
        return dest

    def publish(name: str, path: Path, content_type: str) -> None:
        ref = artefact_refs[name]
        dest = volume_path(ref, root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "rb") as fin, open(dest, "xb") as fout:
                shutil.copyfileobj(fin, fout, 1 << 20)
        except FileExistsError:
            raise TransferError(409, "the object already exists; objects are never overwritten", ref) from None
        written.append(name)
        if len(written) == len(ARTEFACTS):
            commit()

    return fetch, publish


# ---------------------------------------------------------------- functions


@app.function(
    volumes={WEIGHTS_DIR: weights.with_mount_options(read_only=True), DATA_DIR: data_volume},
    **FUNCTION_OPTIONS,
)
def score(job_id: str, source_url: str, artefact_urls: dict, artefact_keys: dict | None = None) -> dict:
    """Score one scan. Input errors come back as {ok: false}; anything else raises so Modal retries once.

    source_url and artefact_urls are presigned URLs, or all volume://<key> for the data Volume.
    """
    from radar_worker import io as wio

    missing = [n for n in ARTEFACTS if n not in artefact_urls]
    if missing:
        return _error(job_id, "input_error", f"missing artefact URLs: {missing}")
    kind = ref_kind(source_url, artefact_urls)
    if kind is None:
        return _error(job_id, "input_error", "mixed volume:// and URL references are not supported")
    if kind == "volume":
        try:
            for ref in [source_url, *artefact_urls.values()]:
                volume_path(ref, DATA_DIR)
        except ValueError as err:
            return _error(job_id, "input_error", str(err))
        data_volume.reload()  # first, before any file on /data is opened
        fetch, publish = volume_io(source_url, artefact_urls, DATA_DIR, data_volume.commit)
        return _plain(run_job(job_id, fetch, publish, artefact_keys))

    def fetch(work: Path, log) -> Path:
        dest = work / _source_name(source_url)
        wio.download(source_url, dest)
        return dest

    def publish(name: str, path: Path, content_type: str) -> None:
        wio.upload(path, artefact_urls[name], content_type)

    return _plain(run_job(job_id, fetch, publish, artefact_keys))


@app.function(
    volumes={WEIGHTS_DIR: weights.with_mount_options(read_only=True), SMOKE_DIR: smoke_volume,
             DATA_DIR: data_volume},
    **dict(FUNCTION_OPTIONS, retries=0),
)
def smoke_score(job_id: str, volume_path: str) -> dict:
    """Same as `score`, reading the scan from and writing artefacts to the scratch Volume."""
    smoke_volume.reload()
    out_dir = Path(SMOKE_DIR) / "out" / job_id

    def fetch(work: Path, log) -> Path:
        src = Path(SMOKE_DIR) / volume_path.lstrip("/")
        dest = work / src.name
        shutil.copyfile(src, dest)
        return dest

    def publish(name: str, path: Path, content_type: str) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, out_dir / Path(path).name)

    result = run_job(job_id, fetch, publish, {n: f"out/{job_id}/{f}" for n, (f, _) in ARTEFACTS.items()})
    smoke_volume.commit()
    return _plain(result)


@app.local_entrypoint()
def smoke(path: str, out_dir: str = "smoke-out"):
    """modal run worker/modal_app.py --path scan.nii.gz [--out-dir smoke-out]"""
    src = Path(path).expanduser().resolve()
    if not src.is_file():
        raise SystemExit(f"no such file: {src}")
    job_id = "smoke-" + time.strftime("%Y%m%d-%H%M%S")
    remote = f"in/{job_id}/{src.name}"
    with smoke_volume.batch_upload(force=True) as batch:
        batch.put_file(str(src), "/" + remote)
    print(f"staged {src.name} on Volume {SMOKE_VOLUME} as /{remote}; scoring on {RADAR_GPU}")
    result = smoke_score.remote(job_id, remote)
    dest = Path(out_dir) / job_id
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    if result.get("ok"):
        for fname, _ in ARTEFACTS.values():
            with open(dest / fname, "wb") as fh:
                fh.writelines(smoke_volume.read_file(f"out/{job_id}/{fname}"))
    print(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"saved to {dest}")
