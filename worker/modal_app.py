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

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from radar_worker import job
from radar_worker.job import ARTEFACTS, ensure_nifti_name, nifti_name  # noqa: F401  kept as public names here
from radar_worker.job import error_result as _error
from radar_worker.job import plain as _plain
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


class ModalScorer:
    """The scorer `job.run_job` drives: weights checked once per container, the model on CUDA."""

    def __init__(self):
        self.loaded = None
        self.checkpoint_sha256 = None

    def load(self, log):
        from radar_worker import infer

        state = weights_state(log)
        if not state["ok"]:
            return {"class": "weights_mismatch", "message": "; ".join(state["problems"])}
        self.checkpoint_sha256 = state["checkpoint_sha256"]
        self.loaded = infer.load_model(WEIGHTS_DIR, device="cuda")
        return None

    def score(self, path: str, log) -> dict:
        from radar_worker import infer

        return infer.score_file(path, self.loaded, log)

    def versions(self) -> dict:
        return versions(self.checkpoint_sha256)


def run_job(job_id: str, fetch_source, publish, artefact_keys: dict | None = None) -> dict:
    """Shared body of `score` and `smoke_score` (radar_worker.job.run_job with the Modal scorer)."""
    return job.run_job(job_id, fetch_source, publish, artefact_keys, ModalScorer())


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
