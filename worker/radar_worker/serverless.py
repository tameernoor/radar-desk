"""The RunPod Serverless handler: one queued job in, one result dict out.

    python3 -u -m radar_worker.serverless

The endpoint starts the image's entrypoint with this module as its command, so the weights are
checked before the RunPod SDK connects. RunPod passes `{"id", "input"}`; the input is what the
app's serverless backend sent: `job_id`, `source`, the five `artefacts`, optional
`artefact_keys`, `result` (where result.json goes) and optional `expected_size`. References are
all `volume://<key>` on the network volume mounted at RADAR_DATA_ROOT, or all presigned URLs.

On the volume the handler is idempotent per job id. A stored result.json is returned without
scoring (a RunPod retry or a duplicate spawn); otherwise whatever an earlier attempt left is
removed, the scan is awaited until the volume shows it at its expected size (an upload through
RunPod's S3 API can take a moment to appear on the mount), the five artefacts are written once
and result.json is written last. This is safe only because the endpoint runs at most one worker,
so two attempts of one job never overlap.

Input errors come back as {ok: false, error}; anything else raises, so RunPod marks the job
FAILED and may retry it. Every answer goes back as {"result": <result>}, because the RunPod SDK
marks a returned dict with a truthy top-level `error` as FAILED and keeps only str(error); the
result.json the handler stores is the unwrapped result. result.json is written to a .tmp file
and renamed into place, and a stored one that does not parse is removed and the job scored again. Env, read at call time: RADAR_DATA_ROOT (default /runpod-volume),
RADAR_SOURCE_WAIT_S (default 60), RADAR_WEIGHTS_RESOLVED else RADAR_WEIGHTS_DIR (default
/runpod-volume/radar-weights), RADAR_DEVICE (auto), RADAR_VENDOR_DIR, RADAR_IMAGE (reported as
versions.image_id).

Standard library and radar_worker only at module level; `runpod` is imported by `main` and torch
by the scorer when it loads, so the handler is tested without either.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
from pathlib import Path

from radar_worker import io as wio
from radar_worker import job as jobmod
from radar_worker.pull import DEFAULT_VENDOR_DIR, MANIFEST, RealScorer, source_name
from radar_worker.volume import ref_kind, volume_io, volume_path

DEFAULT_DATA_ROOT = "/runpod-volume"
DEFAULT_WEIGHTS_DIR = "/runpod-volume/radar-weights"
DEFAULT_SOURCE_WAIT_S = 60.0
SOURCE_POLL_S = 2.0

_SCORER: RealScorer | None = None


def log(job_id: str, msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime())}Z {job_id} {msg}", flush=True)


def _scorer() -> RealScorer:
    """One scorer per process, so the weights check and the model load happen on a worker's first job."""
    global _SCORER
    if _SCORER is None:
        weights_dir = (os.environ.get("RADAR_WEIGHTS_RESOLVED") or os.environ.get("RADAR_WEIGHTS_DIR")
                       or DEFAULT_WEIGHTS_DIR)
        _SCORER = RealScorer(weights_dir, os.environ.get("RADAR_DEVICE", "auto").strip() or "auto", MANIFEST,
                             os.environ.get("RADAR_VENDOR_DIR") or DEFAULT_VENDOR_DIR)
    return _SCORER


def _wait_for_source(src: Path, expected_size, wait_s: float, clock, sleep) -> bool:
    """True once `src` is a file of `expected_size` bytes (any size when None), False after `wait_s`."""
    deadline = clock() + wait_s
    while True:
        if src.is_file() and (expected_size is None or src.stat().st_size == expected_size):
            return True
        if clock() >= deadline:
            return False
        sleep(SOURCE_POLL_S)


def handler(event: dict, **kw) -> dict:
    """Score the job in `event["input"]` and wrap the answer; see the module docstring."""
    return {"result": _handle(event, **kw)}


def _stored(job_id: str, path: Path) -> dict | None:
    """A result.json an earlier attempt finished, or None. One that does not parse is removed."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as err:
        log(job_id, f"stored result.json does not parse, scoring again: {err}")
        path.unlink(missing_ok=True)
        return None


def _handle(event: dict, *, scorer=None, root=None, wait_s=None, clock=time.monotonic, sleep=time.sleep) -> dict:
    data = (event or {}).get("input") or {}
    job_id = data.get("job_id")
    if not job_id:
        return jobmod.error_result("?", "input_error", "missing job_id")
    source = data.get("source")
    artefacts = data.get("artefacts") or {}
    result_ref = data.get("result")
    artefact_keys = data.get("artefact_keys")
    expected_size = data.get("expected_size")

    missing = [n for n in jobmod.ARTEFACTS if n not in artefacts]
    if missing:
        return jobmod.error_result(job_id, "input_error", f"missing artefact references: {missing}")
    if not result_ref:
        return jobmod.error_result(job_id, "input_error", "missing result reference")
    if not source:
        return jobmod.error_result(job_id, "input_error", "missing source reference")
    kind = ref_kind(source, {**artefacts, "result": result_ref})
    if kind is None:
        return jobmod.error_result(job_id, "input_error", "mixed volume:// and URL references are not supported")

    scorer = scorer if scorer is not None else _scorer()

    if kind == "volume":
        root = root if root is not None else os.environ.get("RADAR_DATA_ROOT") or DEFAULT_DATA_ROOT
        wait_s = wait_s if wait_s is not None else float(
            os.environ.get("RADAR_SOURCE_WAIT_S") or DEFAULT_SOURCE_WAIT_S)
        try:
            src = volume_path(source, root)
            paths = [volume_path(artefacts[n], root) for n in jobmod.ARTEFACTS]
            result_path = volume_path(result_ref, root)
        except ValueError as err:
            return jobmod.error_result(job_id, "input_error", str(err))
        try:
            stored = _stored(job_id, result_path)
            if stored is not None:
                log(job_id, f"result already stored at {result_ref}; returning it")
                return stored
            for path in [*paths, result_path]:
                path.unlink(missing_ok=True)  # an earlier attempt that crashed mid-write
            if not _wait_for_source(src, expected_size, wait_s, clock, sleep):
                return jobmod.error_result(job_id, "input_error",
                                           f"source not visible on the volume after {wait_s:g} s: {source}")
            fetch, publish = volume_io(source, artefacts, root, lambda: None)
            result = jobmod.plain(jobmod.run_job(job_id, fetch, publish, artefact_keys, scorer))
            result_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = result_path.with_name(result_path.name + ".tmp")
            tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, result_path)  # a worker killed mid-write leaves no truncated result.json
            return result
        except PermissionError as err:
            # the worker runs as radar unless RADAR_RUN_AS_ROOT=1, which the endpoint script sets for this storage
            return jobmod.error_result(job_id, "permission_error",
                                       f"{err}; the worker cannot write to the volume, run "
                                       "scripts/runpod_endpoint.py update so the endpoint gets RADAR_RUN_AS_ROOT=1")

    def fetch(work: Path, _log) -> Path:
        dest = Path(work) / source_name(urllib.parse.urlsplit(source).path)
        wio.download(source, dest)
        return dest

    def publish(name: str, path: Path, content_type: str) -> None:
        wio.upload(path, artefacts[name], content_type)

    result = jobmod.plain(jobmod.run_job(job_id, fetch, publish, artefact_keys, scorer))
    try:
        wio.upload_bytes(json.dumps(result).encode(), result_ref, "application/json")
    except wio.TransferError as err:
        log(job_id, f"result.json upload failed, the result is returned anyway: {err}")
    return result


def main() -> None:
    import runpod

    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
