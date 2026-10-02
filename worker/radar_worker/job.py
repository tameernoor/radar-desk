"""One scoring job from source to result dict, shared by the Modal function and the pull worker.

`run_job` loads the model through a scorer, fetches the scan, scores it, writes the five
artefacts, publishes each one and returns the result dict the desk stores. Input errors come
back as {ok: false, error: {class, message}}; network failures and 5xx answers raise so the
caller decides (Modal retries once, the pull worker reports the failure to the desk).

The scorer is duck typed and has three methods.

- `load(log)` returns None when the model is ready, or an error dict {"class", "message"}
  (for example `weights_mismatch`) when the job must fail before scoring. It may raise for
  runtime failures. It is timed as `load_s`.
- `score(path, log)` returns exactly what `infer.score_file` returns, including `mask`,
  `affine`, `file_name`, `timings` and `trace`, or {ok: false, error}.
- `versions()` returns the `versions` block of the result.

Standard library plus numpy and nibabel (through geometry) only; torch stays inside the scorer.
"""

from __future__ import annotations

import csv
import io
import json
import re
import shutil
import tempfile
import time
from pathlib import Path

# name -> (file name, content type); the names match Result.artefacts
ARTEFACTS = {
    "scores_json": ("scores.json", "application/json"),
    "scores_csv": ("scores.csv", "text/csv; charset=utf-8"),
    "mask": ("mask.nii.gz", "application/gzip"),
    "trace": ("trace.json", "application/json"),
    "log": ("worker.log", "text/plain; charset=utf-8"),
}


class JobLog:
    """Collects log lines for worker.log and prints them for the platform's own log."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.lines: list[str] = []

    def __call__(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime())}Z {self.job_id} {msg}"
        self.lines.append(line)
        print(line, flush=True)

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def plain(result: dict) -> dict:
    """Round-trip through JSON so the result holds only plain types.

    Modal pickles return values; the API process has no torch or monai, so anything
    else would fail there instead of here.
    """
    return json.loads(json.dumps(result, ensure_ascii=False))


def error_result(job_id: str, klass: str, message: str) -> dict:
    return {"ok": False, "job_id": job_id, "error": {"class": klass, "message": message}}


def scores_csv_text(file_name: str, findings: list) -> str:
    """One row in upstream's CSV format (header `key (English)`, blank for unscored, utf-8-sig on write)."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["file_name"] + [f"{f['key']} ({f['organ']}_{f['finding']})" for f in findings])
    writer.writerow([file_name] + ["" if f["prob"] is None else repr(f["prob"]) for f in findings])
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


def run_job(job_id: str, fetch_source, publish, artefact_keys: dict | None, scorer) -> dict:
    """Score one scan and publish its artefacts.

    fetch_source(workdir, log) -> local path of the scan; raises radar_worker.io.TransferError.
    publish(name, path, content_type) stores one artefact; raises TransferError.
    """
    from radar_worker import geometry
    from radar_worker.io import TransferError

    t_start = time.perf_counter()
    log = JobLog(job_id)
    timings = {}

    t = time.perf_counter()
    refused = scorer.load(log)
    if refused:
        return error_result(job_id, refused.get("class", "runtime_error"), refused.get("message", ""))
    timings["load_s"] = round(time.perf_counter() - t, 3)

    work = Path(tempfile.mkdtemp(prefix=f"radar-{job_id}-"))
    try:
        t = time.perf_counter()
        try:
            src = ensure_nifti_name(Path(fetch_source(work, log)))
        except TransferError as err:
            # a refused or expired URL is the caller's problem; network and 5xx raise so the caller decides
            if 400 <= err.status < 500:
                return error_result(job_id, "input_error", f"download failed: {err}")
            raise
        timings["download_s"] = round(time.perf_counter() - t, 3)
        log(f"source {src.name}, {src.stat().st_size} bytes")

        out = scorer.score(str(src), log)
        if not out.get("ok"):
            err = out.get("error", {})
            log(f"input error: {err.get('message')}")
            return error_result(job_id, err.get("class", "input_error"), err.get("message", ""))
        timings["infer_s"] = out["timings"]["infer_s"]

        t = time.perf_counter()
        vers = scorer.versions()
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
        files["scores_csv"].write_text(scores_csv_text(out["file_name"], out["findings"]), encoding="utf-8-sig")
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
                return error_result(job_id, "artefact_exists", f"artefact already exists, not overwritten: {where}")
            if 400 <= err.status < 500:
                return error_result(job_id, "input_error", f"artefact upload refused: {err}")
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
