"""In-process stand-in for the GPU, for tests and for dev without Modal.

Results from here carry `versions.gpu = "fake"`, so nothing fake can pass for a real score.
"""

from __future__ import annotations

import gzip
import json
import random
import time
from collections.abc import Callable
from typing import Any

import nibabel as nib
import numpy as np

from radar_desk.gpu.backend import Errored, Finished, Pending, PollOutcome, artefact_keys
from radar_desk.records import Job, Scan

ResultValue = dict | BaseException


def canned_result(job_id: str, seed: int = 0, gpu: str = "fake") -> dict:
    """A valid ok result with deterministic probabilities and no artefacts written."""
    from radar_desk.radar import catalog

    rng = random.Random(seed)
    findings = [
        {"key": f["key"], "organ": f["organ"], "finding": f["finding"], "prob": round(rng.random() ** 3, 6)}
        for f in catalog.FINDINGS
    ]
    return {
        "ok": True,
        "job_id": job_id,
        "findings": findings,
        "organs_scored": [],
        "organs_not_found": [],
        "organ_stats": {},
        "timings": {"download_s": 1.0, "load_s": 20.0, "infer_s": 30.0, "postprocess_s": 3.0,
                    "upload_s": 1.0, "total_s": 55.0},
        "versions": {"code_commit": catalog.SOURCE_COMMIT, "gpu": gpu},
        "artefacts": artefact_keys(job_id),
    }


class FakeGpuBackend:
    name = "fake"

    def __init__(
        self,
        results: dict[str, ResultValue] | None = None,
        delay_ticks: int = 0,
        clock: Callable[[], float] | None = None,
        synthesize: Callable[[str], dict] | None = None,
    ) -> None:
        self.results: dict[str, ResultValue] = dict(results or {})
        self.delay_ticks = delay_ticks
        self.clock = clock or time.time
        self.synthesize = synthesize
        self.spawn_error: BaseException | None = None
        self.calls: dict[str, dict[str, Any]] = {}
        self.cancelled: list[str] = []
        self._n = 0

    def set_result(self, job_id: str, value: ResultValue) -> None:
        self.results[job_id] = value

    def spawn(
        self,
        job: Job,
        source_url: str,
        artefact_urls: dict[str, str],
        artefact_keys: dict[str, str] | None = None,
    ) -> str:
        if self.spawn_error is not None:
            raise self.spawn_error
        self._n += 1
        call_id = f"fc-fake-{self._n}"
        self.calls[call_id] = {
            "job_id": job.id,
            "source_url": source_url,
            "artefact_urls": dict(artefact_urls),
            "artefact_keys": dict(artefact_keys or {}),
            "spawned_at": self.clock(),
            "polls": 0,
        }
        return call_id

    def poll(self, call_id: str) -> PollOutcome:
        call = self.calls.get(call_id)
        if call is None:
            return Errored("NotFoundError", f"no call {call_id}")
        if call_id in self.cancelled:
            return Errored("cancelled", "the call was cancelled")
        call["polls"] += 1
        if call["polls"] <= self.delay_ticks:
            return Pending()
        if "outcome" not in call:
            job_id = call["job_id"]
            if job_id in self.results:
                call["outcome"] = self.results[job_id]
            elif self.synthesize is not None:
                try:
                    call["outcome"] = self.synthesize(job_id)
                except Exception as exc:  # noqa: BLE001 - surfaced as Errored
                    call["outcome"] = exc
            else:
                call["outcome"] = canned_result(job_id)
        outcome = call["outcome"]
        if isinstance(outcome, BaseException):
            return Errored(type(outcome).__name__, str(outcome))
        return Finished(outcome)

    def cancel(self, call_id: str) -> None:
        self.cancelled.append(call_id)

    def logs(self, call_id: str, lines: int = 200) -> str:
        call = self.calls.get(call_id, {})
        return f"fake backend: call {call_id} for job {call.get('job_id')}, polled {call.get('polls', 0)} times\n"


def _world_box(affine: np.ndarray, lo: list[int], hi: list[int]) -> list[list[float]]:
    corners = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    world = (affine @ corners.T).T[:, :3]
    return [world.min(axis=0).round(3).tolist(), world.max(axis=0).round(3).tolist()]


def synthesize_result(job: Job, scan: Scan, storage: Any, catalog: Any) -> dict:
    """A deterministic dev result from the scan header: ellipsoid blobs per scored organ, scores from the
    sha256, and the five artefacts written through storage."""
    from radar_desk.services.exports import csv_bytes, csv_row

    if scan.header is None:
        raise ValueError(f"scan {scan.id} has no header")
    dims = [int(d) for d in scan.header.dims]
    affine = np.asarray(scan.header.affine, dtype=float)
    seed = int((scan.sha256 or scan.id.encode().hex())[:16], 16)
    rng = random.Random(seed)

    mask = np.zeros(dims, dtype=np.uint8)
    voxel_ml = float(np.prod(scan.header.spacing_mm)) / 1000.0
    organs = catalog.SCORED_ORGANS
    organs_scored, organ_stats = [], {}
    for i, organ in enumerate(organs):
        frac = [(0.25 + 0.5 * ((i * 7 + k * 5) % 18) / 17) for k in range(3)]
        centre = [f * (d - 1) for f, d in zip(frac, dims, strict=True)]
        radii = [max(1.0, d * 0.06) for d in dims]
        lo = [max(0, int(np.floor(c - r))) for c, r in zip(centre, radii, strict=True)]
        hi = [min(d - 1, int(np.ceil(c + r))) for c, r, d in zip(centre, radii, dims, strict=True)]
        grids = np.ogrid[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]
        inside = sum(((g - c) / r) ** 2 for g, c, r in zip(grids, centre, radii, strict=True)) <= 1.0
        region = mask[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]
        region[inside & (region == 0)] = organ["label"]
        idx = np.argwhere(region == organ["label"]) + np.array(lo)
        if len(idx) == 0:
            continue
        vlo, vhi = idx.min(axis=0).tolist(), idx.max(axis=0).tolist()
        centroid = (affine @ np.append(idx.mean(axis=0), 1.0))[:3].round(3).tolist()
        box = _world_box(affine, vlo, vhi)
        organ_stats[organ["organ"]] = {"voxels": len(idx), "ml": round(len(idx) * voxel_ml, 3),
                                       "centroid_mm": centroid, "bbox_mm": box}
        organs_scored.append({"organ": organ["organ"], "label": organ["label"], "how": "window",
                              "window_index": i % 4, "box_mm": box})

    findings = []
    for f in catalog.FINDINGS:
        prob = None if f["organ"] not in organ_stats else round(rng.random() ** 3, 6)
        findings.append({"key": f["key"], "organ": f["organ"], "finding": f["finding"], "prob": prob})
    not_found = [o["organ"] for o in organs if o["organ"] not in organ_stats]
    timings = {"download_s": 0.5, "load_s": 2.0, "infer_s": 4.0, "postprocess_s": 1.0, "upload_s": 0.5,
               "total_s": 8.0}
    versions = {"code_commit": catalog.SOURCE_COMMIT, "checkpoint_sha256": None, "torch": None,
                "cuda": None, "gpu": "fake", "image_id": None}
    keys = artefact_keys(job.id)

    img = nib.Nifti1Image(mask, affine)
    img.set_sform(affine, code=1)
    img.set_qform(affine, code=1)
    storage.put_bytes(keys["mask"], gzip.compress(img.to_bytes(), compresslevel=1), "application/gzip")
    scores = {"job_id": job.id, "findings": findings, "organs_not_found": not_found,
              "organs_scored": organs_scored, "organ_stats": organ_stats, "versions": versions,
              "timings": timings}
    storage.put_bytes(keys["scores_json"], json.dumps(scores, ensure_ascii=False).encode(), "application/json")
    storage.put_bytes(keys["scores_csv"], csv_bytes([csv_row(scan.filename, [x["prob"] for x in findings])]),
                      "text/csv")
    trace = {"job_id": job.id, "fake": True, "dims": dims, "organs_scored": organs_scored}
    storage.put_bytes(keys["trace"], json.dumps(trace).encode(), "application/json")
    storage.put_bytes(keys["log"], f"fake scoring of {scan.filename} for {job.id}\n".encode(), "text/plain")

    return {"ok": True, "job_id": job.id, "findings": findings, "organs_scored": organs_scored,
            "organs_not_found": not_found, "organ_stats": organ_stats, "timings": timings,
            "versions": versions, "artefacts": keys}
