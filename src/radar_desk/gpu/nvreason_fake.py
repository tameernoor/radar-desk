"""In-process stand-in for NV-Reason-CT, for tests and for the fake compute mode.

Like gpu/fake.py for RADAR, everything it returns says fake: versions.gpu is "fake", the report's first line names
the fake backend and nv_report.fake is true, so nothing here can pass for model output.
"""

from __future__ import annotations

import gzip
from collections.abc import Callable
from typing import Any

import nibabel as nib
import numpy as np

from radar_desk.gpu.backend import Errored, Finished, Pending, PollOutcome
from radar_desk.records import Job, Scan

YES_NO_SUFFIX = "Answer only Yes or No."
FAKE_REPORT_TEXT = (
    "Fake NV-Reason-CT report from the fake backend, not model output.\n\n"
    "FINDINGS:\n"
    "Liver: A small hypodense lesion in the right lobe.\n"
    "Kidneys: A simple cyst in the left kidney. No hydronephrosis.\n"
    "Peritoneum: No ascites.\n"
    "Spleen: Normal in size.\n\n"
    "IMPRESSION:\n"
    "1. Small hypodense liver lesion.\n"
    "2. Left renal cyst.\n"
    "<answer> Liver mass or non-cystic lesion, Renal cyst </answer>"
)


class FakeNvReasonBackend:
    name = "fake"
    priced = False

    def __init__(self, synthesize: Callable[[str], dict] | None = None, delay_ticks: int = 0,
                 answers: dict[str, str] | None = None) -> None:
        self.synthesize = synthesize
        self.delay_ticks = delay_ticks
        self.answers = dict(answers or {})
        self.results: dict[str, dict | BaseException] = {}
        self.spawn_error: BaseException | None = None
        self.ask_error: BaseException | None = None
        self.calls: dict[str, dict[str, Any]] = {}
        self.asks: list[dict[str, Any]] = []
        self.cancelled: list[str] = []
        self._n = 0

    def spawn_report(self, job_id: str, source_ref: str) -> str:
        if self.spawn_error is not None:
            raise self.spawn_error
        self._n += 1
        call_id = f"fc-nvfake-{self._n}"
        self.calls[call_id] = {"job_id": job_id, "source_ref": source_ref, "polls": 0}
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
            try:
                call["outcome"] = self.results[job_id] if job_id in self.results else self.synthesize(job_id)
            except Exception as exc:  # noqa: BLE001 - surfaced as Errored
                call["outcome"] = exc
        outcome = call["outcome"]
        if isinstance(outcome, BaseException):
            return Errored(type(outcome).__name__, str(outcome))
        return Finished(outcome)

    def cancel(self, call_id: str) -> None:
        self.cancelled.append(call_id)

    def logs(self, call_id: str, lines: int = 200) -> str:
        return f"fake NV-Reason-CT: call {call_id}\n"

    def ask(self, source_ref: str, region: str, messages: list[dict], enable_thinking: bool,
            max_new_tokens: int, timeout_s: float) -> dict:
        question = messages[-1]["content"][-1]["text"]
        self.asks.append({"source_ref": source_ref, "region": region, "messages": messages,
                          "enable_thinking": enable_thinking, "max_new_tokens": max_new_tokens,
                          "timeout_s": timeout_s})
        if self.ask_error is not None:
            raise self.ask_error
        answer = self.answers.get(question) or (
            "Yes." if question.endswith(YES_NO_SUFFIX) else f"Fake NV-Reason-CT answer to: {question}")
        return {"thinking": "Fake thinking." if enable_thinking else "", "text": answer, "answer": answer,
                "answer_tag": None, "seconds": 0.1, "new_tokens": 3}


def synthesize_nv_result(job: Job, scan: Scan) -> dict:
    """A fake report result. The crop box is the scan's full x and y extent and the lower two thirds of z, so
    whatever sits in the top third of the scan is outside the fake NV's view."""
    if scan.header is None:
        raise ValueError(f"scan {scan.id} has no header")
    nx, ny, nz = scan.header.dims
    affine = np.asarray(scan.header.affine, dtype=float)
    corners = np.array([[x, y, z, 1.0] for x in (-0.5, nx - 0.5) for y in (-0.5, ny - 0.5)
                        for z in (-0.5, nz - 0.5)]).T
    world = (affine @ corners)[:3]
    lo, hi = world.min(axis=1), world.max(axis=1)
    top = float(hi[2])
    hi[2] = lo[2] + (hi[2] - lo[2]) * 2.0 / 3.0
    crop = {
        "region": "abdomen",
        "box_mm": [[round(float(v), 2) for v in lo], [round(float(v), 2) for v in hi]],
        "crop_offset_vox": [0, 0, 0],
        "scan_length_mm": round(top - float(lo[2]), 1),
        "cut_below_mm": 0.0,
        "cut_above_mm": round(top - float(hi[2]), 1),
        "fallback": False,
    }
    answer, _, tag = FAKE_REPORT_TEXT.partition("<answer>")
    nv_report = {
        "prompt": "Write a structured abdominal CT report.",
        "region": "abdomen",
        "enable_thinking": True,
        "thinking": "Fake thinking.",
        "text": FAKE_REPORT_TEXT,
        "answer": answer.rstrip(),
        "answer_tag": tag.replace("</answer>", "").strip(),
        "seconds": 0.1,
        "new_tokens": 60,
        "fake": True,
    }
    tiny = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4))
    return {
        "ok": True,
        "job_id": job.id,
        "nv_report": nv_report,
        "crop": crop,
        "crop_gz": gzip.compress(tiny.to_bytes()),
        "log": f"fake NV-Reason-CT for {job.id}\n",
        "timings": {"total_s": 0.2},
        "versions": {"gpu": "fake", "model": "nvidia/NV-Reason-CT", "revision": "fake"},
    }
