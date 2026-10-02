"""The RunPod Serverless handler with a fake scorer: no torch, no runpod, no RunPod."""

from __future__ import annotations

import gzip
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import nibabel as nib
import numpy as np
import pytest

from radar_worker import job as jobmod
from radar_worker import serverless

SCAN = gzip.compress(b"not a real scan, the fake scorer never reads it")
KEYS = {n: f"jobs/j1/{f}" for n, (f, _) in jobmod.ARTEFACTS.items()}


class FakeScorer:
    """Answers like `infer.score_file`: a small mask, an identity affine, one finding."""

    def __init__(self, error: Exception | None = None):
        self.error = error
        self.scored: list[str] = []

    def load(self, log):
        return None

    def score(self, path: str, log) -> dict:
        self.scored.append(path)
        if self.error is not None:
            raise self.error
        mask = np.zeros((4, 4, 2), dtype=np.uint8)
        mask[1:3, 1:3, 0] = 1
        return {
            "ok": True,
            "file_name": Path(path).name,
            "findings": [{"key": "k1", "organ": "liver", "finding": "lesion", "prob": 0.25}],
            "organs_scored": [],
            "organs_not_found": [],
            "organ_stats": {},
            "trace": {"fake": True},
            "timings": {"infer_s": 0.01, "postprocess_s": 0.01},
            "mask": mask,
            "affine": np.eye(4),
        }

    def versions(self) -> dict:
        return {"torch": "2.5.1", "gpu": "NVIDIA L4"}


class Clock:
    """A fake monotonic clock; `sleep` records the delay, advances the clock and runs `on_sleep`."""

    def __init__(self, on_sleep=None):
        self.now = 0.0
        self.sleeps: list[float] = []
        self.on_sleep = on_sleep

    def __call__(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.now += s
        if self.on_sleep is not None:
            self.on_sleep(len(self.sleeps))


def volume_event(job_id: str = "j1", **over) -> dict:
    data = {
        "job_id": job_id,
        "source": "volume://scans/s1/source.nii.gz",
        "artefacts": {n: f"volume://jobs/{job_id}/{f}" for n, (f, _) in jobmod.ARTEFACTS.items()},
        "artefact_keys": KEYS,
        "result": f"volume://jobs/{job_id}/result.json",
        "expected_size": len(SCAN),
    }
    data.update(over)
    return {"id": "rp-1", "input": data}


def put_scan(root: Path, data: bytes = SCAN) -> Path:
    src = root / "scans" / "s1" / "source.nii.gz"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(data)
    return src


def unwrap(answer: dict) -> dict:
    """The handler's wire answer is {"result": <result>} and nothing else."""
    assert list(answer) == ["result"], answer
    return answer["result"]


def run(event, root, scorer=None, clock=None, wait_s=10.0):
    clock = clock or Clock()
    return unwrap(serverless.handler(event, scorer=scorer or FakeScorer(), root=root, wait_s=wait_s,
                                     clock=clock, sleep=clock.sleep))


# ---------------------------------------------------------------- volume references


def test_volume_end_to_end(tmp_path):
    put_scan(tmp_path)
    out = run(volume_event(), tmp_path)
    assert out["ok"] is True and out["job_id"] == "j1"
    assert out["artefacts"] == KEYS
    jobdir = tmp_path / "jobs" / "j1"
    for fname, _ in jobmod.ARTEFACTS.values():
        assert (jobdir / fname).is_file(), fname
    assert nib.load(jobdir / "mask.nii.gz").shape == (4, 4, 2)
    assert json.loads((jobdir / "scores.json").read_text())["findings"][0]["key"] == "k1"
    assert json.loads((jobdir / "result.json").read_text()) == out  # stored unwrapped
    assert not (jobdir / "result.json.tmp").exists()


def test_result_json_is_renamed_into_place(tmp_path, monkeypatch):
    """A worker killed between the write and the rename leaves only the .tmp file, never a truncated result."""
    put_scan(tmp_path)

    def killed(src, dst):
        raise KeyboardInterrupt

    monkeypatch.setattr(serverless.os, "replace", killed)
    with pytest.raises(KeyboardInterrupt):
        run(volume_event(), tmp_path)
    jobdir = tmp_path / "jobs" / "j1"
    assert not (jobdir / "result.json").exists() and (jobdir / "result.json.tmp").is_file()


def test_repeat_call_returns_the_stored_result_without_scoring(tmp_path):
    put_scan(tmp_path)
    first = run(volume_event(), tmp_path)
    scorer = FakeScorer(error=AssertionError("must not score again"))
    assert run(volume_event(), tmp_path, scorer=scorer) == first
    assert scorer.scored == []


def test_a_stored_result_json_that_does_not_parse_is_scored_again(tmp_path, capsys):
    put_scan(tmp_path)
    jobdir = tmp_path / "jobs" / "j1"
    jobdir.mkdir(parents=True)
    (jobdir / "result.json").write_text('{"ok": true, "job_')
    scorer = FakeScorer()
    out = run(volume_event(), tmp_path, scorer=scorer)
    assert out["ok"] is True and len(scorer.scored) == 1
    assert json.loads((jobdir / "result.json").read_text()) == out
    assert "does not parse" in capsys.readouterr().out


def test_leftovers_of_a_crashed_attempt_are_removed_first(tmp_path):
    put_scan(tmp_path)
    jobdir = tmp_path / "jobs" / "j1"
    jobdir.mkdir(parents=True)
    (jobdir / "mask.nii.gz").write_bytes(b"half a mask")
    (jobdir / "worker.log").write_text("the first attempt\n")
    out = run(volume_event(), tmp_path)
    assert out["ok"] is True
    assert nib.load(jobdir / "mask.nii.gz").shape == (4, 4, 2)
    assert "the first attempt" not in (jobdir / "worker.log").read_text()


def test_missing_source_waits_then_answers_input_error(tmp_path):
    clock = Clock()
    scorer = FakeScorer()
    out = run(volume_event(), tmp_path, scorer=scorer, clock=clock, wait_s=10.0)
    assert out["ok"] is False and out["error"]["class"] == "input_error"
    assert "volume://scans/s1/source.nii.gz" in out["error"]["message"]
    assert clock.now >= 10.0 and set(clock.sleeps) == {serverless.SOURCE_POLL_S}
    assert scorer.scored == []
    assert not (tmp_path / "jobs" / "j1" / "result.json").exists()


def test_source_appearing_mid_wait_is_scored(tmp_path):
    clock = Clock(on_sleep=lambda n: n == 2 and put_scan(tmp_path))
    out = run(volume_event(), tmp_path, clock=clock)
    assert out["ok"] is True and clock.sleeps == [serverless.SOURCE_POLL_S] * 2


def test_size_mismatch_keeps_waiting_until_it_matches(tmp_path):
    put_scan(tmp_path, SCAN[:10])
    clock = Clock(on_sleep=lambda n: n == 3 and put_scan(tmp_path))
    out = run(volume_event(), tmp_path, clock=clock)
    assert out["ok"] is True and len(clock.sleeps) == 3


def test_wait_and_root_come_from_the_env(tmp_path, monkeypatch):
    monkeypatch.setenv("RADAR_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("RADAR_SOURCE_WAIT_S", "4")
    clock = Clock()
    out = unwrap(serverless.handler(volume_event(), scorer=FakeScorer(), clock=clock, sleep=clock.sleep))
    assert out["error"]["class"] == "input_error" and "after 4 s" in out["error"]["message"]
    put_scan(tmp_path)
    again = unwrap(serverless.handler(volume_event(), scorer=FakeScorer(), clock=clock, sleep=clock.sleep))
    assert again["ok"] is True


@pytest.mark.parametrize("change", [
    lambda d: d.pop("job_id"),
    lambda d: d["artefacts"].pop("trace"),
    lambda d: d.pop("result"),
    lambda d: d["artefacts"].update(log="https://b.example/l?sig=1"),
    lambda d: d.update(result="https://b.example/r?sig=1"),
    lambda d: d.update(source="volume://scans/../etc/passwd"),
    lambda d: d["artefacts"].update(mask="volume:///abs/mask.nii.gz"),
], ids=["no-job-id", "no-artefact", "no-result", "mixed-artefact", "mixed-result", "bad-source-key", "bad-key"])
def test_bad_input_answers_input_error(tmp_path, change):
    put_scan(tmp_path)
    event = volume_event()
    change(event["input"])
    scorer = FakeScorer()
    out = run(event, tmp_path, scorer=scorer)
    assert out["ok"] is False and out["error"]["class"] == "input_error", out
    assert out["job_id"] == event["input"].get("job_id", "?")
    assert scorer.scored == []
    assert not (tmp_path / "jobs").exists()


def test_an_input_error_has_no_top_level_error_key(tmp_path):
    """The RunPod SDK would mark a dict with a truthy top-level `error` FAILED and keep only str(error)."""
    answer = serverless.handler({"input": {}}, scorer=FakeScorer(), root=tmp_path)
    assert "error" not in answer
    assert answer["result"]["ok"] is False and answer["result"]["error"]["class"] == "input_error"


def test_scorer_exception_propagates(tmp_path):
    put_scan(tmp_path)
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        run(volume_event(), tmp_path, scorer=FakeScorer(error=RuntimeError("CUDA out of memory")))
    assert not (tmp_path / "jobs" / "j1" / "result.json").exists()


# ---------------------------------------------------------------- URL references


class _Store:
    objects: ClassVar[dict[str, bytes]] = {}
    types: ClassVar[dict[str, str]] = {}
    refuse: ClassVar[set[str]] = set()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _answer(self, status: int, body: bytes = b"") -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        data = _Store.objects.get(self.path.split("?", 1)[0])
        self._answer(200, data) if data is not None else self._answer(404)

    def do_PUT(self):
        path = self.path.split("?", 1)[0]
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if path in _Store.refuse:
            return self._answer(500, b"<Error>InternalError</Error>")
        _Store.objects[path] = body
        _Store.types[path] = self.headers.get("Content-Type", "")
        self._answer(200)


@pytest.fixture()
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    _Store.objects = {"/scans/s1/source.nii.gz": SCAN}
    _Store.types = {}
    _Store.refuse = set()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def url_event(base: str) -> dict:
    return {"id": "rp-2", "input": {
        "job_id": "j1",
        "source": f"{base}/scans/s1/source.nii.gz?sig=get",
        "artefacts": {n: f"{base}/jobs/j1/{f}?sig=put" for n, (f, _) in jobmod.ARTEFACTS.items()},
        "artefact_keys": KEYS,
        "result": f"{base}/jobs/j1/result.json?sig=put",
        "expected_size": len(SCAN),
    }}


def test_url_references_put_the_artefacts_and_the_result(server, tmp_path):
    scorer = FakeScorer()
    out = run(url_event(server), tmp_path, scorer=scorer)
    assert out["ok"] is True and out["artefacts"] == KEYS
    assert Path(scorer.scored[0]).name == "source.nii.gz"
    for fname, ctype in jobmod.ARTEFACTS.values():
        assert f"/jobs/j1/{fname}" in _Store.objects, fname
        assert _Store.types[f"/jobs/j1/{fname}"] == ctype
    assert _Store.types["/jobs/j1/result.json"] == "application/json"
    assert json.loads(_Store.objects["/jobs/j1/result.json"]) == out
    assert not (tmp_path / "jobs").exists()


def test_refused_result_upload_does_not_fail_the_job(server, tmp_path, capsys):
    _Store.refuse = {"/jobs/j1/result.json"}
    out = run(url_event(server), tmp_path)
    assert out["ok"] is True
    assert "/jobs/j1/result.json" not in _Store.objects
    printed = capsys.readouterr().out
    assert "result.json upload failed" in printed and "sig=put" not in printed


# ---------------------------------------------------------------- main


def test_main_starts_the_runpod_sdk_with_the_handler(monkeypatch):
    started = []
    fake = SimpleNamespace(serverless=SimpleNamespace(start=started.append))
    monkeypatch.setitem(sys.modules, "runpod", fake)
    serverless.main()
    assert started == [{"handler": serverless.handler}]


def test_default_scorer_is_built_once_from_the_env(monkeypatch, tmp_path):
    monkeypatch.setattr(serverless, "_SCORER", None)
    monkeypatch.setenv("RADAR_WEIGHTS_RESOLVED", str(tmp_path / "w"))
    monkeypatch.setenv("RADAR_DEVICE", "cpu")
    first = serverless._scorer()
    assert first.weights_dir == str(tmp_path / "w") and first.device == "cpu"
    assert serverless._scorer() is first
