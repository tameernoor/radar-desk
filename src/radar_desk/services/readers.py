"""The Readers lens: the reading radiologist's outputs for a scan, and (Task 8) the rows of the Readers tab.

Outputs live under readers/{scan_id}/ as JSON. Each records its inputs and their hash and is redone only when an
input changes. A failed call is stored with the same hash, so it is not retried until an input changes or someone
presses retry (force). Calls run in a background thread per scan; tests pass a runner that runs them inline.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import re
import threading
from collections.abc import Callable
from typing import Any

import nibabel as nib
import numpy as np

from radar_desk.radar import catalog
from radar_desk.readers.labels import LABELS, LABELS_SHA256
from radar_desk.readers.prompts import QUERY_PROMPT_SHA, READ_PROMPT_SHA
from radar_desk.readers.reading import norm, read_text, write_queries
from radar_desk.records import now_iso
from radar_desk.services.blobs import read_bytes, read_json, write_json
from radar_desk.services.errors import ServiceError

log = logging.getLogger(__name__)

READS = ("report_read", "nv_read")


def readers_key(scan_id: str, name: str) -> str:
    return f"readers/{scan_id}/{name}.json"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def inputs_hash(inputs: dict) -> str:
    return sha(json.dumps(inputs, sort_keys=True, ensure_ascii=False))


SLAB = 8  # slices of the mask read at a time by organ_coverage
IN_VIEW = 0.5  # NV counts as seeing an organ when at least half its RADAR-mask voxels lie inside the crop box


def organ_coverage(mask_bytes: bytes, box_mm: list[list[float]]) -> dict[str, float]:
    """Share of each scored organ's mask voxels whose centres lie inside the box (RAS mm), from the mask's own
    affine. The same computation as organ_coverage in scripts/nvreason_smoke.py."""
    raw = gzip.decompress(mask_bytes) if mask_bytes[:2] == b"\x1f\x8b" else mask_bytes
    img = nib.Nifti1Image.from_bytes(raw)
    lo, hi = np.asarray(box_mm[0], dtype=float)[:, None], np.asarray(box_mm[1], dtype=float)[:, None]
    total = np.zeros(256, dtype=np.int64)
    kept = np.zeros(256, dtype=np.int64)
    depth = img.shape[2]
    for z0 in range(0, depth, SLAB):  # slabs along the last axis keep the working set small
        slab = np.asarray(img.dataobj[:, :, z0:z0 + SLAB]).astype(np.int16)
        idx = np.nonzero(slab)
        if not idx[0].size:
            continue
        lab = slab[idx]
        vox = np.vstack(idx).astype(np.float64)
        vox[2] += z0
        world = img.affine[:3, :3] @ vox + img.affine[:3, 3:4]
        inside = np.all((world >= lo) & (world <= hi), axis=0)
        total += np.bincount(lab, minlength=256)
        kept += np.bincount(lab[inside], minlength=256)
    return {o["organ"]: round(float(kept[o["label"]] / total[o["label"]]), 3)
            for o in catalog.SCORED_ORGANS if total[o["label"]]}


_FIRST_WORD = re.compile(r"[a-z]+")


def answer_polarity(answer: str) -> str | None:
    """present for an answer starting Yes, absent for one starting No, else None."""
    m = _FIRST_WORD.search(str(answer).lower())
    word = m.group(0) if m else ""
    return {"yes": "present", "no": "absent"}.get(word)


def differs(answer: str, nv_state: str | None) -> bool:
    """NV's direct answer contradicts its own report read: Yes against absent, or No against present."""
    polarity = answer_polarity(answer)
    return polarity is not None and nv_state in ("present", "absent") and polarity != nv_state


def row_cells(row: dict, threshold: float = 0.5) -> dict[str, str]:
    """A row's three states the way web/src/readers.js draws them, at a display line from 0 to 1."""
    radar = row["radar"]
    r = "not_scored" if not radar["scored"] else ("present" if radar["max_prob"] >= threshold else "absent")
    rep = row["report"]["state"] if row["report"] else "none"
    nv = "none" if row["nv"] is None else ("outside" if row["nv"]["in_view"] is False else row["nv"]["state"])
    return {"report": rep, "radar": r, "nv": nv}


def is_disputed(row: dict) -> bool:
    """Flagged at the default display line: one reader says present, another absent."""
    states = set(row_cells(row).values())
    return "present" in states and "absent" in states


def build_rows(labels: list[dict], report_read: dict | None, nv_read: dict | None, result: Any,
               coverage: dict[str, float] | None, asked: list[dict]) -> list[dict]:
    """One row per label: each reader's state and words, RADAR's best score over the row's keys (the client puts it
    against the display line), NV's coverage of the row's organ, and the answers asked for this row."""
    findings = {f.key: f for f in result.findings} if result is not None else {}
    rows = []
    for lab in labels:
        rid, organ = lab["id"], lab["organ"]
        hits = [findings[k] for k in lab["radar_keys"] if k in findings]
        probs = [f.prob for f in hits if f.prob is not None]
        scored = bool(probs)
        nv_cell = (nv_read or {}).get("labels", {}).get(rid) if nv_read else None
        cov = coverage.get(organ) if (coverage is not None and organ) else None
        rows.append({
            "id": rid,
            "title": lab["title"],
            "organ": organ,
            "report": report_read["labels"].get(rid) if report_read else None,
            "radar": {
                "scored": scored,
                "max_prob": max(probs) if scored else None,
                "findings": [{"key": f.key, "organ": f.organ, "finding": f.finding, "prob": f.prob} for f in hits],
            },
            "nv": None if nv_cell is None else {**nv_cell, "coverage": cov,
                                                "in_view": None if cov is None else cov >= IN_VIEW},
            "asked": [{"question": e.get("question"), "answer": e.get("answer"), "seconds": e.get("seconds"),
                       "warm": e.get("warm"), "asked_at": e.get("asked_at"),
                       "differs": differs(e.get("answer") or "", (nv_cell or {}).get("state"))}
                      for e in asked if e.get("label") == rid],
        })
    return rows


def _job_view(job: Any) -> dict | None:
    if job is None:
        return None
    return {"id": job.id, "state": job.state, "hold_reason": job.hold_reason,
            "error": job.error.model_dump(by_alias=True) if job.error else None, "finished_at": job.finished_at}


def _thread_runner(work: Callable[[], None]) -> None:
    threading.Thread(target=work, daemon=True, name="readers").start()


class ReadersService:
    def __init__(self, settings: Any, db: Any, storage: Any, scans: Any, nvreason: Any, llm: Any,
                 runner: Callable[[Callable[[], None]], None] | None = None) -> None:
        self.settings = settings
        self.db = db
        self.storage = storage
        self.scans = scans
        self.nvreason = nvreason
        self.llm = llm  # the chat model client, or None without CHAT_MODEL
        self.runner = runner or _thread_runner
        self._running: set[str] = set()
        self._rerun: dict[str, bool] = {}  # scans to refresh again once the running one ends, with force
        self._deleted: set[str] = set()
        self._lock = threading.Lock()
        self._cov_lock = threading.Lock()
        self._coverage: dict[tuple[str, str], dict[str, float]] = {}

    # What each output reads, and what is stored

    def _sources(self, scan_id: str) -> dict[str, str | None]:
        """The text each read reads: the radiologist report, and NV-Reason-CT's report of its latest done job."""
        nv_text = None
        job = self.nvreason.latest_done(scan_id)
        if job is not None:
            text = (self.nvreason.report(job.id) or {}).get("text")
            nv_text = text if isinstance(text, str) else None
        return {"report_read": self.scans.get_report(scan_id), "nv_read": nv_text}

    def _stage(self, scan_id: str, name: str, inputs: dict | None) -> dict:
        """One output: its inputs and their hash, what is stored, and its status (none, ready, failed or stale)."""
        stored = read_json(self.storage, readers_key(scan_id, name))
        if inputs is None:
            return {"name": name, "inputs": None, "hash": None, "stored": stored, "status": "none"}
        h = inputs_hash(inputs)
        fresh = stored is not None and stored.get("input_hash") == h and stored.get("state") in ("ready", "failed")
        return {"name": name, "inputs": inputs, "hash": h, "stored": stored,
                "status": stored["state"] if fresh else "stale"}

    def plan(self, scan_id: str) -> dict[str, dict]:
        sources = self._sources(scan_id)
        out = {}
        for name in READS:
            text = sources[name]
            inputs = None if text is None else {"text": sha(text), "labels": LABELS_SHA256, "prompt": READ_PROMPT_SHA,
                                                "model": self.settings.chat_model}
            out[name] = {**self._stage(scan_id, name, inputs), "text": text}
        out["queries"] = self.query_stage(scan_id, out)
        return out

    def query_stage(self, scan_id: str, plan: dict) -> dict:
        """The prepared questions as a stage. Their inputs: both reads, RADAR's job, the organs mostly outside NV's
        crop and the disputed rows, so they are written again whenever one of those changes."""
        nv = plan["nv_read"]
        if nv["status"] != "ready":
            return self._stage(scan_id, "queries", None)
        ctx = self.context(scan_id, plan)
        outside = sorted(o for o, share in (ctx["coverage"] or {}).items() if share < IN_VIEW)
        disputed = [r["id"] for r in ctx["rows"] if is_disputed(r)]
        report = plan["report_read"]
        inputs = {
            "nv_read": nv["hash"],
            "report_read": report["hash"] if report["status"] == "ready" else None,
            "radar_job": ctx["radar_job"].id if ctx["radar_job"] else None,
            "outside": outside,
            "disputed": disputed,
            "labels": LABELS_SHA256,
            "prompt": QUERY_PROMPT_SHA,
            "model": self.settings.chat_model,
        }
        stage = self._stage(scan_id, "queries", inputs)
        stage["context"] = {"labels": LABELS, "nv_text": nv["text"], "crop": ctx["crop"], "outside": outside,
                            "disputed": disputed,
                            "nv_findings": [r["id"] for r in ctx["rows"] if r["nv"] and r["nv"]["state"] == "present"]}
        return stage

    @staticmethod
    def queries(plan: dict) -> list[dict]:
        stage = plan["queries"]
        return stage["stored"]["queries"] if stage["status"] == "ready" else []

    @staticmethod
    def read(plan: dict, name: str) -> dict | None:
        """The stored read when it is ready for the current inputs, else None."""
        stage = plan[name]
        return stage["stored"]["read"] if stage["status"] == "ready" else None

    def status(self, scan_id: str, plan: dict | None = None) -> dict[str, dict]:
        """Per output what the Readers tab shows: none, pending, ready, failed (with the error) or no_llm."""
        plan = plan or self.plan(scan_id)
        running = scan_id in self._running
        out = {}
        for name, stage in plan.items():
            s = stage["status"]
            if s == "stale":
                s = "pending" if (self.llm is not None or running) else "no_llm"
            error = (stage["stored"] or {}).get("error") if s == "failed" else None
            out[name] = {"status": s, "error": error}
        return out

    # Running the calls

    def kick(self, scan_id: str, force: bool = False) -> bool:
        """Start a background refresh when an output is stale, has lost its input, or (with force) failed.
        True when a refresh runs."""
        with self._lock:
            if scan_id in self._running:  # the running refresh goes round again, so no change is lost
                self._rerun[scan_id] = self._rerun.get(scan_id, False) or force
                return True
        plan = self.plan(scan_id)
        can_call = self.llm is not None
        due = any((can_call and s["status"] == "stale") or (can_call and force and s["status"] == "failed")
                  or (s["status"] == "none" and s["stored"] is not None) for s in plan.values())
        if not due:
            return False
        with self._lock:
            if scan_id in self._running:
                self._rerun[scan_id] = self._rerun.get(scan_id, False) or force
                return True
            self._running.add(scan_id)

        def work() -> None:
            go, run_force = True, force
            while go:
                try:
                    self.refresh(scan_id, force=run_force)
                except Exception:
                    log.exception("readers: refresh of scan %s failed", scan_id)
                with self._lock:
                    go = scan_id in self._rerun
                    if go:
                        run_force = self._rerun.pop(scan_id)
                    else:
                        self._running.discard(scan_id)

        self.runner(work)
        return True

    def delete(self, scan_id: str) -> None:
        """Remove every stored output of the scan (they quote the patient report) and stop any in-flight refresh from
        writing one back. Scan ids are never reused, so the tombstone stays for the process lifetime."""
        with self._lock:
            self._deleted.add(scan_id)
            for name in (*READS, "queries"):
                self.storage.delete(readers_key(scan_id, name))

    def _store(self, scan_id: str, key: str, value: dict) -> None:
        """Write an output unless the scan was deleted meanwhile."""
        with self._lock:
            if scan_id not in self._deleted:
                write_json(self.storage, key, value)

    def refresh(self, scan_id: str, force: bool = False) -> None:
        """Bring each output up to date with its inputs, in order, synchronously: the reads, then the questions."""
        plan = self.plan(scan_id)
        for name in READS:
            self._run_stage(scan_id, plan[name], force,
                            lambda stage: {"read": read_text(self.llm, stage["text"], LABELS)})
        stage = self.plan(scan_id)["queries"]  # the reads may have changed its inputs
        self._run_stage(scan_id, stage, force, lambda st: {"queries": write_queries(self.llm, st["context"])})

    def _run_stage(self, scan_id: str, stage: dict, force: bool, produce: Callable[[dict], dict]) -> None:
        key = readers_key(scan_id, stage["name"])
        if stage["status"] == "none":
            if stage["stored"] is not None:
                self.storage.delete(key)
            return
        if stage["status"] == "ready" or (stage["status"] == "failed" and not force) or self.llm is None:
            return
        base = {"inputs": stage["inputs"], "input_hash": stage["hash"], "created_at": now_iso()}
        try:
            self._store(scan_id, key, {**base, "state": "ready", **produce(stage)})
        except Exception as exc:  # noqa: BLE001 - recorded; the tab offers a retry
            log.warning("readers: %s for scan %s failed: %s", stage["name"], scan_id, exc)
            self._store(scan_id, key, {**base, "state": "failed",
                                       "error": (str(exc) or type(exc).__name__)[:300]})

    # The rows

    def _radar(self, scan_id: str) -> tuple[Any, Any]:
        """The newest done RADAR job of the scan that has a stored result, and that result."""
        for job in self.db.list_jobs(state="done", scan_id=scan_id, limit=20):
            result = self.db.get_result(job.id)
            if result is not None:
                return job, result
        return None, None

    def _coverage_for(self, result: Any, nv_job: Any, crop: dict | None) -> dict[str, float] | None:
        """NV's coverage per organ, cached per mask and NV job; None without a RADAR mask or an NV crop."""
        if result is None or nv_job is None or not crop or not result.artefacts.mask:
            return None
        key = (result.artefacts.mask, nv_job.id)
        with self._cov_lock:  # one computation per key even when polls arrive together
            if key not in self._coverage:
                data = read_bytes(self.storage, result.artefacts.mask)
                if data is None:
                    return None
                try:
                    self._coverage[key] = organ_coverage(data, crop["box_mm"])
                except Exception as exc:  # noqa: BLE001 - a bad mask must not fail the whole view
                    log.warning("readers: coverage from mask %s failed: %s", key[0], exc)
                    return None
                while len(self._coverage) > 32:
                    self._coverage.pop(next(iter(self._coverage)))
            return self._coverage[key]

    def context(self, scan_id: str, plan: dict) -> dict:
        """Everything the rows are built from, and the rows."""
        radar_job, result = self._radar(scan_id)
        done = self.nvreason.latest_done(scan_id)
        crop = self.nvreason.crop(done.id) if done else None
        coverage = self._coverage_for(result, done, crop)
        asked = self.nvreason.qa_entries(done.id) if done else []
        rows = build_rows(LABELS, self.read(plan, "report_read"), self.read(plan, "nv_read"), result, coverage, asked)
        return {"radar_job": radar_job, "result": result, "nv_done": done, "crop": crop, "coverage": coverage,
                "rows": rows}

    def view(self, scan_id: str) -> dict:
        """GET /scans/{id}/readers. Starts a refresh first when an output is out of date."""
        self.scans.get(scan_id)
        self.kick(scan_id)
        plan = self.plan(scan_id)
        ctx = self.context(scan_id, plan)
        queries = self.queries(plan)
        for row in ctx["rows"]:
            row["queries"] = [q for q in queries if q["label"] == row["id"]]
        status = self.status(scan_id, plan)
        nv_report = self.nvreason.report(ctx["nv_done"].id) if ctx["nv_done"] else None
        report_read, nv_read = self.read(plan, "report_read"), self.read(plan, "nv_read")
        others = [{"reader": "report", "quote": q} for q in (report_read or {}).get("other_mentions", [])]
        others += [{"reader": "nv", "quote": q} for q in (nv_read or {}).get("other_mentions", [])]
        return {
            "scan_id": scan_id,
            "has_report": plan["report_read"]["text"] is not None,
            "report": status["report_read"],
            "nv": {
                "unavailable": self.nvreason.unavailable(),
                "job": _job_view(self.nvreason.latest(scan_id)),
                "read": status["nv_read"],
                "crop": ctx["crop"],
                "fake": bool(nv_report and nv_report.get("fake")),
                "warm": self.nvreason.warm(),
                **self.nvreason.ask_state(scan_id),
            },
            "radar": {"job_id": ctx["radar_job"].id if ctx["radar_job"] else None,
                      "coverage": ctx["coverage"] is not None},
            "rows": ctx["rows"],
            "queries": {**status["queries"], "general": [q for q in queries if q["label"] is None]},
            "other_mentions": others,
        }

    def ask_prepared(self, scan_id: str, query_id: str) -> dict:
        """Check a prepared question can be asked now, then ask it in the background and return at once; the answer
        lands in qa.jsonl and the Readers tab picks it up. A question already in flight is not asked twice."""
        self.scans.get(scan_id)
        q = next((q for q in self.queries(self.plan(scan_id)) if q["id"] == query_id), None)
        if q is None:
            raise ServiceError(404, f"no prepared question {query_id}")
        self.nvreason.precheck(scan_id)
        if not self.nvreason.begin_background(scan_id, query_id):
            return {"asked": query_id}

        def work() -> None:
            failure = None
            try:
                self.nvreason.ask(scan_id, q["text"], label=q["label"])
            except ServiceError as exc:
                failure = exc.detail
            except Exception as exc:
                log.exception("readers: asking %s of scan %s failed", query_id, scan_id)
                failure = str(exc)
            finally:
                self.nvreason.end_background(scan_id, query_id, failure and "could not ask NV-Reason-CT, try again")
                if failure:
                    log.warning("readers: ask %s of scan %s failed: %s", query_id, scan_id, failure)

        self.runner(work)
        return {"asked": query_id}

    def ask(self, scan_id: str, question: str | None = None, query_id: str | None = None) -> dict:
        """Ask NV-Reason-CT a prepared question (by id) or a free one. A free question with the same words as a
        prepared one counts as that one, so its answer shows in the question's row."""
        self.scans.get(scan_id)
        prepared = self.queries(self.plan(scan_id))
        if query_id:
            q = next((q for q in prepared if q["id"] == query_id), None)
            if q is None:
                raise ServiceError(404, f"no prepared question {query_id}")
            return self.nvreason.ask(scan_id, q["text"], label=q["label"])
        if not question or not question.strip():
            raise ServiceError(422, "give a question or a query_id")
        label = next((q["label"] for q in prepared if norm(q["text"]) == norm(question)), None)
        return self.nvreason.ask(scan_id, question, label=label)
