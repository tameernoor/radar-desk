"""The Readers lens: the reading radiologist's outputs for a scan, and (Task 8) the rows of the Readers tab.

Outputs live under readers/{scan_id}/ as JSON. Each records its inputs and their hash and is redone only when an
input changes. A failed call is stored with the same hash, so it is not retried until an input changes or someone
presses retry (force). Calls run in a background thread per scan; tests pass a runner that runs them inline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Callable
from typing import Any

from radar_desk.readers.labels import LABELS, LABELS_SHA256
from radar_desk.readers.prompts import READ_PROMPT_SHA
from radar_desk.readers.reading import read_text
from radar_desk.records import now_iso
from radar_desk.services.blobs import read_json, write_json

log = logging.getLogger(__name__)

READS = ("report_read", "nv_read")


def readers_key(scan_id: str, name: str) -> str:
    return f"readers/{scan_id}/{name}.json"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def inputs_hash(inputs: dict) -> str:
    return sha(json.dumps(inputs, sort_keys=True, ensure_ascii=False))


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
        self._lock = threading.Lock()

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
        return out

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
        plan = self.plan(scan_id)
        can_call = self.llm is not None
        due = any((can_call and s["status"] == "stale") or (can_call and force and s["status"] == "failed")
                  or (s["status"] == "none" and s["stored"] is not None) for s in plan.values())
        if not due:
            return False
        with self._lock:
            if scan_id in self._running:
                return True
            self._running.add(scan_id)

        def work() -> None:
            try:
                self.refresh(scan_id, force=force)
            except Exception:
                log.exception("readers: refresh of scan %s failed", scan_id)
            finally:
                with self._lock:
                    self._running.discard(scan_id)

        self.runner(work)
        return True

    def delete(self, scan_id: str) -> None:
        """Remove every stored output of the scan (they quote the patient report). queries.json is Task 11's."""
        for name in (*READS, "queries"):
            self.storage.delete(readers_key(scan_id, name))

    def refresh(self, scan_id: str, force: bool = False) -> None:
        """Bring each output up to date with its inputs, in order, synchronously."""
        plan = self.plan(scan_id)
        for name in READS:
            self._run_stage(scan_id, plan[name], force,
                            lambda stage: {"read": read_text(self.llm, stage["text"], LABELS)})

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
            write_json(self.storage, key, {**base, "state": "ready", **produce(stage)})
        except Exception as exc:  # noqa: BLE001 - recorded; the tab offers a retry
            log.warning("readers: %s for scan %s failed: %s", stage["name"], scan_id, exc)
            write_json(self.storage, key, {**base, "state": "failed",
                                           "error": (str(exc) or type(exc).__name__)[:300]})
