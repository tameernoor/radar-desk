"""SQLite store for scans, jobs, results and chat executions.

One `Database` per process. WAL mode, one connection shared across threads, every statement under a
lock. Each row keeps the queried columns plus the full record as JSON in `payload`, so a new record
field needs no migration. Migrations are numbered SQL scripts applied once, in order.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from radar_desk.records import ChatExecution, Job, JobState, Result, Scan, ScanState, now_iso

MIGRATIONS: list[str] = [
    # 1: initial schema
    """
    CREATE TABLE scans (
        id TEXT PRIMARY KEY,
        state TEXT NOT NULL,
        created_at TEXT NOT NULL,
        payload TEXT NOT NULL
    );
    CREATE INDEX scans_state ON scans(state, created_at);

    CREATE TABLE jobs (
        id TEXT PRIMARY KEY,
        scan_id TEXT NOT NULL,
        state TEXT NOT NULL,
        modal_call_id TEXT,
        created_at TEXT NOT NULL,
        finished_at TEXT,
        payload TEXT NOT NULL
    );
    CREATE INDEX jobs_state ON jobs(state, created_at);
    CREATE INDEX jobs_scan ON jobs(scan_id, created_at);
    CREATE INDEX jobs_finished ON jobs(finished_at);

    CREATE TABLE results (
        job_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        payload TEXT NOT NULL
    );

    CREATE TABLE chat_executions (
        id TEXT PRIMARY KEY,
        scan_id TEXT NOT NULL,
        job_id TEXT,
        state TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        payload TEXT NOT NULL
    );
    """,
]

# Allowed job state changes. A failed job goes back to queued on retry; done and cancelled are final.
TRANSITIONS: dict[str, set[str]] = {
    "queued": {"submitted", "failed", "cancelled"},
    "submitted": {"done", "failed", "cancelled"},
    "failed": {"queued"},
    "cancelled": set(),
    "done": set(),
}

# Fields from a previous attempt that a retry clears.
_RETRY_CLEARS = (
    "hold_reason",
    "modal_call_id",
    "gpu_used",
    "submitted_at",
    "finished_at",
    "error",
    "timings",
)


class IllegalTransition(ValueError):
    pass


def _merge[M: BaseModel](model: M, fields: dict[str, Any]) -> M:
    """Return a validated copy of `model` with `fields` replaced."""
    unknown = set(fields) - set(type(model).model_fields)
    if unknown:
        raise ValueError(f"unknown fields for {type(model).__name__}: {sorted(unknown)}")
    return type(model).model_validate({**model.model_dump(), **fields})



def _reject_fields(fields: dict[str, Any], *names: str) -> None:
    bad = [n for n in names if n in fields]
    if bad:
        raise ValueError(f"{', '.join(bad)} cannot be changed through this call")


class Database:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._con = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.execute("PRAGMA busy_timeout=5000")
        self._migrate()

    def close(self) -> None:
        with self._lock:
            self._con.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._con.execute("BEGIN IMMEDIATE")
            try:
                yield self._con
            except BaseException:
                self._con.execute("ROLLBACK")
                raise
            self._con.execute("COMMIT")

    def _query(self, sql: str, params: tuple = ()) -> list[tuple]:
        with self._lock:
            return self._con.execute(sql, params).fetchall()

    def _migrate(self) -> None:
        with self._tx() as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY, applied_at TEXT)"
            )
            done = {r[0] for r in con.execute("SELECT version FROM schema_version")}
            for number, sql in enumerate(MIGRATIONS, start=1):
                if number in done:
                    continue
                for statement in sql.split(";"):
                    if statement.strip():
                        con.execute(statement)
                con.execute("INSERT INTO schema_version VALUES (?, ?)", (number, now_iso()))

    # Scans

    def insert_scan(self, scan: Scan) -> None:
        with self._tx() as con:
            con.execute(
                "INSERT INTO scans (id, state, created_at, payload) VALUES (?, ?, ?, ?)",
                (scan.id, scan.state, scan.created_at, scan.model_dump_json()),
            )

    def get_scan(self, scan_id: str) -> Scan | None:
        rows = self._query("SELECT payload FROM scans WHERE id = ?", (scan_id,))
        return Scan.model_validate_json(rows[0][0]) if rows else None

    def list_scans(self, state: ScanState | None = None, limit: int = 100) -> list[Scan]:
        if state is None:
            rows = self._query(
                "SELECT payload FROM scans ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,)
            )
        else:
            rows = self._query(
                "SELECT payload FROM scans WHERE state = ? ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (state, limit),
            )
        return [Scan.model_validate_json(r[0]) for r in rows]

    def update_scan(self, scan_id: str, **fields: Any) -> Scan:
        _reject_fields(fields, "id")
        with self._tx() as con:
            current = self.get_scan(scan_id)
            if current is None:
                raise KeyError(scan_id)
            scan = _merge(current, fields)
            con.execute(
                "UPDATE scans SET state = ?, payload = ? WHERE id = ?",
                (scan.state, scan.model_dump_json(), scan_id),
            )
        return scan

    def delete_scan(self, scan_id: str) -> None:
        with self._tx() as con:
            con.execute("DELETE FROM scans WHERE id = ?", (scan_id,))

    # Jobs

    def _write_job(self, con: sqlite3.Connection, job: Job, insert: bool) -> None:
        values = (job.scan_id, job.state, job.modal_call_id, job.finished_at, job.model_dump_json(), job.id)
        if insert:
            con.execute(
                "INSERT INTO jobs (scan_id, state, modal_call_id, finished_at, payload, id, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (*values, job.queued_at),
            )
        else:
            con.execute(
                "UPDATE jobs SET scan_id = ?, state = ?, modal_call_id = ?, finished_at = ?, payload = ?"
                " WHERE id = ?",
                values,
            )

    def insert_job(self, job: Job) -> None:
        with self._tx() as con:
            self._write_job(con, job, insert=True)

    def get_job(self, job_id: str) -> Job | None:
        rows = self._query("SELECT payload FROM jobs WHERE id = ?", (job_id,))
        return Job.model_validate_json(rows[0][0]) if rows else None

    def list_jobs(
        self, state: JobState | None = None, scan_id: str | None = None, limit: int = 100
    ) -> list[Job]:
        where, params = [], []
        if state is not None:
            where.append("state = ?")
            params.append(state)
        if scan_id is not None:
            where.append("scan_id = ?")
            params.append(scan_id)
        clause = f"WHERE {' AND '.join(where)} " if where else ""
        rows = self._query(
            f"SELECT payload FROM jobs {clause}ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (*params, limit),
        )
        return [Job.model_validate_json(r[0]) for r in rows]

    def jobs_for_scan(self, scan_id: str) -> list[Job]:
        rows = self._query(
            "SELECT payload FROM jobs WHERE scan_id = ? ORDER BY created_at DESC, rowid DESC", (scan_id,)
        )
        return [Job.model_validate_json(r[0]) for r in rows]

    def latest_job_for_scan(self, scan_id: str) -> Job | None:
        jobs = self.list_jobs(scan_id=scan_id, limit=1)
        return jobs[0] if jobs else None

    def update_job(self, job_id: str, **fields: Any) -> Job:
        """Change fields other than the state machine's; use `transition` to change `state`."""
        if "state" in fields:
            raise ValueError("use transition() to change a job's state")
        _reject_fields(fields, "id", "scan_id")
        with self._tx() as con:
            current = self.get_job(job_id)
            if current is None:
                raise KeyError(job_id)
            job = _merge(current, fields)
            self._write_job(con, job, insert=False)
        return job

    def transition(self, job: Job, new_state: JobState, **fields: Any) -> Job:
        """Move a job to `new_state`, checked against the stored state, and persist it."""
        _reject_fields(fields, "state", "id", "scan_id")
        with self._tx() as con:
            current = self.get_job(job.id)
            if current is None:
                raise KeyError(job.id)
            if new_state not in TRANSITIONS[current.state]:
                raise IllegalTransition(f"job {job.id}: {current.state} -> {new_state}")
            now = now_iso()
            changes: dict[str, Any] = {"state": new_state}
            if new_state == "queued":
                changes.update({name: None for name in _RETRY_CLEARS})
                changes["queued_at"] = now
            elif new_state == "submitted":
                changes["submitted_at"] = now
            else:
                changes["finished_at"] = now
            changes.update(fields)
            updated = _merge(current, changes)
            self._write_job(con, updated, insert=False)
        return updated

    def sum_cost_for_month(self, month: str) -> float:
        """Sum of cost_estimate_usd over every job with a cost, in `month` ("YYYY-MM").

        Dated by finished_at, or queued_at while a retry is in flight, so a retried job's earlier
        attempt keeps counting toward the budget.
        """
        rows = self._query(
            "SELECT TOTAL(json_extract(payload, '$.cost_estimate_usd')) FROM jobs"
            " WHERE json_extract(payload, '$.cost_estimate_usd') IS NOT NULL"
            " AND COALESCE(finished_at, json_extract(payload, '$.queued_at')) LIKE ?",
            (f"{month}-%",),
        )
        return float(rows[0][0])

    # Results

    def insert_result(self, result: Result) -> None:
        with self._tx() as con:
            con.execute(
                "INSERT INTO results (job_id, created_at, payload) VALUES (?, ?, ?)",
                (result.job_id, result.created_at, result.model_dump_json()),
            )

    def get_result(self, job_id: str) -> Result | None:
        rows = self._query("SELECT payload FROM results WHERE job_id = ?", (job_id,))
        return Result.model_validate_json(rows[0][0]) if rows else None

    # Chat executions

    def insert_execution(self, ex: ChatExecution) -> None:
        with self._tx() as con:
            con.execute(
                "INSERT INTO chat_executions (id, scan_id, job_id, state, created_at, updated_at, payload)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    ex.execution_id,
                    ex.scan_id,
                    ex.job_id,
                    ex.state,
                    ex.created_at,
                    ex.updated_at,
                    ex.model_dump_json(),
                ),
            )

    def get_execution(self, execution_id: str) -> ChatExecution | None:
        rows = self._query("SELECT payload FROM chat_executions WHERE id = ?", (execution_id,))
        return ChatExecution.model_validate_json(rows[0][0]) if rows else None

    def update_execution(self, execution_id: str, **fields: Any) -> ChatExecution:
        """Change fields and stamp `updated_at`."""
        with self._tx() as con:
            current = self.get_execution(execution_id)
            if current is None:
                raise KeyError(execution_id)
            ex = _merge(current, {"updated_at": now_iso(), **fields})
            con.execute(
                "UPDATE chat_executions SET job_id = ?, state = ?, updated_at = ?, payload = ? WHERE id = ?",
                (ex.job_id, ex.state, ex.updated_at, ex.model_dump_json(), execution_id),
            )
        return ex
