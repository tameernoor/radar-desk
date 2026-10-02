"""Copy the objects the database refers to from one storage backend to another (plan.md, Shared storage).

Settings come from the environment or `.env` in the working directory, like the app:

    uv run python scripts/migrate_storage.py --from modal_volume --to s3 [--dry-run]

The objects are the source of every ready scan and the artefacts of every done job, under the same keys on
both sides. An object already at the destination with the same size is skipped; one with another size is
reported and left alone, because objects are write-once. Each copy streams through a temp file under
DATA_DIR/migrate-tmp, a scan source is checked against the sha256 in the database before it is written, and
the destination size is checked after. The source is never written or deleted.

A storage error on one object is reported and the run goes on to the next. Exit codes are 0 ok, 1 when an
object was missing in the source or had a problem, 2 invalid settings. No secret is printed.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from radar_desk.config import ConfigError, Settings, load_settings
from radar_desk.db import Database
from radar_desk.services.scans import source_key
from radar_desk.storage import ObjectMissing, Storage, StorageError, make_storage

BACKENDS = ("local", "s3", "modal_volume", "runpod_volume")
ALL = 1_000_000


def _print_out(line: str) -> None:
    print(line, flush=True)


def _print_err(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


@dataclass
class Deps:
    settings: Settings | None = None  # None means load_settings()
    out: Callable[[str], None] = field(default=_print_out)
    err: Callable[[str], None] = field(default=_print_err)


@dataclass
class Summary:
    copied: int = 0
    bytes: int = 0
    existing: int = 0
    missing: int = 0
    problems: int = 0
    dry_run: bool = False

    def line(self) -> str:
        verb = "would copy" if self.dry_run else "copied"
        return (f"{verb} {self.copied} ({self.bytes} bytes), skipped {self.existing} existing, "
                f"missing {self.missing}, problems {self.problems}")


def objects(db: Database) -> list[tuple[str, str | None]]:
    """(key, expected sha256 or None) for every ready scan's source and every done job's artefacts."""
    keys = [(source_key(s.id), s.sha256) for s in db.list_scans(state="ready", limit=ALL)]
    for job in db.list_jobs(state="done", limit=ALL):
        result = db.get_result(job.id)
        if result is not None:
            keys += [(k, None) for k in result.artefacts.model_dump().values() if k]
    return keys


def _size(storage: Storage, key: str) -> int | None:
    try:
        return storage.size(key)
    except ObjectMissing:
        return None


def _copy(key: str, sha256: str | None, src: Storage, dst: Storage, tmp_dir: Path, dry_run: bool,
          out: Callable[[str], None], summary: Summary) -> None:
    size = _size(src, key)
    if size is None:
        out(f"missing {key}")
        summary.missing += 1
        return
    there = _size(dst, key)
    if there == size:
        out(f"exists {key}")
        summary.existing += 1
        return
    if there is not None:
        out(f"differs {key} ({size} vs {there} bytes)")
        summary.problems += 1
        return
    if dry_run:
        out(f"would copy {key} {size} bytes")
        summary.copied += 1
        summary.bytes += size
        return
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=tmp_dir)
    tmp = Path(name)
    try:
        h, n = hashlib.sha256(), 0
        with open(fd, "wb") as fh:
            for chunk in src.open_stream(key):
                fh.write(chunk)
                h.update(chunk)
                n += len(chunk)
        if sha256 is not None and h.hexdigest() != sha256:
            out(f"sha256 mismatch {key}")
            summary.problems += 1
            return
        dst.put_file(key, tmp)
        if dst.size(key) != n:
            dst.delete(key)  # written by this run a moment ago, so a rerun can copy it again
            out(f"size check failed {key}, removed")
            summary.problems += 1
            return
        out(f"copied {key} {n} bytes")
        summary.copied += 1
        summary.bytes += n
    finally:
        tmp.unlink(missing_ok=True)


def migrate(db: Database, src: Storage, dst: Storage, data_dir: Path, dry_run: bool,
            out: Callable[[str], None]) -> Summary:
    summary = Summary(dry_run=dry_run)
    tmp_dir = Path(data_dir) / "migrate-tmp"
    for key, sha256 in objects(db):
        try:
            _copy(key, sha256, src, dst, tmp_dir, dry_run, out, summary)
        except StorageError as exc:
            out(f"error {key}: {exc}")
            summary.problems += 1
    try:
        tmp_dir.rmdir()
    except OSError:
        pass  # absent, or something else is in it
    return summary


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scripts/migrate_storage.py",
                                description="Copy the database's objects between storage backends.")
    p.add_argument("--from", dest="src", required=True, choices=BACKENDS)
    p.add_argument("--to", dest="dst", required=True, choices=BACKENDS)
    p.add_argument("--dry-run", action="store_true", help="print what would be copied and write nothing")
    return p


def main(argv: list[str] | None = None, deps: Deps | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if args.src == args.dst:
        p.error("--from and --to must differ")
    deps = deps or Deps()
    try:
        settings = deps.settings or load_settings()
        src = make_storage(settings, backend=args.src)
        dst = make_storage(settings, backend=args.dst)
    except ConfigError as exc:
        deps.err(str(exc))
        return 2
    if not settings.db_path.is_file():
        deps.err(f"no database at {settings.db_path}")
        return 2
    db = Database(settings.db_path)
    try:
        summary = migrate(db, src, dst, settings.data_dir, args.dry_run, deps.out)
    finally:
        db.close()
    deps.out(summary.line())
    return 1 if summary.problems or summary.missing else 0


if __name__ == "__main__":
    sys.exit(main())
