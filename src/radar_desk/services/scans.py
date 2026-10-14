"""Uploads and scan records (design.md, Data flow 1 and Errors)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from radar_desk.radar import header as nifti
from radar_desk.records import Scan, new_id, now_iso
from radar_desk.services.blobs import read_bytes, replace_bytes
from radar_desk.services.costs import iso_at
from radar_desk.services.errors import ServiceError
from radar_desk.storage import ObjectMissing, StorageError

if TYPE_CHECKING:
    from radar_desk.services.fixtures import FixtureService

log = logging.getLogger(__name__)

UPLOAD_URL_S = 15 * 60
VIEW_URL_S = 60 * 60
EXTENSIONS = (".nii.gz", ".nii")


def source_key(scan_id: str) -> str:
    return f"scans/{scan_id}/source.nii.gz"


REPORT_MAX_CHARS = 100_000


def report_key(scan_id: str) -> str:
    return f"scans/{scan_id}/report.txt"


def clean_report(text: str) -> str:
    """Report text as stored: no byte order mark, \\n line ends, no blank edges."""
    text = str(text).replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


class UploadTicket(BaseModel):
    scan_id: str
    put_url: str
    expires_at: str


class ScanService:
    def __init__(self, db: Any, storage: Any, settings: Any, fixtures: FixtureService) -> None:
        self.db = db
        self.storage = storage
        self.settings = settings
        self.fixtures = fixtures
        self.report_listeners: list[Callable[[str], None]] = []
        self.delete_listeners: list[Callable[[str], None]] = []

    # The radiologist report (Readers lens)

    def get_report(self, scan_id: str) -> str | None:
        data = read_bytes(self.storage, report_key(scan_id))
        return None if data is None else data.decode("utf-8", "replace")

    def put_report(self, scan_id: str, text: str) -> str:
        """Add or replace the scan's report. Returns the text as stored."""
        self.get(scan_id)
        clean = clean_report(text)
        if not clean:
            raise ServiceError(422, "the report is empty")
        if len(clean) > REPORT_MAX_CHARS:
            raise ServiceError(413, f"the report is longer than {REPORT_MAX_CHARS:,} characters")
        replace_bytes(self.storage, report_key(scan_id), clean.encode("utf-8"), "text/plain; charset=utf-8")
        self._report_changed(scan_id)
        return clean

    def delete_report(self, scan_id: str) -> None:
        self.get(scan_id)
        self.storage.delete(report_key(scan_id))
        self._report_changed(scan_id)

    def _report_changed(self, scan_id: str) -> None:
        for listener in self.report_listeners:
            try:
                listener(scan_id)
            except Exception:
                log.exception("scans: a report listener failed for scan %s", scan_id)

    def begin_upload(self, filename: str, size_bytes: int, confirmed: bool) -> UploadTicket:
        name = PurePosixPath(str(filename).replace("\\", "/")).name.strip()
        if not name.lower().endswith(EXTENSIONS):
            raise ServiceError(415, "only .nii and .nii.gz files are accepted")
        if size_bytes > self.settings.max_upload_bytes:
            limit_mb = self.settings.max_upload_bytes / (1024 * 1024)
            raise ServiceError(413, f"the file is larger than the {limit_mb:g} MB limit")
        if size_bytes <= 0:
            raise ServiceError(422, "the file is empty")
        if not confirmed:
            raise ServiceError(422, "confirm that the scan is for research use only")
        scan = Scan(
            id=new_id("scan"),
            filename=name,
            size_bytes=int(size_bytes),
            research_only_confirmed_at=now_iso(),
        )
        self.db.insert_scan(scan)
        url = self.storage.put_url(source_key(scan.id), UPLOAD_URL_S)
        return UploadTicket(scan_id=scan.id, put_url=url, expires_at=iso_at(time.time() + UPLOAD_URL_S))

    def get(self, scan_id: str) -> Scan:
        scan = self.db.get_scan(scan_id)
        if scan is None:
            raise ServiceError(404, f"no scan {scan_id}")
        return scan

    def list(self, state: str | None = None, limit: int = 100) -> list[Scan]:
        return self.db.list_scans(state=state, limit=limit)

    def complete_upload(self, scan_id: str) -> Scan:
        """Stream the object once, validate it, and move the scan to ready or rejected.

        A rejection keeps the record with its reason, deletes the object and raises 422.
        """
        scan = self.get(scan_id)
        if scan.state != "uploading":
            return scan
        key = source_key(scan_id)
        if not self.storage.exists(key):
            raise ServiceError(409, "the file has not been uploaded yet")
        try:
            stored = self.storage.size(key)
            if stored > self.settings.max_upload_bytes:
                raise nifti.HeaderError("the uploaded file is larger than the upload limit")
            hdr, sha256, decompressed = nifti.inspect_object(self.storage.open_stream(key))
            nifti.validate(hdr)
        except ObjectMissing:
            raise ServiceError(409, "the file has not been uploaded yet") from None
        except (nifti.HeaderError, StorageError, EOFError, OSError, ValueError) as exc:
            reason = str(exc) or type(exc).__name__
            self.storage.delete(key)
            self.db.update_scan(scan_id, state="rejected", rejected_reason=reason)
            raise ServiceError(422, reason) from None
        return self.db.update_scan(
            scan_id,
            state="ready",
            size_bytes=stored,
            header=hdr,
            sha256=sha256,
            decompressed_bytes=decompressed,
            fixture_id=self.fixtures.id_for_sha256(sha256),
        )

    def delete(self, scan_id: str) -> None:
        """Remove the source object, the report, the readers outputs and the scan record. Refused while a job for the
        scan is queued or running. In v1 the scan's jobs, results and job artefacts are kept."""
        self.get(scan_id)
        active = [j for j in self.db.jobs_for_scan(scan_id, kind=None) if j.state in ("queued", "submitted")]
        if active:
            raise ServiceError(409, f"job {active[0].id} is {active[0].state}; cancel it first")
        self.storage.delete(source_key(scan_id))
        self.storage.delete(report_key(scan_id))
        for listener in self.delete_listeners:
            listener(scan_id)
        self.db.delete_scan(scan_id)

    def view_url(self, scan_id: str) -> str:
        """A one-hour GET URL for the viewer."""
        scan = self.get(scan_id)
        if scan.state != "ready":
            raise ServiceError(409, f"scan {scan_id} is {scan.state}")
        return self.storage.get_url(source_key(scan_id), VIEW_URL_S)
