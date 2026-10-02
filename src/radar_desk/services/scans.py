"""Uploads and scan records (design.md, Data flow 1 and Errors)."""

from __future__ import annotations

import time
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from radar_desk.radar import header as nifti
from radar_desk.records import Scan, new_id, now_iso
from radar_desk.services.costs import iso_at
from radar_desk.services.errors import ServiceError
from radar_desk.storage import ObjectMissing, StorageError

if TYPE_CHECKING:
    from radar_desk.services.fixtures import FixtureService

UPLOAD_URL_S = 15 * 60
VIEW_URL_S = 60 * 60
EXTENSIONS = (".nii.gz", ".nii")


def source_key(scan_id: str) -> str:
    return f"scans/{scan_id}/source.nii.gz"


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
        """Remove the source object and the scan record. Refused while a job for the scan is queued or
        running. In v1 the scan's jobs, results and job artefacts are kept."""
        self.get(scan_id)
        active = [j for j in self.db.jobs_for_scan(scan_id) if j.state in ("queued", "submitted")]
        if active:
            raise ServiceError(409, f"job {active[0].id} is {active[0].state}; cancel it first")
        self.storage.delete(source_key(scan_id))
        self.db.delete_scan(scan_id)

    def view_url(self, scan_id: str) -> str:
        """A one-hour GET URL for the viewer."""
        scan = self.get(scan_id)
        if scan.state != "ready":
            raise ServiceError(409, f"scan {scan_id} is {scan.state}")
        return self.storage.get_url(source_key(scan_id), VIEW_URL_S)
