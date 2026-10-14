"""The radiologist report attached to a scan: plain text at scans/{scan_id}/report.txt."""

from __future__ import annotations

import pytest

from api_client import api_client
from radar_desk.services import ServiceError
from radar_desk.services.scans import REPORT_MAX_CHARS, clean_report, report_key
from test_chat_tools import ready_scan

SYNTHETIC = "FINDINGS: Synthetic test report. Small volume ascites.\nIMPRESSION: Ascites."


def test_put_get_replace_and_delete(make_services, tmp_path):
    svc = make_services()
    scan = ready_scan(svc, tmp_path)
    seen: list[str] = []
    svc.scans.report_listeners.append(seen.append)
    assert svc.scans.get_report(scan.id) is None
    assert svc.scans.put_report(scan.id, SYNTHETIC) == SYNTHETIC
    assert svc.scans.get_report(scan.id) == SYNTHETIC
    svc.scans.put_report(scan.id, "Replaced.")
    assert svc.scans.get_report(scan.id) == "Replaced."
    svc.scans.delete_report(scan.id)
    assert svc.scans.get_report(scan.id) is None
    assert seen == [scan.id, scan.id, scan.id]
    assert report_key(scan.id) == f"scans/{scan.id}/report.txt"


def test_report_text_is_cleaned_and_limited(make_services, tmp_path):
    """Review Focus 2: a Windows .txt with a byte order mark and CRLF, blank text, and a very long paste."""
    svc = make_services()
    scan = ready_scan(svc, tmp_path)
    assert clean_report("﻿Line one.\r\nLine two.\rLine three.  \n\n") == "Line one.\nLine two.\nLine three."
    assert svc.scans.put_report(scan.id, "﻿  Ascites.\r\n") == "Ascites."
    with pytest.raises(ServiceError) as empty:
        svc.scans.put_report(scan.id, " \r\n﻿ ")
    assert empty.value.status == 422 and empty.value.detail == "the report is empty"
    with pytest.raises(ServiceError) as long:
        svc.scans.put_report(scan.id, "x" * (REPORT_MAX_CHARS + 1))
    assert long.value.status == 413
    assert svc.scans.get_report(scan.id) == "Ascites."


def test_unknown_scan_is_404(make_services):
    svc = make_services()
    with pytest.raises(ServiceError) as err:
        svc.scans.put_report("scan_missing", SYNTHETIC)
    assert err.value.status == 404


def test_replacing_works_on_the_write_once_volume(make_services, tmp_path):
    from fake_volume import FakeVolume
    from radar_desk.storage.modal_volume import ModalVolumeStorage

    storage = ModalVolumeStorage("radar-data", "http://127.0.0.1:8000", volume=FakeVolume())
    svc = make_services(storage=storage, storage_backend="modal_volume")
    scan = ready_scan(svc, tmp_path)
    svc.scans.put_report(scan.id, "First.")
    svc.scans.put_report(scan.id, "Second.")
    assert svc.scans.get_report(scan.id) == "Second."


def test_report_routes(make_services, tmp_path):
    svc = make_services()
    scan = ready_scan(svc, tmp_path)
    client = api_client(svc)
    assert client.get(f"/scans/{scan.id}/report").status_code == 404
    r = client.put(f"/scans/{scan.id}/report", json={"text": "﻿Ascites.\r\n"})
    assert r.status_code == 200 and r.json() == {"text": "Ascites."}
    assert client.get(f"/scans/{scan.id}/report").json() == {"text": "Ascites."}
    assert client.put(f"/scans/{scan.id}/report", json={"text": "  "}).status_code == 422
    assert client.delete(f"/scans/{scan.id}/report").status_code == 204
    assert client.get(f"/scans/{scan.id}/report").status_code == 404
