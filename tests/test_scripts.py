"""The command-line scripts that run against the local store."""

from __future__ import annotations

import csv
import io
import os
import subprocess
import sys
from pathlib import Path

from radar_desk.radar import catalog
from test_services import done_job, ready_scan

ROOT = Path(__file__).resolve().parents[1]


def test_export_all_writes_one_row_per_done_job(make_services, tmp_path):
    svc = make_services()
    scan = ready_scan(svc, tmp_path)
    done_job(svc, scan.id)
    done_job(svc, scan.id)
    svc.jobs.create(scan.id)  # queued, not exported

    out = tmp_path / "out" / "scores.csv"
    env = {k: v for k, v in os.environ.items() if k not in ("GPU_BACKEND", "DATA_DIR")}
    env.update(OWNER_TOKEN="t", SESSION_SECRET="s", DATA_DIR=str(tmp_path / "data"), GPU_BACKEND="fake")
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "export_all.py"), str(out)],
                          cwd=tmp_path, env=env, capture_output=True, text=True, check=True)

    assert proc.stdout.startswith("2 rows written")
    rows = list(csv.reader(io.StringIO(out.read_bytes().decode("utf-8-sig"))))
    assert rows[0] == catalog.csv_header()
    assert len(rows) == 3 and all(r[0] == scan.filename for r in rows[1:])
