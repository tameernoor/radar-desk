"""Manual local check: the server's organ coverage against the smoke script's, on a real smoke run.

    NVREASON_SMOKE_RUN=smoke-out/nvreason/nvr-20261014-192111 .venv/bin/pytest -q tests/test_readers_smoke_local.py

Reads run.json and the RADAR masks the smoke run used. Nothing from the run is copied into the repo.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from radar_desk.services.readers import organ_coverage

ROOT = Path(__file__).resolve().parents[1]
RUN = os.environ.get("NVREASON_SMOKE_RUN")

pytestmark = pytest.mark.skipif(not RUN, reason="set NVREASON_SMOKE_RUN to a smoke-out/nvreason/<run> folder")


def test_server_coverage_matches_the_smoke_script():
    checked = 0
    for run_json in sorted((ROOT / RUN).glob("*/run.json")):
        run = json.loads(run_json.read_text(encoding="utf-8"))
        if not run.get("organ_mask"):
            continue
        mask = Path(run["organ_mask"])
        mask = mask if mask.is_absolute() else ROOT / mask
        got = organ_coverage(mask.read_bytes(), [run["box_lo_mm"], run["box_hi_mm"]])
        for organ, share in got.items():
            assert abs(share - run["organ_coverage"][organ]) <= 0.001, (run_json.parent.name, organ)
        checked += 1
    assert checked, "no run.json with an organ_mask"
