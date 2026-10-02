"""Tier-3 parity and the mask sanity checks from the design. Written now, skipped until their inputs exist.

Tier 3 needs radar-web's JSON exports in fixtures/expected/radar-web/<id>.json (open question 5).
The Dice check needs a real mask from a Modal run (a result with versions.gpu other than "fake").
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from radar_desk.radar import catalog

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ROOT / "fixtures" / "expected"
MANIFEST = json.loads((ROOT / "fixtures" / "manifest.json").read_text())
FIXTURE_IDS = ["AC4214dbd", "AC4240fff", "AC4242a2f", "AC4242a55", "image1"]
AC_IDS = FIXTURE_IDS[:4]
RADAR_WEB_TOL = 1e-3

# Real results land here when scripts/parity.py or a Modal job is run by hand and the outputs are copied in.
REAL_RESULTS = EXPECTED / "real"


def _real(fixture_id: str, name: str) -> Path:
    path = REAL_RESULTS / fixture_id / name
    if not path.is_file():
        pytest.skip(f"no real {name} for {fixture_id} under {REAL_RESULTS} (needs a Modal run)")
    return path


@pytest.mark.parametrize("fixture_id", FIXTURE_IDS)
def test_tier3_radar_web_parity(fixture_id):
    ref = EXPECTED / "radar-web" / f"{fixture_id}.json"
    if not ref.is_file():
        pytest.skip(f"radar-web export {ref.name} not saved yet (design open question 5)")
    ours = json.loads(_real(fixture_id, "scores.json").read_text())
    theirs = {f["key"]: f["prob"] for f in json.loads(ref.read_text())["findings"]}
    mine = {f["key"]: f["prob"] for f in ours["findings"]}
    deltas = {k: abs(mine[k] - v) for k, v in theirs.items() if mine.get(k) is not None}
    assert deltas, "no overlapping scored findings"
    worst = max(deltas.values())
    assert worst <= RADAR_WEB_TOL, f"max abs diff {worst:.5f} over {RADAR_WEB_TOL} (known caveat: last-window labels)"


@pytest.mark.parametrize("fixture_id", AC_IDS)
def test_mask_dice_against_damo_resized_masks(fixture_id):
    """Per-organ Dice between our mask and DAMO's TotalSegmentator-derived masks; low on a large organ means a geometry bug."""
    ours = nib.load(_real(fixture_id, "mask.nii.gz"))
    damo_path = ROOT / "fixtures" / next(
        v["damo_mask"] for v in MANIFEST["scans"].values() if v["id"] == fixture_id
    )
    theirs = nib.load(damo_path)
    assert ours.shape == theirs.shape, "the AC fixtures are already 1 x 1 x 5 mm, so the grids must match"
    a = np.asarray(ours.dataobj).astype(np.uint8)
    b = np.asarray(theirs.dataobj).astype(np.uint8)
    for organ in ("Liver", "Kidney", "Spleen"):
        label = catalog.label_for_organ(organ)
        x, y = a == label, b == label
        if not y.any():
            continue
        dice = 2.0 * np.logical_and(x, y).sum() / (x.sum() + y.sum())
        assert dice >= 0.7, f"{organ} Dice {dice:.3f} against DAMO's mask"


@pytest.mark.parametrize("fixture_id", FIXTURE_IDS)
def test_every_scored_organ_has_a_nonempty_mask(fixture_id):
    scores = json.loads(_real(fixture_id, "scores.json").read_text())
    mask = np.asarray(nib.load(_real(fixture_id, "mask.nii.gz")).dataobj)
    present = set(np.unique(mask).tolist())
    for entry in scores["organs_scored"]:
        assert entry["label"] in present, f"{entry['organ']} was scored but has no voxels in the mask"
