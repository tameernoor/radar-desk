"""Catalog of RADAR findings, segmentation labels and scored organs."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CATALOG_JSON = ROOT / "src" / "radar_desk" / "radar" / "catalog.json"

EXPECTED_SCORED = {
    # organ: (label, finding_count)
    "Liver": (21, 18),
    "Large bowel": (6, 18),
    "Kidney": (20, 14),
    "Gallbladder": (11, 14),
    "Small bowel": (29, 13),
    "Lung": (22, 10),
    "Pancreas": (23, 10),
    "Spleen": (30, 8),
    "Heart": (13, 2),
    "Adrenal gland": (1, 6),
    "Stomach": (31, 6),
    "Bladder": (33, 6),
    "Duodenum": (7, 5),
    "Esophagus": (8, 5),
    "Aorta": (2, 4),
    "Rib": (26, 3),
    "Portal vein": (24, 3),
    "Sacrum": (27, 1),
}


def _load_gen_catalog():
    spec = importlib.util.spec_from_file_location("gen_catalog", ROOT / "scripts" / "gen_catalog.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cat():
    from radar_desk.radar import catalog

    return catalog


def test_findings_shape_and_order(cat):
    assert len(cat.FINDINGS) == 146
    first, last = cat.FINDINGS[0], cat.FINDINGS[-1]
    assert first == {
        "index": 0,
        "key": "主动脉_主动脉夹层",
        "organ_zh": "主动脉",
        "organ": "Aorta",
        "finding": "Aortic dissection",
        "english": "Aorta_Aortic dissection",
    }
    assert last["key"] == "骶骨_骨炎"
    assert last["english"] == "Sacrum_Osteitis"
    assert [f["index"] for f in cat.FINDINGS] == list(range(146))
    for f in cat.FINDINGS:
        assert f["key"].split("_", 1)[0] == f["organ_zh"]
        assert f["english"] == f"{f['organ']}_{f['finding']}"


def test_labels(cat):
    labels = [x for x in cat.LABELS if x["label"] != 0]
    assert len(labels) == 36
    assert [x["label"] for x in labels] == list(range(1, 37))
    assert cat.LABELS[0] == {"label": 0, "zh": "background", "en": "background"}
    assert labels[0] == {"label": 1, "zh": "肾上腺", "en": "adrenal gland"}
    assert labels[-1] == {"label": 36, "zh": "胸椎", "en": "thoracic vertebrae"}


def test_scored_organs_match_spec_table(cat):
    assert len(cat.SCORED_ORGANS) == 18
    got = {o["organ"]: (o["label"], o["finding_count"]) for o in cat.SCORED_ORGANS}
    assert got == EXPECTED_SCORED
    assert len({o["label"] for o in cat.SCORED_ORGANS}) == 18
    assert sum(o["finding_count"] for o in cat.SCORED_ORGANS) == 146
    assert cat.scored_organ_names()[0] == "Aorta"
    assert cat.scored_organ_names()[-1] == "Sacrum"


def test_catalog_json_equals_fresh_generation():
    gen = _load_gen_catalog()
    fresh = gen.build_catalog()
    committed = json.loads(CATALOG_JSON.read_text(encoding="utf-8"))
    assert fresh == committed
    assert gen.render(fresh) == CATALOG_JSON.read_text(encoding="utf-8")


def test_csv_header_matches_upstream_demo(cat):
    with (ROOT / "fixtures" / "expected" / "damo-demo.csv").open(encoding="utf-8-sig", newline="") as fh:
        header = next(csv.reader(fh))
    assert cat.csv_header() == header
    assert len(header) == 147


def test_lookups(cat):
    f = cat.finding_by_key("肝_肝囊肿")
    assert f["organ"] == "Liver" and f["finding"] == "Cyst"
    assert len(cat.findings_for_organ("Liver")) == 18
    assert cat.label_for_organ("Kidney") == 20
    assert cat.organ_for_label(20) == "Kidney"
    assert cat.english_to_key("Liver_Cyst") == "肝_肝囊肿"


def test_organ_for_unscored_label_is_none(cat):
    # label 3 (erector spinae) is segmented but not scored
    assert cat.organ_for_label(3) is None


@pytest.mark.parametrize(
    "call",
    [
        lambda c: c.finding_by_key("nope"),
        lambda c: c.findings_for_organ("Brain"),
        lambda c: c.label_for_organ("Brain"),
        lambda c: c.organ_for_label(37),
        lambda c: c.organ_for_label(0),
        lambda c: c.english_to_key("Liver_Nothing"),
    ],
)
def test_lookups_reject_unknowns(cat, call):
    with pytest.raises(KeyError):
        call(cat)
