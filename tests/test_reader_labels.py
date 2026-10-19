"""The curated label list: 22 rows, every RADAR key in the catalog, every organ a scored organ."""

from __future__ import annotations

from radar_desk.radar import catalog
from radar_desk.readers.labels import LABEL_IDS, LABELS, LABELS_SHA256


def test_twenty_two_unique_rows():
    assert len(LABELS) == 22
    assert len(set(LABEL_IDS)) == 22
    assert LABEL_IDS[0] == "liver_lesion" and LABEL_IDS[-1] == "lung_nodule"
    assert len(LABELS_SHA256) == 64


def test_every_radar_key_exists_in_the_catalog():
    known = {f["key"] for f in catalog.FINDINGS}
    for row in LABELS:
        missing = [k for k in row["radar_keys"] if k not in known]
        assert not missing, f"{row['id']}: {missing}"


def test_every_organ_is_a_scored_organ_or_none():
    organs = set(catalog.scored_organ_names())
    for row in LABELS:
        assert row["organ"] is None or row["organ"] in organs, row["id"]
        assert set(row) == {"id", "title", "organ", "radar_keys"}


def test_spot_checks_against_the_spec_table():
    rows = {r["id"]: r for r in LABELS}
    english = {f["key"]: f["english"] for f in catalog.FINDINGS}
    assert [english[k] for k in rows["kidney_stone"]["radar_keys"]] == ["Kidney_Nephrolithiasis"]
    assert [english[k] for k in rows["bile_duct_dilatation"]["radar_keys"]] == [
        "Liver_Intrahepatic bile duct dilatation", "Gallbladder_Extrahepatic bile duct dilatation"]
    assert rows["ascites"] == {"id": "ascites", "title": "Ascites", "organ": None, "radar_keys": []}
    assert rows["lymphadenopathy"]["organ"] is None and len(rows["lymphadenopathy"]["radar_keys"]) == 1
    assert len(rows["bowel_wall"]["radar_keys"]) == 7 and len(rows["liver_lesion"]["radar_keys"]) == 8
