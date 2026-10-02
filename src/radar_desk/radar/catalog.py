"""The RADAR catalog: 146 findings, 36 segmentation labels and the 18 scored organs.

Loaded once from catalog.json, which scripts/gen_catalog.py generates from the
vendored DAMO source. Organ names are the English ones (`Liver`, `Large bowel`);
finding keys are the Chinese upstream keys.
"""

from __future__ import annotations

import json
from pathlib import Path

_DATA = json.loads((Path(__file__).with_name("catalog.json")).read_text(encoding="utf-8"))

SOURCE_COMMIT: str = _DATA["source_commit"]
FINDINGS: list[dict] = _DATA["findings"]
LABELS: list[dict] = _DATA["labels"]
SCORED_ORGANS: list[dict] = _DATA["scored_organs"]

_BY_KEY = {f["key"]: f for f in FINDINGS}
_BY_ENGLISH = {f["english"]: f["key"] for f in FINDINGS}
_ORGAN_BY_NAME = {o["organ"]: o for o in SCORED_ORGANS}
_ORGAN_BY_LABEL = {o["label"]: o["organ"] for o in SCORED_ORGANS}
_LABEL_IDS = {lab["label"] for lab in LABELS if lab["label"] != 0}


def finding_by_key(key: str) -> dict:
    try:
        return _BY_KEY[key]
    except KeyError:
        raise KeyError(f"unknown finding key: {key!r}") from None


def _scored(organ: str) -> dict:
    try:
        return _ORGAN_BY_NAME[organ]
    except KeyError:
        raise KeyError(f"not a scored organ: {organ!r} (expected one of {scored_organ_names()})") from None


def findings_for_organ(organ: str) -> list[dict]:
    _scored(organ)
    return [f for f in FINDINGS if f["organ"] == organ]


def label_for_organ(organ: str) -> int:
    return _scored(organ)["label"]


def organ_for_label(label: int) -> str | None:
    """The scored organ for a segmentation label, or None for a label that is segmented but not scored."""
    if label not in _LABEL_IDS:
        raise KeyError(f"unknown segmentation label: {label!r} (expected 1 to {max(_LABEL_IDS)})")
    return _ORGAN_BY_LABEL.get(label)


def scored_organ_names() -> list[str]:
    return [o["organ"] for o in SCORED_ORGANS]


def csv_header() -> list[str]:
    """The header row of upstream's results CSV."""
    return ["file_name"] + [f"{f['key']} ({f['english']})" for f in FINDINGS]


def english_to_key(english: str) -> str:
    try:
        return _BY_ENGLISH[english]
    except KeyError:
        raise KeyError(f"unknown finding: {english!r}") from None
