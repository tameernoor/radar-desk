"""The Readers lens label list (labels.json): which findings the reads and the queries cover.

Per row: id, title, organ (a scored organ for the viewer jump and the crop check, or None) and radar_keys (catalog
keys; empty when RADAR has no such finding). Rows cost no GPU time; they only steer the reads and the queries.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_PATH = Path(__file__).with_name("labels.json")
_RAW = _PATH.read_bytes()

LABELS: list[dict] = json.loads(_RAW)
LABEL_IDS: list[str] = [row["id"] for row in LABELS]
LABELS_SHA256: str = hashlib.sha256(_RAW).hexdigest()
