"""Synthetic Readers lens data for tests, scripts/seed_dev.py --readers and the Playwright specs. Made-up text only,
never a real report."""

from __future__ import annotations

import json

from radar_desk.readers.labels import LABEL_IDS

SYNTHETIC_REPORT = (
    "Synthetic test report, not a real patient.\n\n"
    "FINDINGS:\n"
    "Liver: A small hypodense lesion in the right lobe.\n"
    "Kidneys: The kidneys are unremarkable.\n"
    "Peritoneum: Small volume ascites in the pelvis.\n"
    "Bones: Degenerative change of the lumbar spine.\n\n"
    "IMPRESSION:\n"
    "1. Small liver lesion.\n"
    "2. Small volume ascites."
)


def read_reply(states: dict[str, tuple[str, str | None]], others: list[str]) -> str:
    """A model reply for a read: every label not_mentioned unless given."""
    labels = {lid: {"state": "not_mentioned", "quote": None} for lid in LABEL_IDS}
    for lid, (state, quote) in states.items():
        labels[lid] = {"state": state, "quote": quote}
    return json.dumps({"labels": labels, "other_mentions": others})


REPORT_READ = read_reply(
    {
        "liver_lesion": ("present", "A small hypodense lesion in the right lobe."),
        "renal_cyst": ("absent", "The kidneys are unremarkable."),
        "ascites": ("present", "Small volume ascites in the pelvis."),
    },
    ["Degenerative change of the lumbar spine."],
)


NV_READ = read_reply(
    {
        "liver_lesion": ("present", "A small hypodense lesion in the right lobe."),
        "renal_cyst": ("present", "A simple cyst in the left kidney."),
        "hydronephrosis": ("absent", "No hydronephrosis."),
        "ascites": ("absent", "No ascites."),
    },
    [],
)


def scripted_replies() -> list[str]:
    """The chat model's replies for one full refresh, in the order the refresh asks: report read, NV read."""
    return [REPORT_READ, NV_READ]
