"""NV-Reason-CT helpers shared by worker/nvreason_app.py and its tests. Standard library only.

The constants mirror src/radar_desk/services/nvreason.py; tests/test_nvreason_service.py pins them equal.
"""

from __future__ import annotations

import re

MODEL_ID = "nvidia/NV-Reason-CT"
REVISION = "386b93e034983f6c1fc841a43833a1b6a0cd9c13"  # main on 2026-10-14; trust_remote_code runs this revision's code
REGION = "abdomen"
REPORT_PROMPT = "Write a structured abdominal CT report."
REPORT_MAX_NEW_TOKENS = 2048
SCALEDOWN_S = 300
CROP = 192

_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>\s*$", re.DOTALL)


def split_output(decoded: str) -> dict:
    """One generation's decoded text as thinking, text (everything after the last </think>), answer (text
    without a trailing <answer>...</answer> tag) and answer_tag (that tag's content, or None)."""
    thinking, _, text = decoded.rpartition("</think>")
    text = text.strip()
    m = _ANSWER_TAG.search(text)
    return {
        "thinking": thinking.replace("<think>", "").strip(),
        "text": text,
        "answer": text[: m.start()].rstrip() if m else text,
        "answer_tag": m.group(1).strip() if m else None,
    }


def crop_summary(box_lo_mm, box_hi_mm, offset_vox, nz: int, spacing_mm: float, fallback: bool, region) -> dict:
    """crop.json: the box NV-Reason-CT reads (RAS mm, voxel edges), its offset and what it cuts off.

    The cut figures are the ones scripts/nvreason_smoke.py computed; z runs inferior to superior."""
    z0 = int(offset_vox[2])
    return {
        "region": region,
        "box_mm": [[round(float(v), 2) for v in box_lo_mm], [round(float(v), 2) for v in box_hi_mm]],
        "crop_offset_vox": [int(v) for v in offset_vox],
        "scan_length_mm": round(nz * spacing_mm, 1),
        "cut_below_mm": round(max(0, z0) * spacing_mm, 1),
        "cut_above_mm": round(max(0, nz - z0 - CROP) * spacing_mm, 1),
        "fallback": bool(fallback),
    }
