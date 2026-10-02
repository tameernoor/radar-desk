"""Weights check against worker/weights.json, standard library only.

Used by the Modal function at container start and by scripts/score_local.py, so both
refuse the same mismatches: every file by size, the checkpoint (or every file with
hash_all) by sha256.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

CHECKPOINT = "checkpoint_radar_pretrain.pth"


def sha256_file(path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def check_weights(weights_dir, manifest: dict, hash_all: bool = False) -> dict:
    """Compare files on disk with the manifest. Sizes always; sha256 of the checkpoint, or of all with hash_all."""
    rows, problems = [], []
    for entry in manifest["files"]:
        path = Path(weights_dir) / entry["path"]
        row = {"path": entry["path"], "expected_size": entry["size"], "expected_sha256": entry.get("sha256")}
        if not path.is_file():
            row.update(size=None, sha256=None, ok=False)
            problems.append(f"{entry['path']}: missing")
            rows.append(row)
            continue
        row["size"] = path.stat().st_size
        ok = row["size"] == entry["size"]
        if not ok:
            problems.append(f"{entry['path']}: size {row['size']} != {entry['size']}")
        if hash_all or entry["path"] == CHECKPOINT:
            row["sha256"] = sha256_file(path)
            if entry.get("sha256") and row["sha256"] != entry["sha256"]:
                ok = False
                problems.append(f"{entry['path']}: sha256 {row['sha256']} != {entry['sha256']}")
        row["ok"] = ok
        rows.append(row)
    ckpt = next((r for r in rows if r["path"] == CHECKPOINT), {})
    return {"ok": not problems, "problems": problems, "rows": rows, "checkpoint_sha256": ckpt.get("sha256")}


def code_commit(vendored_md) -> str | None:
    """The upstream commit named in the vendored tree's VENDORED.md, or None."""
    path = Path(vendored_md)
    if not path.is_file():
        return None
    m = re.search(r"\b([0-9a-f]{40})\b", path.read_text())
    return m.group(1) if m else None
