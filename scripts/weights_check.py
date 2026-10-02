"""Check the `radar-weights` Volume against worker/weights.json.

    modal run scripts/weights_check.py            # size and sha256 of every file; exit 1 on a mismatch
    modal run scripts/weights_check.py --write    # also fill sha256 values that weights.json leaves empty

Hashing runs in a small CPU container with the Volume mounted read-only, so nothing is
downloaded to this machine.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "worker" / "weights.json"
VOLUME_NAME = "radar-weights"
MOUNT = "/weights"

app = modal.App("radar-desk-weights-check")
volume = modal.Volume.from_name(VOLUME_NAME)


@app.function(
    image=modal.Image.debian_slim(python_version="3.10"),
    volumes={MOUNT: volume.with_mount_options(read_only=True)},
    timeout=1800,
    cpu=2.0,
)
def survey(paths: list) -> dict:
    """Size and sha256 of each listed file, plus any file on the Volume that is not listed."""
    rows = []
    for rel in paths:
        p = Path(MOUNT) / rel
        if not p.is_file():
            rows.append({"path": rel, "size": None, "sha256": None})
            continue
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for block in iter(lambda: fh.read(8 << 20), b""):
                h.update(block)
        rows.append({"path": rel, "size": p.stat().st_size, "sha256": h.hexdigest()})
    listed = set(paths)
    extra = []
    for dirpath, _, names in os.walk(MOUNT):
        for n in names:
            rel = os.path.relpath(os.path.join(dirpath, n), MOUNT)
            if rel not in listed:
                extra.append(rel)
    return {"rows": rows, "extra": sorted(extra)}


@app.local_entrypoint()
def main(write: bool = False):
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("volume") != VOLUME_NAME:
        raise SystemExit(f"weights.json names volume {manifest.get('volume')!r}, this script checks {VOLUME_NAME!r}")
    found = survey.remote([f["path"] for f in manifest["files"]])
    bad = 0
    filled = 0
    for entry, row in zip(manifest["files"], found["rows"]):
        if row["size"] is None:
            status = "MISSING"
        elif row["size"] != entry["size"]:
            status = f"SIZE {row['size']} != {entry['size']}"
        elif not entry.get("sha256"):
            status = "ok size, no sha256 in weights.json"
            if write:
                entry["sha256"] = row["sha256"]
                filled += 1
                status += " (filled)"
        elif row["sha256"] != entry["sha256"]:
            status = "SHA256 MISMATCH"
        else:
            status = "ok"
        if not status.startswith("ok"):
            bad += 1
        print(f"{entry['path']:42} {row['size']!s:>12}  {row['sha256'] or '-'}  {status}")
    for rel in found["extra"]:
        print(f"{rel:42} (on the Volume, not in weights.json)")
    if write and filled:
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"wrote {filled} sha256 value(s) to {MANIFEST}")
    if bad:
        print(f"{bad} file(s) do not match")
        sys.exit(1)
    print("all files match")
