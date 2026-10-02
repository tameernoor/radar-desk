"""Vendor a pinned subset of github.com/alibaba-damo-academy/damo-radar.

Fetches the tarball for one commit over HTTPS (no git), verifies its sha256,
and copies the inference code and licence files into worker/vendor/damo-radar/.
Nothing in the copied tree is edited; VENDORED.md records what was taken.

Usage:
    python scripts/vendor_damo.py                 # download and extract
    python scripts/vendor_damo.py --tarball PATH  # use an already downloaded tarball
"""

from __future__ import annotations

import argparse
import hashlib
import io
import shutil
import sys
import tarfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

COMMIT = "0dbf0ece209b77d722588e5e7943014d3db48f7f"
TARBALL_SHA256 = "62a0a2855acd83788c124c8b7a5e82ed4d6fa2cf4e4005ed264ea1efeb65d53b"
URL = f"https://api.github.com/repos/alibaba-damo-academy/damo-radar/tarball/{COMMIT}"

# Paths inside the repository that the worker needs, relative to the repo root.
KEEP_PREFIXES = ("RADAR_inference/",)
KEEP_FILES = (
    "LICENSE",
    "THIRD_PARTY_LICENSES.md",
    "ckpt/infer_text_embedding_radar.pt",
)
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "worker" / "vendor" / "damo-radar"


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(tarball: Path | None) -> bytes:
    if tarball is not None:
        return tarball.read_bytes()
    req = urllib.request.Request(URL, headers={"User-Agent": "radar-desk-vendor"})
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def wanted(member_path: str) -> bool:
    if member_path in KEEP_FILES:
        return True
    return any(member_path.startswith(p) for p in KEEP_PREFIXES)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tarball", type=Path, default=None)
    args = parser.parse_args()

    data = fetch(args.tarball)
    digest = sha256_of(data)
    if digest != TARBALL_SHA256:
        print(f"tarball sha256 {digest} does not match {TARBALL_SHA256}", file=sys.stderr)
        return 1

    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)

    copied = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            rel = member.name.split("/", 1)[1] if "/" in member.name else member.name
            if not wanted(rel):
                continue
            target = DEST / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            assert src is not None
            target.write_bytes(src.read())
            copied += 1

    (DEST / "VENDORED.md").write_text(
        "# Vendored from alibaba-damo-academy/damo-radar\n\n"
        f"Commit `{COMMIT}`, fetched as a tarball on {datetime.now(tz=UTC).date().isoformat()}.\n\n"
        f"Tarball sha256 `{TARBALL_SHA256}`.\n\n"
        "Contents: `RADAR_inference/` (inference code), `ckpt/infer_text_embedding_radar.pt`, "
        "`LICENSE` (CC BY-NC-SA 4.0) and `THIRD_PARTY_LICENSES.md`. "
        "Nothing here is edited. Re-run `scripts/vendor_damo.py` to refresh.\n"
    )
    print(f"copied {copied} files to {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
