"""Make sure the RADAR weights are on disk before a worker starts, outside Modal.

    python scripts/ensure_weights.py [--dir DIR] [--fallback DIR] [--manifest worker/weights.json]

Uses the weights in --dir if they match worker/weights.json. Otherwise fills --dir from
Hugging Face at the pinned revision (HF_TOKEN is sent if set), or --fallback when --dir is
missing or read-only. Prints the report as JSON; exit 1 on failure.

The defaults fit a RunPod pod, where a network volume mounts at /workspace: --dir is
$RADAR_WEIGHTS_DIR or /workspace/radar-weights, --fallback is $RADAR_WEIGHTS_FALLBACK or
~/radar-weights (container disk).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = "/workspace/radar-weights"  # RunPod pods mount the network volume at /workspace
DEFAULT_FALLBACK = str(Path.home() / "radar-weights")
if str(ROOT / "worker") not in sys.path:
    sys.path.insert(0, str(ROOT / "worker"))

from radar_worker.weights import ensure_weights


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dir", default=os.environ.get("RADAR_WEIGHTS_DIR") or DEFAULT_DIR,
                   help="weights directory, usually on a network volume")
    p.add_argument("--fallback", default=os.environ.get("RADAR_WEIGHTS_FALLBACK") or DEFAULT_FALLBACK,
                   help="local directory used when --dir is missing or read-only")
    p.add_argument("--manifest", default=os.environ.get("RADAR_WEIGHTS_MANIFEST") or str(ROOT / "worker" / "weights.json"))
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    log = lambda msg: print(msg, file=sys.stderr, flush=True)
    try:
        manifest = json.loads(Path(args.manifest).read_text())
        report = ensure_weights(args.dir, manifest, fallback_dir=args.fallback, log=log)
    except Exception as exc:  # noqa: BLE001 (report every failure the same way)
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 1
    print(json.dumps({"ok": True, **report}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
