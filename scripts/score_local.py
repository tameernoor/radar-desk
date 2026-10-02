"""Score one scan with the RADAR wrapper on this machine, outside Modal.

    worker/.venv/bin/python scripts/score_local.py --path scan.nii.gz --weights /path/to/weights \\
        --out result.json [--mask mask.nii.gz] [--device auto|cuda|mps|cpu] [--vendor DIR]

`--weights` holds checkpoint_radar_pretrain.pth, infer_text_embedding_radar.pt and
bert-base-chinese/. They are checked against worker/weights.json first (every size, the
checkpoint's sha256), as the Modal function does. Model load, inference and
post-processing are timed with wall clocks; peak memory follows the device's rule (see
radar_worker.infer.MemoryTracker). The result (findings, organs_scored, organs_not_found,
organ_stats, timings, versions, trace) is written as JSON to --out. A weights mismatch or an
input error is printed and written to --out as {ok: false, error}, and the exit code is 1.

On MPS, torch has no max_pool3d (the vendored VisionBranch pools organ masks with it in
every window), so PYTORCH_ENABLE_MPS_FALLBACK defaults to 1 here, set before torch is
imported; that op then runs on the CPU. Set it to 0 to see the missing op instead. Its value
is recorded in versions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "worker") not in sys.path:
    sys.path.insert(0, str(ROOT / "worker"))

from radar_worker import geometry, infer
from radar_worker.weights import check_weights, code_commit

MANIFEST = ROOT / "worker" / "weights.json"
DEFAULT_VENDOR = ROOT / "worker" / "vendor" / "damo-radar" / "RADAR_inference"


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", default="auto", choices=list(infer.DEVICES))
    ap.add_argument("--path", required=True, help="the scan, .nii or .nii.gz")
    ap.add_argument("--weights", required=True, help="folder with the checkpoint, embeddings and bert-base-chinese/")
    ap.add_argument("--out", required=True, help="where to write the result JSON")
    ap.add_argument("--mask", default=None, help="also write the label mask here (.nii.gz)")
    ap.add_argument("--vendor", default=str(DEFAULT_VENDOR), help="the vendored RADAR_inference folder")
    return ap.parse_args(argv)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def _fail(out: Path, klass: str, message: str) -> int:
    err = {"ok": False, "error": {"class": klass, "message": message}}
    _write(out, err)
    print(json.dumps(err, ensure_ascii=False, indent=1))
    return 1


def main(argv=None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # read when torch is imported
    out_path = Path(args.out).expanduser().resolve()
    scan = Path(args.path).expanduser().resolve()
    weights_dir = Path(args.weights).expanduser().resolve()
    vendor = Path(args.vendor).expanduser().resolve()
    if not scan.is_file():
        return _fail(out_path, "input_error", f"no such file: {scan}")
    if not (vendor / "inference_demo.py").is_file():
        return _fail(out_path, "input_error", f"no inference_demo.py under {vendor}")
    os.environ["RADAR_VENDOR_DIR"] = str(vendor)

    def log(msg: str) -> None:
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    t_total = time.perf_counter()
    t = time.perf_counter()
    state = check_weights(weights_dir, json.loads(MANIFEST.read_text()))
    weights_check_s = time.perf_counter() - t
    if not state["ok"]:
        return _fail(out_path, "weights_mismatch", "; ".join(state["problems"]))
    log(f"weights ok in {weights_check_s:.1f}s")

    try:
        device = infer.resolve_device(args.device)
    except ValueError as err:
        return _fail(out_path, "device_error", str(err))
    t = time.perf_counter()
    loaded = infer.load_model(str(weights_dir), device=device)
    load_s = time.perf_counter() - t
    log(f"model loaded on {device} in {load_s:.1f}s")

    out = infer.score_file(str(scan), loaded, log)
    if not out.get("ok"):
        err = out.get("error", {})
        return _fail(out_path, err.get("class", "input_error"), err.get("message", ""))

    t = time.perf_counter()
    if args.mask:
        geometry.write_mask(out["mask"], out["affine"], Path(args.mask).expanduser().resolve())
    write_s = time.perf_counter() - t

    import torch

    mps = getattr(getattr(torch, "backends", None), "mps", None)
    versions = {
        "code_commit": code_commit(vendor.parent / "VENDORED.md"),
        "checkpoint_sha256": state["checkpoint_sha256"],
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda) if torch.version.cuda else None,
        "gpu": infer.device_name(device),
        "device": device.type,
        "mps_available": bool(mps is not None and mps.is_available()),
        "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        "python": sys.version.split()[0],
        "image_id": None,
    }
    timings = {
        "weights_check_s": round(weights_check_s, 3),
        "load_s": round(load_s, 3),
        "infer_s": out["timings"]["infer_s"],
        "postprocess_s": out["timings"]["postprocess_s"],
        "mask_write_s": round(write_s, 3),
        "total_s": round(time.perf_counter() - t_total, 3),
    }
    result = {
        "ok": True,
        "file_name": out["file_name"],
        "findings": out["findings"],
        "organs_scored": out["organs_scored"],
        "organs_not_found": out["organs_not_found"],
        "organ_stats": out["organ_stats"],
        "timings": timings,
        "versions": versions,
        "memory": out["trace"]["memory"],
        "trace": out["trace"],
    }
    _write(out_path, result)
    scored = sum(f["prob"] is not None for f in out["findings"])
    log(f"{scored}/146 findings scored, {len(out['organs_scored'])} organs; timings {timings}")
    log(f"memory {out['trace']['memory']}")
    log(f"wrote {out_path}" + (f" and {args.mask}" if args.mask else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
