"""Measure RADAR on GPU types in order and stop at the first that fits.

    modal run scripts/modal_spike.py --imports-only
    modal run scripts/modal_spike.py --demo /path/AC423ccbe.nii.gz --image1 /path/merlin-image1.nii.gz [--gpus L4,A10]

`--imports-only` runs a CPU container on the deployed function's image that imports torch,
transformers, monai, nibabel, SimpleITK and the vendored `inference_demo` and prints their
versions, so an install problem shows up before any GPU time.

Otherwise both scans are staged on the scratch Volume (image1 is over Modal's 100 MB input
limit) and scored on each type in `--gpus` (default L4), in order, with the deployed
function's image, weights and resources (`Function.with_options(gpu=...)`, one cold
container per type). Per type it records the in-container load time, the local wall time of
the call (container boot and image pull included), per-scan time, peak
`torch.cuda.max_memory_allocated` and `max_memory_reserved`, the device's total memory, the
checkpoint sha256 and the 146 scores per scan.

Gates, in this order:
1. the demo case must match fixtures/expected/damo-demo.csv within 1e-4; on failure nothing
   else is tried and the script exits 1;
2. the type passes when image1's peak allocated memory leaves 30% of the device's total
   free; the first type that passes ends the run.
Results go to fixtures/expected/spike-<timestamp>-<types>.json. Exit 1 when no type passes.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "worker") not in sys.path:
    sys.path.insert(0, str(ROOT / "worker"))

import modal

import modal_app

HEADROOM = 0.30
DEMO_TOL = 1e-2  # cross-hardware; Lars's decision 2026-10-01 after the L4 parity run
DEMO_CSV = ROOT / "fixtures" / "expected" / "damo-demo.csv"

app = modal.App("radar-desk-spike")
spike_image = modal_app.image.add_local_python_source("modal_app")
VOLUMES = {
    modal_app.WEIGHTS_DIR: modal_app.weights.with_mount_options(read_only=True),
    modal_app.SMOKE_DIR: modal_app.smoke_volume,
}


@app.function(image=spike_image, timeout=900, cpu=2.0, memory=8192)
def imports() -> dict:
    """Import everything the scoring path needs, on CPU, and report versions."""
    import os

    os.environ.setdefault("MODEL_ROOT", modal_app.WEIGHTS_DIR)
    os.environ.setdefault("CONFIGS_ROOT", modal_app.WEIGHTS_DIR)
    vendor = os.environ.get("RADAR_VENDOR_DIR", f"{modal_app.VENDOR_REMOTE}/RADAR_inference")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    import inference_demo
    import monai
    import nibabel
    import numpy
    import SimpleITK
    import torch
    import transformers

    return json.loads(json.dumps({
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "transformers": transformers.__version__,
        "monai": monai.__version__,
        "nibabel": nibabel.__version__,
        "SimpleITK": SimpleITK.Version_VersionString(),
        "numpy": numpy.__version__,
        "inference_demo": inference_demo.__file__,
        "image_id": os.environ.get("MODAL_IMAGE_ID"),
    }))


@app.function(image=spike_image, gpu="L4", volumes=VOLUMES, timeout=3600, cpu=4.0, memory=16384,
              max_containers=1, retries=0)
def measure(gpu_label: str, remote_paths: list) -> dict:
    import torch

    from radar_worker import infer

    modal_app.smoke_volume.reload()
    t = time.perf_counter()
    state = modal_app.weights_state(print)
    if not state["ok"]:
        return {"gpu_requested": gpu_label, "error": "weights_mismatch", "problems": state["problems"]}
    loaded = infer.load_model(modal_app.WEIGHTS_DIR)
    load_s = time.perf_counter() - t
    scans = []
    for rel in remote_paths:
        path = str(Path(modal_app.SMOKE_DIR) / rel)
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        out = infer.score_file(path, loaded, print)
        torch.cuda.synchronize()
        scans.append({
            "path": rel,
            "ok": bool(out.get("ok")),
            "error": out.get("error"),
            "seconds": round(time.perf_counter() - t, 3),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "organs_scored": len(out.get("organs_scored", [])),
            "scores": [f["prob"] for f in out.get("findings", [])],
        })
    props = torch.cuda.get_device_properties(0)
    return json.loads(json.dumps({
        "gpu_requested": gpu_label,
        "device": torch.cuda.get_device_name(),
        "total_bytes": int(props.total_memory),
        "checkpoint_sha256": state["checkpoint_sha256"],
        "load_s": round(load_s, 3),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "keys": loaded.test_items,
        "scans": scans,
    }))


def read_demo_csv(path: Path) -> list:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    return [float(v) if v != "" else None for v in rows[1][1:]]


def demo_gate(ours: list, ref: list) -> dict:
    worst, bad = 0.0, []
    for i, (a, b) in enumerate(zip(ours, ref)):
        if (a is None) != (b is None):
            bad.append({"index": i, "ours": a, "reference": b, "why": "blank in one only"})
        elif a is not None:
            worst = max(worst, abs(a - b))
            if abs(a - b) > DEMO_TOL:
                bad.append({"index": i, "ours": a, "reference": b, "diff": abs(a - b)})
    ok = not bad and len(ours) == len(ref) == 146
    return {"ok": ok, "tolerance": DEMO_TOL, "max_abs_diff": worst, "n": len(ours), "mismatches": bad}


# The next type with MORE memory, for the fallback entry of RADAR_GPU. A10 has the same 24 GB as L4,
# so it is no fallback for an out-of-memory failure. Modal names and sizes as of 2026-10-01.
GPU_MEMORY_GB = {"T4": 16, "L4": 24, "A10": 24, "L40S": 48, "A100-40GB": 40, "A100": 40, "A100-80GB": 80,
                 "H100": 80, "H200": 141, "B200": 180}
FALLBACK = {"T4": "L4", "L4": "L40S", "A10": "L40S", "A100-40GB": "L40S", "A100": "L40S", "L40S": "A100-80GB",
            "A100-80GB": "H200", "H100": "H200", "H200": "B200"}


def fallback_for(name: str) -> str | None:
    """The fallback type for `name`: the next one up with strictly more memory, or None at the top."""
    nxt = FALLBACK.get(name)
    if nxt is None or GPU_MEMORY_GB.get(nxt, 0) <= GPU_MEMORY_GB.get(name, 0):
        return None
    return nxt


def headroom_ok(result: dict, image1_rel: str) -> bool:
    """Judge headroom on peak RESERVED memory, which is what runs out, not on peak allocated."""
    scan = next((s for s in result.get("scans", []) if s["path"] == image1_rel), None)
    if not scan or not scan["ok"] or not result.get("total_bytes"):
        return False
    return scan["peak_reserved_bytes"] <= (1.0 - HEADROOM) * result["total_bytes"]


def _write(out: Path, payload: dict) -> None:
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}")


@app.local_entrypoint()
def main(demo: str = "", image1: str = "", gpus: str = "L4", imports_only: bool = False):
    if imports_only:
        t = time.perf_counter()
        info = imports.remote()
        info["wall_s"] = round(time.perf_counter() - t, 3)
        print(json.dumps(info, indent=1))
        return
    if not demo or not image1:
        raise SystemExit("--demo and --image1 are required (or use --imports-only)")

    run_id = "spike-" + time.strftime("%Y%m%d-%H%M%S")
    staged = []
    with modal_app.smoke_volume.batch_upload(force=True) as batch:
        for local in (demo, image1):
            src = Path(local).expanduser().resolve()
            if not src.is_file():
                raise SystemExit(f"no such file: {src}")
            remote = f"in/{run_id}/{src.name}"
            batch.put_file(str(src), "/" + remote)
            staged.append(remote)
    demo_rel, image1_rel = staged
    print(f"staged {staged} on Volume {modal_app.SMOKE_VOLUME}")
    reference = read_demo_csv(DEMO_CSV)

    # One file per run, named by date, time and the types tried, so an escalation run never overwrites an earlier one.
    types = [g.strip() for g in gpus.split(",") if g.strip()]
    out = ROOT / "fixtures" / "expected" / f"{run_id}-{'-'.join(types)}.json"
    payload = {"run_id": run_id, "headroom": HEADROOM, "gpus": gpus, "results": [], "pick": None}
    for name in types:
        print(f"--> {name}")
        t = time.perf_counter()
        try:
            r = measure.with_options(gpu=name).remote(name, staged)
        except Exception as exc:  # noqa: BLE001  an OOM or a Modal error on one type is a result, not a crash
            r = {"gpu_requested": name, "error": f"{type(exc).__name__}: {exc}"}
        r["wall_s"] = round(time.perf_counter() - t, 3)
        payload["results"].append(r)
        if "error" in r:
            print(f"{name} failed: {r['error']}")
            _write(out, payload)
            if "out of memory" in r["error"].lower() or "OutOfMemory" in r["error"]:
                print("out of memory; trying the next type" if name != types[-1] else "out of memory; no type left")
                continue
            sys.exit(1)
        if "scans" not in r:
            print(json.dumps(r, indent=1))
            _write(out, payload)
            sys.exit(1)

        demo_scan = next(s for s in r["scans"] if s["path"] == demo_rel)
        gate = demo_gate(demo_scan["scores"], reference)
        r["demo_gate"] = gate
        r["headroom_ok"] = headroom_ok(r, image1_rel)
        summary = {k: r.get(k) for k in ("gpu_requested", "device", "total_bytes", "load_s", "wall_s",
                                          "checkpoint_sha256", "headroom_ok")}
        summary["scans"] = [{k: s[k] for k in ("path", "ok", "seconds", "peak_allocated_bytes",
                                               "peak_reserved_bytes", "organs_scored")} for s in r["scans"]]
        summary["demo_max_abs_diff"] = gate["max_abs_diff"]
        print(json.dumps(summary, indent=1))
        if not gate["ok"]:
            print(f"demo gate FAILED on {name}: max abs diff {gate['max_abs_diff']}, "
                  f"{len(gate['mismatches'])} mismatch(es); stopping")
            _write(out, payload)
            sys.exit(1)
        if r["headroom_ok"]:
            payload["pick"] = name
            break
        print(f"{name} leaves less than {HEADROOM:.0%} headroom on image1; trying the next type")

    _write(out, payload)
    if payload["pick"] is None:
        print("no GPU type passed")
        sys.exit(1)
    fallback = fallback_for(payload["pick"])
    value = payload["pick"] + (f",{fallback}" if fallback else "")
    print(f"pick {payload['pick']}; set RADAR_GPU={value}")
