"""Self parity: upstream `inference_demo.py`, unchanged, against `radar_worker.infer` on one scan.

    modal run scripts/parity.py --path /path/AC423ccbe.nii.gz [--case AC423ccbe]

In one GPU container: run upstream as a subprocess (MODEL_ROOT and CONFIGS_ROOT pointing at
the weights Volume, --img_dir a folder holding only this scan, cwd RADAR_inference so its
`../ckpt/infer_text_embedding_radar.pt` resolves to the vendored copy), then load our wrapper
and score the same file. All 146 scores must agree within 1e-6 and be blank in the same
places. For the demo case the scores are also compared with upstream's published
`results/RADAR_infer_results_demo.csv` (fixtures/expected/damo-demo.csv) at 1e-2.
Writes fixtures/expected/parity-<case>.json and exits 1 on a failure.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "worker") not in sys.path:
    sys.path.insert(0, str(ROOT / "worker"))

import modal

import modal_app

SELF_TOL = 1e-6
DEMO_TOL = 1e-2  # cross-hardware; Lars's decision 2026-10-01 after the L4 parity run
DEMO_CSV = ROOT / "fixtures" / "expected" / "damo-demo.csv"

app = modal.App("radar-desk-parity")
parity_image = modal_app.image.add_local_python_source("modal_app")


def read_upstream_csv(path) -> tuple:
    """(header, first row) of an upstream results CSV; values as float or None."""
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        return (rows[0] if rows else []), None, None
    header, row = rows[0], rows[1]
    values = [float(v) if v != "" else None for v in row[1:]]
    return header, row[0], values


def compare_values(keys: list, ours: list, theirs: list, tol: float) -> dict:
    worst, mismatches = 0.0, []
    for key, a, b in zip(keys, ours, theirs):
        if (a is None) != (b is None):
            mismatches.append({"key": key, "ours": a, "upstream": b, "why": "blank in one only"})
            continue
        if a is None:
            continue
        diff = abs(a - b)
        worst = max(worst, diff)
        if diff > tol:
            mismatches.append({"key": key, "ours": a, "upstream": b, "diff": diff})
    return {"n": len(keys), "tolerance": tol, "max_abs_diff": worst, "mismatches": mismatches,
            "ok": not mismatches and len(ours) == len(theirs)}


@app.function(
    image=parity_image,
    gpu=modal_app.RADAR_GPU,
    volumes={
        modal_app.WEIGHTS_DIR: modal_app.weights.with_mount_options(read_only=True),
        modal_app.SMOKE_DIR: modal_app.smoke_volume,
    },
    timeout=3600,
    cpu=4.0,
    memory=16384,
    max_containers=1,
)
def compare(volume_path: str) -> dict:
    from radar_worker import infer

    modal_app.smoke_volume.reload()
    src = Path(modal_app.SMOKE_DIR) / volume_path
    state = modal_app.weights_state(print)
    if not state["ok"]:
        return {"ok": False, "error": "weights_mismatch", "problems": state["problems"]}

    work = Path(tempfile.mkdtemp(prefix="parity-"))
    img_dir, save_dir = work / "img", work / "results"
    img_dir.mkdir()
    scan = img_dir / src.name
    shutil.copyfile(src, scan)

    # 1. upstream, unchanged
    env = dict(os.environ, MODEL_ROOT=modal_app.WEIGHTS_DIR, CONFIGS_ROOT=modal_app.WEIGHTS_DIR)
    t = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "inference_demo.py", "--img_dir", str(img_dir), "--save_dir", str(save_dir),
         "--save_tag", "parity"],
        cwd=f"{modal_app.VENDOR_REMOTE}/RADAR_inference", env=env, capture_output=True, text=True, check=False,
    )
    upstream_s = time.perf_counter() - t
    if proc.returncode != 0:
        return {"ok": False, "error": "upstream failed", "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:]}
    csv_path = save_dir / "RADAR_infer_results_parity.csv"
    if not csv_path.is_file():
        return {"ok": False, "error": "upstream wrote no row", "stdout": proc.stdout[-4000:]}
    header, up_name, up_values = read_upstream_csv(csv_path)
    if up_values is None:
        return {"ok": False, "error": "upstream wrote no row", "stdout": proc.stdout[-4000:]}

    # 2. ours, same file, same container
    t = time.perf_counter()
    loaded = infer.load_model(modal_app.WEIGHTS_DIR)
    out = infer.score_file(str(scan), loaded, print)
    ours_s = time.perf_counter() - t
    if not out.get("ok"):
        return {"ok": False, "error": "wrapper input error", "detail": out.get("error"), "upstream": up_values}
    ours = [f["prob"] for f in out["findings"]]
    result = compare_values(loaded.test_items, ours, up_values, SELF_TOL)
    result.update({
        "header_matches": header == loaded.csv_header(),
        "upstream_file_name": up_name,
        "upstream_s": round(upstream_s, 3),
        "ours_s": round(ours_s, 3),
        "keys": loaded.test_items,
        "ours": ours,
        "upstream": up_values,
        "organs_scored": out["organs_scored"],
        "organs_not_found": out["organs_not_found"],
        "crops": out["trace"]["crops"],
        "versions": modal_app.versions(state["checkpoint_sha256"]),
    })
    result["ok"] = result["ok"] and result["header_matches"]
    shutil.rmtree(work, ignore_errors=True)
    return json.loads(json.dumps(result))  # plain types only; the laptop has no torch


@app.local_entrypoint()
def main(path: str, case: str = ""):
    src = Path(path).expanduser().resolve()
    if not src.is_file():
        raise SystemExit(f"no such file: {src}")
    case = case or src.name.split(".")[0]
    remote = f"in/parity-{time.strftime('%Y%m%d-%H%M%S')}/{src.name}"
    with modal_app.smoke_volume.batch_upload(force=True) as batch:
        batch.put_file(str(src), "/" + remote)
    result = compare.remote(remote)

    if result.get("ours") and case.startswith("AC423ccbe") and DEMO_CSV.is_file():
        _, _, demo = read_upstream_csv(DEMO_CSV)
        result["demo_reference"] = compare_values(result["keys"], result["ours"], demo, DEMO_TOL)

    out = ROOT / "fixtures" / "expected" / f"parity-{case}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: result.get(k) for k in ("ok", "max_abs_diff", "mismatches", "header_matches",
                                                  "demo_reference", "error")}, ensure_ascii=False, indent=1))
    print(f"wrote {out}")
    ok = result.get("ok") and result.get("demo_reference", {"ok": True})["ok"]
    if not ok:
        sys.exit(1)
