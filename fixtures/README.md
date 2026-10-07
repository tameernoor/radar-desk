# Fixtures

The scans are not committed (about 470 MB together). `manifest.json` holds their sha256, so an upload of one of them is recognised and tagged, and the viewer shows the reference column.

| id | file | where it comes from |
|---|---|---|
| AC4214dbd, AC4240fff, AC4242a2f, AC4242a55 | `merlin-<id>.nii.gz` | `data/merlin_data_train_demo/resized_images/` in github.com/alibaba-damo-academy/damo-radar (already 1 x 1 x 5 mm, float32) |
| image1 | `merlin-image1.nii.gz` | huggingface.co/stanfordmimi/Merlin |
| AC423ccbe | `AC423ccbe.nii.gz` | `data/demo_cases/` in damo-radar, the upstream demo case |

Tests that need the files read `RADAR_FIXTURE_DIR` (default this `fixtures/` folder, where `*.nii.gz` is gitignored) and skip when it is missing.

## expected/

- `damo-demo.csv`. Upstream's `results/RADAR_infer_results_demo.csv`, one row for AC423ccbe with all 146 scores. Tolerance 1e-2 across hardware (set by Lars on 2026-10-01; tier 1, upstream against the wrapper on the same GPU, stays at 1e-6).
- `tally.json` (optional, gitignored). Your own spot checks per scan, `{"scans": {"<id>": {"Organ_Finding": prob}}}`. Shown as a reference column when present.
- `radar-web/`. Full radar-web JSON exports for the five scans. Pending; the parity tests that need them are skipped until the files are dropped in as `<id>.json`.
- `damo-resized-masks/`. The four TotalSegmentator-derived 36-label masks that ship with the AC cases, for the mask Dice check. Not reference output, training supervision.
- `spike-*.json`, `parity-*.json`. Written by `scripts/modal_spike.py` and `scripts/parity.py` when they run.

Model and data are CC BY-NC-SA 4.0 (Alibaba DAMO Academy). Research use only.
