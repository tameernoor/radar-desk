"""Tests for worker/radar_worker/geometry.py: upstream preprocessing geometry and its inverse."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from radar_worker import geometry as g
from synth import affine_for, ellipsoid, make_label_volume

ROOT = Path(__file__).resolve().parents[1]
VENDORED_DEMO = ROOT / "worker/vendor/damo-radar/RADAR_inference/inference_demo.py"


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    denom = a.sum() + b.sum()
    return 1.0 if denom == 0 else 2.0 * np.logical_and(a, b).sum() / denom


def body_volume(shape, block=None, air=-1000.0, tissue=40.0):
    """Air everywhere, tissue inside `block` ((i0,i1),(j0,j1),(k0,k1)), or a body cylinder if None."""
    vol = np.full(shape, air, dtype=np.float32)
    if block is None:
        cx, cy = (shape[0] - 1) / 2, (shape[1] - 1) / 2
        r = min(shape[0], shape[1]) * 0.45
        vol[ellipsoid(shape, (cx, cy, (shape[2] - 1) / 2), (r, r, shape[2]))] = tissue
    else:
        (i0, i1), (j0, j1), (k0, k1) = block
        vol[i0:i1, j0:j1, k0:k1] = tissue
    return vol


# ---------------------------------------------------------------- labels


def test_label_list_matches_vendored_source():
    tree = ast.parse(VENDORED_DEMO.read_text(encoding="utf-8"))
    organs = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "DataFolder":
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Assign) and isinstance(sub.targets[0], ast.Attribute)
                        and sub.targets[0].attr == "organs"):
                    organs = ast.literal_eval(sub.value)
    assert organs is not None
    assert g.LABELS_ZH == organs
    assert len(g.LABELS) == 36
    assert g.LABELS[20] == "Liver" and g.label_for("Liver") == 21
    assert g.LABELS[5] == "Large bowel" and g.LABELS[0] == "Adrenal gland"


def test_scored_organ_names_agree_with_english_mapping():
    tree = ast.parse(VENDORED_DEMO.read_text(encoding="utf-8"))
    mapping = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Attribute)
                and node.targets[0].attr == "english_mapping"):
            mapping = ast.literal_eval(node.value)
            break
    assert mapping is not None and len(mapping) == 146
    organs = {}
    for key, english in mapping.items():
        organs.setdefault(key.split("_")[0], set()).add(english.split("_", 1)[0])
    assert len(organs) == 18
    for zh, names in organs.items():
        assert names == {g.LABELS[g.LABELS_ZH.index(zh)]}, (zh, names)


# ---------------------------------------------------------------- forward plan


def test_target_size_reproduces_upstream_index_swap():
    vol = body_volume((100, 200, 30))
    plan = g.forward_plan(vol, (0.8, 0.6, 2.0), "LAS")
    # upstream: [int(h * sy), int(w * sx), int(d * sz / 5)]
    assert plan.target_size == [int(100 * 0.6), int(200 * 0.8), int(30 * 2.0 / 5.0)] == [60, 160, 12]
    assert plan.orig_shape_ijk == [100, 200, 30]


def test_crop_box_margins_and_end_padding():
    shape = (100, 120, 20)
    vol = body_volume(shape, block=((30, 60), (40, 80), (5, 10)))
    plan = g.forward_plan(vol, (1.0, 1.0, 5.0), "LAS")
    assert plan.target_size == [100, 120, 20]
    # transposed space (D', W', H') = (k, j, i); non-zero box is k 5..9, j 40..79, i 30..59 inclusive
    assert plan.crop_min_dwh == [0, 20, 10]
    # upstream adds the margin to the inclusive max and uses it as an exclusive bound, capped at the shape
    assert plan.crop_max_dwh == [14, 99, 79]
    assert plan.crop_shape == [14, 79, 69]
    assert plan.padded_shape == [96, 256, 384]
    assert plan.windows == [[[0, 96], [0, 256], [0, 384]]]
    assert plan.orientation_in == "LAS" and plan.las_transform is None


def test_crop_zero_only_where_value_equals_minimum():
    shape = (60, 60, 10)
    # no voxel at or below -300: the minimum is soft tissue, so only it normalises to zero
    vol = np.full(shape, 40.0, dtype=np.float32)
    vol[25:35, 20:30, 4:6] = 100.0
    plan = g.forward_plan(vol, (1.0, 1.0, 5.0), "LAS")
    assert plan.crop_min_dwh == [0, 0, 5]
    assert plan.crop_max_dwh == [10, 49, 54]
    # values below -300 are clipped to -300 and count as zero, values just above do not
    vol2 = np.full(shape, -500.0, dtype=np.float32)
    vol2[10:12, 10:12, 3] = -299.0
    plan2 = g.forward_plan(vol2, (1.0, 1.0, 5.0), "LAS")
    assert plan2.crop_min_dwh == [0, 0, 0]
    assert plan2.crop_max_dwh == [8, 31, 31]


def test_constant_volume_is_rejected():
    with pytest.raises(ValueError):
        g.forward_plan(np.zeros((20, 20, 5), np.float32), (1.0, 1.0, 5.0), "LAS")


def test_padded_shape_keeps_larger_axes():
    plan = g.make_plan([400, 300, 130], [1.0, 1.0, 5.0], [0, 0, 0], [120, 300, 390], "LAS")
    assert plan.crop_shape == [120, 300, 390]
    assert plan.padded_shape == [120, 300, 390]
    assert g.max_axis(plan) == 390


def test_plan_is_json_serialisable_and_round_trips():
    plan = g.forward_plan(body_volume((40, 40, 10)), (1.0, 1.0, 5.0), "RAS", [[0, -1], [1, 1], [2, 1]])
    text = json.dumps(plan.to_dict())
    back = g.Plan.from_dict(json.loads(text))
    assert back == plan


# ---------------------------------------------------------------- sliding windows


def test_scan_interval_copy():
    assert g.get_scan_interval((96, 256, 384), (96, 256, 384), 0.25) == (96, 256, 384)
    assert g.get_scan_interval((100, 300, 400), (96, 256, 384), 0.25) == (72, 192, 288)
    assert g.get_scan_interval((96, 300, 384), (96, 256, 384), 0.25) == (96, 192, 384)


def test_windows_single():
    assert g.sliding_windows((96, 256, 384)) == [[[0, 96], [0, 256], [0, 384]]]


def test_windows_two_by_one_by_two():
    wins = g.sliding_windows((100, 256, 400))
    assert wins == [
        [[0, 96], [0, 256], [0, 384]],
        [[0, 96], [0, 256], [16, 400]],
        [[4, 100], [0, 256], [0, 384]],
        [[4, 100], [0, 256], [16, 400]],
    ]


def test_windows_three_by_two_by_three():
    wins = g.sliding_windows((200, 300, 800))
    zs = sorted({w[0][0] for w in wins})
    ys = sorted({w[1][0] for w in wins})
    xs = sorted({w[2][0] for w in wins})
    assert (zs, ys, xs) == ([0, 72, 104], [0, 44], [0, 288, 416])
    assert len(wins) == 18
    # MONAI order: first axis slowest
    assert wins[0] == [[0, 96], [0, 256], [0, 384]]
    assert wins[1] == [[0, 96], [0, 256], [288, 672]]
    assert wins[3] == [[0, 96], [44, 300], [0, 384]]
    assert wins[-1] == [[104, 200], [44, 300], [416, 800]]


# ---------------------------------------------------------------- resampling


def test_linear_resample_matches_torch_half_pixel_rule():
    a = np.array([0.0, 1.0], dtype=np.float32).reshape(2, 1, 1)
    up = g.linear_resample(a, (4, 1, 1)).ravel()
    np.testing.assert_allclose(up, [0.0, 0.25, 0.75, 1.0], atol=1e-7)
    b = np.arange(4, dtype=np.float32).reshape(1, 4, 1)
    down = g.linear_resample(b, (1, 2, 1)).ravel()
    np.testing.assert_allclose(down, [0.5, 2.5], atol=1e-7)


def test_nearest_resample_half_pixel_centres():
    a = np.arange(6, dtype=np.uint8).reshape(6, 1, 1)
    # src = floor((dst + 0.5) * 6 / 4) -> 0, 2, 3, 5
    assert g.nearest_resample(a, (4, 1, 1)).ravel().tolist() == [0, 2, 3, 5]
    # src = floor((dst + 0.5) * 3 / 7) -> 0, 0, 1, 1, 1, 2, 2
    b = np.arange(3, dtype=np.uint8).reshape(1, 1, 3)
    assert g.nearest_resample(b, (1, 1, 7)).ravel().tolist() == [0, 0, 1, 1, 1, 2, 2]
    c = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    assert np.array_equal(g.nearest_resample(c, (2, 3, 4)), c)


# ---------------------------------------------------------------- round trip


BLOBS_FULL = [
    ((20, 22, 8), (9, 10, 4), 21),   # Liver
    ((44, 40, 10), (8, 9, 4), 30),   # Spleen
    ((32, 44, 4), (7, 6, 3), 20),    # Kidney
]


def _round_trip(shape, spacing, blobs):
    vol = body_volume(shape)
    labels = make_label_volume(shape, blobs)
    plan = g.forward_plan(vol, spacing, "LAS")
    stitched = g.forward_labels(labels, plan)
    assert list(stitched.shape) == plan.padded_shape
    back = g.inverse_mask(stitched, plan)
    return labels, back, plan


def test_round_trip_identity_when_target_equals_original():
    labels, back, plan = _round_trip((64, 64, 16), (1.0, 1.0, 5.0), BLOBS_FULL)
    assert plan.target_size == [64, 64, 16]
    assert back.shape == labels.shape and back.dtype == np.uint8
    for _, _, lab in BLOBS_FULL:
        assert dice(labels == lab, back == lab) == 1.0
    assert np.array_equal(back, labels)


def test_round_trip_two_times_downsample():
    # nearest down then up moves about half a voxel of every surface, so blobs need radii near 30
    shape = (160, 160, 64)
    blobs = [((50, 56, 30), (30, 32, 26), 21), ((112, 104, 34), (30, 30, 26), 30)]
    labels, back, plan = _round_trip(shape, (0.5, 0.5, 2.5), blobs)
    assert plan.target_size == [80, 80, 32]
    assert back.shape == labels.shape and back.dtype == np.uint8
    for _, _, lab in blobs:
        assert dice(labels == lab, back == lab) >= 0.95, lab


def test_write_mask_keeps_original_affine(tmp_path):
    shape = (64, 64, 16)
    aff = affine_for((1.0, 1.0, 5.0), "LAS", origin=(10.0, -20.0, 30.0))
    labels, back, _ = _round_trip(shape, (1.0, 1.0, 5.0), BLOBS_FULL)
    path = g.write_mask(back, aff, tmp_path / "mask.nii.gz")
    img = nib.load(str(path))
    assert img.shape == shape
    assert img.get_data_dtype() == np.uint8
    np.testing.assert_allclose(img.affine, aff)
    assert np.array_equal(np.asanyarray(img.dataobj), labels)


# ---------------------------------------------------------------- reorientation


def test_to_las_leaves_las_alone():
    img = nib.Nifti1Image(np.zeros((4, 5, 6), np.int16), affine_for((1.0, 1.0, 5.0), "LAS"))
    out, ornt = g.to_las(img)
    assert ornt is None and out is img


def test_ras_input_round_trips_through_las():
    shape = (64, 64, 16)
    aff = affine_for((1.0, 1.0, 5.0), "RAS", origin=(-30.0, 5.0, 0.0))
    ct = body_volume(shape)
    labels = make_label_volume(shape, BLOBS_FULL)
    img = nib.Nifti1Image(ct, aff)
    las_img, ornt = g.to_las(img)
    assert ornt is not None
    assert "".join(nib.aff2axcodes(las_img.affine)) == "LAS"
    las_ct = np.asanyarray(las_img.dataobj)
    # the first axis is flipped
    assert np.array_equal(las_ct, ct[::-1])
    plan = g.forward_plan(las_ct, (1.0, 1.0, 5.0), "RAS", ornt)
    las_labels = labels[::-1]
    back = g.inverse_mask(g.forward_labels(las_labels, plan), plan)
    assert back.shape == labels.shape
    assert np.array_equal(back, labels)
    # the same voxel lands on the same world point
    ijk = np.array([20, 22, 8, 1.0])
    las_ijk = np.array([shape[0] - 1 - 20, 22, 8, 1.0])
    np.testing.assert_allclose(aff @ ijk, las_img.affine @ las_ijk)


def test_permuted_input_round_trips():
    # axes stored as (j, i, k) with flips: PRS-like orientation
    shape = (40, 48, 12)
    aff = np.array([[0.0, -1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 5.0, 0.0], [0, 0, 0, 1.0]])
    labels = make_label_volume(shape, [((20, 24, 6), (8, 9, 3), 21)])
    img = nib.Nifti1Image(body_volume(shape), aff)
    las_img, ornt = g.to_las(img)
    las_ct = np.asanyarray(las_img.dataobj)
    assert las_ct.shape == (48, 40, 12)
    plan = g.forward_plan(las_ct, (1.0, 1.0, 5.0), "".join(nib.aff2axcodes(aff)), ornt)
    las_labels = nib.orientations.apply_orientation(labels, np.array(ornt))
    back = g.inverse_mask(g.forward_labels(las_labels, plan), plan)
    assert np.array_equal(back, labels)


# ---------------------------------------------------------------- stats and boxes


def test_organ_stats_hand_computed():
    mask = np.zeros((20, 20, 10), np.uint8)
    mask[2:6, 4:8, 1:3] = 21
    aff = np.diag([-2.0, 2.0, 5.0, 1.0])
    aff[0, 3] = 10.0
    stats = g.organ_stats(mask, aff)
    assert list(stats) == ["Liver"]
    s = stats["Liver"]
    assert s["voxels"] == 32
    assert s["ml"] == pytest.approx(0.64)
    assert s["centroid_mm"] == pytest.approx([3.0, 11.0, 7.5])
    assert s["bbox_mm"] == [pytest.approx([-1.0, 7.0, 2.5]), pytest.approx([7.0, 15.0, 12.5])]


def test_organ_stats_skips_background_and_unknown_labels():
    mask = np.zeros((5, 5, 5), np.uint8)
    mask[0, 0, 0] = 40
    mask[1, 1, 1] = 1
    stats = g.organ_stats(mask, np.eye(4))
    assert list(stats) == ["Adrenal gland"]


def test_box_to_mm_hand_computed():
    shape = (100, 100, 20)
    vol = body_volume(shape, block=((30, 50), (30, 50), (7, 12)))
    plan = g.forward_plan(vol, (1.0, 1.0, 5.0), "LAS")
    assert plan.crop_min_dwh == [2, 10, 10] and plan.crop_max_dwh == [16, 69, 69]
    aff = np.diag([-1.0, 1.0, 5.0, 1.0])
    box = g.box_to_mm([[0, 5], [0, 50], [0, 60]], plan, aff)
    # padded -> resampled: z 2..7, y 10..60, x 10..69 (x clipped to the crop); edges at index - 0.5
    assert box == [pytest.approx([-68.5, 9.5, 7.5]), pytest.approx([-9.5, 59.5, 32.5])]


def test_box_to_mm_scales_back_to_original_grid():
    shape = (80, 80, 10)
    vol = body_volume(shape)
    plan = g.forward_plan(vol, (0.5, 0.5, 2.5), "LAS")
    assert plan.target_size == [40, 40, 5]
    aff = np.diag([0.5, 0.5, 2.5, 1.0])
    z0, y0, x0 = plan.crop_min_dwh
    box = g.box_to_mm([[0, 2], [0, 4], [0, 6]], plan, aff)
    # resampled edges (x0..x0+6, y0..y0+4, z0..z0+2) scale by 2 in voxels, then 0.5/0.5/2.5 mm
    lo = [(2 * x0 - 0.5) * 0.5, (2 * y0 - 0.5) * 0.5, (2 * z0 - 0.5) * 2.5]
    hi = [(2 * (x0 + 6) - 0.5) * 0.5, (2 * (y0 + 4) - 0.5) * 0.5, (2 * (z0 + 2) - 0.5) * 2.5]
    assert box == [pytest.approx(lo), pytest.approx(hi)]


def test_box_to_mm_undoes_las_transform():
    shape = (64, 64, 16)
    aff = affine_for((1.0, 1.0, 5.0), "RAS")
    img = nib.Nifti1Image(body_volume(shape), aff)
    las_img, ornt = g.to_las(img)
    las_ct = np.asanyarray(las_img.dataobj)
    plan_ras = g.forward_plan(las_ct, (1.0, 1.0, 5.0), "RAS", ornt)
    plan_las = g.forward_plan(las_ct, (1.0, 1.0, 5.0), "LAS", None)
    box = [[0, 4], [0, 10], [0, 12]]
    assert g.box_to_mm(box, plan_ras, aff) == g.box_to_mm(box, plan_las, las_img.affine)
