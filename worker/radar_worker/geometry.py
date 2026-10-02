"""Upstream RADAR preprocessing geometry, and its inverse, with numpy and nibabel only.

Mirrors `DataFolder.__getitem__` and the sliding window set-up in the vendored
`RADAR_inference/inference_demo.py`:

1. target size (H', W', D') = [int(h * sy), int(w * sx), int(d * sz / 5)], with
   upstream's swap of the x and y spacings,
2. trilinear resize to the target, transpose (0, 3, 2, 1) so array axes become (D', W', H'),
3. HU clip to [-300, 400] and min-max normalisation,
4. crop to the non-zero box of the normalised image with margins (5, 20, 20),
5. end padding to at least (96, 256, 384),
6. sliding windows of (96, 256, 384) with 25% overlap (MONAI `dense_patch_slices`).

The inverse puts a label mask from the padded space back on the original voxel grid.
Runs on Python 3.10 and 3.12.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel import orientations as nio

REF_SPACING = (1.0, 1.0, 5.0)
HU_MIN, HU_MAX = -300.0, 400.0
EXTEND_DWH = (5, 20, 20)
ROI_SIZE = (96, 256, 384)
OVERLAP = 0.25
LAS = ("L", "A", "S")

# Segmentation labels 1..36 in upstream order (`DataFolder.organs`), with the English
# names from upstream's commented `organ_dict`, first letter capitalised.
LABELS_ZH = [
    "肾上腺", "主动脉", "竖脊肌", "脑", "锁骨", "大肠", "十二指肠", "食管", "面部", "股骨",
    "胆囊", "臀肌", "心脏", "髋关节", "肱骨", "髂动脉", "髂静脉", "髂腰肌", "下腔静脉", "肾",
    "肝", "肺", "胰腺", "门静脉", "肺动脉", "肋骨", "骶骨", "肩胛骨", "小肠", "脾",
    "胃", "气管", "膀胱", "颈椎", "腰椎", "胸椎",
]
LABELS = [
    "Adrenal gland", "Aorta", "Erector spinae muscle", "Brain", "Clavicle", "Large bowel",
    "Duodenum", "Esophagus", "Face", "Femur", "Gallbladder", "Gluteus muscle", "Heart",
    "Hip joint", "Humerus", "Iliac artery", "Iliac vena", "Iliopsoas muscle",
    "Inferior vena cava", "Kidney", "Liver", "Lung", "Pancreas", "Portal vein",
    "Pulmonary artery", "Rib", "Sacrum", "Scapula", "Small bowel", "Spleen", "Stomach",
    "Trachea", "Bladder", "Cervical vertebrae", "Lumbar vertebrae", "Thoracic vertebrae",
]


def label_for(organ: str) -> int:
    """Segmentation label (1..36) for an English or Chinese organ name."""
    if organ in LABELS:
        return LABELS.index(organ) + 1
    return LABELS_ZH.index(organ) + 1


@dataclass
class Plan:
    """Everything needed to map between the original grid and upstream's padded space.

    Shapes in (H', W', D') order are the resampled original axes (i, j, k). Shapes and
    boxes named dwh are in the transposed space (D', W', H') = (k, j, i).
    """

    orig_shape_ijk: list
    spacing_xyz: list
    target_size: list
    crop_min_dwh: list
    crop_max_dwh: list
    crop_shape: list
    padded_shape: list
    windows: list
    orientation_in: str
    las_transform: list | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Plan:
        return cls(**d)


# ---------------------------------------------------------------- forward geometry


def target_size_for(shape_ijk, spacing_xyz) -> list:
    """Upstream's resize target, including its swap of the x and y spacing."""
    h, w, d = (int(s) for s in shape_ijk)
    scale = [float(spacing_xyz[i]) / REF_SPACING[i] for i in range(3)]
    return [int(h * scale[1]), int(w * scale[0]), int(d * scale[2])]


def get_scan_interval(image_size, roi_size=ROI_SIZE, overlap=OVERLAP) -> tuple:
    """Copy of upstream `_get_scan_interval`."""
    out = []
    for i in range(len(image_size)):
        if roi_size[i] == image_size[i]:
            out.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - overlap))
            out.append(interval if interval > 0 else 1)
    return tuple(out)


def dense_patch_slices(image_size, patch_size, scan_interval) -> list:
    """Copy of MONAI 1.4.0 `monai.data.utils.dense_patch_slices` (Apache-2.0), returning (start, end) pairs."""
    nd = len(image_size)
    patch_size = tuple(min(int(i), int(p) if p else int(i)) for i, p in zip(image_size, patch_size))
    scan_num = []
    for i in range(nd):
        if scan_interval[i] == 0:
            scan_num.append(1)
        else:
            num = math.ceil(float(image_size[i]) / scan_interval[i])
            scan_dim = next(
                (d for d in range(num) if d * scan_interval[i] + patch_size[i] >= image_size[i]), None
            )
            scan_num.append(scan_dim + 1 if scan_dim is not None else 1)
    starts = []
    for dim in range(nd):
        dim_starts = []
        for idx in range(scan_num[dim]):
            start = idx * scan_interval[dim]
            start -= max(start + patch_size[dim] - image_size[dim], 0)
            dim_starts.append(start)
        starts.append(dim_starts)
    grid = np.asarray([x.flatten() for x in np.meshgrid(*starts, indexing="ij")]).T
    return [tuple((int(s), int(s) + patch_size[d]) for d, s in enumerate(x)) for x in grid]


def sliding_windows(padded_shape) -> list:
    """Windows as [[z0, z1], [y0, y1], [x0, x1]] in the padded (D', W', H') space."""
    size = [int(s) for s in padded_shape]
    interval = get_scan_interval(size, ROI_SIZE, OVERLAP)
    return [[list(p) for p in win] for win in dense_patch_slices(size, ROI_SIZE, interval)]


def _linear_axis(a: np.ndarray, axis: int, out_n: int) -> np.ndarray:
    in_n = a.shape[axis]
    if in_n == out_n:
        return a
    # torch linear interpolation, align_corners=False: src = (dst + 0.5) * in / out - 0.5, floored at 0
    src = (np.arange(out_n, dtype=np.float64) + 0.5) * (in_n / out_n) - 0.5
    src = np.maximum(src, 0.0)
    i0 = np.minimum(np.floor(src).astype(np.int64), in_n - 1)
    i1 = np.where(i0 < in_n - 1, i0 + 1, i0)
    l1 = (src - i0).astype(np.float32)
    shape = [1] * a.ndim
    shape[axis] = out_n
    l1 = l1.reshape(shape)
    return np.take(a, i0, axis=axis) * (1.0 - l1) + np.take(a, i1, axis=axis) * l1


def linear_resample(vol: np.ndarray, out_shape) -> np.ndarray:
    """Trilinear resize equivalent to torch `interpolate(mode="trilinear", align_corners=False)`.

    Separable, in float32. Close to torch, not bit-identical. Only `forward_plan` uses it, to
    find the crop box without torch (tests and offline checks); the GPU side takes the box
    from upstream's own tensors.
    """
    out = np.asarray(vol, dtype=np.float32)
    for axis, n in enumerate(out_shape):
        out = _linear_axis(out, axis, int(n))
    return out.astype(np.float32, copy=False)


def nearest_resample(labels: np.ndarray, out_shape) -> np.ndarray:
    """Nearest neighbour with half-pixel centres: src = floor((dst + 0.5) * in / out), exact in integers."""
    idx = []
    for in_n, out_n in zip(labels.shape, out_shape):
        dst = np.arange(int(out_n), dtype=np.int64)
        src = ((2 * dst + 1) * int(in_n)) // (2 * int(out_n))
        idx.append(np.minimum(src, int(in_n) - 1))
    return labels[np.ix_(*idx)]


def crop_box_dwh(normalised_or_clipped_dwh: np.ndarray):
    """Upstream's non-zero box with margins, from the clipped (or normalised) (D', W', H') image.

    The normalised image is zero exactly where the clipped image equals its minimum.
    Returns (min_dwh, max_dwh) with max used as an exclusive bound, as upstream does.
    """
    img = normalised_or_clipped_dwh
    nz = img != img.min()
    if not nz.any():
        raise ValueError("the image has no voxel above its minimum after the HU clip")
    mins, maxs = [], []
    for axis in range(3):
        other = tuple(a for a in range(3) if a != axis)
        hit = np.nonzero(nz.any(axis=other))[0]
        mins.append(int(hit[0]))
        maxs.append(int(hit[-1]))
    lo = [max(m - e, 0) for m, e in zip(mins, EXTEND_DWH)]
    hi = [min(m + e, s) for m, e, s in zip(maxs, EXTEND_DWH, img.shape)]
    return lo, hi


def make_plan(orig_shape_ijk, spacing_xyz, crop_min_dwh, crop_max_dwh, orientation_in="LAS",
              las_transform=None) -> Plan:
    """Build a Plan from a known crop box (the GPU side passes upstream's real indices)."""
    target = target_size_for(orig_shape_ijk, spacing_xyz)
    lo = [int(v) for v in crop_min_dwh]
    hi = [int(v) for v in crop_max_dwh]
    crop_shape = [b - a for a, b in zip(lo, hi)]
    padded = [max(c, r) for c, r in zip(crop_shape, ROI_SIZE)]
    return Plan(
        orig_shape_ijk=[int(s) for s in orig_shape_ijk],
        spacing_xyz=[float(s) for s in spacing_xyz],
        target_size=target,
        crop_min_dwh=lo,
        crop_max_dwh=hi,
        crop_shape=crop_shape,
        padded_shape=padded,
        windows=sliding_windows(padded),
        orientation_in=str(orientation_in) if isinstance(orientation_in, str) else "".join(orientation_in),
        las_transform=None if las_transform is None else [[float(a), float(b)] for a, b in las_transform],
    )


def preprocess_dwh(volume_ijk: np.ndarray, spacing_xyz) -> np.ndarray:
    """Resize, transpose to (D', W', H') and clip to [-300, 400], as upstream does before normalising."""
    target = target_size_for(volume_ijk.shape, spacing_xyz)
    resized = linear_resample(volume_ijk, target)
    return np.clip(resized.transpose(2, 1, 0), HU_MIN, HU_MAX)


def forward_plan(volume_ijk: np.ndarray, spacing_xyz, orientation_codes="LAS", las_transform=None) -> Plan:
    """Upstream's preprocessing geometry for a volume as upstream loads it (already LAS).

    `orientation_codes` is the file's own orientation before any reorientation and
    `las_transform` the nibabel transform `to_las` applied, or None.
    """
    vol = np.asarray(volume_ijk)
    if vol.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {vol.shape}")
    lo, hi = crop_box_dwh(preprocess_dwh(vol, spacing_xyz))
    return make_plan(vol.shape, spacing_xyz, lo, hi, orientation_codes, las_transform)


def forward_labels(labels_ijk: np.ndarray, plan: Plan) -> np.ndarray:
    """Push a label volume through the same steps (nearest resize, transpose, crop, pad) into the padded space."""
    lab = nearest_resample(np.asarray(labels_ijk), plan.target_size).transpose(2, 1, 0)
    lo, hi = plan.crop_min_dwh, plan.crop_max_dwh
    crop = lab[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    out = np.zeros(plan.padded_shape, dtype=lab.dtype)
    out[: crop.shape[0], : crop.shape[1], : crop.shape[2]] = crop
    return out


def max_axis(plan: Plan) -> int:
    return max(plan.padded_shape)


# ---------------------------------------------------------------- inverse


def _invert_ornt(ornt) -> np.ndarray:
    ornt = np.asarray(ornt, dtype=float)
    inv = np.zeros_like(ornt)
    for in_axis, (out_axis, flip) in enumerate(ornt):
        inv[int(out_axis)] = [in_axis, flip]
    return inv


def _file_shape(plan: Plan) -> list:
    """Shape of the file's own array before `to_las` reoriented it."""
    if plan.las_transform is None:
        return list(plan.orig_shape_ijk)
    return [plan.orig_shape_ijk[int(out_axis)] for out_axis, _ in plan.las_transform]


def inverse_mask(mask_padded_dwh: np.ndarray, plan: Plan) -> np.ndarray:
    """Labels from the padded (D', W', H') space back on the file's original voxel grid, as uint8."""
    mask = np.asarray(mask_padded_dwh)
    cs = plan.crop_shape
    if any(m < c for m, c in zip(mask.shape, cs)):
        raise ValueError(f"mask {mask.shape} is smaller than the crop {cs}")
    h, w, d = plan.target_size
    full = np.zeros((d, w, h), dtype=np.uint8)
    lo = plan.crop_min_dwh
    full[lo[0]:lo[0] + cs[0], lo[1]:lo[1] + cs[1], lo[2]:lo[2] + cs[2]] = mask[: cs[0], : cs[1], : cs[2]]
    out = nearest_resample(full.transpose(2, 1, 0), plan.orig_shape_ijk)
    if plan.las_transform is not None:
        out = nio.apply_orientation(out, _invert_ornt(plan.las_transform))
    return np.ascontiguousarray(out, dtype=np.uint8)


def to_las(img: nib.Nifti1Image):
    """Reorient to LAS. Returns (image, transform) with transform None when the file is already LAS."""
    codes = nib.aff2axcodes(img.affine)
    if tuple(codes) == LAS:
        return img, None
    ornt = nio.ornt_transform(nio.io_orientation(img.affine), nio.axcodes2ornt(LAS))
    return img.as_reoriented(ornt), ornt.tolist()


def write_mask(mask_ijk: np.ndarray, affine, path) -> Path:
    """Save a uint8 label mask as NIfTI with the given (original) affine."""
    aff = np.asarray(affine, dtype=float)
    img = nib.Nifti1Image(np.asarray(mask_ijk, dtype=np.uint8), aff)
    img.set_data_dtype(np.uint8)
    img.set_sform(aff, code=1)
    img.set_qform(aff, code=1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(path))
    return path


# ---------------------------------------------------------------- stats and boxes


def _corners_mm(lo_idx, hi_idx, affine) -> list:
    """World-mm axis-aligned box of the index-space box [lo, hi], corners sorted per axis."""
    pts = np.array([[x, y, z, 1.0] for x in (lo_idx[0], hi_idx[0])
                    for y in (lo_idx[1], hi_idx[1]) for z in (lo_idx[2], hi_idx[2])])
    mm = (np.asarray(affine, dtype=float) @ pts.T).T[:, :3]
    return [[round(float(v), 3) for v in mm.min(axis=0)], [round(float(v), 3) for v in mm.max(axis=0)]]


def organ_stats(mask_ijk: np.ndarray, affine, catalog_labels=None) -> dict:
    """Per label present: voxel count, volume in mL, centroid and bounding box (voxel edges) in world mm."""
    names = list(catalog_labels) if catalog_labels is not None else LABELS
    mask = np.asarray(mask_ijk)
    aff = np.asarray(affine, dtype=float)
    voxel_ml = abs(float(np.linalg.det(aff[:3, :3]))) / 1000.0
    counts = np.bincount(mask.ravel(), minlength=len(names) + 1)
    out = {}
    for label in range(1, len(names) + 1):
        if label >= len(counts) or counts[label] == 0:
            continue
        coords = np.nonzero(mask == label)
        centre = [float(c.mean()) for c in coords] + [1.0]
        lo = [float(c.min()) - 0.5 for c in coords]
        hi = [float(c.max()) + 0.5 for c in coords]
        out[names[label - 1]] = {
            "voxels": int(counts[label]),
            "ml": round(int(counts[label]) * voxel_ml, 3),
            "centroid_mm": [round(float(v), 3) for v in (aff @ np.array(centre))[:3]],
            "bbox_mm": _corners_mm(lo, hi, aff),
        }
    return out


def box_to_mm(box_padded_dwh, plan: Plan, affine) -> list:
    """Map a window or crop [[z0, z1], [y0, y1], [x0, x1]] in padded space to world mm.

    The box is clipped to the crop (outside it the network saw padding), moved by the crop
    offset, transposed to (i, j, k), scaled from the 1 x 1 x 5 grid to original voxel edges,
    taken back through the LAS transform if one was applied, and mapped by the file's affine.
    """
    box = np.asarray(box_padded_dwh, dtype=float)
    lo_c = np.asarray(plan.crop_min_dwh, dtype=float)
    hi_c = np.asarray(plan.crop_max_dwh, dtype=float)
    lo = np.clip(box[:, 0] + lo_c, lo_c, hi_c)
    hi = np.clip(box[:, 1] + lo_c, lo, hi_c)
    lo_ijk, hi_ijk = lo[::-1], hi[::-1]
    scale = np.asarray(plan.orig_shape_ijk, dtype=float) / np.asarray(plan.target_size, dtype=float)
    lo_idx = lo_ijk * scale - 0.5
    hi_idx = hi_ijk * scale - 0.5
    aff = np.asarray(affine, dtype=float)
    if plan.las_transform is not None:
        aff = aff @ nio.inv_ornt_aff(np.asarray(plan.las_transform), _file_shape(plan))
    return _corners_mm(lo_idx, hi_idx, aff)
