"""RADAR inference that mirrors upstream `evaluate()` and keeps what it throws away.

Runs only in the GPU container. `score_file` follows `DataFolder.__getitem__` and
`evaluate()` in the vendored `RADAR_inference/inference_demo.py` step by step, with the same
constants, and calls upstream's own `forward_test_win`, `center_crop`, `_get_scan_interval`,
pad functions and MONAI's `dense_patch_slices`. On top it records the crop indices, the
window list, which window or centred crop scored each organ, the stitched mask and the raw
probabilities. Nothing in the vendored tree is edited.

Deviations from upstream, all outside the numeric path:
- one file per call instead of a DataLoader over a folder (upstream uses batch size 1 and
  runs `__getitem__` in worker processes; the values are the same);
- the text embeddings are loaded once from `weights_dir` instead of `../ckpt/` per call;
- a file that is not LAS is reoriented to LAS first (upstream has no reorientation);
- the unused intact-organ computation after stitching is left out (its result is never read);
- input problems return {ok: false, error: input_error} where upstream would crash or skip.

torch, monai and the vendored module are imported inside functions so this file parses
and imports without them.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

import nibabel as nib
import numpy as np

from . import geometry

DEFAULT_VENDOR_DIR = "/root/damo-radar/RADAR_inference"
TEXT_EMBEDDINGS = "infer_text_embedding_radar.pt"

# upstream constants (DataFolder.__getitem__ and evaluate)
REF_SPACING = (1.0, 1.0, 5.0)
EXTEND_D = 5
EXTEND_HW = 20
SW_BATCH_SIZE = 1
OVERLAP = 0.25
ROI_SIZE = (96, 256, 384)
MAX_AXIS = 1000

_LOADED = None


class Loaded:
    """Everything `load_model` builds once per container."""

    def __init__(self, upstream, pad_func, model, text_feat_dict, datafolder, weights_dir):
        self.upstream = upstream
        self.pad_func = pad_func
        self.model = model
        self.text_feat_dict = text_feat_dict
        self.datafolder = datafolder
        self.weights_dir = weights_dir
        self.test_items = list(datafolder.test_items)
        self.organs = list(datafolder.organs)
        self.english_mapping = dict(datafolder.english_mapping)
        self.test_organs = datafolder.test_organs
        # organ_zh -> English organ name, from the finding keys
        self.organ_en = {}
        for item in self.test_items:
            self.organ_en.setdefault(item.split("_")[0], self.english_mapping[item].split("_", 1)[0])

    def csv_header(self) -> list:
        """Upstream's CSV columns."""
        return ["file_name"] + [f"{k} ({self.english_mapping[k]})" for k in self.test_items]


def load_model(weights_dir: str) -> Loaded:
    """Load RADAR once per process, the way upstream `initialize()` does, and cache it."""
    global _LOADED
    if _LOADED is not None and _LOADED.weights_dir == weights_dir:
        return _LOADED
    os.environ["MODEL_ROOT"] = weights_dir
    os.environ["CONFIGS_ROOT"] = weights_dir
    vendor = os.environ.get("RADAR_VENDOR_DIR", DEFAULT_VENDOR_DIR)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)

    import inference_demo as upstream  # reads MODEL_ROOT and CONFIGS_ROOT at import
    import torch

    pad_func, model = upstream.initialize()
    # upstream: torch.load('../ckpt/infer_text_embedding_radar.pt'), same call, absolute path
    text_feat_dict = torch.load(os.path.join(weights_dir, TEXT_EMBEDDINGS))
    empty = tempfile.mkdtemp(prefix="radar-catalog-")
    try:
        datafolder = upstream.DataFolder(empty)
    finally:
        shutil.rmtree(empty, ignore_errors=True)
    _LOADED = Loaded(upstream, pad_func, model, text_feat_dict, datafolder, weights_dir)
    return _LOADED


def _input_error(message: str) -> dict:
    return {"ok": False, "error": {"class": "input_error", "message": message}}


def _center_crop_box(masks_to_boxes_3d, image, mask, crop_size) -> list:
    """Local copy of upstream `center_crop`'s index arithmetic, on the same tensors.

    Returns [[z_start, z_end], [y_start, y_end], [x_start, x_end]].
    """
    x_min, y_min, z_min, x_max, y_max, z_max = masks_to_boxes_3d(mask)[0].long()

    crop_d, crop_h, crop_w = (max(crop_size[0], z_max - z_min), max(crop_size[1], y_max - y_min),
                              max(crop_size[2], x_max - x_min))

    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    cz = (z_min + z_max) // 2

    d, h, w = image.shape[-3:]

    x_start = max(0, cx - crop_w // 2)
    x_end = min(w, x_start + crop_w)
    if x_end - x_start < crop_w:
        x_start = max(0, x_end - crop_w)

    y_start = max(0, cy - crop_h // 2)
    y_end = min(h, y_start + crop_h)
    if y_end - y_start < crop_h:
        y_start = max(0, y_end - crop_h)

    z_start = max(0, cz - crop_d // 2)
    z_end = min(d, z_start + crop_d)
    if z_end - z_start < crop_d:
        z_start = max(0, z_end - crop_d)

    return [[int(z_start), int(z_end)], [int(y_start), int(y_end)], [int(x_start), int(x_end)]]


def _preprocess(src_path: str, loaded: Loaded):
    """Mirror of `DataFolder.__getitem__`. Returns (padded tensor [C, D, W, H], facts) or an error string."""
    import torch
    from monai import transforms

    # load image
    data = {"image": src_path}
    res = transforms.LoadImaged(keys=["image"], image_only=False, ensure_channel_first=True)(data)
    image = res["image"]

    affine = res["image_meta_dict"]["affine"]
    spacing = (
        abs(affine[0, 0].item()),
        abs(affine[1, 1].item()),
        abs(affine[2, 2].item())
    )
    if image.dim() != 4:
        return None, f"expected one 3D volume, got array shape {tuple(image.shape)}"
    _, h, w, d = image.shape

    ref_spacing = REF_SPACING
    scale = [spacing[i] / ref_spacing[i] for i in range(3)]
    target_size = [int(h * scale[1]), int(w * scale[0]), int(d * scale[2])]  # [H', W', D']
    if min(target_size) < 1:
        return None, f"spacing {spacing} gives an empty resize target {target_size}"

    trans = transforms.Compose(
        [
            transforms.Resized(spatial_size=target_size, keys=["image"], mode="trilinear"),
            transforms.Transposed(keys=["image"], indices=(0, 3, 2, 1)),
        ]
    )
    resized_data = trans(res)

    img_resized = resized_data["image"]   # [C, D', W', H']
    image = img_resized
    image[image > 400] = 400
    image[image < -300] = -300
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    img = image

    # crop non-zero region in image
    roi_coords = np.nonzero(img[0].cpu().numpy())
    if roi_coords[0].size == 0:
        return None, "the scan has no voxel above its minimum after the HU clip"
    min_dhw = torch.from_numpy(np.min(roi_coords, axis=1))
    max_dhw = torch.from_numpy(np.max(roi_coords, axis=1))

    extend_d = EXTEND_D
    extend_hw = EXTEND_HW

    min_dhw = torch.max(
        min_dhw - torch.tensor([extend_d, extend_hw, extend_hw]),
        torch.tensor([0, 0, 0]),
    )
    max_dhw = torch.min(
        max_dhw + torch.tensor([extend_d, extend_hw, extend_hw]),
        torch.tensor([img.shape[1], img.shape[2], img.shape[3]]),
    )

    cropped_image = img[
        :,
        min_dhw[0]: max_dhw[0],
        min_dhw[1]: max_dhw[1],
        min_dhw[2]: max_dhw[2]
    ]
    crop_shape_dhw = tuple(cropped_image.shape[1:])

    # pad data to [96, 256, 384] if smaller
    data["image"] = cropped_image
    data_pad = loaded.datafolder.pad_func(data)
    data = data_pad

    facts = {
        "orig_shape_hwd": [int(h), int(w), int(d)],
        "spacing": [float(s) for s in spacing],
        "target_size": [int(t) for t in target_size],
        "crop_min_dwh": [int(v) for v in min_dhw.tolist()],
        "crop_max_dwh": [int(v) for v in max_dhw.tolist()],
        "crop_shape_dwh": [int(v) for v in crop_shape_dhw],
    }
    return data["image"].as_tensor(), facts


def score_file(nifti_path: str, loaded: Loaded, log=print) -> dict:
    """Score one NIfTI file. Returns the scoring part of the worker result plus `trace`,
    `mask` (uint8 numpy in the file's own grid), `affine`, `file_name` and `timings`,
    or {ok: false, error: {class: input_error, message}}."""
    import torch

    t0 = time.perf_counter()
    workdir = tempfile.mkdtemp(prefix="radar-score-")
    try:
        # 0. read the header with nibabel; reorient to LAS when needed
        try:
            img = nib.load(nifti_path)
            file_affine = np.asarray(img.affine, dtype=float)
            file_shape = tuple(int(s) for s in img.shape)
        except Exception as err:  # noqa: BLE001  nibabel raises many types for a bad file
            return _input_error(f"unreadable NIfTI: {err}")
        if len(file_shape) == 4 and file_shape[3] == 1:
            file_shape = file_shape[:3]
        if len(file_shape) != 3:
            return _input_error(f"expected a 3D volume, got shape {file_shape}")
        orientation_in = "".join(str(c) for c in nib.aff2axcodes(file_affine))
        las_img, ornt = geometry.to_las(img)
        if ornt is None:
            src = nifti_path
        else:
            src = os.path.join(workdir, "las.nii.gz")
            nib.save(las_img, src)
            log(f"reoriented {orientation_in} -> LAS for upstream")

        # 1. DataFolder.__getitem__
        try:
            image, facts = _preprocess(src, loaded)
        except Exception as err:  # noqa: BLE001  MONAI's loader raises many types for a bad file
            return _input_error(f"could not load the scan: {err}")
        if image is None:
            return _input_error(facts)

        plan = geometry.make_plan(facts["orig_shape_hwd"], facts["spacing"], facts["crop_min_dwh"],
                                  facts["crop_max_dwh"], orientation_in, ornt)
        assert plan.target_size == facts["target_size"], (plan.target_size, facts["target_size"])
        assert plan.crop_shape == facts["crop_shape_dwh"], (plan.crop_shape, facts["crop_shape_dwh"])
        log(f"orig {facts['orig_shape_hwd']} spacing {facts['spacing']} target {plan.target_size} "
            f"crop {plan.crop_min_dwh}..{plan.crop_max_dwh} padded {list(image.shape[1:])}")

        # 2. evaluate(): skip rule
        for tmp_s in image.shape[1:]:
            if tmp_s > MAX_AXIS:
                return _input_error(f"padded axis of {int(tmp_s)} voxels is over {MAX_AXIS}; upstream skips the case")
        assert list(image.shape[1:]) == plan.padded_shape, (list(image.shape[1:]), plan.padded_shape)

        with torch.inference_mode():
            out = _evaluate(image, loaded, plan, log)
        t_infer = time.perf_counter()

        # 3. back to the file's grid
        mask_ijk = geometry.inverse_mask(out["stitched_mask"], plan)
        if mask_ijk.shape != file_shape:
            raise RuntimeError(f"inverse mask shape {mask_ijk.shape} != file shape {file_shape}")
        stats_all = geometry.organ_stats(mask_ijk, file_affine, geometry.LABELS)
        result = _assemble(out, plan, file_affine, stats_all, loaded)
        result["mask"] = mask_ijk
        result["affine"] = file_affine
        result["file_name"] = os.path.basename(nifti_path)
        result["timings"] = {"infer_s": round(t_infer - t0, 3),
                             "postprocess_s": round(time.perf_counter() - t_infer, 3)}
        if not result["organs_scored"]:
            return _input_error("no scored organ was found in the scan")
        return result
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _evaluate(image, loaded: Loaded, plan: geometry.Plan, log) -> dict:
    """Mirror of the per-case body of upstream `evaluate()`."""
    import torch
    import torch.nn.functional as F
    from monai.data.utils import dense_patch_slices

    up = loaded.upstream
    model = loaded.model
    pad_func = loaded.pad_func
    text_feat_dict = loaded.text_feat_dict
    test_items = loaded.test_items
    test_organs = loaded.test_organs

    sw_batch_size = SW_BATCH_SIZE
    overlap = OVERLAP
    roi_size = ROI_SIZE

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()  # so the trace's peak is this scan's
    organ_feat_dict = {}

    image = image[None].cuda()

    image_size = list(image.shape[2:])
    num_spatial_dims = len(image.shape) - 2

    scan_interval = up._get_scan_interval(
        image_size, roi_size, num_spatial_dims, overlap
    )
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)
    windows = [[[int(s.start), int(s.stop)] for s in sl] for sl in slices]
    assert windows == plan.windows, "MONAI windows differ from geometry.sliding_windows"
    organ_logits = dict(zip(test_items, [[] for _ in test_items]))

    # get full mask
    full_mask = torch.zeros((1, 37) + tuple(image_size)).cuda()
    count_map = torch.zeros_like(full_mask).cuda()

    source = {}            # organ_zh -> {"how": "window", "window_index": i} or {"how": "centered_crop", ...}
    window_scored = []     # per window, the organs scored in it

    for slice_g in range(0, num_win, sw_batch_size):
        slice_range = range(slice_g, min(slice_g + sw_batch_size, num_win))
        unravel_slice = [
            [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]

        window_patches = torch.cat([image[win_slice] for win_slice in unravel_slice]).cuda()

        before = set(organ_feat_dict)
        organ_logits, pred_window_seg_prob = model.forward_test_win(
            window_patches,
            None,
            organ_logits,
            test_organs,
            text_feat_dict,
            organ_feat_dict,
            None
        )
        new = [k for k in organ_feat_dict if k not in before]
        for name in new:
            source[name] = {"how": "window", "window_index": slice_g}
        window_scored.append(new)

        # interpolate
        interpolated_seg_prob = F.interpolate(pred_window_seg_prob, size=window_patches.shape[2:], mode='trilinear')

        for ii, slice_idx in enumerate(slice_range):
            full_slice = unravel_slice[ii]
            full_mask[full_slice] += interpolated_seg_prob[ii]
            count_map[full_slice] += 1
    log(f"{num_win} windows, organs scored in windows: {sum(len(s) for s in window_scored)}")

    # Avoid division by zero by ensuring count_map is at least 1 everywhere
    count_map = torch.clamp(count_map, min=1)
    stitched_mask = full_mask / count_map  # argmax
    stitched_mask = stitched_mask.argmax(1).unsqueeze(0)

    # upstream computes intact_organ_ids here and never reads them; left out

    crops = []
    for k, v in organ_logits.items():
        if not len(v):
            organ_name = k.split('_')[0]
            organ_id = loaded.datafolder.organs.index(organ_name)

            # An organ absent from the stitched mask gets an inf box from masks_to_boxes_3d. On CUDA
            # inf converts to INT64_MAX, so the crop lands in the far (end) corner of the padded image;
            # on CPU it would convert differently and land in the origin corner. The wrapper must
            # therefore run on CUDA, as upstream does, for its crops to match.
            organ_mask = torch.eq(stitched_mask, organ_id + 1)
            present = bool(organ_mask.any().item())
            box = _center_crop_box(up.masks_to_boxes_3d, image, organ_mask, roi_size)
            window_patch, window_mask = up.center_crop(
                image,
                organ_mask,
                crop_size=roi_size
            )
            got = [int(s) for s in window_patch.shape[-3:]]
            assert got == [b - a for a, b in box], (got, box)
            window_mask = window_mask.float()
            window_mask[window_mask == 1] = organ_id + 1

            pad_data = pad_func({'image': window_patch[0], 'label': window_mask[0]})
            window_patch, window_mask = pad_data['image'], pad_data['label']

            before = set(organ_feat_dict)
            organ_logits, _ = model.forward_test_win(
                window_patch[None],
                None,
                organ_logits,
                test_organs,
                text_feat_dict,
                organ_feat_dict,
                None,
                skip_organ=organ_id
            )
            new = [n for n in organ_feat_dict if n not in before]
            crop_index = len(crops)
            for name in new:
                source[name] = {"how": "centered_crop", "crop_index": crop_index}
            crops.append({
                "index": crop_index,
                "for_item": k,
                "for_organ_zh": organ_name,
                "label": organ_id + 1,
                "absent": not present,  # crop in the far corner on CUDA, see above
                "box": box,
                "padded_shape": [int(s) for s in window_patch.shape[-3:]],
                "scored_zh": new,
            })
    log(f"{len(crops)} centred crops")

    probs = {item: p for item, p in organ_logits.items() if len(p) > 0}
    scores = {}
    for item in test_items:
        if item in probs:
            scores[item] = float(np.concatenate(probs[item]).mean(0)[1])  # get average of one organ in multi-widows
        else:
            scores[item] = None

    peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    return {
        "scores": scores,
        "raw_probs": probs,
        "windows": windows,
        "window_scored": window_scored,
        "crops": crops,
        "source": source,
        "stitched_mask": stitched_mask[0, 0].to(torch.uint8).cpu().numpy(),
        "peak_gpu_bytes": peak,
    }


def _assemble(out: dict, plan: geometry.Plan, affine, stats_all: dict, loaded: Loaded) -> dict:
    """Build findings, organs_scored, organs_not_found, organ_stats and the trace.

    organ_stats keeps the 18 scored organs; the trace keeps all 36 labels.
    """
    en = loaded.organ_en
    scored_names = set(en.values())
    stats = {k: v for k, v in stats_all.items() if k in scored_names}
    findings = []
    for item in loaded.test_items:
        organ, finding = loaded.english_mapping[item].split("_", 1)
        findings.append({"key": item, "organ": organ, "finding": finding, "prob": out["scores"][item]})

    window_boxes = [geometry.box_to_mm(w, plan, affine) for w in plan.windows]
    crop_boxes = [geometry.box_to_mm(c["box"], plan, affine) for c in out["crops"]]

    scored_order = []
    for item in loaded.test_items:
        zh = item.split("_")[0]
        if zh not in scored_order:
            scored_order.append(zh)

    organs_scored, organs_not_found, per_organ = [], [], {}
    for zh in scored_order:
        label = loaded.organs.index(zh) + 1
        src = out["source"].get(zh)
        if src is None:
            organs_not_found.append(en[zh])
            per_organ[en[zh]] = {"how": None, "label": label}
            continue
        if src["how"] == "window":
            box_mm = window_boxes[src["window_index"]]
            entry = {"organ": en[zh], "label": label, "how": "window",
                     "window_index": src["window_index"], "box_mm": box_mm}
        else:
            box_mm = crop_boxes[src["crop_index"]]
            entry = {"organ": en[zh], "label": label, "how": "centered_crop",
                     "window_index": None, "box_mm": box_mm}
        organs_scored.append(entry)
        per_organ[en[zh]] = dict(src, label=label)

    trace = {
        "upstream": {"roi_size": list(ROI_SIZE), "overlap": OVERLAP, "sw_batch_size": SW_BATCH_SIZE,
                     "ref_spacing": list(REF_SPACING), "extend_dwh": [EXTEND_D, EXTEND_HW, EXTEND_HW],
                     "hu_clip": [-300, 400], "crop_pad": "DivisiblePadd(k=32, end)"},
        "plan": plan.to_dict(),
        "windows": [{"index": i, "box": w, "box_mm": window_boxes[i],
                     "scored": [en.get(z, z) for z in out["window_scored"][i]]}
                    for i, w in enumerate(plan.windows)],
        "crops": [{"index": c["index"], "for_item": c["for_item"], "for_organ": en.get(c["for_organ_zh"]),
                   "label": c["label"], "absent": c["absent"], "box": c["box"], "box_mm": crop_boxes[i],
                   "padded_shape": c["padded_shape"], "scored": [en.get(z, z) for z in c["scored_zh"]]}
                  for i, c in enumerate(out["crops"])],
        "organs": per_organ,
        "organ_stats_all": stats_all,
        "raw_probs": out["raw_probs"],
        "peak_gpu_bytes": out["peak_gpu_bytes"],
    }
    return {
        "ok": True,
        "findings": findings,
        "organs_scored": organs_scored,
        "organs_not_found": organs_not_found,
        "organ_stats": stats,
        "trace": trace,
    }
