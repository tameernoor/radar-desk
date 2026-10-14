"""Modal app `radar-desk-nvreason`: NVIDIA's NV-Reason-CT as a second reader for the Readers lens.

Deploy with `modal deploy worker/nvreason_app.py` (MODAL_DATA_VOLUME names the data Volume, default radar-data, the
same one the radar-desk app mounts). One class, NvReason, loads the model once per container and has two methods.

- report(job_id, source_ref): the crop the model reads with its world box, and one structured report with
  thinking on. No Yes/No questions run here.
- ask(source_ref, region, messages, enable_thinking, max_new_tokens): one answer in a conversation that the API
  builds (the image, the report prompt, NV-Reason-CT's own report, earlier questions, the new question).

source_ref is a presigned GET URL or volume://<key> on the data Volume. The image, the pinned revision, the crop
offset check and generate() come from scripts/nvreason_smoke.py, which proved them on an L4.

Importing this module defines objects only; nothing talks to Modal until deploy.
"""

from __future__ import annotations

import gzip
import os
import shutil
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from radar_worker.job import JobLog, ensure_nifti_name, error_result, nifti_name
from radar_worker.nvreason import (
    CROP,
    MODEL_ID,
    REGION,
    REPORT_MAX_NEW_TOKENS,
    REPORT_PROMPT,
    REVISION,
    SCALEDOWN_S,
    crop_summary,
    split_output,
)
from radar_worker.volume import VOLUME_SCHEME, volume_path

APP_NAME = "radar-desk-nvreason"
CLASS_NAME = "NvReason"
CACHE_DIR = "/cache"
DATA_DIR = "/data"
WORK_DIR = "/tmp/nvreason"
DATA_VOLUME = os.environ.get("MODAL_DATA_VOLUME", "radar-data")  # read at deploy time

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch==2.11.0", "torchvision==0.26.0", index_url="https://download.pytorch.org/whl/cu128")
    # the model card's requirements.txt at REVISION, with torch already in place
    .uv_pip_install(
        "transformers==5.6.2",
        "monai>=1.5.1",
        "nibabel>=5.3.3",
        "dynamic-network-architectures==0.4.3",
        "scipy",
        "numpy",
        "Pillow",
        "timm",
        "einops",
        "safetensors",
    )
    .env({"HF_HOME": f"{CACHE_DIR}/hf", "PYTHONUNBUFFERED": "1"})
    # local files last: attached at container start, after the build steps
    .add_local_python_source("radar_worker")
)

cache = modal.Volume.from_name("nvreason-hf-cache", create_if_missing=True)
data_volume = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)

app = modal.App(APP_NAME)

CLASS_OPTIONS = {
    "image": image,
    "gpu": "L4",
    "volumes": {CACHE_DIR: cache, DATA_DIR: data_volume},
    "timeout": 1200,
    "cpu": 4.0,
    "memory": 32768,
    "max_containers": 1,
    "retries": 0,
    "scaledown_window": SCALEDOWN_S,
}


@app.cls(**CLASS_OPTIONS)
class NvReason:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        t = time.perf_counter()
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            revision=REVISION,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).eval().to("cuda")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
        self.load_s = time.perf_counter() - t
        self.sources: dict[str, str] = {}
        cache.commit()

    def fetch(self, source_ref: str) -> str:
        """A local path to the scan: read in place on the data Volume, else downloaded once per container."""
        if source_ref.startswith(VOLUME_SCHEME):
            data_volume.reload()  # before any file on /data is opened
            src = volume_path(source_ref, DATA_DIR)
            if not src.is_file():
                raise FileNotFoundError(f"no such object on the data Volume: {source_ref}")
            if nifti_name(src) == src.name:
                return str(src)
            dest = Path(WORK_DIR) / nifti_name(src)  # never rename on the Volume
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            return str(dest)
        key = source_ref.split("?", 1)[0]
        if key not in self.sources:
            from radar_worker import io as wio

            dest = Path(WORK_DIR) / f"scan-{len(self.sources)}.nii.gz"
            wio.download(source_ref, dest)
            self.sources[key] = str(ensure_nifti_name(dest))
        return self.sources[key]

    def crop(self, path: str, region: str | None) -> tuple[dict, bytes]:
        """The model's crop as crop.json and crop.nii.gz bytes (world-aligned, overlays the original).

        scripts/nvreason_smoke.py, Reader.crop, without the coronal picture."""
        import nibabel as nib
        import numpy as np

        tfs = self.processor.image_processor_3d.get_transforms().transforms
        i = [type(t).__name__ for t in tfs].index("AnatomySpatialCropd")
        cropper = tfs[i]
        resolved = cropper.resolve_region(region)
        d = {"image": path, "anatomy_region": resolved}
        for t in tfs[:i]:  # load, LPS, 2 mm, pad: the space the crop is cut from
            d = t(d)
        full = d["image"]
        crop = cropper(d)["image"]

        # The region crop slices the MetaTensor, which keeps the full volume's affine, so the offset is recomputed
        # the way `_crop_one` (or MONAI's centre crop, for no region) picks it, then checked voxel for voxel.
        shape = [int(s) for s in full.shape[1:]]
        roi = [int(r) for r in cropper.roi_size]
        bounds = None
        if resolved is None:
            off = [s // 2 - r // 2 for s, r in zip(shape, roi)]
        else:
            off = [cropper._axis_start(shape[a], roi[a], (shape[a] - 1) / 2.0) for a in (0, 1)]
            bounds = cropper._region_bounds(full, resolved)
            z_center = None if bounds is None else (float(bounds[0]) + float(bounds[1])) / 2.0
            off.append(shape[2] - roi[2] if bounds is None else cropper._axis_start(shape[2], roi[2], z_center))
        vol = full[0].numpy()
        crop_np = crop[0].numpy()
        x0, y0, z0 = off
        if not np.array_equal(vol[x0:x0 + CROP, y0:y0 + CROP, z0:z0 + CROP], crop_np):
            raise RuntimeError(f"crop offset {off} does not reproduce the processor's crop")
        a_full = np.asarray(full.affine, dtype=np.float64)
        a_crop = a_full.copy()
        a_crop[:3, 3] = a_full[:3, :3] @ np.array(off, dtype=np.float64) + a_full[:3, 3]

        edges = np.array([[x, y, z, 1.0] for x in (-0.5, CROP - 0.5) for y in (-0.5, CROP - 0.5)
                          for z in (-0.5, CROP - 0.5)]).T
        world = (a_crop @ edges)[:3]
        spacing = float(self.processor.image_processor_3d.pixdim[2])
        summary = crop_summary(world.min(axis=1), world.max(axis=1), off, shape[2], spacing,
                               fallback=resolved is not None and bounds is None, region=region)
        crop_gz = gzip.compress(nib.Nifti1Image(crop_np.astype(np.float32), a_crop).to_bytes())
        return summary, crop_gz

    def generate(self, path: str, region: str | None, messages: list, enable_thinking: bool,
                 max_new_tokens: int) -> dict:
        """scripts/nvreason_smoke.py's generate, taking the whole conversation instead of one prompt."""
        import torch

        processor, model = self.processor, self.model
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        inputs = processor(text=prompt, images3d=[path], anatomy_region=region, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t
        new = ids[:, inputs.input_ids.shape[1]:]
        decoded = processor.batch_decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return {
            **split_output(decoded),
            "enable_thinking": enable_thinking,
            "seconds": round(seconds, 2),
            "input_tokens": int(inputs.input_ids.shape[1]),
            "new_tokens": int(new.shape[1]),
            "hit_max_new_tokens": int(new.shape[1]) >= max_new_tokens,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }

    def versions(self) -> dict:
        import torch
        import transformers

        return {
            "model": MODEL_ID,
            "revision": REVISION,
            "transformers": transformers.__version__,
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "gpu": torch.cuda.get_device_name(),
            "image_id": os.environ.get("MODAL_IMAGE_ID"),
        }

    @modal.method()
    def report(self, job_id: str, source_ref: str) -> dict:
        """crop.json, crop.nii.gz and one structured report with thinking on. An unreadable scan comes back as
        {ok: false}; anything else raises and the job fails with the exception's class."""
        from radar_worker.io import TransferError

        log = JobLog(job_id)
        t0 = time.perf_counter()
        try:
            path = self.fetch(source_ref)
        except (TransferError, FileNotFoundError, ValueError) as err:
            log(f"could not read the scan: {err}")
            return error_result(job_id, "input_error", str(err))
        download_s = time.perf_counter() - t0
        t = time.perf_counter()
        crop, crop_gz = self.crop(path, REGION)
        crop_s = time.perf_counter() - t
        log(f"crop {crop['crop_offset_vox']}, cut {crop['cut_below_mm']} mm below, {crop['cut_above_mm']} mm above")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": REPORT_PROMPT}]}]
        out = self.generate(path, REGION, messages, True, REPORT_MAX_NEW_TOKENS)
        log(f"report in {out['seconds']} s, {out['new_tokens']} tokens")
        return {
            "ok": True,
            "job_id": job_id,
            "nv_report": {"prompt": REPORT_PROMPT, "region": REGION, **out},
            "crop": crop,
            "crop_gz": crop_gz,
            "log": log.text(),
            "timings": {
                "load_s": round(self.load_s, 2),
                "download_s": round(download_s, 2),
                "postprocess_s": round(crop_s, 2),
                "infer_s": out["seconds"],
                "total_s": round(time.perf_counter() - t0, 2),
            },
            "versions": self.versions(),
        }

    @modal.method()
    def ask(self, source_ref: str, region: str, messages: list, enable_thinking: bool, max_new_tokens: int) -> dict:
        """One answer in a conversation that starts with the image and NV-Reason-CT's own report."""
        return self.generate(self.fetch(source_ref), region, messages, enable_thinking, max_new_tokens)
