"""Small synthetic NIfTI volumes for tests.

Builds CT-like volumes with ellipsoid blobs at known places, in a chosen
orientation, so header parsing, geometry and the API can be tested without
the real scans.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

Blob = tuple[tuple[float, float, float], tuple[float, float, float], float]
"""(centre_ijk, radii_ijk, value)"""

ORIENT_SIGNS = {
    "LAS": (-1, 1, 1),
    "RAS": (1, 1, 1),
    "LPS": (-1, -1, 1),
    "RPI": (1, -1, -1),
}


def affine_for(spacing: tuple[float, float, float], orientation: str = "LAS",
               origin: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    signs = ORIENT_SIGNS[orientation]
    aff = np.eye(4)
    for i in range(3):
        aff[i, i] = signs[i] * spacing[i]
        aff[i, 3] = origin[i]
    return aff


def ellipsoid(shape: tuple[int, int, int], centre, radii) -> np.ndarray:
    grids = np.ogrid[: shape[0], : shape[1], : shape[2]]
    acc = np.zeros(shape, dtype=float)
    for g, c, r in zip(grids, centre, radii):
        acc = acc + ((g - c) / max(r, 1e-6)) ** 2
    return acc <= 1.0


def make_volume(shape=(32, 32, 16), blobs: list[Blob] | None = None,
                background: float = -1000.0, body: float = 40.0,
                body_radius_frac: float = 0.45, dtype=np.int16) -> np.ndarray:
    """A body cylinder of soft tissue on air, with optional blobs."""
    vol = np.full(shape, background, dtype=float)
    cx, cy = (shape[0] - 1) / 2, (shape[1] - 1) / 2
    r = min(shape[0], shape[1]) * body_radius_frac
    body_mask = ellipsoid(shape, (cx, cy, (shape[2] - 1) / 2), (r, r, shape[2]))
    vol[body_mask] = body
    for centre, radii, value in blobs or []:
        vol[ellipsoid(shape, centre, radii)] = value
    return vol.astype(dtype)


def make_label_volume(shape=(32, 32, 16), blobs: list[tuple] | None = None) -> np.ndarray:
    """uint8 label volume: blobs are (centre_ijk, radii_ijk, label)."""
    lab = np.zeros(shape, dtype=np.uint8)
    for centre, radii, label in blobs or []:
        lab[ellipsoid(shape, centre, radii)] = label
    return lab


def write_nifti(path: Path, data: np.ndarray, spacing=(1.0, 1.0, 5.0), orientation="LAS",
                origin=(0.0, 0.0, 0.0), sform: bool = True, qform: bool = True,
                affine: np.ndarray | None = None) -> Path:
    aff = affine if affine is not None else affine_for(spacing, orientation, origin)
    img = nib.Nifti1Image(data, aff)
    img.set_sform(aff if sform else None, code=1 if sform else 0)
    img.set_qform(aff if qform else None, code=1 if qform else 0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(path))
    return path


def make_nifti(path: Path, shape=(32, 32, 16), spacing=(1.0, 1.0, 5.0), orientation="LAS",
               blobs: list[Blob] | None = None, dtype=np.int16, **kw) -> Path:
    return write_nifti(path, make_volume(shape, blobs, dtype=dtype), spacing, orientation, **kw)


def oblique_affine(spacing=(1.0, 1.0, 5.0), degrees: float = 10.0) -> np.ndarray:
    """An affine rotated about z by `degrees`, to test the oblique rejection."""
    t = np.deg2rad(degrees)
    rot = np.array([[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]])
    aff = np.eye(4)
    aff[:3, :3] = rot @ np.diag([-spacing[0], spacing[1], spacing[2]])
    return aff
