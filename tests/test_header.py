"""NIfTI header parsing, validation and the single-pass object inspection."""

from __future__ import annotations

import gzip
import hashlib
import io

import nibabel as nib
import numpy as np
import pytest

from radar_desk.radar.header import Header, HeaderError, inspect_object, read_header, validate
from synth import affine_for, make_nifti, make_volume, oblique_affine, write_nifti

REAL = {
    "AC4214dbd": ((437, 437, 87), (1.00, 1.00, 5.03), "float32"),
    "AC4240fff": ((460, 460, 106), (1.00, 1.00, 5.01), "float32"),
    "AC4242a2f": ((474, 474, 100), (1.00, 1.00, 5.03), "float32"),
    "AC4242a55": ((448, 448, 95), (1.00, 1.00, 5.01), "float32"),
    "image1": ((512, 512, 489), (0.98, 0.98, 1.00), "int16"),
}
AC4242A55_SHA256 = "6e39c5ff4a9f397a018de2f0faef8edb4e389e3d97dae79a3459e8e096ca49e9"


def _reason(header: Header) -> str:
    with pytest.raises(HeaderError) as exc:
        validate(header)
    return str(exc.value)


# Real scans


@pytest.mark.parametrize("fixture_id", sorted(REAL))
def test_real_scan_headers(fixture_paths, fixture_id):
    dims, spacing, dtype = REAL[fixture_id]
    h = read_header(fixture_paths[fixture_id])
    assert tuple(h.dims) == dims
    assert tuple(round(s, 2) for s in h.spacing_mm) == spacing
    assert h.dtype == dtype
    assert h.orientation == "LAS"
    assert h.nifti_version == 1
    validate(h)


def test_real_scan_from_open_stream(fixture_paths):
    with fixture_paths["AC4214dbd"].open("rb") as fh:
        h = read_header(fh)
    assert tuple(h.dims) == (437, 437, 87)


def test_inspect_object_real_scan_sha256(fixture_paths):
    path = fixture_paths["AC4242a55"]
    with path.open("rb") as fh:
        h, sha, nbytes = inspect_object(fh)
    assert sha == AC4242A55_SHA256
    assert tuple(h.dims) == (448, 448, 95)
    assert nbytes == 352 + 448 * 448 * 95 * 4  # vox_offset 352, no extensions


# Synthetic: happy paths


def test_synthetic_gzip_header(tmp_path):
    p = make_nifti(tmp_path / "a.nii.gz", shape=(32, 30, 16), spacing=(0.8, 0.9, 2.5))
    h = read_header(p)
    assert h.dims == [32, 30, 16]
    assert [round(s, 3) for s in h.spacing_mm] == [0.8, 0.9, 2.5]
    assert h.dtype == "int16"
    assert h.orientation == "LAS"
    assert h.nifti_version == 1
    assert len(h.affine) == 4 and all(len(r) == 4 for r in h.affine)
    assert h.affine[0][0] == pytest.approx(-0.8)
    validate(h)


def test_plain_nii(tmp_path):
    p = make_nifti(tmp_path / "a.nii", orientation="RAS")
    h = read_header(p)
    assert h.orientation == "RAS"
    validate(h)


def test_file_like_source(tmp_path):
    p = make_nifti(tmp_path / "a.nii.gz")
    h = read_header(io.BytesIO(p.read_bytes()))
    assert h.dims == [32, 32, 16]


def test_nifti2(tmp_path):
    data = make_volume((20, 22, 12), dtype=np.float32)
    aff = affine_for((1.0, 1.0, 3.0))
    img = nib.Nifti2Image(data, aff)
    img.set_sform(aff, code=1)
    p = tmp_path / "n2.nii.gz"
    nib.save(img, str(p))
    h = read_header(p)
    assert h.nifti_version == 2
    assert h.dims == [20, 22, 12]
    assert h.dtype == "float32"
    validate(h)
    h2, _, _ = inspect_object(io.BytesIO(p.read_bytes()))
    assert h2 == h


def test_4d_single_frame_passes(tmp_path):
    data = make_volume((16, 16, 8))[..., None]
    p = write_nifti(tmp_path / "one.nii.gz", data)
    h = read_header(p)
    assert h.dims == [16, 16, 8]
    validate(h)


@pytest.mark.parametrize("dtype", [np.int16, np.uint16, np.float32, np.float64])
def test_allowed_dtypes(tmp_path, dtype):
    p = make_nifti(tmp_path / "d.nii.gz", shape=(8, 8, 4), dtype=dtype)
    validate(read_header(p))


# Synthetic: rejections


def test_reject_4d_many_frames(tmp_path):
    data = np.stack([make_volume((16, 16, 8))] * 3, axis=-1)
    p = write_nifti(tmp_path / "many.nii.gz", data)
    assert "3D" in _reason(read_header(p))


def test_reject_2d(tmp_path):
    data = make_volume((16, 16, 1))[..., 0]
    p = write_nifti(tmp_path / "flat.nii.gz", data)
    assert "3D" in _reason(read_header(p))


def test_reject_axis_over_1000(tmp_path):
    p = make_nifti(tmp_path / "big.nii.gz", shape=(1001, 2, 2))
    assert "1000" in _reason(read_header(p))


def test_axis_of_exactly_1000_passes(tmp_path):
    p = make_nifti(tmp_path / "edge.nii.gz", shape=(1000, 2, 2))
    validate(read_header(p))


@pytest.mark.parametrize("spacing", [(0.2, 1.0, 1.0), (1.0, 1.0, 11.0)])
def test_reject_spacing_out_of_range(tmp_path, spacing):
    p = make_nifti(tmp_path / "s.nii.gz", shape=(8, 8, 4), spacing=spacing)
    assert "spacing" in _reason(read_header(p))


def test_reject_no_sform_no_qform(tmp_path):
    p = make_nifti(tmp_path / "noform.nii.gz", shape=(8, 8, 4), sform=False, qform=False)
    assert "sform" in _reason(read_header(p))


def test_qform_only_passes(tmp_path):
    p = make_nifti(tmp_path / "q.nii.gz", shape=(8, 8, 4), sform=False, qform=True)
    validate(read_header(p))


def test_reject_oblique(tmp_path):
    p = write_nifti(tmp_path / "obl.nii.gz", make_volume((8, 8, 4)), affine=oblique_affine(degrees=10.0))
    assert "oblique" in _reason(read_header(p))


def test_reject_uint8(tmp_path):
    p = make_nifti(tmp_path / "u8.nii.gz", shape=(8, 8, 4), dtype=np.uint8)
    assert "uint8" in _reason(read_header(p))


def test_reject_not_nifti():
    with pytest.raises(HeaderError):
        read_header(io.BytesIO(b"\x00" * 600))
    with pytest.raises(HeaderError):
        read_header(io.BytesIO(gzip.compress(b"hello")))


def test_reject_truncated_header(tmp_path):
    p = make_nifti(tmp_path / "a.nii")
    with pytest.raises(HeaderError):
        read_header(io.BytesIO(p.read_bytes()[:200]))


# inspect_object


def _chunks(data: bytes, size: int):
    for i in range(0, len(data), size):
        yield data[i : i + size]


def test_inspect_object_gzip(tmp_path):
    p = make_nifti(tmp_path / "a.nii.gz", shape=(40, 40, 20))
    raw = p.read_bytes()
    h, sha, nbytes = inspect_object(io.BytesIO(raw), chunk_size=1000)
    assert sha == hashlib.sha256(raw).hexdigest()
    assert nbytes == len(gzip.decompress(raw))
    assert h == read_header(p)


def test_inspect_object_plain_nii(tmp_path):
    p = make_nifti(tmp_path / "a.nii")
    raw = p.read_bytes()
    h, sha, nbytes = inspect_object(io.BytesIO(raw))
    assert sha == hashlib.sha256(raw).hexdigest()
    assert nbytes == len(raw)
    assert h.dims == [32, 32, 16]


def test_inspect_object_accepts_byte_iterator(tmp_path):
    p = make_nifti(tmp_path / "a.nii.gz")
    raw = p.read_bytes()
    h, sha, nbytes = inspect_object(_chunks(raw, 1))
    assert sha == hashlib.sha256(raw).hexdigest()
    assert nbytes == len(gzip.decompress(raw))
    assert h.dims == [32, 32, 16]


def test_inspect_object_bounds_decompression_of_highly_compressible_data(tmp_path):
    # 64 MB of zeros compresses to about 64 KB; one input chunk must not inflate in one go.
    data = np.zeros((400, 400, 200), dtype=np.int16)
    p = write_nifti(tmp_path / "zeros.nii.gz", data)
    raw = p.read_bytes()
    h, _, nbytes = inspect_object(io.BytesIO(raw), chunk_size=len(raw))
    assert nbytes == 352 + data.nbytes
    assert h.dims == [400, 400, 200]


def test_inspect_object_rejects_truncated_gzip(tmp_path):
    p = make_nifti(tmp_path / "a.nii.gz", shape=(40, 40, 20))
    raw = p.read_bytes()
    with pytest.raises(HeaderError):
        inspect_object(io.BytesIO(raw[: len(raw) // 2]))


def test_reject_permuted_axes(tmp_path):
    """An axis permutation keeps every column unit length but puts zeros on the diagonal,
    which upstream would read as a spacing of zero."""
    aff = np.zeros((4, 4))
    aff[0, 2], aff[1, 0], aff[2, 1], aff[3, 3] = 5.0, -1.0, 1.0, 1.0
    p = write_nifti(tmp_path / "perm.nii.gz", make_volume((8, 8, 4)), affine=aff)
    assert "permuted" in _reason(read_header(p))


def test_spacing_comes_from_the_affine_not_pixdim(tmp_path):
    p = write_nifti(tmp_path / "aff.nii.gz", make_volume((8, 8, 4)), affine=affine_for((0.5, 0.5, 2.0)))
    img = nib.load(p)
    img.header["pixdim"][1:4] = [1.0, 1.0, 5.0]
    nib.save(img, p)
    h = read_header(p)
    assert [round(s, 3) for s in h.spacing_mm] == [0.5, 0.5, 2.0]
