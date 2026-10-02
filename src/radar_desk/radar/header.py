"""NIfTI header reading and upload validation.

`read_header` reads just enough bytes to parse the header. `inspect_object`
streams a whole object once, hashing the stored bytes and counting the
decompressed size, with memory bounded by the chunk size.
"""

from __future__ import annotations

import hashlib
import os
import zlib
from collections.abc import Iterable, Iterator
from typing import Annotated, BinaryIO, Literal

import nibabel as nib
import numpy as np
from pydantic import BaseModel, Field, PrivateAttr

GZIP_MAGIC = b"\x1f\x8b"
_N1_SIZE, _N2_SIZE = 348, 540
_MAX_OUT = 1 << 20  # decompressed bytes produced per zlib call
ALLOWED_DTYPES = ("int16", "uint16", "float32", "float64")
MAX_AXIS = 1000
SPACING_MM = (0.3, 10.0)


class HeaderError(ValueError):
    """The object is not a NIfTI volume this app accepts. The message is the reason."""


class Header(BaseModel):
    dims: Annotated[list[int], Field(min_length=3, max_length=3)]
    spacing_mm: Annotated[list[float], Field(min_length=3, max_length=3)]
    dtype: str
    orientation: str
    affine: Annotated[list[list[float]], Field(min_length=4, max_length=4)]
    nifti_version: Literal[1, 2]

    # Facts from the raw header that validate() needs but records do not keep.
    # None on a Header rebuilt from storage, where those checks are skipped.
    _ndim: int | None = PrivateAttr(default=None)
    _frames: int | None = PrivateAttr(default=None)
    _sform_code: int | None = PrivateAttr(default=None)
    _qform_code: int | None = PrivateAttr(default=None)


def _parse(raw: bytes) -> Header:
    if len(raw) < 4:
        raise HeaderError("file is too short to be NIfTI")
    size = int.from_bytes(raw[:4], "little")
    if size not in (_N1_SIZE, _N2_SIZE):
        size = int.from_bytes(raw[:4], "big")
    if size not in (_N1_SIZE, _N2_SIZE):
        raise HeaderError("not a NIfTI file (header size is neither 348 nor 540)")
    if len(raw) < size:
        raise HeaderError("file ends inside the NIfTI header")
    version = 1 if size == _N1_SIZE else 2
    klass = nib.Nifti1Header if version == 1 else nib.Nifti2Header
    hdr = klass(raw[:size], check=False)

    magic = hdr["magic"].item().rstrip(b"\x00")
    if magic in (b"ni1", b"ni2"):
        raise HeaderError("separate .hdr/.img pairs are not supported; upload a single .nii or .nii.gz")
    if magic not in (b"n+1", b"n+2"):
        raise HeaderError("not a NIfTI file (bad magic)")

    dim = [int(x) for x in hdr["dim"]]
    ndim = dim[0]
    if not 1 <= ndim <= 7:
        raise HeaderError(f"invalid NIfTI dimension count {ndim}")
    try:
        dtype = hdr.get_data_dtype().name
    except Exception as exc:  # nibabel raises several types for unknown codes
        raise HeaderError(f"unknown NIfTI data type code {int(hdr['datatype'])}") from exc

    affine = hdr.get_best_affine()
    try:
        orientation = "".join(c or "?" for c in nib.aff2axcodes(affine))
    except (ValueError, np.linalg.LinAlgError):
        orientation = "???"

    header = Header(
        dims=dim[1:4],
        # Upstream reads spacing from the affine diagonal, not pixdim, so the record does too.
        spacing_mm=[abs(float(affine[i, i])) for i in range(3)],
        dtype=dtype,
        orientation=orientation,
        affine=[[float(v) for v in row] for row in affine],
        nifti_version=version,
    )
    header._ndim = ndim
    header._frames = dim[4] if ndim >= 4 else 1
    header._sform_code = int(hdr["sform_code"])
    header._qform_code = int(hdr["qform_code"])
    return header


class _Decoder:
    """Feeds stored bytes, gunzipping when they start with the gzip magic.

    Keeps the first header-sized bytes of the decompressed stream and a count of
    the rest. Output per zlib call is capped, so a highly compressible chunk
    never inflates in one piece.
    """

    def __init__(self) -> None:
        self.gzip: bool | None = None
        self._pending = b""
        self._z: zlib._Decompress | None = None
        self.head = bytearray()
        self.nbytes = 0

    def _emit(self, data: bytes) -> None:
        self.nbytes += len(data)
        room = _N2_SIZE - len(self.head)
        if room > 0:
            self.head += data[:room]

    def feed(self, chunk: bytes) -> None:
        if self.gzip is None:
            self._pending += chunk
            if len(self._pending) < 2:
                return
            self.gzip = self._pending[:2] == GZIP_MAGIC
            chunk, self._pending = self._pending, b""
        if not self.gzip:
            self._emit(chunk)
            return
        data = chunk
        try:
            while data:
                if self._z is None:
                    self._z = zlib.decompressobj(16 + zlib.MAX_WBITS)
                self._emit(self._z.decompress(data, _MAX_OUT))
                if self._z.eof:  # end of one gzip member; another may follow
                    data = self._z.unused_data
                    self._z = None
                else:
                    data = self._z.unconsumed_tail
        except zlib.error as exc:
            raise HeaderError(f"corrupt gzip data: {exc}") from exc

    def finish(self) -> None:
        if self.gzip is None:
            self.gzip = False
            self._emit(self._pending)
            return
        if self._z is not None:
            try:
                while True:
                    out = self._z.decompress(self._z.unconsumed_tail, _MAX_OUT)
                    if not out:
                        break
                    self._emit(out)
                self._emit(self._z.flush())
            except zlib.error as exc:
                raise HeaderError(f"corrupt gzip data: {exc}") from exc
            if not self._z.eof:
                raise HeaderError("gzip data ends early (truncated upload?)")


def _chunks(stream: BinaryIO | Iterable[bytes], size: int) -> Iterator[bytes]:
    read = getattr(stream, "read", None)
    if read is None:
        yield from stream
        return
    while chunk := read(size):
        yield chunk


def read_header(source: str | os.PathLike | BinaryIO) -> Header:
    """Parse the header of a .nii or .nii.gz from a path or a binary file-like."""
    if isinstance(source, (str, os.PathLike)):
        with open(source, "rb") as fh:
            return read_header(fh)
    dec = _Decoder()
    for chunk in _chunks(source, 64 * 1024):
        dec.feed(chunk)
        if len(dec.head) >= _N2_SIZE:
            break
    else:
        if len(dec.head) < _N2_SIZE:
            dec.finish()
    return _parse(bytes(dec.head))


def inspect_object(stream: BinaryIO | Iterable[bytes], chunk_size: int = 1 << 20) -> tuple[Header, str, int]:
    """Stream an object once: (header, sha256 of the stored bytes, decompressed size)."""
    digest = hashlib.sha256()
    dec = _Decoder()
    for chunk in _chunks(stream, chunk_size):
        digest.update(chunk)
        dec.feed(chunk)
    dec.finish()
    return _parse(bytes(dec.head)), digest.hexdigest(), dec.nbytes


def validate(header: Header) -> None:
    """Raise HeaderError with a reason when the scan cannot be scored."""
    ndim, frames = header._ndim, header._frames
    if ndim is not None and not (ndim == 3 or (ndim == 4 and frames == 1)):
        raise HeaderError(f"not a 3D volume (dim[0] = {ndim}, frames = {frames})")
    if any(d < 1 for d in header.dims):
        raise HeaderError(f"not a 3D volume (dims {header.dims})")
    if any(d > MAX_AXIS for d in header.dims):
        raise HeaderError(f"an axis has more than {MAX_AXIS} voxels (dims {header.dims})")
    if header._sform_code is not None and header._sform_code <= 0 and (header._qform_code or 0) <= 0:
        raise HeaderError("no sform or qform, so the scan has no position in space")
    rot = np.asarray(header.affine, dtype=float)[:3, :3]
    norms = np.linalg.norm(rot, axis=0)
    if np.any(norms == 0):
        raise HeaderError("the affine is degenerate (a zero column)")
    # Upstream reads spacing from affine[0,0], [1,1], [2,2], so the dominant entry of every
    # column must sit on the diagonal: a rotated or axis-permuted affine would be scored wrongly.
    if not np.all(np.abs(np.diag(rot)) / norms > 0.999):
        raise HeaderError("oblique or permuted affine: the voxel axes do not line up with the scanner "
                          "axes, and RADAR reads spacing from the affine diagonal only")
    lo, hi = SPACING_MM
    if any(not lo <= s <= hi for s in header.spacing_mm):
        spacing = ", ".join(f"{s:g}" for s in header.spacing_mm)
        raise HeaderError(f"voxel spacing {spacing} mm is outside {lo:g} to {hi:g} mm")
    if header.dtype not in ALLOWED_DTYPES:
        raise HeaderError(f"data type {header.dtype} is not supported ({', '.join(ALLOWED_DTYPES)})")
