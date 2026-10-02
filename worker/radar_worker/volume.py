"""volume://<key> references: the scan and the artefacts on a mounted data volume instead of URLs.

Shared by the Modal function (the data Volume at /data) and the RunPod serverless handler (the
network volume at /runpod-volume). The key rules match the API's validate_key, so a reference
the app built always maps to a path under the mount and never outside it.

Standard library plus radar_worker.job and radar_worker.io only.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from radar_worker.job import ARTEFACTS, nifti_name

VOLUME_SCHEME = "volume://"


def ref_kind(source_ref: str, artefact_refs: dict):
    """"volume" when every reference is volume://, "url" when none is, None when they are mixed."""
    on_volume = [r.startswith(VOLUME_SCHEME) for r in [source_ref, *artefact_refs.values()]]
    if all(on_volume):
        return "volume"
    return "url" if not any(on_volume) else None


def volume_path(ref: str, root) -> Path:
    """Where volume://<key> sits under the mount; the key rules match the API's validate_key."""
    if not ref.startswith(VOLUME_SCHEME):
        raise ValueError(f"not a volume reference: {ref!r}")
    key = ref[len(VOLUME_SCHEME):]
    if (not key or key.startswith("/") or "\\" in key or any(ord(c) < 32 or ord(c) == 127 for c in key)
            or any(part in ("", ".", "..") for part in key.split("/"))):
        raise ValueError(f"bad volume key: {key!r}")
    return Path(root) / key


def volume_io(source_ref: str, artefact_refs: dict, root, commit):
    """fetch and publish for run_job when the scan and the artefacts live on a mounted volume.

    On Modal the caller reloads the Volume before anything on it is opened (a reload fails while
    files are open). Keys are written once; an existing artefact is refused, never overwritten
    (concurrent writes to one file are last write wins). `commit` runs once after the fifth
    artefact and before the call returns. On Modal, background commits and the shutdown commit can
    persist some artefacts earlier, so this is no all-or-nothing write; the guarantee is that the
    poller reads the artefacts only after FunctionCall.get returns, and by then commit() has run.
    The RunPod handler passes a no-op `commit`, since its network volume has no commit step.
    """
    from radar_worker.io import TransferError

    written = []

    def fetch(work: Path, log) -> Path:
        src = volume_path(source_ref, root)
        if not src.is_file():
            raise TransferError(404, "no such object on the data Volume", source_ref)
        name = nifti_name(src)
        if name == src.name:
            return src  # read in place; run_job's ensure_nifti_name then leaves it alone
        dest = Path(work) / name  # never rename on the Volume
        shutil.copyfile(src, dest)
        return dest

    def publish(name: str, path: Path, content_type: str) -> None:
        ref = artefact_refs[name]
        dest = volume_path(ref, root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "rb") as fin, open(dest, "xb") as fout:
                shutil.copyfileobj(fin, fout, 1 << 20)
        except FileExistsError:
            raise TransferError(409, "the object already exists; objects are never overwritten", ref) from None
        written.append(name)
        if len(written) == len(ARTEFACTS):
            commit()

    return fetch, publish
