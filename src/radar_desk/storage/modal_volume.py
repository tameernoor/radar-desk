"""Objects on a Modal Volume, for running against Modal without a bucket.

The API reaches the Volume through the modal client: `read_file`, `batch_upload`, `listdir` and
`remove_file` (ref_Volume.md lines 385-489 in the Modal docs). A batch upload is durable when its
`with` block exits; no commit is needed from outside a container (guide_volumes.md lines 143-154).
Key `a/b` lives at `/a/b` on the Volume.

The browser cannot reach a Volume, so `put_url` and `get_url` are the relative path `/_volume/{key}`
on the API (routes/volume_storage.py), which needs the owner's login instead of a signature. Relative,
so the browser sends them to the origin it is on and its cookie goes along. The worker mounts the same
Volume and gets `volume://{key}` references.
"""

from __future__ import annotations

import io
import itertools
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from radar_desk.storage.base import ObjectExists, ObjectMissing, StorageError, validate_key

VOLUME_SCHEME = "volume://"


class ModalVolumeStorage:
    name = "modal_volume"
    browser_via_api = True  # app.py mounts routes/volume_storage.py

    def __init__(self, volume_name: str, public_base_url: str, volume: Any = None,
                 credentials: tuple[str, str] | None = None) -> None:
        self.volume_name = volume_name
        self.public_base_url = public_base_url.rstrip("/")
        self._volume = volume
        self._credentials = credentials
        self._lock = threading.Lock()

    def _vol(self) -> Any:
        """The Volume handle, looked up on first use. `from_name` is lazy itself (ref_Volume.md line 248)."""
        if self._volume is None:
            with self._lock:
                if self._volume is None:
                    import modal

                    kw: dict[str, Any] = {}
                    if self._credentials is not None:
                        kw["client"] = modal.Client.from_credentials(*self._credentials)
                    self._volume = modal.Volume.from_name(self.volume_name, create_if_missing=True, **kw)
        return self._volume

    @staticmethod
    def _path(key: str) -> str:
        return "/" + validate_key(key)

    # URLs

    @staticmethod
    def _url(key: str) -> str:
        return f"/_volume/{validate_key(key)}"

    def put_url(self, key: str, expires_s: int) -> str:
        return self._url(key)

    def get_url(self, key: str, expires_s: int) -> str:
        return self._url(key)

    def worker_ref(self, key: str, method: str, expires_s: int) -> str:
        if method not in ("GET", "PUT"):
            raise ValueError(f"method must be GET or PUT, not {method!r}")
        return VOLUME_SCHEME + validate_key(key)

    # Reads

    def _entry(self, key: str) -> Any:
        """The listdir entry of this file, or None. For a file path listdir returns that file's entry
        (ref_Volume.md lines 391-395); a missing path raises NotFoundError, the client's mapping of
        the server's NOT_FOUND (modal/_grpc_client.py, _STATUS_TO_EXCEPTION). Entry paths carry no
        leading slash (modal/cli/_download.py, producer)."""
        from modal.exception import NotFoundError

        path = self._path(key)
        try:
            entries = self._vol().listdir(path)
        except NotFoundError:
            return None
        return next((e for e in entries if str(e.path).lstrip("/") == key), None)

    def exists(self, key: str) -> bool:
        return self._entry(key) is not None

    def size(self, key: str) -> int:
        entry = self._entry(key)
        if entry is None:
            raise ObjectMissing(key)
        return int(entry.size)

    def open_stream(self, key: str) -> Iterator[bytes]:
        """The client's blocks as they come. A missing object raises ObjectMissing here, not on
        iteration: read_file raises FileNotFoundError on its first step (modal/volume.py, read_file)."""
        chunks = iter(self._vol().read_file(self._path(key)))
        try:
            first = next(chunks, b"")
        except FileNotFoundError:
            raise ObjectMissing(key) from None
        return itertools.chain((first,), chunks)

    # Writes. batch_upload(force=False) refuses to overwrite and raises FileExistsError on exit
    # (ref_Volume.md lines 473-476; modal/volume.py, _VolumeUploadContextManager.__aexit__).

    def _upload(self, key: str, source: Any) -> None:
        path = self._path(key)
        try:
            with self._vol().batch_upload(force=False) as batch:
                batch.put_file(source, path)
        except FileExistsError:
            raise ObjectExists(f"{key} already exists; objects are never overwritten") from None

    def put_bytes(self, key: str, data: bytes, content_type: str | None = None) -> None:
        self._upload(key, io.BytesIO(data))

    def put_file(self, key: str, path: Path, content_type: str | None = None) -> None:
        """The client reads the file from disk while uploading; it is never loaded whole."""
        if not Path(path).is_file():
            raise StorageError(f"not a file: {path}")
        self._upload(key, str(path))

    def delete(self, key: str) -> None:
        """Deleting a missing object is fine. Modal 1.6 reports it as InvalidError("No such file or directory.")."""
        from modal.exception import InvalidError

        try:
            self._vol().remove_file(self._path(key))
        except FileNotFoundError:
            pass
        except InvalidError as err:
            if "no such file" not in str(err).lower():
                raise
