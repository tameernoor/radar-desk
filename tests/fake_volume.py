"""An in-memory stand-in for `modal.Volume` as a local client sees it (modal 1.6.0).

Missing paths raise what the installed client raises. `listdir` raises
`modal.exception.NotFoundError`, because the server answers NOT_FOUND and
`modal/_grpc_client.py` maps that status to NotFoundError. `read_file` and `remove_file`
raise `FileNotFoundError` (`modal/volume.py`, read_file and remove_file). A batch that would
overwrite a file without `force` raises `FileExistsError` on exit and writes nothing, as the
real batch sends one VolumePutFiles request. Entry paths carry no leading slash, as in
`modal/cli/_download.py`.
"""

from __future__ import annotations

import contextlib
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Self

from modal.exception import NotFoundError

CHUNK = 1024 * 1024


class FakeVolume:
    def __init__(self, chunk: int = CHUNK, keep: bool = True) -> None:
        self.chunk = chunk
        self.keep = keep  # False stores only size and sha256, for files too big to hold
        self.files: dict[str, bytes | None] = {}
        self.sizes: dict[str, int] = {}
        self.sha256: dict[str, str] = {}
        self.max_read = 0
        self.remote_paths: list[str] = []

    @staticmethod
    def _norm(path: str) -> str:
        return path.lstrip("/")

    def listdir(self, path: str, *, recursive: bool = False) -> list:
        p = self._norm(path)
        if p in self.sizes:
            return [SimpleNamespace(path=p, type=1, mtime=0, size=self.sizes[p])]
        children = sorted(k for k in self.sizes if k.startswith(p + "/"))
        if not children:
            raise NotFoundError(f"No such file or directory: {path}")
        return [SimpleNamespace(path=k, type=1, mtime=0, size=self.sizes[k]) for k in children]

    def read_file(self, path: str):
        p = self._norm(path)
        if p not in self.files:
            raise FileNotFoundError(path)
        data = self.files[p] or b""
        for i in range(0, len(data), self.chunk):
            yield data[i:i + self.chunk]

    def remove_file(self, path: str, recursive: bool = False) -> None:
        p = self._norm(path)
        if p not in self.sizes:
            raise FileNotFoundError(path)
        for d in (self.files, self.sizes, self.sha256):
            d.pop(p, None)

    def batch_upload(self, force: bool = False) -> _Batch:
        return _Batch(self, force)

    def _store(self, source, remote: str) -> None:
        p = self._norm(remote)
        self.remote_paths.append(remote)
        h, size, parts = hashlib.sha256(), 0, []
        with contextlib.ExitStack() as stack:
            fh = stack.enter_context(open(source, "rb")) if isinstance(source, (str, Path)) else source
            while True:
                block = fh.read(self.chunk)
                if not block:
                    break
                self.max_read = max(self.max_read, len(block))
                h.update(block)
                size += len(block)
                if self.keep:
                    parts.append(block)
        self.files[p] = b"".join(parts) if self.keep else None
        self.sizes[p] = size
        self.sha256[p] = h.hexdigest()


class _Batch:
    def __init__(self, volume: FakeVolume, force: bool) -> None:
        self.volume = volume
        self.force = force
        self.pending: list[tuple[object, str]] = []

    def __enter__(self) -> Self:
        return self

    def put_file(self, local_file, remote_path: str, mode: int | None = None) -> None:
        if str(remote_path).endswith("/"):
            raise ValueError("remote_path must refer to a file")
        self.pending.append((local_file, str(remote_path)))

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            return
        if not self.force:
            taken = [r for _, r in self.pending if FakeVolume._norm(r) in self.volume.sizes]
            if taken:
                raise FileExistsError(f"files already exist: {taken}")
        for source, remote in self.pending:
            self.volume._store(source, remote)
