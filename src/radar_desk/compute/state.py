"""The compute switch's state file: ids and pids only, never a secret, mode 600."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def empty() -> dict:
    return {"tunnel": None, "pod": None, "token_id": None, "worker_id": None}


def load(path: Path | str) -> dict:
    path = Path(path)
    state = empty()
    if path.exists():
        state.update(json.loads(path.read_text(encoding="utf-8")))
    return state


def save(path: Path | str, state: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            f.write("\n")
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
