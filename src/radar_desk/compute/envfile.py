"""Read and edit a `.env` file without touching anything but the one key.

`read_env` follows the app's reading of `.env`: `KEY=value`, an optional `export`, comments, blank lines,
matching quotes stripped, no interpolation, the last definition wins.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _strip_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return re.split(r"\s+#", value, maxsplit=1)[0].rstrip()


def read_env(path: Path | str) -> dict[str, str]:
    """The keys and values in `path`, or {} when the file does not exist."""
    path = Path(path)
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        line = line.removeprefix("export ")
        key, _, value = line.partition("=")
        key = key.strip()
        if _KEY.match(key):
            out[key] = _strip_value(value)
    return out


def set_key(path: Path | str, key: str, value: str) -> bool:
    """Set `key` to `value` in `path` and return whether the file's bytes changed.

    Every active line for the key is rewritten after its `=`, keeping its line ending; without one,
    `KEY=value` is appended on a new line. The write is atomic and keeps the file's mode, 600 for a new file.
    """
    path = Path(path)
    exists = path.exists()
    old = path.read_bytes().decode("utf-8") if exists else ""
    mode = (path.stat().st_mode & 0o777) if exists else 0o600
    active = re.compile(rf"^(\s*(?:export\s+)?{re.escape(key)}\s*=[ \t]*)(.*?)(\r?\n)?$", re.DOTALL)
    lines = old.splitlines(keepends=True)
    found = False
    for i, line in enumerate(lines):
        m = active.match(line)
        if m:
            found = True
            lines[i] = f"{m.group(1)}{value}{m.group(3) or ''}"
    if not found:
        newline = "\r\n" if lines and lines[0].endswith("\r\n") else "\n"
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += newline
        lines.append(f"{key}={value}{newline}")
    new = "".join(lines)
    if exists and new == old:
        return False
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(new.encode("utf-8"))
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return True
