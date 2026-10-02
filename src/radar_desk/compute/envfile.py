"""Read a `.env` file the way the app does.

`read_env` follows the app's reading of `.env`: `KEY=value`, an optional `export`, comments, blank lines,
matching quotes stripped, no interpolation, the last definition wins.
"""

from __future__ import annotations

import re
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
