"""Small objects the app reads and rewrites in storage: JSON documents, a text file, an append-only log.

Objects on the Modal Volume are write-once (ObjectExists), so a rewrite deletes and writes again there.
"""

from __future__ import annotations

import json
from typing import Any

from radar_desk.storage import ObjectExists, ObjectMissing


def read_bytes(storage: Any, key: str) -> bytes | None:
    try:
        return b"".join(storage.open_stream(key))
    except ObjectMissing:
        return None


def read_json(storage: Any, key: str) -> dict | None:
    """The object as a JSON object, or None when it is missing or not a JSON object."""
    data = read_bytes(storage, key)
    if data is None:
        return None
    try:
        value = json.loads(data)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def replace_bytes(storage: Any, key: str, data: bytes, content_type: str | None = None) -> None:
    try:
        storage.put_bytes(key, data, content_type)
    except ObjectExists:
        storage.delete(key)
        storage.put_bytes(key, data, content_type)


def write_json(storage: Any, key: str, value: dict) -> None:
    replace_bytes(storage, key, json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json")
