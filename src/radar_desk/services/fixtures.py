"""The fixture manifest and the reference scores that compare() reads."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from radar_desk.radar import catalog

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "fixtures"


class FixtureService:
    def __init__(self, root: Path = DEFAULT_ROOT) -> None:
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        data = json.loads(manifest_path.read_text("utf-8")) if manifest_path.is_file() else {"scans": {}}
        self._by_sha = data.get("scans", {})
        self._by_id = {entry["id"]: {**entry, "sha256": sha} for sha, entry in self._by_sha.items()}

    def id_for_sha256(self, sha256: str | None) -> str | None:
        entry = self._by_sha.get(sha256 or "")
        return entry["id"] if entry else None

    def get(self, fixture_id: str) -> dict | None:
        return self._by_id.get(fixture_id)

    def list(self) -> list[dict]:
        """The body of GET /fixtures: each entry with `available` on its references."""
        out = []
        for entry in self._by_id.values():
            refs = [{**r, "available": self.load_reference(entry["id"], r) is not None}
                    for r in entry.get("references", [])]
            out.append({**entry, "references": refs})
        return out

    def load_reference(self, fixture_id: str, ref: dict) -> dict[str, float | None] | None:
        """Reference probabilities by finding key, or None when the file is absent or lacks this scan."""
        path = self.root / ref["file"]
        if not path.is_file():
            return None
        source = ref.get("source")
        if source == "tally":
            scans = json.loads(path.read_text("utf-8")).get("scans", {})
            if fixture_id not in scans:
                return None
            return {catalog.english_to_key(k): float(v) for k, v in scans[fixture_id].items()}
        if path.suffix == ".csv":
            return self._csv_reference(path, fixture_id)
        data = json.loads(path.read_text("utf-8"))
        return {f["key"]: (None if f.get("prob") is None else float(f["prob"])) for f in data["findings"]}

    def _csv_reference(self, path: Path, fixture_id: str) -> dict[str, float | None] | None:
        with path.open(encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.reader(fh))
        header, body = rows[0], rows[1:]
        filename = (self.get(fixture_id) or {}).get("filename")
        row = next((r for r in body if r and r[0] == filename), body[0] if len(body) == 1 else None)
        if row is None:
            return None
        keys = [h.split(" (", 1)[0] for h in header[1:]]
        return {k: (float(v) if v != "" else None) for k, v in zip(keys, row[1:], strict=True)}
