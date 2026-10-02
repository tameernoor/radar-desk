"""Write the scores of every finished job to one CSV, the same content as GET /export/scores.csv.

Runs against the local database and storage with the app's settings (environment or a .env in the
working directory), like seed_dev.py:

    .venv/bin/python scripts/export_all.py [out.csv]

The default path is exports/scores-<date>.csv under DATA_DIR (data/exports/... by default). Prints the number of rows written.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from datetime import UTC, datetime
from pathlib import Path

from radar_desk.config import ConfigError, load_settings
from radar_desk.services import build_services


def write_export(services, out: Path) -> int:
    """Write the CSV to `out` and return the number of data rows (the header not counted)."""
    data = services.exports.export_all_csv()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return max(len(list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))) - 1, 0)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export the scores of every finished job to a CSV.")
    parser.add_argument("out", nargs="?", type=Path, help="default: DATA_DIR/exports/scores-<date>.csv")
    args = parser.parse_args(argv)
    try:
        settings = load_settings()
    except ConfigError as exc:
        sys.exit(f"export_all: {exc}")
    out = args.out or Path(settings.data_dir) / "exports" / f"scores-{datetime.now(UTC).date().isoformat()}.csv"
    rows = write_export(build_services(settings), out)
    print(f"{rows} rows written to {out}")


if __name__ == "__main__":
    main()
