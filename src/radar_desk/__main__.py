"""`python -m radar_desk`: serve the app with uvicorn on HOST:PORT (default 127.0.0.1:8000)."""

from __future__ import annotations

import os
import sys

import uvicorn

from radar_desk.app import create_app
from radar_desk.config import ConfigError, load_settings


def main() -> None:
    try:
        app = create_app(load_settings())
    except ConfigError as exc:
        sys.exit(f"radar-desk: {exc}")
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port, proxy_headers=False)


if __name__ == "__main__":
    main()
