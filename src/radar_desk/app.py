"""The FastAPI app: JSON API, local or Volume storage routes, and the built front end at "/"."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from radar_desk.auth import Auth
from radar_desk.chat.agent import ChatAgent, make_llm
from radar_desk.config import ConfigError, load_settings
from radar_desk.gpu.poller import Poller
from radar_desk.routes import (
    auth,
    catalog,
    chat,
    compute,
    export,
    fixtures,
    gpu,
    health,
    jobs,
    local_storage,
    results,
    scans,
    uploads,
    volume_storage,
    worker,
    workers,
)
from radar_desk.routes.health import VERSION
from radar_desk.services import ServiceError, Services, build_services
from radar_desk.services.compute import MODAL_NEEDS_STORAGE
from radar_desk.storage import LocalStorage, ModalVolumeStorage, storage_backend

log = logging.getLogger(__name__)

WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"
LOCAL_HOSTS = ["localhost", "127.0.0.1", "::1", "[::1]"]


def create_app(settings: Any = None, services: Services | None = None, start_poller: bool = True,
               web_dist: Path = WEB_DIST) -> FastAPI:
    """Build the app. Services are built here when not given, so routes work with or without the lifespan;
    the lifespan only runs the poller."""
    settings = settings if settings is not None else (services.settings if services else load_settings())
    if settings.gpu_backend == "modal" and storage_backend(settings) == "local":
        raise ConfigError(MODAL_NEEDS_STORAGE)
    services = services if services is not None else build_services(settings)
    if services.compute.changeable and services.compute.mode == "modal" and storage_backend(settings) == "local":
        services.compute.fall_back_from_modal()  # refusing would leave the owner no way to switch back

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        stop = asyncio.Event()
        task = None
        if start_poller:
            try:
                await asyncio.to_thread(services.compute.reconcile_at_start, time.time())
            except Exception:
                log.exception("compute: reconcile at start failed")
            task = asyncio.create_task(Poller(services).run(stop), name="radar-poller")
            log.info("poller started in compute mode %s", services.compute.mode)
        try:
            yield
        finally:
            stop.set()
            if task is not None:
                with contextlib.suppress(asyncio.CancelledError, TimeoutError):
                    await asyncio.wait_for(task, timeout=settings.gpu_poll_interval_s + 5)

    app = FastAPI(title="radar-desk", version=VERSION, lifespan=lifespan,
                  description="RADAR abdominal CT scoring. Research use only.")
    app.state.settings = settings
    app.state.services = services
    app.state.auth = Auth(settings)
    app.state.chat_agent = ChatAgent(services, make_llm(settings))
    app.state.sse_interval_s = 1.0

    if settings.app_hostname:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=[settings.app_hostname, *LOCAL_HOSTS])

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        parts = []
        for err in exc.errors():
            where = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
            parts.append(f"{where}: {err.get('msg')}" if where else str(err.get("msg")))
        return JSONResponse({"detail": "; ".join(parts) or "invalid request"}, status_code=422)

    for module in (health, auth, uploads, scans, jobs, results, export, gpu, catalog, fixtures, chat, worker,
                   workers, compute):
        app.include_router(module.router)
    if isinstance(services.storage, LocalStorage):
        app.include_router(local_storage.router)
    if isinstance(services.storage, ModalVolumeStorage):
        app.include_router(volume_storage.router)

    if Path(web_dist).is_dir():
        app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")
    return app
