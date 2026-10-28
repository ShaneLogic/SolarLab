"""Explicit local routes over a read handle and optional shared execution owner.

The factory has no import-time/file/network side effects. ASGI lifespan opens
and closes its own handle to an EXISTING local RunStore; the coordinating
writer remains external. Opening the reader never recovers attempts, cancels
work, or starts a solver. Do not expose this unauthenticated preparation on a
public interface: deployment/authentication and the production switch are
separate integration work.

Runs and attempt records include stored opaque execution metadata and results.
Event pages span a run's history; SSE URLs explicitly select one attempt.
Submission requires an explicit source-checked legacy execution catalog.
Server shutdown and subscriber disconnect never cancel the external owner.
Unregistered experiments and the new scientific engine remain unavailable.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import math
from pathlib import Path
import sqlite3
from typing import cast

from fastapi import FastAPI, Request
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from solarlab.io.artifacts import ArtifactIntegrityError
from solarlab.io.run_store import RunStore, StateConflict
from solarlab_server.configuration import ConfigurationPreviewContext
from solarlab_server.job_service import JobService
from solarlab_server.routes.configs import router as configuration_router
from solarlab_server.routes.jobs import router as jobs_router
from solarlab_server.routes.materials import router as materials_router
from solarlab_server.routes.schema import router as schema_router
from solarlab_server.routes.results import install_results, verified_artifact
from solarlab_server.routes.legacy import install_legacy


def _store(request: Request) -> RunStore:
    return cast(RunStore, request.app.state.run_store)


_verified_artifact = verified_artifact  # Preserve the previous internal reader name.


def create_app(
    store_root: str | Path, *, event_page_size: int = 100,
    poll_interval: float = 0.1, keepalive_interval: float = 15.0,
    max_artifact_bytes: int = 16 * 1024**2,
    configuration_preview: ConfigurationPreviewContext | None = None,
    job_service: JobService | None = None,
) -> FastAPI:
    """Build a local reader; start/stop its owned handle with ASGI lifespan.

    ``store_root`` must name an existing canonical absolute local store.
    Limits describe transport, not numerical controls or physical defaults.
    JSON pages accept ``after``/``limit``; SSE accepts ``after`` and gives
    ``Last-Event-ID`` precedence. Artifact bodies are exact validated bytes.
    Preview POSTs resolve DTOs only when the caller supplies trusted catalog
    and resource data through ``configuration_preview``; they never execute.
    ``job_service`` explicitly adds checked legacy submissions through an
    existing shared supervisor. Its lifecycle is owned by the caller.
    """
    if type(event_page_size) is not int or not 1 <= event_page_size <= 1000:
        raise ValueError("event_page_size must be between 1 and 1000")
    for value in (poll_interval, keepalive_interval):
        if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
            raise ValueError("poll/keepalive intervals must be positive and finite")
    if type(max_artifact_bytes) is not int or max_artifact_bytes < 0:
        raise ValueError("max_artifact_bytes must be a nonnegative integer")
    if configuration_preview is not None and not isinstance(configuration_preview, ConfigurationPreviewContext):
        raise TypeError("configuration_preview must be an explicit ConfigurationPreviewContext")
    root = Path(store_root)
    if not root.is_absolute():
        raise ValueError("store_root must be an explicit absolute local path")
    if job_service is not None:
        if type(job_service) is not JobService or job_service.supervisor.store.root != root:
            raise ValueError("job service must share this exact local store and external process owner")
        job_service.catalog.verify()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if not (root / "store.sqlite3").is_file():
            raise ValueError("the read service requires an existing RunStore")
        store = await run_in_threadpool(RunStore, root, read_only=True)
        app.state.run_store = store
        try:
            yield
        finally:
            await run_in_threadpool(store.close)
            app.state.run_store = None

    app = FastAPI(title="SolarLab stored runs", lifespan=lifespan)
    app.state.run_store = None
    app.state.configuration_preview = configuration_preview
    app.state.job_service = job_service
    app.include_router(schema_router)
    app.include_router(configuration_router)
    app.include_router(materials_router)
    app.include_router(jobs_router)

    @app.exception_handler(KeyError)
    async def not_found(request: Request, error: KeyError) -> JSONResponse:
        return JSONResponse({"detail": "unknown stored identifier"}, status_code=404)

    @app.exception_handler(ArtifactIntegrityError)
    async def artifact_integrity(request: Request, error: ArtifactIntegrityError) -> JSONResponse:
        return JSONResponse({"detail": {"code": "artifact_integrity", "message": str(error)}}, status_code=409)

    @app.exception_handler(StateConflict)
    async def conflict(request: Request, error: StateConflict) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=409)

    @app.exception_handler(ValueError)
    async def invalid(request: Request, error: ValueError) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=400)

    @app.exception_handler(sqlite3.OperationalError)
    @app.exception_handler(TimeoutError)
    async def unavailable(request: Request, error: Exception) -> JSONResponse:
        return JSONResponse({"detail": "stored data temporarily unavailable"}, status_code=503)

    install_results(app, lambda request: _store(request), event_page_size=event_page_size,
        poll_interval=poll_interval, keepalive_interval=keepalive_interval, max_artifact_bytes=max_artifact_bytes)
    install_legacy(app, max_artifact_bytes=max_artifact_bytes, event_page_size=event_page_size,
                   poll_interval=poll_interval, keepalive_interval=keepalive_interval)
    return app
