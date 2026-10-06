"""Explicit local read service, prepared_pending_dependencies for P06-06.

The factory has no import-time/file/network side effects. ASGI lifespan opens
and closes its own handle to an EXISTING local RunStore; the coordinating
writer remains external. Opening the reader never recovers attempts, cancels
work, or starts a solver. Do not expose this unauthenticated preparation on a
public interface: deployment/authentication and the production switch are
separate integration work.

Runs and attempt records include stored opaque execution metadata and results.
Event pages span a run's history; SSE URLs explicitly select one attempt.
Legacy result/error/done framing, submission, configuration editing and registry
routes remain upstream integration obligations.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import math
from pathlib import Path
import sqlite3
from typing import Any, cast

from fastapi import FastAPI, Header, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse, Response, StreamingResponse

from solarlab.io.artifacts import ArtifactIntegrityError
from solarlab.io.run_store import RunStore, StateConflict
from solarlab_server.events import MAX_SEQUENCE, attempt_record, event_stream, replay_cursor
from solarlab_server.schema import router as schema_router


def _store(request: Request) -> RunStore:
    return cast(RunStore, request.app.state.run_store)


def _verified_artifact(
    store: RunStore, artifact_id: str, maximum_bytes: int,
) -> tuple[dict[str, Any], bytes]:
    record = store.artifact(artifact_id)
    if record["phase"] != "committed":
        raise StateConflict("artifact is not a committed result")
    if record["size_bytes"] > maximum_bytes:
        raise HTTPException(413, detail="artifact exceeds this reader's byte limit")
    # Return this verified independent byte snapshot, never a file that would
    # be opened again after validation. Metadata-only reads also validate it.
    return record, store.read_artifact(artifact_id)


def create_app(
    store_root: str | Path, *, event_page_size: int = 100,
    poll_interval: float = 0.1, keepalive_interval: float = 15.0,
    max_artifact_bytes: int = 16 * 1024**2,
) -> FastAPI:
    """Build a local reader; start/stop its owned handle with ASGI lifespan.

    ``store_root`` must name an existing canonical absolute local store.
    Limits describe transport, not numerical controls or physical defaults.
    JSON pages accept ``after``/``limit``; SSE accepts ``after`` and gives
    ``Last-Event-ID`` precedence. Artifact bodies are exact validated bytes.
    """
    if type(event_page_size) is not int or not 1 <= event_page_size <= 1000:
        raise ValueError("event_page_size must be between 1 and 1000")
    for value in (poll_interval, keepalive_interval):
        if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
            raise ValueError("poll/keepalive intervals must be positive and finite")
    if type(max_artifact_bytes) is not int or max_artifact_bytes < 0:
        raise ValueError("max_artifact_bytes must be a nonnegative integer")
    root = Path(store_root)
    if not root.is_absolute():
        raise ValueError("store_root must be an explicit absolute local path")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if not (root / "store.sqlite3").is_file():
            raise ValueError("the read service requires an existing RunStore")
        store = await run_in_threadpool(RunStore, root)
        app.state.run_store = store
        try:
            yield
        finally:
            await run_in_threadpool(store.close)
            app.state.run_store = None

    app = FastAPI(title="SolarLab stored runs", lifespan=lifespan)
    app.state.run_store = None
    app.include_router(schema_router)

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

    @app.get("/runs/{run_id}")
    def run(request: Request, run_id: str) -> JSONResponse:
        return JSONResponse(_store(request).get_run(run_id))

    @app.get("/runs/{run_id}/attempts/{attempt_id}")
    def attempt(request: Request, run_id: str, attempt_id: str) -> JSONResponse:
        return JSONResponse(attempt_record(_store(request), run_id, attempt_id))

    @app.get("/runs/{run_id}/events")
    def events(
        request: Request, run_id: str,
        after: int = Query(0, ge=0, le=MAX_SEQUENCE),
        limit: int = Query(event_page_size, ge=1, le=event_page_size),
    ) -> JSONResponse:
        store = _store(request)
        store.get_run(run_id)
        page = store.events(after=after, limit=limit, run_id=run_id)
        return JSONResponse({"store_id": store.store_id, "events": page,
                             "next_cursor": page[-1]["sequence"] if page else after})

    @app.get("/runs/{run_id}/attempts/{attempt_id}/events")
    async def stream(
        request: Request, run_id: str, attempt_id: str,
        after: int = Query(0, ge=0, le=MAX_SEQUENCE),
        last_event_id: str | None = Header(None, alias="Last-Event-ID"),
    ) -> Response:
        store = _store(request)
        await run_in_threadpool(attempt_record, store, run_id, attempt_id)
        cursor, terminal_acknowledged = await run_in_threadpool(
            replay_cursor, store, run_id, attempt_id, after, last_event_id,
        )
        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                   "X-RunStore-ID": store.store_id}
        if terminal_acknowledged:
            return Response(status_code=204, headers=headers)
        return StreamingResponse(
            event_stream(request, store, run_id, attempt_id, after=cursor,
                         page_size=event_page_size, poll_interval=poll_interval,
                         keepalive_interval=keepalive_interval),
            media_type="text/event-stream", headers=headers,
        )

    @app.get("/artifacts/{artifact_id}")
    def artifact(request: Request, artifact_id: str) -> JSONResponse:
        record, _ = _verified_artifact(_store(request), artifact_id, max_artifact_bytes)
        return JSONResponse(record)

    @app.get("/artifacts/{artifact_id}/content")
    def content(request: Request, artifact_id: str) -> Response:
        record, data = _verified_artifact(_store(request), artifact_id, max_artifact_bytes)
        return Response(data, media_type="application/octet-stream",
                        headers={"ETag": f'"{record["sha256"]}"'})

    return app
