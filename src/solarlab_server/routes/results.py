"""Stored results and canonical durable event routes, moved without wire changes."""
from __future__ import annotations
from collections.abc import Callable
from typing import Any
from fastapi import FastAPI, Header, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse, Response, StreamingResponse
from solarlab.io.run_store import RunStore, StateConflict
from solarlab_server.events import MAX_SEQUENCE, attempt_record, event_stream, replay_cursor


def verified_artifact(store: RunStore, artifact_id: str, maximum_bytes: int) -> tuple[dict[str, Any], bytes]:
    record = store.artifact(artifact_id)
    if record["phase"] != "committed":
        raise StateConflict("artifact is not a committed result")
    if record["size_bytes"] > maximum_bytes:
        raise HTTPException(413, detail="artifact exceeds this reader's byte limit")
    return record, store.read_artifact(artifact_id)


def result_payload(store: RunStore, run_id: str, attempt_id: str, maximum_bytes: int) -> dict[str, Any]:
    attempt = attempt_record(store, run_id, attempt_id)
    if attempt["state"] not in {"succeeded", "failed", "cancelled"}:
        raise HTTPException(409, "attempt has no committed terminal result")
    artifacts = []
    for reference in attempt["terminal"]["artifacts"]:
        record, _ = verified_artifact(store, reference["artifact_id"], maximum_bytes)
        if (record["attempt_id"] != attempt_id or record["run_id"] != run_id
                or record["generation"] != attempt["generation"] or record["manifest_sha256"] != reference["manifest_sha256"]):
            raise StateConflict("terminal artifact belongs to a different attempt/generation")
        artifacts.append({**record, "content_url": f"/artifacts/{record['artifact_id']}/content"})
    return {"run_id": run_id, "attempt_id": attempt_id, "state": attempt["state"],
            "result": attempt["result"], "artifacts": artifacts}


def install_results(app: FastAPI, get_store: Callable[[Request], RunStore], *, event_page_size: int,
                    poll_interval: float, keepalive_interval: float, max_artifact_bytes: int) -> None:
    @app.get("/runs/{run_id}")
    def run(request: Request, run_id: str) -> JSONResponse:
        store = get_store(request)
        return JSONResponse(store.get_run(run_id), headers={"X-RunStore-ID": store.store_id})

    @app.get("/runs/{run_id}/attempts/{attempt_id}")
    def attempt(request: Request, run_id: str, attempt_id: str) -> JSONResponse:
        store = get_store(request)
        return JSONResponse(attempt_record(store, run_id, attempt_id),
                            headers={"X-RunStore-ID": store.store_id})

    @app.get("/runs/{run_id}/events")
    def events(
        request: Request, run_id: str,
        after: int = Query(0, ge=0, le=MAX_SEQUENCE),
        limit: int = Query(event_page_size, ge=1, le=event_page_size),
    ) -> JSONResponse:
        store = get_store(request)
        store.get_run(run_id)
        page = store.events(after=after, limit=limit, run_id=run_id)
        return JSONResponse({"store_id": store.store_id, "events": page,
                             "next_cursor": page[-1]["sequence"] if page else after},
                            headers={"X-RunStore-ID": store.store_id})

    @app.get("/runs/{run_id}/attempts/{attempt_id}/events")
    async def stream(
        request: Request, run_id: str, attempt_id: str,
        after: int = Query(0, ge=0, le=MAX_SEQUENCE),
        last_event_id: str | None = Header(None, alias="Last-Event-ID"),
    ) -> Response:
        store = get_store(request)
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
        store = get_store(request)
        record, _ = verified_artifact(store, artifact_id, max_artifact_bytes)
        return JSONResponse(record, headers={"X-RunStore-ID": store.store_id})

    @app.get("/artifacts/{artifact_id}/content")
    def content(request: Request, artifact_id: str) -> Response:
        store = get_store(request)
        record, data = verified_artifact(store, artifact_id, max_artifact_bytes)
        return Response(data, media_type="application/octet-stream",
                        headers={"ETag": f'"{record["sha256"]}"', "X-RunStore-ID": store.store_id})

    @app.get("/results/{run_id}/attempts/{attempt_id}")
    def result(request: Request, run_id: str, attempt_id: str) -> JSONResponse:
        store = get_store(request)
        return JSONResponse(result_payload(store, run_id, attempt_id, max_artifact_bytes),
                            headers={"X-RunStore-ID": store.store_id})
