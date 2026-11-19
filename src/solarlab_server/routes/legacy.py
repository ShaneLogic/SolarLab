"""Solver-free historical RunStore readback for old stored runs; legacy execution retired."""
from __future__ import annotations

from collections.abc import AsyncIterator
import json
import math
import struct
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse, Response, StreamingResponse

from solarlab_server.events import attempt_record, event_stream, replay_cursor
from solarlab_server.job_service import legacy_job_target
from solarlab_server.routes.results import result_payload


def _old_json(value: Any) -> Any:
    """Project verified W03 words to the old JSON view; raw artifacts survive."""
    import numpy as np
    if isinstance(value, list):
        return [_old_json(x) for x in value]
    if type(value) is dict:
        if set(value) == {"type", "fields"}:
            return _old_json(value["fields"])
        if set(value) == {"binary64_be"}:
            number = struct.unpack(">d", bytes.fromhex(value["binary64_be"]))[0]
            return number if math.isfinite(number) else None
        if set(value) in ({"dtype", "shape", "raw_hex"}, {"dtype", "raw_hex"}):
            dtype = np.dtype(value["dtype"])
            if dtype.hasobject or dtype.fields:
                raise ValueError("unsupported legacy result dtype")
            raw = bytes.fromhex(value["raw_hex"])
            shape = value.get("shape", [])
            if (any(type(n) is not int or n < 0 for n in shape)
                    or math.prod(shape) * dtype.itemsize != len(raw)):
                raise ValueError("legacy result words differ from shape")
            return _old_json(np.frombuffer(raw, dtype=dtype).reshape(shape).tolist())
        return {key: _old_json(item) for key, item in value.items()}
    if type(value) is float and not math.isfinite(value):
        return None
    return value


def legacy_result(store: Any, run_id: str, attempt_id: str, maximum_bytes: int) -> dict[str, Any]:
    view = result_payload(store, run_id, attempt_id, maximum_bytes)
    payload = next((a for a in view["artifacts"] if a["name"] == "legacy_result"), None)
    if payload is not None and payload["size_bytes"] <= 256 * 1024:
        decoded = json.loads(store.read_artifact(payload["artifact_id"]))
        if isinstance(decoded, dict) and decoded.get("schema") == "solarlab.legacy-raw-result.v1" and "result" in decoded:
            decoded = decoded["result"]
        result = _old_json(decoded)
        if type(result) is not dict:
            result = {"value": result}
    else:
        result = {"payload_inline": False}
    return {**result, "artifacts": view["artifacts"], "execution": view["result"],
            "run_id": run_id, "attempt_id": attempt_id}


def legacy_cursor(store: Any, run_id: str, attempt_id: str, last: str | None) -> tuple[int, str | None, bool]:
    if last is None:
        return 0, None, False
    words = last.split(".")
    if len(words) != 2 or words[1] not in {"event", "result", "error", "done"}:
        raise ValueError("legacy cursor requires a durable sequence and frame phase")
    sequence, terminal = replay_cursor(store, run_id, attempt_id, 0, words[0])
    if sequence == 0:
        raise ValueError("legacy phase cursor must identify a committed event")
    if not terminal:
        if words[1] != "event":
            raise ValueError("nonterminal event cannot acknowledge a terminal frame")
        return sequence, None, False
    event = store.events(after=sequence - 1, limit=1, run_id=run_id)[0]
    expected = "result" if event["payload"]["state"] == "succeeded" else "error"
    if words[1] not in {expected, "done"}:
        raise ValueError("cursor phase differs from the committed terminal outcome")
    return sequence - 1, words[1], words[1] == "done"


def install_legacy(app: FastAPI, *, max_artifact_bytes: int, event_page_size: int,
                   poll_interval: float, keepalive_interval: float) -> None:
    @app.post("/api/jobs")
    async def retired_jobs() -> JSONResponse:
        raise HTTPException(
            410,
            detail={
                "code": "legacy_execution_retired",
                "message": "POST /api/jobs is retired; submit V2 jobs via POST /jobs with engine='solarlab'",
            },
        )

    @app.get("/api/jobs/{job_id}/result")
    def read_historical_result(request: Request, job_id: str) -> JSONResponse:
        store = request.app.state.run_store
        run_id, attempt_id = legacy_job_target(job_id)
        return JSONResponse(
            legacy_result(store, run_id, attempt_id, max_artifact_bytes),
            headers={"X-RunStore-ID": store.store_id},
        )

    @app.get("/api/jobs/{job_id}/events")
    async def events(request: Request, job_id: str, last_event_id: str | None = Header(None, alias="Last-Event-ID")) -> Response:
        # Read-only historical RunStore replay requires zero live execution service or legacy solver imports.
        store = request.app.state.run_store
        run_id, attempt_id = legacy_job_target(job_id)
        await run_in_threadpool(attempt_record, store, run_id, attempt_id)
        cursor, phase, acknowledged = await run_in_threadpool(legacy_cursor, store, run_id, attempt_id, last_event_id)
        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-RunStore-ID": store.store_id}
        if acknowledged:
            return Response(status_code=204, headers=headers)

        async def stream() -> AsyncIterator[str]:
            async for block in event_stream(request, store, run_id, attempt_id, after=cursor,
                    page_size=event_page_size, poll_interval=poll_interval, keepalive_interval=keepalive_interval):
                if block.startswith(":"):
                    yield block
                    continue
                record = json.loads(next(line[6:] for line in block.splitlines() if line.startswith("data: ")))
                sequence = record["sequence"]
                if record["kind"] == "terminal":
                    event = "result" if record["payload"]["state"] == "succeeded" else "error"
                    if phase is None:
                        if event == "result":
                            data = await run_in_threadpool(legacy_result, store, run_id, attempt_id, max_artifact_bytes)
                        else:
                            result = record["payload"]["result"]
                            cause = result.get("worker_result", result)
                            data = {"message": cause.get("reason", record["payload"]["state"]),
                                    "state": record["payload"]["state"], "result": result}
                        yield f"id: {sequence}.{event}\nevent: {event}\ndata: {json.dumps(data, allow_nan=False)}\n\n"
                    yield f"id: {sequence}.done\nevent: done\ndata: {{}}\n\n"
                    return
                data = record["payload"].get("ui_progress", {"stage": record["kind"]})
                yield f"id: {sequence}.event\nevent: progress\ndata: {json.dumps(data, allow_nan=False)}\n\n"
        return StreamingResponse(stream(), media_type="text/event-stream", headers=headers)

    @app.post("/api/{retired_path:path}")
    async def retired_synchronous(retired_path: str) -> JSONResponse:
        raise HTTPException(
            410,
            detail={
                "code": "legacy_execution_retired",
                "message": f"POST /api/{retired_path} is retired; submit V2 jobs via POST /jobs with engine='solarlab'",
            },
        )
