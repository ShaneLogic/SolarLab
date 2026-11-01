"""Opt-in old HTTP framing; durable sequence plus phase prevents terminal gaps."""
from __future__ import annotations

from dataclasses import asdict
from collections.abc import AsyncIterator, Awaitable, Callable
import json
import math
import struct
from typing import Any
import uuid

from fastapi import FastAPI, Header, HTTPException, Request
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse, Response, StreamingResponse

from solarlab.experiments.execution import JobHandle
from solarlab_server.events import attempt_record, event_stream, replay_cursor
from solarlab_server.job_service import JobSubmission, legacy_job_target
from solarlab_server.routes.jobs import body, service, submitted
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
        result = _old_json(json.loads(store.read_artifact(payload["artifact_id"])))
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
    async def submit_legacy(request: Request, *, fixed_key: tuple[str, int] | None = None) -> dict[str, Any]:
        context = service(request)
        if not context.allow_legacy_routes:
            raise HTTPException(404, "legacy compatibility routes are not enabled")
        value = await body(request)
        if fixed_key is None:
            if set(value) - {"kind", "device", "config_path", "params"} or "kind" not in value:
                raise HTTPException(422, "legacy jobs require kind/device/config_path/params")
            registration = next((r for r in context.catalog.registrations if value["kind"] in r.legacy_kinds), None)
            parameters = {"device": value.get("device"), "config_path": value.get("config_path"), "params": value.get("params", {})}
        else:
            registration = next((r for r in context.catalog.registrations
                                 if (r.experiment_id, r.version) == fixed_key), None)
            parameters = {"device": value.get("device"), "config_path": value.get("config_path"),
                          "params": {k: v for k, v in value.items() if k not in {"device", "config_path"}}}
        if registration is None:
            raise HTTPException(422, "legacy caller has not migrated; retain its original entry")
        declaration = JobSubmission.model_validate({"schema_version": "solarlab.job-submission.v1", "run_id": uuid.uuid4().hex,
            "experiment_id": registration.experiment_id, "version": registration.version, "engine": "legacy",
            "input": parameters, "limits": asdict(context.catalog.ceiling)})
        result = await run_in_threadpool(submitted, context, declaration)
        return {**result, "status": "ok", "legacy_events_url": f"/api/jobs/{result['job_id']}/events"}

    @app.post("/api/jobs")
    async def jobs(request: Request) -> JSONResponse:
        return JSONResponse(await submit_legacy(request), status_code=202)

    @app.get("/api/jobs/{job_id}/events")
    async def events(request: Request, job_id: str, last_event_id: str | None = Header(None, alias="Last-Event-ID")) -> Response:
        # A read-only restart needs no live execution service to replay old jobs.
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

    context = app.state.job_service
    if context is not None and context.allow_legacy_routes:
        def endpoint(experiment_id: str, version: int) -> Callable[[Request], Awaitable[JSONResponse]]:
            async def synchronous(request: Request) -> JSONResponse:
                submitted_job = await submit_legacy(request, fixed_key=(experiment_id, version))
                handle = JobHandle(submitted_job["store_id"], submitted_job["run_id"], submitted_job["attempt_id"])
                try:
                    receipt = await run_in_threadpool(context.supervisor.wait, handle, context.synchronous_wait_seconds)
                except TimeoutError:
                    return JSONResponse({**submitted_job, "status": "pending"}, status_code=202)
                if receipt.data["store_state"] != "succeeded":
                    return JSONResponse({"status": "error", "job": submitted_job, "result": receipt.data}, status_code=422)
                result = await run_in_threadpool(legacy_result, context.supervisor.store, handle.run_id, handle.attempt_id, max_artifact_bytes)
                return JSONResponse({"status": "ok", "result": result})
            return synchronous
        for registration in context.catalog.registrations:
            for path in registration.synchronous_paths:
                app.add_api_route(path, endpoint(registration.experiment_id, registration.version), methods=["POST"])
