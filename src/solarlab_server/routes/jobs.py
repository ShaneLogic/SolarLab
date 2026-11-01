"""Stable-ID submissions and exact-attempt cancellation on the shared owner."""
from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from solarlab.experiments.execution import JobHandle, QueueFull
from solarlab.io.run_store import StateConflict
from solarlab_server.configuration import _constant, _number, _pairs
from solarlab_server.job_service import JobService, JobSubmission, legacy_job_id

router = APIRouter()


def service(request: Request) -> JobService:
    value = cast(JobService | None, request.app.state.job_service)
    if value is None:
        raise HTTPException(503, detail={"code": "execution_unavailable", "message": "No trusted execution catalog/owner supplied"})
    return value


async def body(request: Request, limit: int = 1024**2) -> dict[str, Any]:
    import json
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        raise HTTPException(415, "Expected application/json")
    raw = bytearray()
    async for chunk in request.stream():
        if len(raw) + len(chunk) > limit:
            raise HTTPException(413, "job submission exceeds the bounded input limit")
        raw.extend(chunk)
    try:
        value = json.loads(raw, object_pairs_hook=_pairs, parse_float=_number, parse_constant=_constant,
                           parse_int=lambda word: -0.0 if word == "-0" else int(word))
        if type(value) is not dict:
            raise ValueError("job body must be an object")
        return value
    except (ValueError, UnicodeError, RecursionError) as error:
        raise HTTPException(400, str(error)) from error


def submitted(service: JobService, declaration: JobSubmission) -> dict[str, Any]:
    try:
        handle = service.submit(declaration)
    except ValidationError as error:
        raise HTTPException(422, error.errors(include_url=False, include_context=False)) from error
    except ValueError as error:
        raise HTTPException(422, {"code": "capability_or_binding_refused", "message": str(error)}) from error
    except QueueFull as error:
        raise HTTPException(429, {"code": "queue_full", "message": str(error)}, headers={"Retry-After": "1"}) from error
    return {"status": "accepted", "store_id": handle.store_id, "run_id": handle.run_id,
        "attempt_id": handle.attempt_id, "job_id": legacy_job_id(handle),
        "events_url": f"/runs/{handle.run_id}/attempts/{handle.attempt_id}/events",
        "result_url": f"/results/{handle.run_id}/attempts/{handle.attempt_id}",
        "execution_catalog_sha256": service.catalog.document.sha256,
        "job_schema_sha256": service.catalog.schema.sha256, "scientific_qualification": False}


@router.get("/experiments")
def experiments(request: Request) -> JSONResponse:
    import json
    catalog = service(request).catalog
    return JSONResponse(json.loads(catalog.document.content), headers={"ETag": '"' + catalog.document.sha256 + '"',
        "X-Solarlab-Registry": catalog.document.sha256, "X-Solarlab-Job-Schema": catalog.schema.sha256})


@router.post("/jobs", status_code=202)
async def submit(request: Request) -> JSONResponse:
    context = service(request)
    for name, expected in (("X-Solarlab-Registry", context.catalog.document.sha256),
                           ("X-Solarlab-Job-Schema", context.catalog.schema.sha256)):
        declared = request.headers.get(name)
        if declared is None:
            raise HTTPException(428, "required submission identity header: " + name)
        if declared != expected:
            raise HTTPException(409, {"code": "schema_registry_mismatch", "header": name})
    try:
        declaration = JobSubmission.model_validate(await body(request))
    except ValidationError as error:
        raise HTTPException(422, error.errors(include_url=False, include_context=False)) from error
    result = await run_in_threadpool(submitted, context, declaration)
    return JSONResponse(result, status_code=202, headers={"Location": result["events_url"], "Cache-Control": "no-store"})


class CancelInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    expected_generation: int = Field(ge=0)
    reason: str = Field(min_length=1, max_length=4096)


@router.post("/jobs/{run_id}/attempts/{attempt_id}/cancel")
async def cancel(request: Request, run_id: str, attempt_id: str) -> dict[str, Any]:
    context = service(request)
    try:
        value = CancelInput.model_validate(await body(request, 8192))
    except ValidationError as error:
        raise HTTPException(422, error.errors(include_url=False, include_context=False)) from error
    row = await run_in_threadpool(context.supervisor.store.get_run, run_id)
    if row["current_attempt_id"] != attempt_id or row["generation"] != value.expected_generation:
        raise StateConflict("cancellation target is a stale attempt/generation")
    # The store repeats the comparison in its transaction. The attempt can be
    # claimed/retried between this HTTP read and the cancellation operation.
    await run_in_threadpool(context.supervisor.store.cancel, run_id,
        expected_attempt_id=attempt_id, expected_generation=value.expected_generation, reason=value.reason)
    try:
        stop_requested = await run_in_threadpool(context.supervisor.cancel,
            JobHandle(context.supervisor.store.store_id, run_id, attempt_id), reason=value.reason)
    except KeyError:
        stop_requested = False  # Fence old ownership; never signal an unowned PID.
    return {"requested": True, "owner_stop_requested": stop_requested, "run_id": run_id, "attempt_id": attempt_id,
            "state": context.supervisor.store.get_run(run_id)["state"], "process_exit_inferred": False}
