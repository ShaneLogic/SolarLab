"""Pure configuration preview using explicitly supplied trusted preparation data."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Literal, cast

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from solarlab.config.resolve_device import resolve_device, resolve_tandem
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.inputs import DeviceInput, TandemInput
from solarlab.device.resolved import PreparedDevice, PreparedTandem
from solarlab.materials.resources import ResourceLibrary
from solarlab.materials.source import SourceDocument

router = APIRouter()
Kind = Literal["device", "tandem"]


@dataclass(frozen=True, slots=True)
class ConfigurationPreviewContext:
    """App-owned data, never decoded from a preview request or discovered by path.

    ``sources`` are optional original documents supplied by the trusted caller.
    An empty tuple means that no original document provenance was supplied.
    The byte limit is a transport limit, not a physical configuration default.
    """

    defaults: DefaultCatalog
    resources: ResourceLibrary
    sources: tuple[SourceDocument, ...] = ()
    max_input_bytes: int = 1024**2

    def __post_init__(self) -> None:
        if not isinstance(self.defaults, DefaultCatalog) or not isinstance(self.resources, ResourceLibrary):
            raise TypeError("preview requires an explicit DefaultCatalog and ResourceLibrary")
        sources = tuple(self.sources)
        if any(not isinstance(source, SourceDocument) for source in sources):
            raise TypeError("preview sources must be explicit SourceDocument records")
        if len({source.id for source in sources}) != len(sources):
            raise ValueError("preview source IDs must be unique")
        if type(self.max_input_bytes) is not int or self.max_input_bytes <= 0:
            raise ValueError("max_input_bytes must be a positive transport byte limit")
        object.__setattr__(self, "sources", sources)


def _error(status: int, code: str, message: str, fields: list[dict[str, Any]] | None = None) -> HTTPException:
    return HTTPException(status, detail={
        "code": code, "message": message, "status": "unresolved", "can_execute": False,
        "field_errors": fields if fields is not None else [
            {"loc": ["body"], "type": code, "msg": message}],
    })


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    if len({name for name, _ in items}) != len(items):
        raise ValueError("duplicate JSON configuration field")
    return dict(items)


def _number(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("configuration JSON numbers must be finite")
    return value


def _constant(text: str) -> Any:
    raise ValueError(f"non-JSON numeric constant: {text}")


def _resolve(kind: Kind, data: Any, context: ConfigurationPreviewContext) -> dict[str, Any]:
    try:
        prepared: PreparedDevice | PreparedTandem
        if kind == "device":
            prepared = resolve_device(DeviceInput.model_validate(data), context.defaults,
                                      context.resources, sources=context.sources)
        else:
            prepared = resolve_tandem(TandemInput.model_validate(data), context.defaults,
                                      context.resources, sources=context.sources)
    except ValidationError as error:
        fields = [dict(field) for field in error.errors(include_url=False, include_context=False)]
        raise _error(422, "configuration_validation", "Invalid configuration input", fields) from error
    except ValueError as error:
        # The core currently supplies textual semantic errors, not structured
        # paths. Preserve that message at the document level; do not guess a path.
        raise _error(422, "configuration_resolution", str(error)) from error
    return {
        "schema": "solarlab.configuration-preview.v1", "kind": kind,
        "status": "prepared_pending_dependencies", "can_execute": prepared.can_execute,
        "input": prepared.to_input().editing_data(), "resolved": prepared.to_mapping(),
        "identity": {
            "scope": "configuration_content",
            "content_sha256": prepared.content_sha256,
            "default_catalog_sha256": context.defaults.content_sha256,
            "resource_library_sha256": context.resources.content_sha256,
        },
    }


async def _preview(request: Request, kind: Kind) -> JSONResponse:
    context = cast(ConfigurationPreviewContext | None, request.app.state.configuration_preview)
    if context is None:
        raise _error(503, "configuration_context_missing",
                     "Configuration preview requires an explicitly supplied trusted catalog and resource library")
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        raise _error(415, "configuration_content_type", "Expected application/json")
    raw = bytearray()
    async for chunk in request.stream():
        if len(raw) + len(chunk) > context.max_input_bytes:
            raise _error(413, "configuration_input_limit", "Configuration exceeds the preview byte limit")
        raw.extend(chunk)
    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                          parse_float=_number, parse_constant=_constant,
                          parse_int=lambda word: -0.0 if word == "-0" else int(word))
    except (ValueError, UnicodeError, RecursionError) as error:
        raise _error(400, "configuration_json", str(error)) from error
    result = await run_in_threadpool(_resolve, kind, data, context)
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


@router.post("/configuration-preview/device")
async def preview_device(request: Request) -> JSONResponse:
    """Resolve a DeviceInput JSON document; never construct an execution request."""
    return await _preview(request, "device")


@router.post("/configuration-preview/tandem")
async def preview_tandem(request: Request) -> JSONResponse:
    """Resolve a TandemInput with supplied subcells; references are not file paths."""
    return await _preview(request, "tandem")
