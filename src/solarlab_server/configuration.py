"""Pure configuration preview using explicitly supplied trusted preparation data."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Literal, cast

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from solarlab.config.resolve_device import resolve_device, resolve_tandem
from solarlab.config.legacy_wire import prepare_legacy_wire
from solarlab.config.schema import configuration_schema_representation
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.inputs import DeviceInput, TandemInput
from solarlab.device.resolved import PreparedDevice, PreparedTandem
from solarlab.experiments.two_dimensional.inputs import SpatialExperimentInput
from solarlab.experiments.two_dimensional.preparation import PreparedSpatialExperiment, prepare_spatial_experiment
from solarlab.experiments.jv.inputs import JVExperimentInput
from solarlab.experiments.jv.preparation import PreparedJVExperiment, prepare_jv_experiment
from solarlab.experiments.inputs import invalid
from solarlab.sweeps.inputs import SweepInput
from solarlab.sweeps.preparation import PreparedSweep, prepare_sweep
from solarlab.sweeps.reference import SweepReference
from solarlab.materials.resources import ResourceLibrary
from solarlab.materials.source import SourceDocument

router = APIRouter()
Kind = Literal["device", "tandem", "experiment", "sweep", "legacy-wire"]


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
    sweep_references: tuple[SweepReference, ...] = ()

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
        references = tuple(self.sweep_references)
        if any(not isinstance(item, SweepReference) for item in references) or len({item.id for item in references}) != len(references):
            raise ValueError("sweep references require unique trusted coverage records")
        object.__setattr__(self, "sweep_references", references)


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


class _SourceIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class _LegacyWireRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["solarlab.legacy-wire-preview-request.v1"]
    # Both existing models validate their own literal schema version with a
    # before-validator, which Pydantic's field discriminator cannot wrap.
    input: DeviceInput | TandemInput
    source: _SourceIdentity
    references: dict[str, _SourceIdentity] = Field(default_factory=dict)


def _legacy_wire(data: Any, context: ConfigurationPreviewContext) -> dict[str, Any]:
    declaration = _LegacyWireRequest.model_validate(data)
    trusted = {source.id: source for source in context.sources}

    def source(binding: _SourceIdentity, path: list[str]) -> SourceDocument:
        original = trusted.get(binding.id)
        if original is None:
            raise _error(422, "configuration_source_unknown", "Choose an explicitly supplied source ID", [
                {"loc": [*path, "id"], "type": "source_unknown", "msg": "Unknown trusted source ID"}])
        if original.sha256 != binding.sha256:
            raise _error(409, "configuration_source_mismatch", "Original source content identity changed", [
                {"loc": [*path, "sha256"], "type": "source_mismatch", "msg": "SHA-256 does not match the selected trusted source"}])
        return original

    original = source(declaration.source, ["source"])
    expected = ({declaration.input.top_cell_reference, declaration.input.bottom_cell_reference}
                if isinstance(declaration.input, TandemInput) else set())
    if set(declaration.references) != expected:
        raise _error(422, "configuration_source_references", "Supply exactly the declared tandem reference bindings", [
            {"loc": ["references"], "type": "reference_identity", "msg": "Reference names must match the declared subcells; device exports use no references"}])
    references = {name: source(binding, ["references", name]) for name, binding in declaration.references.items()}
    result = prepare_legacy_wire(declaration.input, source=original, defaults=context.defaults, references=references)
    report = result.export()
    documents = []
    if result.document is not None:
        for name, document in [(None, result.document), *result.references]:
            documents.append({
                "role": "source" if name is None else "reference", "reference": name,
                "source_id": document.id, "sha256": document.sha256, "size_bytes": len(document.content),
                "media_type": "application/yaml", "utf8": document.content.decode("utf-8"),
            })
    return {
        "schema": "solarlab.legacy-wire-preview.v1", "kind": "legacy-wire",
        "status": report["status"], "can_execute": False,
        "input": report["input_declaration"], "report": report, "documents": documents,
        "identity": {"scope": "legacy_wire_preparation", "source": declaration.source.model_dump(),
            "references": {name: binding.model_dump() for name, binding in declaration.references.items()},
            "default_catalog_sha256": context.defaults.content_sha256,
            "resource_library_sha256": context.resources.content_sha256,
            "configuration_schema_sha256": configuration_schema_representation()[1]},
    }


def _resolve(kind: Kind, data: Any, context: ConfigurationPreviewContext) -> dict[str, Any]:
    try:
        if kind == "legacy-wire":
            return _legacy_wire(data, context)
        prepared: PreparedDevice | PreparedTandem | PreparedSpatialExperiment | PreparedJVExperiment | PreparedSweep
        if kind == "device":
            prepared = resolve_device(DeviceInput.model_validate(data), context.defaults,
                                      context.resources, sources=context.sources)
        elif kind == "tandem":
            prepared = resolve_tandem(TandemInput.model_validate(data), context.defaults,
                                      context.resources, sources=context.sources)
        elif kind == "sweep":
            sweep = SweepInput.model_validate(data)
            if isinstance(sweep.base, JVExperimentInput) and not any(name == "jv_jobs" for name, _ in context.defaults.experiment_defaults):
                raise _error(503, "configuration_experiment_context_missing", "J-V sweep preview requires explicit source-bound J-V defaults")
            prepared = prepare_sweep(sweep, context.defaults, context.resources, sources=context.sources, references=context.sweep_references)
        else:
            declaration = data.get("experiment") if isinstance(data, dict) else None
            experiment_kind = declaration.get("kind") if isinstance(declaration, dict) else None
            if experiment_kind in {"jv", "dark_jv"}:
                if not any(name == "jv_jobs" for name, _ in context.defaults.experiment_defaults):
                    raise _error(503, "configuration_experiment_context_missing", "J-V preview requires explicit source-bound J-V defaults")
                prepared = prepare_jv_experiment(JVExperimentInput.model_validate(data), context.defaults,
                                                 context.resources, sources=context.sources)
            elif experiment_kind in {"jv_2d", "voc_grain_sweep"}:
                if not any(name == "jv_2d" for name, _ in context.defaults.experiment_defaults):
                    raise _error(503, "configuration_experiment_context_missing", "Spatial preview requires explicit source-bound spatial defaults")
                prepared = prepare_spatial_experiment(SpatialExperimentInput.model_validate(data), context.defaults,
                                                      context.resources, sources=context.sources)
            else:
                invalid("ExperimentPreview", ("experiment", "kind"), "choose jv, dark_jv, jv_2d or voc_grain_sweep", experiment_kind)
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
            **({"configuration_schema_sha256": configuration_schema_representation()[1]} if kind in {"experiment", "sweep"} else {}),
        },
    }


async def _preview(request: Request, kind: Kind) -> JSONResponse:
    context = cast(ConfigurationPreviewContext | None, request.app.state.configuration_preview)
    if context is None:
        raise _error(503, "configuration_context_missing",
                     "Configuration preview requires an explicitly supplied trusted catalog and resource library")
    if kind in {"experiment", "sweep", "legacy-wire"}:
        if kind == "experiment" and not context.defaults.experiment_defaults:
            raise _error(503, "configuration_experiment_context_missing", "Experiment preview requires explicit source-bound experiment defaults")
        declared_schema = request.headers.get("x-solarlab-configuration-schema")
        if declared_schema is None:
            raise _error(428, "configuration_schema_required", "Send the generated configuration schema SHA-256 in X-Solarlab-Configuration-Schema")
        if declared_schema != configuration_schema_representation()[1]:
            raise _error(409, "configuration_schema_mismatch", "Configuration schema identity does not match this server")
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


@router.post("/configuration-preview/experiment")
async def preview_experiment(request: Request) -> JSONResponse:
    """Prepare declared geometry/protocol controls without generating a mesh or state."""
    return await _preview(request, "experiment")


@router.post("/configuration-preview/sweep")
async def preview_sweep(request: Request) -> JSONResponse:
    """Expand bounded serial declarations; no numerical work, reuse or run submission."""
    return await _preview(request, "sweep")


@router.post("/configuration-preview/legacy-wire")
async def preview_legacy_wire(request: Request) -> JSONResponse:
    """Prepare exact UTF-8 wire documents from selected trusted source bytes."""
    return await _preview(request, "legacy-wire")
