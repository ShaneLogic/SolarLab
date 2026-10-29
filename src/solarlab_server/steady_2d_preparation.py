"""Explicit uniform 2D steady installation on the shared HTTP job owner."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import inspect
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from solarlab.config.identity import BuildSource, ExecutionEngine
from solarlab.device.resolved import PreparedDevice
from solarlab.experiments.two_dimensional.inputs import JV2DInput
from solarlab.experiments.two_dimensional.steady_worker import (
    INITIAL_POLICY, REQUEST_SCHEMA, RESULT_SCHEMA, Steady2DJob, Steady2DProtocol,
    prepare_steady_2d_job, run_steady_2d_worker, steady_2d_worker_binding,
)
from solarlab.experiments.two_dimensional.steady_results import read_steady_2d_job
from solarlab.experiments.worker import ResourceLimits, WorkerContext, WorkerRequest, WorkerResult
from solarlab.materials.source import SourceDocument
from solarlab.physics.registry import ModelRegistry, metadata_digest
from solarlab_server.ionic_jv_preparation import _same_fixed_policy
from solarlab_server.steady_preparation import _device_binding, _implementation_binding
from solarlab_server.transient_preparation import _server_runtime

if TYPE_CHECKING:
    from solarlab.io.run_store import RunStore
    from solarlab.solvers.nonlinear.steady_2d import Steady2DOptions
    from solarlab_server.job_service import ExecutionCatalog

__all__ = ["Steady2DJobInput", "Steady2DPreparationContext", "steady_2d_execution_catalog",
           "prepare_steady_2d_input", "read_registered_steady_2d_job"]
EXPERIMENT_ID = "steady_2d"
SCOPE = "uniform_no_grain_classical_2d_steady"
GEOMETRY = {"lateral_length_m": 1e-6, "Nx": 4, "Ny_per_layer": 4, "lateral_bc": "neumann",
            "geometry_policy": "rectangular_half_endpoint_neumann_2d_v1"}
_SERVER_PATHS = ("steady_2d_preparation.py", "steady_preparation.py", "transient_preparation.py",
                 "ionic_jv_preparation.py", "job_service.py", "registered_worker.py", "app.py",
                 "routes/jobs.py", "routes/results.py", "events.py")
_NUMERICAL_SOURCES = ("solarlab/device/compile_2d.py", "solarlab/equations/dc_2d.py",
                      "solarlab/mesh/geometry_2d.py", "solarlab/solvers/nonlinear/steady_2d.py")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _options(insufficient_budget: bool = False) -> Steady2DOptions:
    from solarlab.solvers.nonlinear.steady_1d import SteadyOptions
    from solarlab.solvers.nonlinear.steady_2d import Steady2DOptions
    controls = SteadyOptions()
    if insufficient_budget:
        controls = replace(controls, max_iterations=1, max_residual_evaluations=1, max_jacobian_evaluations=1)
    return Steady2DOptions(controls, 30.)


def _controls(insufficient_budget: bool = False) -> dict[str, Any]:
    from solarlab.solvers.nonlinear.steady_1d import SteadyOptions
    return {"auxiliary_1d": asdict(SteadyOptions()), "full_2d": asdict(_options(insufficient_budget))}


def _policy() -> dict[str, Any]:
    return {"spatial": dict(GEOMETRY), "voltage_V": .002, "generation_scales": [[0.], [1.], [0., 1.]],
        "controls": _controls(), "insufficient_budget_controls": _controls(True), "initial": INITIAL_POLICY,
        "scope": SCOPE, "shape_Ny_Nx": [13, 5], "numerical_coordinates": 195,
        "auxiliary_seed_is_2d_completion": False, "scientific_qualification": False}


def _control_schema(schema: dict[str, Any]) -> None:
    schema["anyOf"] = [{"const": _controls()}, {"const": _controls(True)}]


class _SpatialInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    lateral_length_m: float = Field(ge=1e-6, le=1e-6)
    Nx: int = Field(ge=4, le=4)
    Ny_per_layer: int = Field(ge=4, le=4)
    lateral_bc: Literal["neumann"]
    geometry_policy: Literal["rectangular_half_endpoint_neumann_2d_v1"]

    def declaration(self) -> JV2DInput:
        return JV2DInput(kind="jv_2d", lateral_length=f"{self.lateral_length_m} m", Nx=self.Nx,
                         Ny_per_layer=self.Ny_per_layer, lateral_bc=self.lateral_bc)


class Steady2DJobInput(BaseModel):
    """Installed data IDs and full explicit fixed controls, with no file/code inputs."""
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    device_id: str = Field(pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}$")
    voltage_V: float = Field(ge=.002, le=.002)
    generation_scales: list[Annotated[float, Field(ge=0, le=1)]] = Field(
        min_length=1, max_length=2, json_schema_extra={"enum": [[0.], [1.], [0., 1.]]})
    spatial: _SpatialInput
    controls: dict[str, Any] = Field(json_schema_extra=_control_schema)
    initial: dict[str, Any] = Field(json_schema_extra={"const": INITIAL_POLICY})

    @model_validator(mode="after")
    def fixed_scope(self) -> Steady2DJobInput:
        Steady2DProtocol(self.voltage_V, tuple(self.generation_scales))
        if (not _same_fixed_policy(self.initial, INITIAL_POLICY)
                or not any(_same_fixed_policy(self.controls, _controls(failure)) for failure in (False, True))):
            raise ValueError("2D steady v1 requires the full unchanged initial recipe and default or explicit one-count controls")
        return self

    def options(self) -> Steady2DOptions:
        return _options(_same_fixed_policy(self.controls, _controls(True)))


def _server_documents() -> tuple[SourceDocument, ...]:
    root = Path(__file__).resolve().parent
    return tuple(SourceDocument("solarlab_server/" + name, (root / name).read_bytes()) for name in _SERVER_PATHS)


def _server_bindings() -> list[dict[str, str]]:
    return [{"id": source.id, "sha256": source.sha256} for source in _server_documents()]


@dataclass(frozen=True, slots=True)
class Steady2DPreparationContext:
    devices: tuple[tuple[str, PreparedDevice], ...]
    implementation: BuildSource
    current_repository_head: str
    models: ModelRegistry = field(init=False)
    server_sources: tuple[SourceDocument, ...] = field(init=False)

    def __post_init__(self) -> None:
        devices = tuple(tuple(item) for item in self.devices)
        if (not devices or any(len(item) != 2 or type(item[0]) is not str
                or re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}", item[0]) is None
                or type(item[1]) is not PreparedDevice for item in devices)
                or len({item[0] for item in devices}) != len(devices)
                or len({cast(PreparedDevice, item[1]).content_sha256 for item in devices}) != len(devices)):
            raise ValueError("2D installation requires unique IDs and distinct actual prepared devices")
        if (type(self.implementation) is not BuildSource or type(self.current_repository_head) is not str
                or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", self.current_repository_head) is None):
            raise ValueError("2D installation requires immutable build sources and an explicit maintenance context")
        object.__setattr__(self, "devices", tuple(sorted(devices, key=lambda item: item[0])))
        models = []
        for _, prepared in self.devices:
            # Public preparation checks the actual no-grain scope and sources;
            # it creates neither a compiled device nor a numerical solve.
            job = prepare_steady_2d_job(prepared, _SpatialInput.model_validate(GEOMETRY).declaration(),
                Steady2DProtocol(.002, (0., 1.)), options=_options(), implementation=self.implementation,
                current_repository_head=self.current_repository_head, engine=ExecutionEngine("solarlab"))
            models.append(job.execution.inputs.registry)
        if len({model.content_sha256 for model in models}) != 1:
            raise ValueError("2D devices must share this installed physical model registry")
        object.__setattr__(self, "models", models[0])
        object.__setattr__(self, "server_sources", _server_documents())

    def device(self, id: str) -> PreparedDevice:
        for key, prepared in self.devices:
            if key == id:
                return prepared
        raise ValueError("device is not installed in the trusted2D catalog")

    def verify_sources(self) -> None:
        if self.server_sources != _server_documents():
            raise ValueError("2D server source changed after installation")

    def export(self) -> dict[str, Any]:
        self.verify_sources()
        return {"devices": [_device_binding(id, p) for id, p in self.devices],
            "implementation": _implementation_binding(self.implementation), "current_repository_head": self.current_repository_head,
            "server_sources": [{"id": s.id, "sha256": s.sha256} for s in self.server_sources],
            "server_runtime": _server_runtime(), "fixed_numerical_policy": _policy(),
            "request_transport": "unchanged_core_request_with_explicit_installed_catalog",
            "historical_process_readback": "original_process_identity_without_relabelling"}


def prepare_steady_2d_input(input: Steady2DJobInput, context: Steady2DPreparationContext) -> Steady2DJob:
    context.verify_sources()
    return prepare_steady_2d_job(context.device(input.device_id), input.spatial.declaration(),
        Steady2DProtocol(input.voltage_V, tuple(input.generation_scales)), options=input.options(),
        implementation=context.implementation, current_repository_head=context.current_repository_head,
        engine=ExecutionEngine("solarlab"))


def steady_2d_execution_catalog(context: Steady2DPreparationContext, *, ceiling: ResourceLimits) -> ExecutionCatalog:
    from solarlab.experiments.registry import ExperimentDefinition, ExperimentRegistry
    from solarlab.experiments.two_dimensional.steady_results import steady_2d_payload
    from solarlab.physics.registry import CapabilityRule, EvidenceRef
    from solarlab_server.job_service import ExecutionCatalog, Steady2DRegistration
    from solarlab_server.registered_worker import run_steady_2d
    if type(context) is not Steady2DPreparationContext or ceiling.wall_seconds > 30:
        raise ValueError("explicit2D installation and <=30s job ceiling required")
    context.verify_sources()
    documents = {s.id: s for s in context.server_sources}

    def reference(id: str, function: Any) -> EvidenceRef:
        source = SourceDocument(id, Path(inspect.getfile(function)).read_bytes())
        documents[id] = source
        return EvidenceRef(id + "#" + function.__name__, source.sha256, "source")

    schema = SourceDocument("solarlab.steady-2d-job-input.schema.v1", _canonical(Steady2DJobInput.model_json_schema()))
    documents[schema.id] = schema
    factory = reference("solarlab_server/steady_2d_preparation.py", prepare_steady_2d_input)
    payload = reference("solarlab/experiments/two_dimensional/steady_results.py", steady_2d_payload)
    executor = reference("solarlab_server/registered_worker.py", run_steady_2d)
    reader = reference("solarlab_server/steady_2d_preparation.py", read_registered_steady_2d_job)
    documents.update({s.id: s for s in context.implementation.files if s.id in _NUMERICAL_SOURCES})
    definition = ExperimentDefinition(EXPERIMENT_ID, 1, "Uniform no-grain full2D steady",
        EvidenceRef(schema.id, schema.sha256, "source"), payload, factory, executor, "steady_2d_arrays", reader,
        CapabilityRule("installed_uniform_2d_steady", (2,), ("dc",), (factory,), backends=(EXPERIMENT_ID,),
                       statistics=("maxwell_boltzmann",)), tuple(m.ref for m in context.models.definitions))
    registry = ExperimentRegistry((definition,))
    return ExecutionCatalog(registry, context.models, tuple(documents.values()),
        (Steady2DRegistration(EXPERIMENT_ID, 1, context),), registry.content_sha256, context.models.content_sha256, ceiling)


def _registration(request: WorkerRequest) -> tuple[dict[str, Any], dict[str, Any]]:
    body, catalog = json.loads(request.request.content), json.loads(request.registry.content)
    entries = [e for e in catalog.get("bindings", ()) if e.get("driver") == EXPERIMENT_ID]
    if (request.registry.id != "solarlab.execution-catalog.v1" or request.schema.id != "solarlab.job-submission.schema.v1"
            or catalog.get("schema") != "solarlab.execution-catalog.v1" or catalog.get("job_schema_sha256") != request.schema.sha256
            or body.get("schema") != REQUEST_SCHEMA.id or body.get("engine") != "solarlab" or len(entries) != 1
            or entries[0].get("experiment_id") != EXPERIMENT_ID or entries[0].get("version") != 1
            or entries[0].get("engine") != "solarlab" or entries[0].get("scope") != SCOPE
            or entries[0].get("result_schema") != RESULT_SCHEMA):
        raise ValueError("2D HTTP request/catalog/schema differs from the installed route")
    return entries[0], body


def execute_registered_steady_2d(request: WorkerRequest, context: WorkerContext) -> WorkerResult:
    context.check_cancel()
    entry, body = _registration(request)
    if (entry["worker_source_sha256"] != steady_2d_worker_binding().source_sha256
            or entry["input_schema"] != Steady2DJobInput.model_json_schema()):
        raise ValueError("2D worker/schema source differs from the installed registration")
    binding = context.consume_manifest(request, engine="solarlab")
    prepared = binding.inputs.prepared
    installation = entry["preparation_context"]
    selected = [p for p in installation["devices"] if p["prepared_sha256"] == prepared.content_sha256]
    if len(selected) != 1:
        raise ValueError("2D request requires exactly one installed prepared identity")
    device_id = selected[0]["device_id"]
    if (selected != [_device_binding(device_id, prepared)]
            or entry["model_registry_sha256"] != binding.inputs.registry.content_sha256
            or binding.inputs.selection.context.dimension != 2
            or installation["implementation"] != _implementation_binding(binding.execution.implementation)
            or installation["current_repository_head"] != binding.execution.current_repository_head
            or installation["server_sources"] != _server_bindings() or installation["server_runtime"] != _server_runtime()
            or installation["fixed_numerical_policy"] != _policy()):
        raise ValueError("2D device/geometry/model/source/runtime differs from the consumed manifest")
    declared = Steady2DJobInput.model_validate({"device_id": device_id, **body["protocol"],
        "spatial": installation["fixed_numerical_policy"]["spatial"], "controls": body["controls"], "initial": body["initial"]})
    if body["spatial"] != declared.spatial.declaration().editing_data():
        raise ValueError("2D core geometry differs from the explicit installed geometry")
    output = run_steady_2d_worker(request, context)
    return WorkerResult.from_dict({**json.loads(output.result_json), "experiment_id": EXPERIMENT_ID, "experiment_version": 1,
        "registered_device_id": device_id, "execution_catalog_sha256": request.registry.sha256,
        "registered_input_schema_sha256": metadata_digest(entry["input_schema"]),
        "registered_model_registry_sha256": entry["model_registry_sha256"],
        "registered_numerical_policy": installation["fixed_numerical_policy"],
        "registered_spatial_declaration": declared.spatial.model_dump(),
        "server_sources": installation["server_sources"], "server_runtime": installation["server_runtime"],
        "http_request_sha256": request.request.sha256}, artifacts=output.artifacts, state=output.state)


def read_registered_steady_2d_job(store: RunStore, run_id: str, *, attempt_id: str | None = None) -> dict[str, Any]:
    """Check actual saved HTTP transport, then read its unchanged core result."""
    request = WorkerRequest.from_bytes(_canonical(store.get_run(run_id)["request"]["input_metadata"]["transport"]))
    entry, body = _registration(request)
    loaded = read_steady_2d_job(store, run_id, attempt_id=attempt_id)
    result, installation = loaded["worker_result"], entry["preparation_context"]
    manifest = loaded["manifest"]
    bound = manifest["document"]["execution_identity"]["payload"]
    implementation = {"build_source_commit": bound["implementation"]["build_source_commit"],
        "source_manifest_sha256": metadata_digest([{"id": s["id"], "sha256": s["sha256"]}
            for s in manifest["document"]["implementation_content"]])}
    selected = [p for p in installation["devices"] if p["prepared_sha256"] == loaded["document"]["prepared_sha256"]]
    policy = installation["fixed_numerical_policy"]
    if (result["schema"] != RESULT_SCHEMA or result["experiment_id"] != EXPERIMENT_ID or result["experiment_version"] != 1
            or len(selected) != 1 or result["registered_device_id"] != selected[0]["device_id"]
            or result["execution_catalog_sha256"] != request.registry.sha256
            or result["registered_input_schema_sha256"] != metadata_digest(entry["input_schema"])
            or result["registered_model_registry_sha256"] != entry["model_registry_sha256"]
            or entry["model_registry_sha256"] != bound["effective_binding"]["registry_sha256"]
            or installation["implementation"] != implementation
            or installation["current_repository_head"] != manifest["current_repository_head"]
            or result["registered_numerical_policy"] != policy or result["server_sources"] != installation["server_sources"]
            or result["server_runtime"] != installation["server_runtime"]
            or result["registered_spatial_declaration"] != policy["spatial"]
            or result["http_request_sha256"] != request.request.sha256
            or result["capture"]["request_sha256"] != request.request.sha256
            or body["protocol"]["voltage_V"] != policy["voltage_V"]
            or body["protocol"]["generation_scales"] not in policy["generation_scales"]
            or body["controls"] not in (policy["controls"], policy["insufficient_budget_controls"])
            or body["initial"] != policy["initial"]):
        raise ValueError("stored2D HTTP/catalog/request bindings differ")
    return loaded
