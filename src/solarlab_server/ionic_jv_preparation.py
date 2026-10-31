"""Installed illuminated shared-site ionic J-V on the existing HTTP owner.

HTTP selects registered prepared data and supplies the explicit verified
protocol/controls. The actual core request body is preserved while the
installed catalog binds HTTP configuration, source and runtime identities.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import inspect
import json
import math
from pathlib import Path
import re
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from solarlab.config.identity import BuildSource, ExecutionEngine
from solarlab.device.resolved import PreparedDevice
from solarlab.experiments.jv.ionic_worker import (
    REGISTERED_WALL_SECONDS, IonicJVInitial, IonicJVJob, IonicJVProtocol,
    ionic_jv_identity_inputs, ionic_jv_worker_binding, prepare_ionic_jv_job, read_ionic_jv_job, run_ionic_jv_worker,
)
from solarlab.experiments.worker import ResourceLimits, WorkerContext, WorkerRequest, WorkerResult
from solarlab.materials.source import SourceDocument
from solarlab.physics.registry import ModelRegistry, metadata_digest
from solarlab_server.steady_preparation import _device_binding, _implementation_binding
from solarlab_server.transient_preparation import _server_runtime

if TYPE_CHECKING:
    from solarlab.experiments.jv.ionic_steady import IonicSteadyJVOptions
    from solarlab.io.run_store import RunStore
    from solarlab_server.job_service import ExecutionCatalog

__all__ = ["IonicJVJobInput", "IonicJVPreparationContext", "ionic_jv_execution_catalog",
           "prepare_ionic_jv_input", "read_registered_ionic_jv_job"]
EXPERIMENT_ID = "ionic_steady_jv"
_CORE_REQUEST_SCHEMA_ID = "solarlab.ionic-jv.worker-request.v1"
_CORE_RESULT_SCHEMA_ID = "solarlab.ionic-jv.worker-result.v1"
SCOPE = "illuminated_shared_site_positive_D_single_blocking_domain_ionic_steady_jv"
_REGISTERED_WALL_DEVICE_ID = "research_1d_shared"
# Content-only approval for this device and the exact fixed v1 recipe. Source
# locations and the catalog-origin hashes derived from them are normalized;
# the ordinary prepared/compiled identities retain their full provenance.
_REGISTERED_WALL_PROFILE_SHA256 = "b051cb0cc9f99205c3c2361f0540162a412114f983b4d98bfc0ac873bab4908b"
VOLTAGES_V = tuple(tick / 40. for tick in range(51))
_SERVER_PATHS = ("ionic_jv_preparation.py", "ionic_transient_preparation.py", "transient_preparation.py",
                 "steady_preparation.py", "job_service.py", "registered_worker.py", "app.py",
                 "routes/jobs.py", "routes/results.py", "events.py")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _same_fixed_policy(value: Any, expected: Any) -> bool:
    """JSON numeric values are equal across int/float spelling, never booleans."""
    if type(expected) in (int, float):
        return bool(type(value) in (int, float) and (type(value) is int or math.isfinite(value))
                    and value == expected)
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return bool(value.keys() == expected.keys() and all(_same_fixed_policy(value[k], v) for k, v in expected.items()))
    if isinstance(expected, list):
        return len(value) == len(expected) and all(_same_fixed_policy(a, b) for a, b in zip(value, expected))
    return bool(value == expected)


def _options() -> IonicSteadyJVOptions:
    from solarlab.experiments.jv.ionic_steady import IonicSteadyJVOptions
    from solarlab.solvers.nonlinear.ionic_1d import IonicSteadyOptions
    from solarlab.solvers.nonlinear.problem import SolveBudget
    from solarlab.solvers.nonlinear.steady_1d import SteadyOptions
    return IonicSteadyJVOptions(IonicSteadyOptions(SteadyOptions(), 1e-12, 1e6, 1e6, 1., 12,
                               SolveBudget(60, 600, 60, 24), 30.), max_points=51, wall_s=25.)


def _initial() -> IonicJVInitial:
    return IonicJVInitial("dark_direct_density_seed_v1", .002, 25.)


def _policy() -> dict[str, Any]:
    return {"protocol": {"voltage_V": list(VOLTAGES_V), "generation_scale": 1., "incident_power_W_m2": None},
        "intervals_per_layer": [12, 12, 12], "controls": asdict(_options()),
        "initial": {"policy": "dark_direct_density_seed_v1", "voltage_V": .002, "wall_s": 25.}}


def _wall_profile_sha256(prepared: PreparedDevice, ion_domain_layer: str,
                         ion_domain_endpoints: tuple[str, ...]) -> str:
    """Bind all prepared values and the fixed recipe without source locations."""
    data = prepared.to_mapping()
    source_origin = "catalog:" + prepared.defaults.content_sha256
    data["input_sources"] = [{"sha256": item["sha256"]}
        for item in sorted(data["input_sources"], key=lambda item: item["sha256"])]
    data["default_catalog"]["evidence"] = sorted(sha for _, sha in data["default_catalog"]["evidence"])
    content_origin = "catalog:" + metadata_digest(data["default_catalog"])
    for layer in data["layers"]:
        for item in layer["provenance"]:
            if item["origin"] == source_origin:
                item["origin"] = content_origin
    for item in data["resource_bindings"]:
        del item["source_id"]  # Source bytes and resource content digests remain bound.
    return metadata_digest({"schema": "solarlab.ionic-jv.wall-profile.v1",
        "device_id": _REGISTERED_WALL_DEVICE_ID, "input": json.loads(prepared.input_json),
        "prepared": data, "ion_domain_layer": ion_domain_layer,
        "ion_domain_endpoints": list(ion_domain_endpoints), "fixed_numerical_policy": _policy()})


def _wall_ceiling_seconds(device_id: str, prepared: PreparedDevice, ion_domain_layer: str,
                          ion_domain_endpoints: tuple[str, ...], wall_profile_sha256: str | None) -> float:
    if wall_profile_sha256 is None:
        return 30.
    if type(wall_profile_sha256) is not str or wall_profile_sha256 != _REGISTERED_WALL_PROFILE_SHA256:
        raise ValueError("ionic J-V wall profile is not the approved installed request")
    if device_id != _REGISTERED_WALL_DEVICE_ID:
        return 30.
    if _wall_profile_sha256(prepared, ion_domain_layer, ion_domain_endpoints) != wall_profile_sha256:
        raise ValueError("ionic J-V wall profile differs from the approved prepared device and recipe")
    return REGISTERED_WALL_SECONDS


def _control_schema(schema: dict[str, Any]) -> None:
    schema["const"] = _policy()["controls"]


class _InitialInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    policy: Literal["dark_direct_density_seed_v1"]
    voltage_V: float = Field(ge=.002, le=.002)
    wall_s: float = Field(ge=25., le=25.)


class IonicJVJobInput(BaseModel):
    """The first verified HTTP protocol; scientific controls cannot be changed."""
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    device_id: str = Field(pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}$")
    voltage_V: list[Annotated[float, Field(ge=0, le=1.25)]] = Field(
        min_length=51, max_length=51, json_schema_extra={"const": list(VOLTAGES_V)})
    generation_scale: float = Field(ge=1, le=1)
    incident_power_W_m2: None
    intervals_per_layer: list[Annotated[int, Field(ge=12, le=12)]] = Field(min_length=3, max_length=3)
    controls: dict[str, Any] = Field(description="Exact complete controls advertised in fixed_numerical_policy.controls",
                                   json_schema_extra=_control_schema)
    initial: _InitialInput

    @model_validator(mode="after")
    def fixed_verified_protocol(self) -> IonicJVJobInput:
        declared = _policy()
        if (tuple(self.voltage_V) != VOLTAGES_V or not _same_fixed_policy(self.controls, declared["controls"])
                or not _same_fixed_policy(self.initial.model_dump(), declared["initial"])):
            raise ValueError("ionic J-V v1 requires the exact installed51-point protocol and unchanged full controls/initial recipe")
        return self


def _server_documents() -> tuple[SourceDocument, ...]:
    root = Path(__file__).resolve().parent
    return tuple(SourceDocument("solarlab_server/" + name, (root / name).read_bytes()) for name in _SERVER_PATHS)


def _server_bindings() -> list[dict[str, str]]:
    return [{"id": s.id, "sha256": s.sha256} for s in _server_documents()]


@dataclass(frozen=True, slots=True)
class IonicJVPreparationContext:
    devices: tuple[tuple[str, PreparedDevice], ...]
    ion_domain_layer: str
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
            raise ValueError("ionic J-V installation needs unique IDs and unique actual prepared devices")
        if (type(self.implementation) is not BuildSource or type(self.current_repository_head) is not str
                or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", self.current_repository_head) is None):
            raise ValueError("immutable implementation and independently supplied maintenance context required")
        object.__setattr__(self, "devices", tuple(sorted(devices, key=lambda item: item[0])))
        for _, prepared in self.devices:
            layers = [p for p in prepared.layers if p.role != "substrate"]
            selected = [p for p in layers if p.id == self.ion_domain_layer]
            settings = prepared.to_mapping()["settings"]
            if len(layers) != 3 or len(selected) != 1 or not settings["ion_steric_shared_site"] or not settings["ion_steric_diffusion_only"]:
                raise ValueError("this installation requires the verified three-layer shared-site blocking-domain scope")
            parameters = dict(selected[0].parameters)
            values: dict[str, float] = {}
            for name in ("P0", "P0_neg", "P_lim", "P_lim_neg", "D_ion", "D_ion_neg"):
                value = parameters[name]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("shared-site populations/capacities/diffusivities must be numeric")
                values[name] = float(value)
            for suffix in ("", "_neg"):
                reference, capacity, diffusion = (float(values[name + suffix]) for name in ("P0", "P_lim", "D_ion"))
                if not all(math.isfinite(v) for v in (reference, capacity, diffusion)) or not 0 < reference < capacity or diffusion <= 0:
                    raise ValueError("both shared-site species require positive connected diffusivities and unsaturated populations")
            if values["P_lim"] != values["P_lim_neg"] or values["P0"] + values["P0_neg"] >= values["P_lim"]:
                raise ValueError("shared-site populations require the same available capacity")
            if any(any(dict(p.parameters)[key] != 0 for key in ("P0", "P0_neg", "D_ion", "D_ion_neg"))
                   for p in prepared.layers if p.id != self.ion_domain_layer):
                raise ValueError("outside-domain ionic populations are unsupported")
        object.__setattr__(self, "models", ionic_jv_identity_inputs(self.devices[0][1], self.implementation).registry)
        object.__setattr__(self, "server_sources", _server_documents())

    def device(self, id: str) -> PreparedDevice:
        for key, prepared in self.devices:
            if key == id:
                return prepared
        raise ValueError("device is not installed in the trusted ionic J-V catalog")

    def registered_wall_profile(self) -> str:
        """Approve only the installed device content and the exact v1 recipe."""
        profile = _wall_profile_sha256(self.device(_REGISTERED_WALL_DEVICE_ID),
                                      self.ion_domain_layer, ("blocking", "blocking"))
        if profile != _REGISTERED_WALL_PROFILE_SHA256:
            raise ValueError("ionic J-V wall profile differs from the approved prepared device and recipe")
        return profile

    def wall_ceiling_seconds(self, device_id: str, *, wall_profile_sha256: str | None = None) -> float:
        return _wall_ceiling_seconds(device_id, self.device(device_id), self.ion_domain_layer,
                                     ("blocking", "blocking"), wall_profile_sha256)

    def verify_sources(self) -> None:
        if self.server_sources != _server_documents():
            raise ValueError("ionic J-V server source changed after installation")

    def export(self) -> dict[str, Any]:
        self.verify_sources()
        return {"devices": [_device_binding(id, p) for id, p in self.devices], "ion_domain_layer": self.ion_domain_layer,
            "ion_domain_endpoints": ["blocking", "blocking"], "implementation": _implementation_binding(self.implementation),
            "current_repository_head": self.current_repository_head,
            "server_sources": [{"id": s.id, "sha256": s.sha256} for s in self.server_sources], "server_runtime": _server_runtime(),
            "fixed_numerical_policy": _policy(), "request_transport": "unchanged_core_request_with_unique_registered_prepared_identity",
            "submission_initial_policies": ["dark_direct_density_seed_v1"],
            "historical_process_readback": "unchanged_original_process_identity"}


def prepare_ionic_jv_input(input: IonicJVJobInput, context: IonicJVPreparationContext) -> IonicJVJob:
    context.verify_sources()
    return prepare_ionic_jv_job(context.device(input.device_id),
        IonicJVProtocol(tuple(input.voltage_V), float(input.generation_scale), input.incident_power_W_m2),
        intervals_per_layer=tuple(input.intervals_per_layer), ion_domain_layer=context.ion_domain_layer,
        ion_domain_endpoints=("blocking", "blocking"), options=_options(), initial=_initial(),
        implementation=context.implementation, current_repository_head=context.current_repository_head, engine=ExecutionEngine("solarlab"))


def ionic_jv_execution_catalog(context: IonicJVPreparationContext, *, ceiling: ResourceLimits,
                               wall_profile_sha256: str | None = None) -> ExecutionCatalog:
    from solarlab.experiments.jv.ionic_results import ionic_jv_payload
    from solarlab.experiments.registry import ExperimentDefinition, ExperimentRegistry
    from solarlab.physics.registry import CapabilityRule, EvidenceRef
    from solarlab_server.job_service import ExecutionCatalog, IonicJVRegistration
    from solarlab_server.registered_worker import run_ionic_jv
    if type(context) is not IonicJVPreparationContext or ceiling.wall_seconds > 30:
        raise ValueError("explicit ionic J-V installation and <=30s ordinary job ceiling required")
    context.verify_sources()
    documents = {s.id: s for s in context.server_sources}

    def reference(id: str, function: Any) -> EvidenceRef:
        source = SourceDocument(id, Path(inspect.getfile(function)).read_bytes())
        documents[id] = source
        return EvidenceRef(id + "#" + function.__name__, source.sha256, "source")

    schema = SourceDocument("solarlab.ionic-jv-job-input.schema.v1", _canonical(IonicJVJobInput.model_json_schema()))
    documents[schema.id] = schema
    factory = reference("solarlab_server/ionic_jv_preparation.py", prepare_ionic_jv_input)
    payload = reference("solarlab/experiments/jv/ionic_results.py", ionic_jv_payload)
    executor = reference("solarlab_server/registered_worker.py", run_ionic_jv)
    reader = reference("solarlab_server/ionic_jv_preparation.py", read_registered_ionic_jv_job)
    for source in context.implementation.files:
        if source.id in {"solarlab/device/compile_ionic_1d.py", "solarlab/equations/ionic_1d.py",
                         "solarlab/solvers/nonlinear/ionic_1d.py", "solarlab/experiments/ionic_initialization.py",
                         "solarlab/experiments/jv/ionic_steady.py"}:
            documents[source.id] = source
    definition = ExperimentDefinition(EXPERIMENT_ID, 1, "Illuminated shared-site ionic steady J-V",
        EvidenceRef(schema.id, schema.sha256, "source"), payload, factory, executor, "ionic_steady_jv_arrays", reader,
        CapabilityRule("installed_shared_site_ionic_jv", (1,), ("dc",), (factory,), backends=(EXPERIMENT_ID,),
                       statistics=("maxwell_boltzmann",)), tuple(m.ref for m in context.models.definitions))
    registry = ExperimentRegistry((definition,))
    return ExecutionCatalog(registry, context.models, tuple(documents.values()),
        (IonicJVRegistration(EXPERIMENT_ID, 1, context, wall_profile_sha256),),
        registry.content_sha256, context.models.content_sha256, ceiling)


def _registration(request: WorkerRequest) -> tuple[dict[str, Any], dict[str, Any]]:
    body, catalog = json.loads(request.request.content), json.loads(request.registry.content)
    entries = [e for e in catalog.get("bindings", ()) if e.get("driver") == EXPERIMENT_ID]
    if (catalog.get("schema") != "solarlab.execution-catalog.v1" or catalog.get("job_schema_sha256") != request.schema.sha256
            or body.get("schema") != _CORE_REQUEST_SCHEMA_ID or body.get("engine") != "solarlab" or len(entries) != 1
            or entries[0]["experiment_id"] != EXPERIMENT_ID or entries[0]["version"] != 1
            or entries[0]["engine"] != "solarlab" or entries[0]["scope"] != SCOPE):
        raise ValueError("ionic J-V request/catalog/schema differs from the installed route")
    return entries[0], body


def execute_registered_ionic_jv(request: WorkerRequest, context: WorkerContext) -> WorkerResult:
    context.check_cancel()
    entry, body = _registration(request)
    if (entry["worker_source_sha256"] != ionic_jv_worker_binding().source_sha256
            or entry["input_schema"] != IonicJVJobInput.model_json_schema()):
        raise ValueError("ionic J-V worker/schema source differs from installed registration")
    binding = context.consume_manifest(request, engine="solarlab")
    prepared = binding.inputs.prepared
    installation = entry["preparation_context"]
    selected = [p for p in installation["devices"] if p["prepared_sha256"] == prepared.content_sha256]
    if len(selected) != 1:
        raise ValueError("ionic J-V requires one exact registered prepared identity")
    device_id = selected[0]["device_id"]
    if (selected != [_device_binding(device_id, prepared)]
            or entry["model_registry_sha256"] != binding.inputs.registry.content_sha256
            or installation["implementation"] != _implementation_binding(binding.execution.implementation)
            or installation["current_repository_head"] != binding.execution.current_repository_head
            or installation["server_sources"] != _server_bindings() or installation["server_runtime"] != _server_runtime()
            or installation["fixed_numerical_policy"] != _policy()):
        raise ValueError("ionic J-V device/source/model/runtime/policy differs from consumed manifest")
    declared = IonicJVJobInput.model_validate({"device_id": device_id, **body["protocol"],
        "intervals_per_layer": body["grid"]["intervals_per_layer"], "controls": body["controls"],
        "initial": {k: body["initial"][k] for k in ("policy", "voltage_V", "wall_s")}})
    if (body["initial"]["artifact_content_hex"] is not None or body["initial"]["artifact_sha256"] is not None
            or body["prepared_sha256"] != prepared.content_sha256
            or body["grid"]["ion_domain_layer"] != installation["ion_domain_layer"]
            or body["grid"]["ion_domain_endpoints"] != installation["ion_domain_endpoints"]):
        raise ValueError("ionic J-V initial/domain differs from explicit installed recipe")
    allowance = entry.get("whole_job_wall_allowance")
    profile = None
    if allowance is not None:
        if (type(allowance) is not dict or set(allowance) != {"profile_sha256", "wall_seconds"}
                or type(allowance["profile_sha256"]) is not str
                or type(allowance["wall_seconds"]) not in {int, float}
                or allowance["wall_seconds"] != REGISTERED_WALL_SECONDS):
            raise ValueError("ionic J-V wall allowance is not the bounded registered profile")
        profile = allowance["profile_sha256"]
    wall_ceiling = _wall_ceiling_seconds(device_id, prepared, body["grid"]["ion_domain_layer"],
        tuple(body["grid"]["ion_domain_endpoints"]), profile)
    output = run_ionic_jv_worker(request, context, wall_ceiling_seconds=wall_ceiling)
    return WorkerResult.from_dict({**json.loads(output.result_json), "experiment_id": EXPERIMENT_ID, "experiment_version": 1,
        "registered_device_id": declared.device_id, "execution_catalog_sha256": request.registry.sha256,
        "registered_input_schema_sha256": metadata_digest(entry["input_schema"]),
        "registered_model_registry_sha256": entry["model_registry_sha256"],
        "registered_numerical_policy": installation["fixed_numerical_policy"],
        "server_sources": installation["server_sources"], "server_runtime": installation["server_runtime"],
        "http_request_sha256": request.request.sha256}, artifacts=output.artifacts, state=output.state)


def read_registered_ionic_jv_job(store: RunStore, run_id: str, *, attempt_id: str | None = None) -> dict[str, Any]:
    """Validate original HTTP transport and the source-bound core result reader."""
    request = WorkerRequest.from_bytes(_canonical(store.get_run(run_id)["request"]["input_metadata"]["transport"]))
    entry, body = _registration(request)
    loaded = read_ionic_jv_job(store, run_id, attempt_id=attempt_id)
    result = loaded["worker_result"]
    installation = entry["preparation_context"]
    manifest = loaded["manifest"]
    bound = manifest["document"]["execution_identity"]["payload"]
    implementation = {"build_source_commit": bound["implementation"]["build_source_commit"],
        "source_manifest_sha256": metadata_digest([{"id": s["id"], "sha256": s["sha256"]}
            for s in manifest["document"]["implementation_content"]])}
    selected = [p for p in installation["devices"] if p["prepared_sha256"] == loaded["document"]["prepared_sha256"]]
    policy = installation["fixed_numerical_policy"]
    if (result["schema"] != _CORE_RESULT_SCHEMA_ID or result["experiment_id"] != EXPERIMENT_ID or result["experiment_version"] != 1
            or len(selected) != 1 or result["registered_device_id"] != selected[0]["device_id"]
            or result["execution_catalog_sha256"] != request.registry.sha256
            or result["registered_input_schema_sha256"] != metadata_digest(entry["input_schema"])
            or result["registered_model_registry_sha256"] != entry["model_registry_sha256"]
            or entry["model_registry_sha256"] != bound["effective_binding"]["registry_sha256"]
            or installation["implementation"] != implementation
            or installation["current_repository_head"] != manifest["current_repository_head"]
            or result["registered_numerical_policy"] != policy or result["server_sources"] != installation["server_sources"]
            or result["server_runtime"] != installation["server_runtime"]
            or result["http_request_sha256"] != request.request.sha256
            or result["capture"]["request_sha256"] != request.request.sha256
            or body["protocol"] != policy["protocol"] or body["controls"] != policy["controls"]
            or body["grid"]["intervals_per_layer"] != policy["intervals_per_layer"]
            or {k: body["initial"][k] for k in ("policy", "voltage_V", "wall_s")} != policy["initial"]):
        raise ValueError("stored ionic J-V HTTP/catalog/request bindings differ")
    return loaded
