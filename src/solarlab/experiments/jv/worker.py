"""Explicit classical DC jobs on the existing manifest, supervisor and store.

The child returns named payloads. Only ProcessSupervisor claims attempts,
publishes artifacts and commits a terminal. Preparation/cache admission flags
retain their existing meaning; successful execution is not qualification.
"""
from __future__ import annotations

from solarlab.reproducibility.job_provenance import (
    JobProvenance, worker_result_provenance,
)

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
from typing import TYPE_CHECKING, Any

from solarlab.config.identity import (
    BuildSource, ExecutionDescription, ExecutionEngine, ExecutionManifest, IdentityInputs, ModelInputs,
    RuntimeEnvironment, execution_manifest, input_identity, physics_identity,
)
from solarlab.config.model_parameters import device_topology, layer_parameter_source
from solarlab.device.resolved import PreparedDevice
from solarlab.experiments.worker import (
    ManifestBinding, ResourceLimits, WorkerBinding, WorkerContext, WorkerRequest, WorkerResult,
)
from solarlab.materials.source import SourceDocument
from solarlab.physics.registry import (
    CapabilityContext, CapabilityRule, EvidenceRef, ModelDefinition, ModelRef, ModelRegistry, Scope,
)
from solarlab.physics.selection import ModelChoice, select_models
from solarlab.physics.semantic_projection import SemanticProjection

if TYPE_CHECKING:
    from solarlab.experiments.execution import JobHandle, ProcessSupervisor
    from solarlab.io.run_store import RunStore

__all__ = ["SteadyJVProtocol", "SteadyJVJob", "prepare_steady_jv_job",
           "steady_jv_worker_binding", "read_steady_jv_job", "run_steady_jv_worker",
           "steady_jv_identity_inputs", "validate_steady_jv_scope"]

_ID = "steady_jv"
_SCHEMA = SourceDocument("solarlab.steady-jv.worker-request.v1", b'{"engine":"solarlab","fields":["schema","engine","prepared_sha256","protocol"],"schema":"solarlab.steady-jv.worker-request.v1"}')
_REGISTRY = SourceDocument("solarlab.steady-jv.worker-registry.v1", b'{"engine":"solarlab","worker":"steady_jv","scope":"static_classical_ion_free_1d_dc","fallback":"forbidden"}')
_THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
_PACKAGES = ("numpy", "scipy", "pydantic", "threadpoolctl")
_ADAPTERS = ("solarlab/experiments/jv/worker.py", "solarlab/experiments/jv/steady.py",
             "solarlab/sweeps/steady_worker.py", "solarlab/experiments/transient_worker.py",
             "solarlab/experiments/ionic_transient_worker.py", "solarlab/experiments/ionic_transient_results.py",
             "solarlab/experiments/jv/ionic_worker.py", "solarlab/experiments/jv/ionic_results.py",
             "solarlab/experiments/worker.py", "solarlab/experiments/execution.py")


def _document(name: str, value: Any) -> SourceDocument:
    return SourceDocument(name, json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def _package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_path(name: str) -> Path:
    if not name.startswith("solarlab/"):
        raise ValueError("implementation IDs must be relative to the installed solarlab package")
    path = _package_root() / name.removeprefix("solarlab/")
    if not path.resolve().is_relative_to(_package_root()) or path.suffix != ".py":
        raise ValueError("invalid implementation source path")
    return path


def _adapter_sources(implementation: BuildSource) -> list[dict[str, Any]]:
    base = {item.id: item for item in implementation.files}
    # These actively authored adapters are bound separately from the
    # supplied committed core. Never label new author bytes as that commit.
    return [{"id": name, "sha256": source.sha256, "content_hex": source.content.hex(),
             "base_sha256": base[name].sha256 if name in base else None}
            for name in _ADAPTERS
            for source in (SourceDocument(name, _source_path(name).read_bytes()),)]


def _verify_sources(execution: ExecutionDescription) -> None:
    expected = {item.id: item.content for item in execution.implementation.files}
    adapters = json.loads(execution.preparation.content)["adapter_sources"]
    if adapters != _adapter_sources(execution.implementation):
        raise ValueError("installed worker/result encoder differs from its execution source binding")
    expected.update({item["id"]: bytes.fromhex(item["content_hex"]) for item in adapters})
    for name, content in expected.items():
        if _source_path(name).read_bytes() != content:
            raise ValueError("installed core implementation differs: " + name)
    for name, module in tuple(sys.modules.items()):
        if name != "solarlab" and not name.startswith("solarlab."):
            continue
        origin = module.__file__
        if origin is None:
            raise ValueError("loaded core module has no verifiable source origin")
        path = Path(origin).resolve()
        if not path.is_relative_to(_package_root()):
            raise ValueError("loaded core module is outside the bound installation")
        source_id = "solarlab/" + path.relative_to(_package_root()).as_posix()
        if path.read_bytes() and source_id not in expected:
            raise ValueError("loaded core module lacks a source binding: " + source_id)


@dataclass(frozen=True, slots=True)
class SteadyJVProtocol:
    voltage_V: tuple[float, ...]
    intervals_per_layer: int
    generation_scale: float
    incident_power_W_m2: float | None

    def __post_init__(self) -> None:
        values = tuple(self.voltage_V)
        if (not 2 <= len(values) <= 101 or any(type(v) not in {int, float} or not math.isfinite(v) for v in values)
                or not (all(b > a for a, b in zip(values, values[1:]))
                        or all(b < a for a, b in zip(values, values[1:])))):
            raise ValueError("steady protocol requires 2..101 finite strictly monotonic voltages")
        if type(self.intervals_per_layer) is not int or self.intervals_per_layer < 2:
            raise ValueError("explicit integer intervals_per_layer >= 2 required")
        if type(self.generation_scale) not in {int, float} or not math.isfinite(self.generation_scale) or self.generation_scale < 0:
            raise ValueError("generation scale must be finite and nonnegative")
        if self.incident_power_W_m2 is not None and (type(self.incident_power_W_m2) not in {int, float}
                or not math.isfinite(self.incident_power_W_m2) or self.incident_power_W_m2 <= 0):
            raise ValueError("incident power must be finite and positive when supplied")
        object.__setattr__(self, "voltage_V", tuple(float(v) for v in values))


def validate_steady_jv_scope(prepared: PreparedDevice) -> None:
    """Refuse unsupported configured physics before numerical preparation."""
    from solarlab.device.compile_1d import unsupported

    if type(prepared) is not PreparedDevice:
        raise ValueError("an actual explicit PreparedDevice is required")
    # Reuse the compiler's static check, before geometry/operators or a solve.
    unsupported(prepared, prepared.to_mapping(), tuple(p for p in prepared.layers if p.role != "substrate"))


def steady_jv_identity_inputs(prepared: PreparedDevice, implementation: BuildSource) -> IdentityInputs:
    """Bind the actual prepared input and compiler source to classical models."""
    source = next((item for item in implementation.files if item.id == "solarlab/device/compile_1d.py"), None)
    if source is None:
        raise ValueError("the actual core compiler source must be bound")
    ref = ModelRef("classical_1d_dc", 1)
    capability = CapabilityRule("source_bound_classical_dc", (1,), ("dc",),
        (EvidenceRef(source.id, source.sha256, "source"),),
        backends=("steady_jv_worker",), statistics=("maxwell_boltzmann",))
    definition = ModelDefinition(ref, "Fixed classical 1D core source binding", "classical_dc", ("layer",), "node",
        ("mu_n", "mu_p", "tau_n", "tau_p", "alpha", "B_rad", "C_n", "C_p", "N_A", "N_D"), (), (),
        capability, parameter_schema="full_layer")
    registry = ModelRegistry((definition,))
    defaults = {"layer": (ModelChoice("classical_dc", ref, parameter_schema="full_layer"),)}
    topology = device_topology(prepared)
    sources = {Scope("layer", (layer.id,)): layer_parameter_source(layer)
               for layer in prepared.layers if layer.id in topology.electrical_layer_ids}
    context = CapabilityContext(1, "dc", "steady_jv_worker", "maxwell_boltzmann",
                                temperature_K=prepared.to_mapping()["settings"]["T"])
    selection = select_models(registry, topology, context, defaults, sources, {})
    return IdentityInputs(prepared, selection, registry,
        ModelInputs.from_mappings(defaults=defaults, overrides={}),
        (SemanticProjection(ref, "all_values_v1", (source,)),))


def _roles(prepared: PreparedDevice, protocol: SteadyJVProtocol, implementation: BuildSource) -> dict[str, SourceDocument]:
    from solarlab.solvers.nonlinear.steady_1d import SteadyOptions

    return {
        "protocol": _document("solarlab.steady-jv.protocol.v1", asdict(protocol)),
        "preparation": _document("solarlab.steady-jv.preparation.v1", {
            "prepared_sha256": prepared.content_sha256, "adapter_sources": _adapter_sources(implementation),
            "source_scope": "supplied_committed_core_plus_explicit_adapter_bytes",
            "cold_compile_per_job": True, "fallback": "forbidden"}),
        "initial_state": _document("solarlab.steady-jv.initial-recipe.v1", {
            "prepared_sha256": prepared.content_sha256, "initial_state": None,
            "recipe": "solve_steady_state cold initialization; continuation within this curve only",
            "supplied_working_point": False, "cross_job_reuse": False}),
        "grid": _document("solarlab.steady-jv.grid-recipe.v1", {
            "intervals_per_layer": protocol.intervals_per_layer,
            "electrical_grid": prepared.to_mapping()["electrical_grid"]}),
        "numeric_controls": _document("solarlab.steady-jv.controls.v1", asdict(SteadyOptions())),
    }


def _body(prepared: PreparedDevice, protocol: SteadyJVProtocol) -> dict[str, Any]:
    return {"schema": _SCHEMA.id, "engine": "solarlab", "prepared_sha256": prepared.content_sha256,
            "protocol": json.loads(_document("protocol", asdict(protocol)).content)}


@dataclass(frozen=True, slots=True)
class SteadyJVJob:
    request: WorkerRequest
    execution: ManifestBinding

    def submit(self, supervisor: ProcessSupervisor, run_id: str, limits: ResourceLimits) -> JobHandle:
        return supervisor.submit(run_id, _ID, self.request, limits, execution=self.execution)


def prepare_steady_jv_job(prepared: PreparedDevice, protocol: SteadyJVProtocol, *,
                          implementation: BuildSource, current_repository_head: str,
                          engine: ExecutionEngine) -> SteadyJVJob:
    """Bind explicit data and actual source/runtime; do not execute a model."""
    if type(engine) is not ExecutionEngine or engine.name != "solarlab":
        raise ValueError("the steady worker requires the explicit solarlab engine")
    if type(protocol) is not SteadyJVProtocol or type(implementation) is not BuildSource:
        raise ValueError("explicit steady protocol and actual implementation sources required")
    validate_steady_jv_scope(prepared)
    inputs = steady_jv_identity_inputs(prepared, implementation)
    from solarlab.equations.implementation import selection_setting

    # The child installs these before any engine import: a queued or spawned
    # job runs the equation selection of the parent that prepared it.
    settings = tuple(sorted([*((name, "1") for name in _THREADS), selection_setting()]))
    environment = RuntimeEnvironment(platform.python_implementation(), platform.python_version(),
        platform.system(), platform.machine(), tuple((name, importlib.metadata.version(name)) for name in _PACKAGES),
        tuple((name, json.dumps(value)) for name, value in settings))
    roles = _roles(prepared, protocol, implementation)
    description = ExecutionDescription(**roles, implementation=implementation, environment=environment,
                                       current_repository_head=current_repository_head)
    _verify_sources(description)
    manifest = execution_manifest(inputs, description, engine=engine)
    binding = ManifestBinding(manifest, inputs, description, engine, manifest.sha256, settings)
    identity = manifest.export()["document"]["execution_identity"]
    request = WorkerRequest(_document(_SCHEMA.id, _body(prepared, protocol)), _REGISTRY, _SCHEMA,
        _document("solarlab.steady-jv.execution-identity.v1", identity),
        identity["payload"]["H_input"], identity["payload"]["H_physics"])
    binding.validate_request(request)
    return SteadyJVJob(request, binding)


def steady_jv_worker_binding() -> WorkerBinding:
    return WorkerBinding.from_function(_ID, run_steady_jv_worker, registry_sha256=_REGISTRY.sha256,
                                       schema_sha256=_SCHEMA.sha256)


def run_steady_jv_worker(request: WorkerRequest, worker: WorkerContext) -> WorkerResult:
    """Run the explicitly bound cold job using the supplied existing context."""
    worker.check_cancel()
    binding = worker.consume_manifest(request, engine="solarlab")
    from solarlab.equations.implementation import verify_selection

    verify_selection(binding.environment)
    body = json.loads(request.request.content)
    protocol = SteadyJVProtocol(**body["protocol"])
    prepared = binding.inputs.prepared
    if body != _body(prepared, protocol):
        raise ValueError("request differs from the explicit prepared steady invocation")
    _verify_sources(binding.execution)
    validate_steady_jv_scope(prepared)
    expected = steady_jv_identity_inputs(prepared, binding.execution.implementation)
    if (input_identity(expected).sha256 != request.H_input or physics_identity(expected).sha256 != request.H_physics
            or any(getattr(binding.execution, name) != value for name, value in
                   _roles(prepared, protocol, binding.execution.implementation).items())):
        raise ValueError("steady input/model/protocol roles differ from the manifest")
    worker.check_cancel()
    from solarlab.device.compile_1d import compile_device_1d
    from solarlab.equations.implementation import provenance_binding
    from solarlab.experiments.jv.steady import run_steady_jv, steady_jv_payload
    from solarlab.solvers.nonlinear.steady_1d import SteadyOptions
    from threadpoolctl import threadpool_info, threadpool_limits

    _verify_sources(binding.execution)  # Check newly imported numerical modules before native work.
    worker.progress({"stage": "steady_jv_native", "voltage_points": len(protocol.voltage_V)})
    with threadpool_limits(limits=1):
        compiled = compile_device_1d(prepared, intervals_per_layer=protocol.intervals_per_layer)
    # Admission names the bound equation system the job evaluates, so the device is compiled first.
    with worker.admit_provenance(request, **provenance_binding("DC1DProblem.steady_jv", compiled)):
        with threadpool_limits(limits=1):
            pools = threadpool_info()
            if any(pool["num_threads"] != 1 for pool in pools):
                raise ValueError("child numerical thread limit was not installed")
            result = run_steady_jv(compiled, protocol.voltage_V, generation_scale=protocol.generation_scale,
                incident_power_W_m2=protocol.incident_power_W_m2, options=SteadyOptions())
        worker.check_cancel()
        document, payloads = steady_jv_payload(result)
        return WorkerResult.from_dict({"schema": "solarlab.steady-jv.worker-result.v1",
            "engine": "solarlab", "prepared_sha256": prepared.content_sha256,
            "H_input": request.H_input, "H_physics": request.H_physics,
            "H_execution": request.execution_identity.sha256, "manifest_sha256": binding.expected_sha256,
            "steady_result": document, "array_payloads": {name: payload.sha256 for name, payload in payloads},
            "threadpools": pools, "scientific_qualification": False}, artifacts=payloads)


def read_steady_jv_job(store: RunStore, run_id: str, *, attempt_id: str | None = None,
    current_identity: JobProvenance | dict[str, Any] | None = None, strict_same_source: bool = False) -> dict[str, Any]:
    """Bind real committed artifact IDs/attempts, then use the existing reader."""
    from solarlab.experiments.jv.steady import read_steady_jv

    run = store.get_run(run_id)
    selected = run["current_attempt_id"] if attempt_id is None else attempt_id
    attempt = next((a for a in store.attempts(run_id) if a["attempt_id"] == selected), None)
    if attempt is None or attempt["state"] != "succeeded":
        raise ValueError("a succeeded attempt of this run is required")
    result = attempt["result"]["worker_result"]
    identity = run["request"]["identity"]
    if result.get("schema") != "solarlab.steady-jv.worker-result.v1" or result.get("engine") != "solarlab":
        raise ValueError("not a new-core steady worker result")
    if any(result[name] != identity[name] for name in ("H_input", "H_physics", "H_execution")):
        raise ValueError("steady result belongs to a different execution identity")
    metadata = run["request"]["input_metadata"]
    if result["manifest_sha256"] != metadata["execution_manifest"]["manifest_sha256"]:
        raise ValueError("steady result manifest mismatch")
    records = [store.artifact(item["artifact_id"]) for item in attempt["terminal"]["artifacts"]]
    by_name = {item["name"]: item for item in records}
    if len(by_name) != len(records) or set(by_name) != {*result["array_payloads"], "execution_manifest"}:
        raise ValueError("committed terminal does not contain the exact named result payloads")
    for name, item in by_name.items():
        if (item["phase"] != "committed" or item["run_id"] != run_id
                or item["attempt_id"] != selected or item["generation"] != attempt["generation"]):
            raise ValueError("artifact belongs to a different committed attempt")
        if name in result["array_payloads"] and item["sha256"] != result["array_payloads"][name]:
            raise ValueError("committed array differs from the child's actual payload")
    manifest = store.read_artifact(by_name["execution_manifest"]["artifact_id"])
    # Aggregate manifests have their own bound, separate from each source file.
    if hashlib.sha256(manifest).hexdigest() != result["manifest_sha256"]:
        raise ValueError("committed manifest bytes differ from the execution binding")
    saved = ExecutionManifest(manifest, metadata["execution_manifest"]["current_repository_head"])
    saved_document = saved.export()["document"]
    saved_identity = saved_document["execution_identity"]["payload"]
    if saved.execution_sha256 != identity["H_execution"] or any(
        saved_identity[name] != identity[name] for name in ("H_input", "H_physics")
    ):
        raise ValueError("committed manifest belongs to a different input/physics/execution identity")
    roles = {name: json.loads(bytes.fromhex(item["content_hex"]))
             for name, item in saved_document["content_roles"].items()}
    protocol = roles["protocol"]
    if (result["prepared_sha256"] != roles["preparation"]["prepared_sha256"]
            or result["steady_result"]["solver_controls"] != roles["numeric_controls"]):
        raise ValueError("result differs from its prepared input or numeric controls")
    document = {**result["steady_result"], "run_id": run_id, "attempt_id": selected,
                "generation": attempt["generation"],
                "arrays": {name: by_name[name]["artifact_id"] for name in result["array_payloads"]},
                "artifact_ids": [by_name[name]["artifact_id"] for name in result["array_payloads"]]}
    arrays = read_steady_jv(store, document)
    if (arrays["voltage_V"].tolist() != protocol["voltage_V"]
            or document["generation_scale"] != protocol["generation_scale"]
            or document["incident_power_W_m2"] != protocol["incident_power_W_m2"]):
        raise ValueError("committed result differs from its exact protocol")
    return {"document": document, "arrays": arrays, "worker_result": result, "provenance_comparison": worker_result_provenance(result, run, attempt, store_id=store.store_id,
            current=current_identity, strict_same_source=strict_same_source,
            provenance_ref=by_name["execution_manifest"]["metadata"].get("job_provenance_sha256"), execution_document=saved_document)}
