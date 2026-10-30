"""Explicit illuminated ionic J-V jobs using the existing process owner/store.

The child compiles the declared device and admits an exact saved initial or
runs the existing cold-dark/direct-density/coupled initialization. Only the
supervisor publishes artifacts and commits a terminal; no numerical fallback.
"""
from __future__ import annotations

from solarlab.reproducibility.job_provenance import JobProvenance

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import platform
from typing import TYPE_CHECKING, Any

from solarlab.config.identity import (
    BuildSource, ExecutionDescription, ExecutionEngine, IdentityInputs, ModelInputs,
    RuntimeEnvironment, execution_manifest, input_identity, physics_identity,
)
from solarlab.config.model_parameters import device_topology, layer_parameter_source
from solarlab.device.resolved import PreparedDevice
from solarlab.experiments.jv.worker import _PACKAGES, _THREADS, _adapter_sources, _document, _verify_sources
from solarlab.experiments.worker import (
    ManifestBinding, ResourceLimits, WorkerBinding, WorkerContext, WorkerRequest, WorkerResult, observe_environment,
)
from solarlab.materials.source import SourceDocument
from solarlab.physics.registry import CapabilityContext, CapabilityRule, EvidenceRef, ModelDefinition, ModelRef, ModelRegistry, Scope
from solarlab.physics.selection import ModelChoice, select_models
from solarlab.physics.semantic_projection import SemanticProjection

if TYPE_CHECKING:
    from solarlab.experiments.execution import JobHandle, ProcessSupervisor
    from solarlab.experiments.jv.ionic_steady import IonicSteadyJVOptions
    from solarlab.io.run_store import RunStore

__all__ = ["IonicJVProtocol", "IonicJVInitial", "IonicJVJob", "prepare_ionic_jv_job",
           "ionic_jv_worker_binding", "run_ionic_jv_worker", "read_ionic_jv_job", "ionic_jv_identity_inputs"]
WORKER_ID = "ionic_steady_jv"
REQUEST_SCHEMA = _document("solarlab.ionic-jv.worker-request.v1", {
    "engine": "solarlab", "fields": ["schema", "engine", "prepared_sha256", "protocol", "grid", "controls", "initial"]})
REGISTRY = _document("solarlab.ionic-jv.worker-registry.v1", {
    "engine": "solarlab", "worker": WORKER_ID, "scope": "illuminated_positive_D_one_blocking_domain_ionic_steady_jv",
    "fallback": "forbidden"})
RESULT_SCHEMA = "solarlab.ionic-jv.worker-result.v1"


@dataclass(frozen=True, slots=True)
class IonicJVProtocol:
    voltage_V: tuple[float, ...]
    generation_scale: float
    incident_power_W_m2: float | None

    def __post_init__(self) -> None:
        values = tuple(self.voltage_V)
        if (not 1 <= len(values) <= 101 or any(type(v) not in {int, float} or not math.isfinite(v) for v in values)
                or not (all(b > a for a, b in zip(values, values[1:])) or all(b < a for a, b in zip(values, values[1:])))):
            raise ValueError("ionic J-V requires 1..101 finite strictly monotonic voltages")
        if (type(self.generation_scale) not in {int, float} or not math.isfinite(self.generation_scale)
                or self.generation_scale <= 0):
            raise ValueError("this worker requires explicit positive illumination")
        power = self.incident_power_W_m2
        if power is not None and (type(power) not in {int, float} or not math.isfinite(power) or power <= 0):
            raise ValueError("incident power must be finite and positive when supplied")
        object.__setattr__(self, "voltage_V", tuple(float(v) for v in values))


@dataclass(frozen=True, slots=True)
class IonicJVInitial:
    policy: str
    voltage_V: float
    wall_s: float
    artifact_bytes: bytes | None = None
    artifact_sha256: str | None = None

    def __post_init__(self) -> None:
        if (self.policy not in {"dark_direct_density_seed_v1", "saved_accepted"}
                or not math.isfinite(self.voltage_V) or not math.isfinite(self.wall_s) or not 0 < self.wall_s <= 30):
            raise ValueError("explicit initial policy, finite voltage and wall ceiling <=30s required")
        if self.policy == "saved_accepted":
            if (type(self.artifact_bytes) is not bytes or not self.artifact_bytes
                    or hashlib.sha256(self.artifact_bytes).hexdigest() != self.artifact_sha256):
                raise ValueError("saved initial requires exact artifact bytes and matching SHA-256")
            SourceDocument("ionic-jv-saved-initial", self.artifact_bytes)
            if not isinstance(json.loads(self.artifact_bytes), dict):
                raise ValueError("saved initial must be an explicit result document")
        elif self.artifact_bytes is not None or self.artifact_sha256 is not None:
            raise ValueError("cold initialization cannot contain a hidden saved initial")


def _initial_body(initial: IonicJVInitial) -> dict[str, Any]:
    return {"policy": initial.policy, "voltage_V": initial.voltage_V, "wall_s": initial.wall_s,
        "artifact_content_hex": None if initial.artifact_bytes is None else initial.artifact_bytes.hex(),
        "artifact_sha256": initial.artifact_sha256}


def _identity_inputs(prepared: PreparedDevice, implementation: BuildSource) -> IdentityInputs:
    names = ("solarlab/device/compile_ionic_1d.py", "solarlab/equations/ionic_1d.py",
             "solarlab/solvers/nonlinear/ionic_1d.py", "solarlab/experiments/ionic_initialization.py",
             "solarlab/experiments/jv/ionic_steady.py")
    sources = tuple(s for s in implementation.files if s.id in names)
    if {s.id for s in sources} != set(names):
        raise ValueError("actual ionic compiler/equation/steady/initialization/J-V sources must be bound")
    ref = ModelRef("classical_ionic_steady_jv", 1)
    rule = CapabilityRule("source_bound_ionic_steady_jv", (1,), ("dc",),
        tuple(EvidenceRef(s.id, s.sha256, "source") for s in sources),
        backends=(WORKER_ID,), statistics=("maxwell_boltzmann",))
    definition = ModelDefinition(ref, "Source-bound illuminated ionic steady J-V", "classical_ionic_dc", ("layer",), "node",
        ("mu_n", "mu_p", "tau_n", "tau_p", "alpha", "B_rad", "C_n", "C_p", "N_A", "N_D",
         "P0", "P0_neg", "P_lim", "P_lim_neg", "D_ion", "D_ion_neg", "E_a_ion"), (), (), rule,
        parameter_schema="full_layer")
    registry = ModelRegistry((definition,))
    defaults = {"layer": (ModelChoice("classical_ionic_dc", ref, parameter_schema="full_layer"),)}
    topology = device_topology(prepared)
    parameters = {Scope("layer", (layer.id,)): layer_parameter_source(layer)
                  for layer in prepared.layers if layer.id in topology.electrical_layer_ids}
    context = CapabilityContext(1, "dc", WORKER_ID, "maxwell_boltzmann", temperature_K=prepared.to_mapping()["settings"]["T"])
    selection = select_models(registry, topology, context, defaults, parameters, {})
    return IdentityInputs(prepared, selection, registry, ModelInputs.from_mappings(defaults=defaults, overrides={}),
                          (SemanticProjection(ref, "all_values_v1", sources),))


def ionic_jv_identity_inputs(prepared: PreparedDevice, implementation: BuildSource) -> IdentityInputs:
    """Expose the unchanged identity recipe for explicitly installed adapters."""
    return _identity_inputs(prepared, implementation)


def _validate(prepared: PreparedDevice, protocol: IonicJVProtocol, intervals: tuple[int, ...], domain: str,
              endpoints: tuple[str, str], options: IonicSteadyJVOptions, initial: IonicJVInitial) -> None:
    from solarlab.device.compile_1d import unsupported
    from solarlab.experiments.jv.ionic_steady import IonicSteadyJVOptions

    if type(prepared) is not PreparedDevice or type(protocol) is not IonicJVProtocol or type(initial) is not IonicJVInitial:
        raise ValueError("explicit prepared device, J-V protocol and initial policy required")
    if type(options) is not IonicSteadyJVOptions:
        raise ValueError("explicit ionic steady J-V controls required")
    layers = tuple(layer for layer in prepared.layers if layer.role != "substrate")
    if (len(intervals) != len(layers) or any(type(v) is not int or v < 2 for v in intervals)
            or domain not in {layer.id for layer in layers} or endpoints != ("blocking", "blocking")):
        raise ValueError("explicit layer intervals and one named blocking domain required")
    count = intervals[next(i for i, layer in enumerate(layers) if layer.id == domain)] + 1
    if (type(options.steady.inventory_row_node) is not int or not 0 <= options.steady.inventory_row_node < count
            or len(protocol.voltage_V) > options.max_points or options.wall_s > 25 or options.steady.wall_s > 30):
        raise ValueError("invalid inventory row/point count or curve wall ceiling >25s")
    data = prepared.to_mapping()
    populations = ("P0", "P0_neg", "D_ion", "D_ion_neg")
    for layer in prepared.layers:
        parameters = dict(layer.parameters)
        if layer.id != domain and any(parameters[name] != 0 for name in populations):
            raise ValueError("outside-domain ions or multiple ionic domains are unsupported")
    p: dict[str, Any] = dict(next(layer for layer in layers if layer.id == domain).parameters)
    dual = p["P0_neg"] != 0 or p["D_ion_neg"] != 0
    for suffix in ("", "_neg") if dual else ("",):
        reference, capacity, diffusion = (float(p[name + suffix]) for name in ("P0", "P_lim", "D_ion"))
        if not (all(math.isfinite(x) for x in (reference, capacity, diffusion)) and 0 < reference < capacity and diffusion > 0):
            raise ValueError("ionic J-V requires unsaturated connected positive-D species")
    if data["settings"]["ion_steric_shared_site"] and (not dual or not data["settings"]["ion_steric_diffusion_only"]
            or p["P_lim"] != p["P_lim_neg"] or p["P0"] + p["P0_neg"] >= p["P_lim"]):
        raise ValueError("shared sites require two unsaturated populations and one diffusion-only capacity")
    screen = {**data, "settings": {**data["settings"], "ion_steric_shared_site": False, "ion_steric_diffusion_only": False}}
    screened = tuple(replace(layer, parameters=tuple((k, 0.0 if k in populations else v) for k, v in layer.parameters)) for layer in layers)
    unsupported(prepared, screen, screened)


def _roles(prepared: PreparedDevice, body: dict[str, Any], implementation: BuildSource) -> dict[str, SourceDocument]:
    return {"protocol": _document("solarlab.ionic-jv.protocol.v1", body["protocol"]),
        "preparation": _document("solarlab.ionic-jv.preparation.v1", {
            "prepared_sha256": prepared.content_sha256, "resolved_experiment": prepared.to_mapping(),
            "source_experiment": [{"id": s.id, "sha256": s.sha256, "content_hex": s.content.hex()} for s in prepared.sources],
            "adapter_sources": _adapter_sources(implementation), "cold_compile_per_job": True,
            "source_scope": "committed_core_plus_explicit_active_adapters", "fallback": "forbidden"}),
        "initial_state": _document("solarlab.ionic-jv.initial.v1", body["initial"]),
        "grid": _document("solarlab.ionic-jv.grid.v1", body["grid"]),
        "numeric_controls": _document("solarlab.ionic-jv.controls.v1", body["controls"])}


def _limits(limits: ResourceLimits) -> None:
    if limits.accepted_steps is not None or limits.calls is not None or limits.wall_seconds > 30:
        raise ValueError("ionic J-V requires wall<=30s and no unobserved whole-job counter caps")


@dataclass(frozen=True, slots=True)
class IonicJVJob:
    request: WorkerRequest
    execution: ManifestBinding

    def submit(self, supervisor: ProcessSupervisor, run_id: str, limits: ResourceLimits) -> JobHandle:
        _limits(limits)
        return supervisor.submit(run_id, WORKER_ID, self.request, limits, execution=self.execution)


def prepare_ionic_jv_job(prepared: PreparedDevice, protocol: IonicJVProtocol, *, intervals_per_layer: tuple[int, ...],
    ion_domain_layer: str, ion_domain_endpoints: tuple[str, str], options: IonicSteadyJVOptions, initial: IonicJVInitial,
    implementation: BuildSource, current_repository_head: str, engine: ExecutionEngine,
) -> IonicJVJob:
    """Bind explicit data, artifact bytes and actual sources before child work."""
    if type(engine) is not ExecutionEngine or engine.name != "solarlab" or type(implementation) is not BuildSource:
        raise ValueError("explicit solarlab engine and immutable core sources required")
    intervals = tuple(intervals_per_layer)
    _validate(prepared, protocol, intervals, ion_domain_layer, ion_domain_endpoints, options, initial)
    inputs = _identity_inputs(prepared, implementation)
    body = {"schema": REQUEST_SCHEMA.id, "engine": "solarlab", "prepared_sha256": prepared.content_sha256,
        "protocol": asdict(protocol), "grid": {"intervals_per_layer": list(intervals), "ion_domain_layer": ion_domain_layer,
        "ion_domain_endpoints": list(ion_domain_endpoints)}, "controls": asdict(options), "initial": _initial_body(initial)}
    from solarlab.equations.implementation import selection_setting

    # The child installs these before any engine import: a queued or spawned
    # job runs the equation selection of the parent that prepared it.
    settings = tuple(sorted([*((name, "1") for name in _THREADS), selection_setting()]))
    environment = RuntimeEnvironment(platform.python_implementation(), platform.python_version(), platform.system(), platform.machine(),
        tuple((name, importlib.metadata.version(name)) for name in _PACKAGES), tuple((name, json.dumps(value)) for name, value in settings))
    description = ExecutionDescription(**_roles(prepared, body, implementation), implementation=implementation,
        environment=environment, current_repository_head=current_repository_head)
    _verify_sources(description)
    manifest = execution_manifest(inputs, description, engine=engine)
    execution = ManifestBinding(manifest, inputs, description, engine, manifest.sha256, settings)
    identity = manifest.export()["document"]["execution_identity"]
    request = WorkerRequest(_document(REQUEST_SCHEMA.id, body), REGISTRY, REQUEST_SCHEMA,
        _document("solarlab.ionic-jv.execution-identity.v1", identity), identity["payload"]["H_input"], identity["payload"]["H_physics"])
    execution.validate_request(request)
    return IonicJVJob(request, execution)


def ionic_jv_worker_binding() -> WorkerBinding:
    return WorkerBinding.from_function(WORKER_ID, run_ionic_jv_worker, registry_sha256=REGISTRY.sha256, schema_sha256=REQUEST_SCHEMA.sha256)


def run_ionic_jv_worker(request: WorkerRequest, worker: WorkerContext) -> WorkerResult:
    worker.check_cancel()
    binding = worker.consume_manifest(request, engine="solarlab")
    from solarlab.equations.implementation import verify_selection

    verify_selection(binding.environment)
    _verify_sources(binding.execution)
    _limits(worker._limits)
    from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
    from solarlab.equations.implementation import provenance_binding
    from solarlab.experiments.ionic_initialization import IonicInitializationOptions, initialize_ionic_steady_state
    from solarlab.experiments.ionic_transient import IonicTransientInitial, _admit_evidence
    from solarlab.experiments.jv.ionic_results import initial_document, ionic_jv_payload, result_digest
    from solarlab.experiments.jv.ionic_steady import IonicSteadyJVOptions, run_ionic_steady_jv
    from solarlab.experiments.transient_results import _fingerprint
    from solarlab.solvers.nonlinear.ionic_1d import IonicSteadyOptions, solve_ionic_steady_state
    from solarlab.solvers.nonlinear.problem import SolveBudget
    from solarlab.solvers.nonlinear.steady_1d import SteadyOptions
    from threadpoolctl import threadpool_info, threadpool_limits

    body = json.loads(request.request.content)
    controls, grid, initial_data = body["controls"], body["grid"], body["initial"]
    steady_data = controls["steady"]
    steady = IonicSteadyOptions(**{**steady_data, "electronic": SteadyOptions(**steady_data["electronic"]),
                                  "budget": SolveBudget(**steady_data["budget"])})
    options = IonicSteadyJVOptions(steady, controls["max_points"], controls["wall_s"])
    protocol = IonicJVProtocol(**body["protocol"])
    initial = IonicJVInitial(initial_data["policy"], initial_data["voltage_V"], initial_data["wall_s"],
        None if initial_data["artifact_content_hex"] is None else bytes.fromhex(initial_data["artifact_content_hex"]),
        initial_data["artifact_sha256"])
    intervals, domain, endpoints = tuple(grid["intervals_per_layer"]), grid["ion_domain_layer"], tuple(grid["ion_domain_endpoints"])
    prepared = binding.inputs.prepared
    _validate(prepared, protocol, intervals, domain, endpoints, options, initial)
    inputs = _identity_inputs(prepared, binding.execution.implementation)
    if (body["schema"] != REQUEST_SCHEMA.id or body["engine"] != "solarlab" or body["prepared_sha256"] != prepared.content_sha256
            or input_identity(inputs).sha256 != request.H_input or physics_identity(inputs).sha256 != request.H_physics
            or any(getattr(binding.execution, k) != v for k, v in _roles(prepared, body, binding.execution.implementation).items())):
        raise ValueError("ionic J-V request/initial/configuration differs from manifest")
    _verify_sources(binding.execution)
    started_utc = datetime.now(timezone.utc).isoformat()
    worker.progress({"stage": "ionic_jv_compile"})
    with threadpool_limits(limits=1):
        compiled = compile_ionic_device_1d(prepared, intervals_per_layer=intervals,
                                         ion_domain_layer=domain, ion_domain_endpoints=endpoints)
    # Admission names the bound equation system the job evaluates, so the device is compiled first.
    with worker.admit_provenance(request, **provenance_binding("Ionic1DProblem.steady_jv", compiled,
                                                         inventory_row_node=steady.inventory_row_node)):
        capture: dict[str, Any] = {"scope": "ionic_compile_initialization_and_steady_jv_in_worker",
            "started_utc": started_utc, "prepared_sha256": prepared.content_sha256,
            "request_sha256": request.request.sha256, "manifest_sha256": binding.expected_sha256,
            "build_source_commit": binding.execution.implementation.build_source_commit}
        dark = illumination = curve = admitted = None
        with threadpool_limits(limits=1):
            pools = threadpool_info()
            if any(pool["num_threads"] != 1 for pool in pools):
                raise ValueError("child thread limit was not installed")
            capture.update(runtime=asdict(observe_environment(_PACKAGES, tuple(name for name, _ in binding.environment))), threadpools=pools)
            worker.check_cancel()
            if initial.policy == "saved_accepted":
                assert initial.artifact_bytes is not None and initial.artifact_sha256 is not None
                initial_bytes = initial.artifact_bytes
                state, generation, _ = _admit_evidence(compiled, initial_bytes.decode(), initial.artifact_sha256, steady)
                if generation != protocol.generation_scale or state.voltage_V != initial.voltage_V:
                    raise ValueError("saved initial voltage/illumination differs from explicit recipe")
                admitted = IonicTransientInitial(state, generation, steady, "request-supplied:ionic_jv_initial",
                                                initial.artifact_sha256, initial_bytes.decode())
            else:
                worker.progress({"stage": "ionic_jv_cold_dark", "electrical_nodes": compiled.node_count, "ion_nodes": compiled.ion_node_count})
                dark = solve_ionic_steady_state(compiled, voltage_V=initial.voltage_V, generation_scale=0., options=steady)
                initial_bytes = initial_document(compiled.identity, dark, initial.voltage_V, 0.)
                worker.check_cancel()
                if dark.accepted:
                    assert dark.state is not None
                    dark_input = IonicTransientInitial(dark.state, 0., steady, "worker-produced:ionic_jv_dark",
                        hashlib.sha256(initial_bytes).hexdigest(), initial_bytes.decode())
                    worker.progress({"stage": "ionic_jv_direct_density_initialization"})
                    illumination = initialize_ionic_steady_state(compiled, initial=dark_input,
                        voltage_V=initial.voltage_V, generation_scale=protocol.generation_scale,
                        options=IonicInitializationOptions(steady, "direct_density_seed_v1", initial.wall_s))
                    initial_bytes = initial_document(compiled.identity, illumination.coupled_result, initial.voltage_V,
                                                    protocol.generation_scale)
                    if illumination.accepted:
                        assert illumination.state is not None
                        admitted = IonicTransientInitial(illumination.state, protocol.generation_scale, steady,
                            "worker-produced:ionic_jv_initial", hashlib.sha256(initial_bytes).hexdigest(), initial_bytes.decode())
            worker.check_cancel()
            if admitted is not None:
                worker.progress({"stage": "ionic_jv_curve", "requested_points": len(protocol.voltage_V)})
                curve = run_ionic_steady_jv(compiled, protocol.voltage_V, generation_scale=protocol.generation_scale,
                    options=options, initial_policy="saved_accepted", initial=admitted, incident_power_W_m2=protocol.incident_power_W_m2)
            worker.check_cancel()
        summary, payloads = ionic_jv_payload(compiled, body, dark, illumination, curve, initial_bytes)
        sources = {s.id: s.sha256 for s in binding.execution.implementation.files}
        sources.update({s["id"]: s["sha256"] for s in json.loads(binding.execution.preparation.content)["adapter_sources"]})
        capture.update(finished_utc=datetime.now(timezone.utc).isoformat(), compiled_identity=compiled.identity,
            initial_artifact_sha256=hashlib.sha256(initial_bytes).hexdigest(),
            initial_state=None if admitted is None else _fingerprint(admitted.state.u),
            source_sha256=sources, result_sha256=result_digest(summary, payloads), curve_started=curve is not None)
        capture = json.loads(_document("ionic-jv-capture", capture).content)
        return WorkerResult.from_dict({"schema": RESULT_SCHEMA, "engine": "solarlab", "H_input": request.H_input,
            "H_physics": request.H_physics, "H_execution": request.execution_identity.sha256, "manifest_sha256": binding.expected_sha256,
            "prepared_sha256": prepared.content_sha256, "ionic_jv_result": summary, "reason": summary["reason"],
            "capture": capture, "payloads": {name: {"sha256": p.sha256, "format": p.format, "metadata": json.loads(p.metadata_json)}
            for name, p in payloads.items()}, "scientific_qualification": False}, artifacts=tuple(payloads.items()),
            state="succeeded" if summary["outcome"] == "completed" else "failed")


def read_ionic_jv_job(store: RunStore, run_id: str, *, attempt_id: str | None = None,
    current_identity: JobProvenance | dict[str, Any] | None = None, strict_same_source: bool = False) -> dict[str, Any]:
    """Read exact committed data without importing a compiler/equation/solver."""
    from solarlab.experiments.jv.ionic_results import read_ionic_jv_job as read
    return read(store, run_id, attempt_id=attempt_id, current_identity=current_identity, strict_same_source=strict_same_source)
