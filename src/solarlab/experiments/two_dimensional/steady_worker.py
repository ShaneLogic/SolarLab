"""Explicit cold-seed/full-2D jobs on the existing process owner and store.

Prepared spatial declarations keep their existing preview meaning. This
adapter binds a separate dimension-two execution, without a legacy fallback.
"""
from __future__ import annotations

from solarlab.reproducibility.job_provenance import JobProvenance

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import TYPE_CHECKING, Any

from solarlab.config.identity import (
    BuildSource, ExecutionDescription, ExecutionEngine, IdentityInputs, ModelInputs,
    RuntimeEnvironment, execution_manifest, input_identity, physics_identity,
)
from solarlab.config.model_parameters import device_topology, layer_parameter_source
from solarlab.device.resolved import PreparedDevice
from solarlab.experiments.two_dimensional.inputs import JV2DInput
from solarlab.experiments.worker import (
    ManifestBinding, ResourceLimits, WorkerBinding, WorkerContext, WorkerRequest, WorkerResult, observe_environment,
)
from solarlab.materials.source import SourceDocument
from solarlab.physics.registry import CapabilityContext, CapabilityRule, EvidenceRef, ModelDefinition, ModelRef, ModelRegistry, Scope
from solarlab.physics.selection import ModelChoice, select_models
from solarlab.physics.semantic_projection import SemanticProjection

if TYPE_CHECKING:
    import numpy as np
    from solarlab.equations.dc_2d import CompiledDevice2D, State2D
    from solarlab.experiments.execution import JobHandle, ProcessSupervisor
    from solarlab.io.run_store import RunStore
    from solarlab.solvers.nonlinear.steady_2d import Steady2DOptions

__all__ = ["INITIAL_POLICY", "REQUEST_SCHEMA", "RESULT_SCHEMA",
           "Steady2DProtocol", "Steady2DJob", "prepare_steady_2d_job", "steady_2d_worker_binding",
           "run_steady_2d_worker", "read_steady_2d_job"]
WORKER_ID = "steady_2d"
RESULT_SCHEMA = "solarlab.steady-2d.worker-result.v1"
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
PACKAGES = ("numpy", "scipy", "pydantic", "threadpoolctl")
ADAPTERS = ("solarlab/experiments/two_dimensional/steady_worker.py",
            "solarlab/experiments/two_dimensional/steady_results.py",
            "solarlab/config/identity.py")
INITIAL_POLICY = {
    "policy": "cold_auxiliary_1d_lateral_v1", "initial_state": None,
    "auxiliary_seed_is_2d_completion": False, "cross_target_seed_reuse": False,
    "perturbation": {"phi_V": .0007, "z_n": .013, "z_p": -.011,
        "shape": "stack sine times lateral sine/cosine; contact perturbations exactly zero"},
}


def document(name: str, value: Any) -> SourceDocument:
    return SourceDocument(name, json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


REQUEST_SCHEMA = document("solarlab.steady-2d.worker-request.v1", {
    "type": "object", "additionalProperties": False,
    "required": ["schema", "engine", "prepared_sha256", "protocol", "spatial", "controls", "initial"],
    "properties": {"schema": {"const": "solarlab.steady-2d.worker-request.v1"}, "engine": {"const": "solarlab"},
        "prepared_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "protocol": {"type": "object", "additionalProperties": False,
            "required": ["voltage_V", "generation_scales"],
            "properties": {"voltage_V": {"const": .002}, "generation_scales": {"enum": [[0.], [1.], [0., 1.]]}}},
        "spatial": {"type": "object"}, "controls": {"type": "object"},
        "initial": {"const": INITIAL_POLICY}},
    "scope": "uniform_no_grain_classical_2d_13_by_5_fixed_reservoir_neumann"})
REGISTRY = document("solarlab.steady-2d.worker-registry.v1", {
    "worker": WORKER_ID, "engine": "solarlab", "dimension": 2, "fallback": "forbidden",
    "scientific_qualification": False})


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_path(name: str) -> Path:
    path = _root() / name.removeprefix("solarlab/")
    if not name.startswith("solarlab/") or path.suffix != ".py" or not path.resolve().is_relative_to(_root()):
        raise ValueError("invalid bound core source identifier")
    return path


def _adapter_sources(implementation: BuildSource) -> list[dict[str, Any]]:
    base = {s.id: s for s in implementation.files}
    return [{"id": name, "sha256": s.sha256, "content_hex": s.content.hex(),
             "base_sha256": base[name].sha256 if name in base else None}
            for name in ADAPTERS for s in (SourceDocument(name, _source_path(name).read_bytes()),)]


def _verify_sources(execution: ExecutionDescription) -> None:
    expected = {s.id: s.content for s in execution.implementation.files}
    adapters = json.loads(execution.preparation.content)["adapter_sources"]
    if adapters != _adapter_sources(execution.implementation):
        raise ValueError("2D worker/reader differs from the bound adapter bytes")
    expected.update({s["id"]: bytes.fromhex(s["content_hex"]) for s in adapters})
    for name, raw in expected.items():
        if _source_path(name).read_bytes() != raw:
            raise ValueError("installed core differs from bound source: " + name)
    for name, module in tuple(sys.modules.items()):
        if name != "solarlab" and not name.startswith("solarlab."):
            continue
        origin = getattr(module, "__file__", None)
        if origin is None or not Path(origin).resolve().is_relative_to(_root()):
            raise ValueError("loaded core is outside the bound source installation")
        path = Path(origin).resolve()
        identifier = "solarlab/" + path.relative_to(_root()).as_posix()
        if path.read_bytes() and identifier not in expected:
            raise ValueError("loaded core source is unbound: " + identifier)


@dataclass(frozen=True, slots=True)
class Steady2DProtocol:
    voltage_V: float
    generation_scales: tuple[float, ...]

    def __post_init__(self) -> None:
        values = tuple(self.generation_scales)
        if (type(self.voltage_V) not in (int, float) or self.voltage_V != .002
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                or values not in ((0.,), (1.,), (0., 1.))):
            raise ValueError("2D process scope is explicit 2mV dark/G1 targets in order")
        object.__setattr__(self, "generation_scales", tuple(float(v) for v in values))


def _validate(prepared: PreparedDevice, spatial: JV2DInput, options: Steady2DOptions) -> None:
    from solarlab.device.compile_1d import unsupported
    from solarlab.solvers.nonlinear.steady_1d import SteadyOptions
    from solarlab.solvers.nonlinear.steady_2d import Steady2DOptions
    from solarlab.units import normalize_quantity

    if type(prepared) is not PreparedDevice or type(spatial) is not JV2DInput or type(options) is not Steady2DOptions:
        raise ValueError("explicit prepared device, geometry and controls required")
    if (spatial.model_fields_set - {"kind", "lateral_length", "Nx", "Ny_per_layer", "lateral_bc"}
            or spatial.Nx != 4 or spatial.Ny_per_layer != 4 or spatial.lateral_bc != "neumann"
            or spatial.lateral_length is None or normalize_quantity(spatial.lateral_length, "m") != 1e-6):
        raise ValueError("fixed 13x5 geometry requires Nx4/Ny_per_layer4/1um/Neumann, without execution overrides")
    data = prepared.to_mapping()
    if len(prepared.layers) != 3 or data["grain_boundaries"] or data["settings"]["band_grading"]:
        raise ValueError("first 2D process requires three uniform layers and no grain declarations")
    if data["declaration"].get("spectrum") is not None or data["declaration"].get("fixed_generation") is not None:
        raise ValueError("first 2D process has no spectral/fixed generation binding")
    for layer in prepared.layers:
        p = dict(layer.parameters)
        if (layer.role == "substrate" or layer.optical is not None or p["n_optical"] is not None or p["incoherent"]
                or any(p[k] != 0 for k in ("v_sat_n", "v_sat_p", "pf_gamma_n", "pf_gamma_p"))):
            raise ValueError("2D process requires constant-mobility electrical layers without optical-only layers")
    unsupported(prepared, data, prepared.layers)
    expected, supplied = asdict(SteadyOptions()), asdict(options.electronic)
    for key in ("max_iterations", "max_residual_evaluations", "max_jacobian_evaluations"):
        value = supplied.pop(key)
        if type(value) is not int or not 1 <= value <= expected.pop(key):
            raise ValueError("explicit 2D Newton budgets may only decrease the accepted defaults")
    if document("gates", expected) != document("gates", supplied) or options.wall_s != 30.:
        raise ValueError("2D physical gates, step controls and wall ceiling must retain accepted defaults")


def _identity_inputs(prepared: PreparedDevice, spatial: JV2DInput, implementation: BuildSource) -> IdentityInputs:
    from solarlab.units import normalize_quantity
    names = ("solarlab/device/compile_2d.py", "solarlab/equations/dc_2d.py",
             "solarlab/mesh/geometry_2d.py", "solarlab/solvers/nonlinear/steady_2d.py")
    sources = tuple(s for s in implementation.files if s.id in names)
    if {s.id for s in sources} != set(names):
        raise ValueError("actual 2D compiler/equation/geometry/solver source bindings required")
    ref = ModelRef("classical_2d_dc", 1)
    rule = CapabilityRule("source_bound_classical_2d_dc", (2,), ("dc",),
        tuple(EvidenceRef(s.id, s.sha256, "source") for s in sources),
        backends=(WORKER_ID,), statistics=("maxwell_boltzmann",))
    definition = ModelDefinition(ref, "Source-bound connected classical 2D DC", "classical_2d_dc", ("layer",), "node",
        ("mu_n", "mu_p", "tau_n", "tau_p", "alpha", "B_rad", "C_n", "C_p", "N_A", "N_D"), (), (), rule,
        parameter_schema="full_layer")
    registry = ModelRegistry((definition,))
    defaults = {"layer": (ModelChoice("classical_2d_dc", ref, parameter_schema="full_layer"),)}
    topology = device_topology(prepared)
    parameters = {Scope("layer", (layer.id,)): layer_parameter_source(layer)
                  for layer in prepared.layers if layer.id in topology.electrical_layer_ids}
    context = CapabilityContext(2, "dc", WORKER_ID, "maxwell_boltzmann", temperature_K=prepared.to_mapping()["settings"]["T"])
    selection = select_models(registry, topology, context, defaults, parameters, {})
    schema = "solarlab.uniform-no-grain-2d-geometry.v1"
    geometry = document(schema, {"schema": schema, "lateral_length_m": normalize_quantity(spatial.lateral_length, "m"),
        "Nx": spatial.Nx, "Ny_per_layer": spatial.Ny_per_layer, "lateral_bc": spatial.lateral_bc,
        "geometry_policy": "rectangular_half_endpoint_neumann_2d_v1",
        "electrical_layer_ids": [layer.id for layer in prepared.layers if layer.role != "substrate"]})
    return IdentityInputs(prepared, selection, registry, ModelInputs.from_mappings(defaults=defaults, overrides={}),
                          (SemanticProjection(ref, "all_values_v1", sources),), spatial_geometry=geometry)


def _body(prepared: PreparedDevice, spatial: JV2DInput, protocol: Steady2DProtocol, options: Steady2DOptions) -> dict[str, Any]:
    from solarlab.solvers.nonlinear.steady_1d import SteadyOptions
    return {"schema": REQUEST_SCHEMA.id, "engine": "solarlab", "prepared_sha256": prepared.content_sha256,
        "protocol": {"voltage_V": protocol.voltage_V, "generation_scales": list(protocol.generation_scales)},
        "spatial": spatial.editing_data(), "controls": {"auxiliary_1d": asdict(SteadyOptions()), "full_2d": asdict(options)},
        "initial": INITIAL_POLICY}


def _roles(prepared: PreparedDevice, body: dict[str, Any], implementation: BuildSource) -> dict[str, SourceDocument]:
    return {"protocol": document("solarlab.steady-2d.protocol.v1", body["protocol"]),
        "preparation": document("solarlab.steady-2d.preparation.v1", {
            "prepared_sha256": prepared.content_sha256, "resolved_device": prepared.to_mapping(),
            "adapter_sources": _adapter_sources(implementation), "cold_compile_per_job": True,
            "dimension": 2, "source_scope": "committed_core_plus_explicit_active_adapters", "fallback": "forbidden"}),
        "grid": document("solarlab.steady-2d.geometry.v1", body["spatial"]),
        "numeric_controls": document("solarlab.steady-2d.controls.v1", body["controls"]),
        "initial_state": document("solarlab.steady-2d.initial.v1", body["initial"])}


@dataclass(frozen=True, slots=True)
class Steady2DJob:
    request: WorkerRequest
    execution: ManifestBinding

    def submit(self, supervisor: ProcessSupervisor, run_id: str, limits: ResourceLimits) -> JobHandle:
        if limits.wall_seconds > 30 or limits.calls is not None or limits.accepted_steps is not None:
            raise ValueError("2D jobs require wall<=30s and no unobserved whole-job counter caps")
        return supervisor.submit(run_id, WORKER_ID, self.request, limits, execution=self.execution)


def prepare_steady_2d_job(prepared: PreparedDevice, spatial: JV2DInput, protocol: Steady2DProtocol, *,
                          options: Steady2DOptions, implementation: BuildSource,
                          current_repository_head: str, engine: ExecutionEngine) -> Steady2DJob:
    if type(engine) is not ExecutionEngine or engine.name != "solarlab" or type(protocol) is not Steady2DProtocol:
        raise ValueError("explicit new-core engine and 2D target protocol required")
    _validate(prepared, spatial, options)
    inputs = _identity_inputs(prepared, spatial, implementation)
    from solarlab.equations.implementation import selection_setting

    # The child installs these before any engine import: a queued or spawned
    # job runs the equation selection of the parent that prepared it.
    environment = tuple(sorted([*((name, "1") for name in THREADS), selection_setting()]))
    runtime = RuntimeEnvironment(platform.python_implementation(), platform.python_version(),
        platform.system(), platform.machine(), tuple((name, importlib.metadata.version(name)) for name in PACKAGES),
        tuple((name, json.dumps(value)) for name, value in environment))
    body = _body(prepared, spatial, protocol, options)
    execution = ExecutionDescription(**_roles(prepared, body, implementation), implementation=implementation,
        environment=runtime, current_repository_head=current_repository_head)
    _verify_sources(execution)
    manifest = execution_manifest(inputs, execution, engine=engine)
    binding = ManifestBinding(manifest, inputs, execution, engine, manifest.sha256, environment)
    identity = manifest.export()["document"]["execution_identity"]
    request = WorkerRequest(document(REQUEST_SCHEMA.id, body), REGISTRY, REQUEST_SCHEMA,
        document("solarlab.steady-2d.execution-identity.v1", identity), identity["payload"]["H_input"], identity["payload"]["H_physics"])
    binding.validate_request(request)
    return Steady2DJob(request, binding)


def steady_2d_worker_binding() -> WorkerBinding:
    return WorkerBinding.from_function(WORKER_ID, run_steady_2d_worker,
                                      registry_sha256=REGISTRY.sha256, schema_sha256=REQUEST_SCHEMA.sha256)


def _request_contract(request: WorkerRequest, worker_source_sha256: str | None) -> bool:
    """Admit original process requests or the explicitly installed HTTP route."""
    if request.registry == REGISTRY and request.schema == REQUEST_SCHEMA:
        return True
    catalog = json.loads(request.registry.content)
    if not isinstance(catalog, dict):
        return False
    entries = [entry for entry in catalog.get("bindings", ())
               if isinstance(entry, dict) and entry.get("driver") == WORKER_ID]
    return bool(request.registry.id == "solarlab.execution-catalog.v1"
        and request.schema.id == "solarlab.job-submission.schema.v1"
        and catalog.get("schema") == "solarlab.execution-catalog.v1"
        and catalog.get("job_schema_sha256") == request.schema.sha256 and len(entries) == 1
        and entries[0].get("experiment_id") == WORKER_ID and entries[0].get("version") == 1
        and entries[0].get("engine") == "solarlab" and entries[0].get("result_schema") == RESULT_SCHEMA
        and entries[0].get("scope") == "uniform_no_grain_classical_2d_steady"
        and worker_source_sha256 is not None and entries[0].get("worker_source_sha256") == worker_source_sha256)


def _lateral_initial(device: CompiledDevice2D, column_u: np.ndarray, *, voltage_V: float) -> State2D:
    """The accepted research_2d.steady lateral seed recipe, unchanged."""
    import numpy as np
    g, N = device.geometry, device.node_count
    y = (device.column.nodes_m - device.column.nodes_m[0]) / np.ptp(device.column.nodes_m)
    x = (g.x_m - g.x_m[0]) / g.lateral_length_m
    wave = np.sin(np.pi * y)
    wave[[0, -1]] = 0.0
    u = device.extrude_coordinates(column_u)
    u[:N] += (.0007 * wave[:, None] * np.cos(np.pi * x)[None, :]).ravel()
    u[N:2 * N] += (.013 * wave[:, None] * np.sin(np.pi * x)[None, :]).ravel()
    u[2 * N:] += (-.011 * wave[:, None] * np.cos(2 * np.pi * x)[None, :]).ravel()
    return device.state(u, voltage_V=voltage_V)


def run_steady_2d_worker(request: WorkerRequest, worker: WorkerContext) -> WorkerResult:
    worker.check_cancel()
    binding = worker.consume_manifest(request, engine="solarlab")
    from solarlab.equations.implementation import verify_selection

    verify_selection(binding.environment)
    _verify_sources(binding.execution)
    from solarlab.device.compile_2d import compile_device_2d
    from solarlab.equations.implementation import provenance_binding
    from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, SteadyStateError, solve_steady_state
    from solarlab.solvers.nonlinear.steady_2d import Steady2DOptions, solve_steady_state_2d
    from solarlab.experiments.two_dimensional.steady_results import steady_2d_payload, result_digest
    from threadpoolctl import threadpool_info, threadpool_limits

    body = json.loads(request.request.content)
    protocol = Steady2DProtocol(**body["protocol"])
    spatial = JV2DInput.model_validate(body["spatial"])
    supplied = body["controls"]["full_2d"]
    options = Steady2DOptions(SteadyOptions(**supplied["electronic"]), supplied["wall_s"])
    prepared = binding.inputs.prepared
    _validate(prepared, spatial, options)
    expected = _identity_inputs(prepared, spatial, binding.execution.implementation)
    if (body != _body(prepared, spatial, protocol, options)
            or not _request_contract(request, steady_2d_worker_binding().source_sha256)
            or input_identity(expected).sha256 != request.H_input or physics_identity(expected).sha256 != request.H_physics
            or any(getattr(binding.execution, name) != value for name, value in
                   _roles(prepared, body, binding.execution.implementation).items())):
        raise ValueError("2D request/input/model/controls differ from the bound execution")
    _verify_sources(binding.execution)
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    with threadpool_limits(limits=1):
        device = compile_device_2d(prepared, spatial)
    # Admission names the bound equation systems the job evaluates, so the device is compiled first.
    with worker.admit_provenance(request, **provenance_binding("DC2DProblem.steady", device)):
        attempts = []
        capture: dict[str, Any] = {"started_utc": started_utc,
            "build_source_commit": binding.execution.implementation.build_source_commit,
            "runtime": json.loads(json.dumps(asdict(observe_environment(
                PACKAGES, tuple(name for name, _ in binding.environment))))),
            "manifest_sha256": binding.expected_sha256, "request_sha256": request.request.sha256,
            "prepared_sha256": prepared.content_sha256, "scope": "cold_auxiliary_seed_then_connected_2d_newton_in_worker"}
        with threadpool_limits(limits=1):
            pools = threadpool_info()
            if any(pool["num_threads"] != 1 for pool in pools):
                raise ValueError("requested native thread ceiling was not installed")
            deadline = started + options.wall_s
            for scale in protocol.generation_scales:
                worker.check_cancel()
                target = {"voltage_V": protocol.voltage_V, "generation_scale": scale}
                seed, result, failed_seed = None, None, None
                seed_called, full_called = False, False
                try:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("whole2D path exhausted before auxiliary seed")
                    worker.progress({"stage": "steady_2d_auxiliary_seed", "target": target})
                    seed_called = True
                    seed = solve_steady_state(device.column, **target, initial_state=None, options=SteadyOptions())
                    initial = _lateral_initial(device, seed.state.u, voltage_V=protocol.voltage_V)
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("whole2D path exhausted during auxiliary seed")
                    worker.check_cancel()
                    worker.progress({"stage": "steady_2d_full_newton", "target": target, "coordinates": 3 * device.node_count})
                    full_called = True
                    result = solve_steady_state_2d(device, **target, options=replace(options, wall_s=remaining),
                        initial_state=initial, initial_provenance={
                            "kind": "lateral perturbation of auxiliary1D numerical seed", "auxiliary_seed_is_2d_completion": False,
                            "auxiliary_1d_compiled_identity": device.column.identity,
                            "auxiliary_1d_state_sha256": hashlib.sha256(seed.state.u.tobytes()).hexdigest(),
                            "perturbation": INITIAL_POLICY["perturbation"]})
                    accepted, termination, error = result.accepted, result.termination, result.error
                except Exception as exc:
                    accepted, termination, error = False, "preparation_or_evaluation_failed", f"{type(exc).__name__}: {str(exc)[:600]}"
                    if isinstance(exc, SteadyStateError):
                        failed_seed = exc.result
                attempts.append({"target": target, "accepted": accepted, "termination": termination, "error": error,
                    "auxiliary_1d_seed_called": seed_called, "full_2d_adapter_called": full_called,
                    "auxiliary_1d_seed": seed, "failed_auxiliary_seed_newton": failed_seed, "result_2d": result,
                    "auxiliary_seed_is_2d_completion": False})
                if not accepted:
                    break
            worker.check_cancel()
        summary, payloads = steady_2d_payload(device, prepared, body, attempts, time.monotonic() - started)
        sources = {s.id: s.sha256 for s in binding.execution.implementation.files}
        sources.update({s["id"]: s["sha256"] for s in json.loads(binding.execution.preparation.content)["adapter_sources"]})
        capture.update(finished_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sources,
            compiled_identity=device.identity, geometry_identity=device.geometry.sha256,
            result_sha256=result_digest(summary, payloads), threadpools=pools, native_thread_counts_observed=bool(pools))
        return WorkerResult.from_dict({"schema": RESULT_SCHEMA, "engine": "solarlab", "H_input": request.H_input,
            "H_physics": request.H_physics, "H_execution": request.execution_identity.sha256,
            "manifest_sha256": binding.expected_sha256, "prepared_sha256": prepared.content_sha256,
            "steady_2d_result": summary, "capture": capture, "reason": summary["reason"],
            "payloads": {name: {"sha256": p.sha256, "format": p.format, "metadata": json.loads(p.metadata_json)}
                         for name, p in payloads.items()}, "scientific_qualification": False},
            artifacts=tuple(payloads.items()), state="succeeded" if summary["completed"] else "failed")


def read_steady_2d_job(store: RunStore, run_id: str, *, attempt_id: str | None = None,
    current_identity: JobProvenance | dict[str, Any] | None = None, strict_same_source: bool = False) -> dict[str, Any]:
    from solarlab.experiments.two_dimensional.steady_results import read_steady_2d_job as read
    return read(store, run_id, attempt_id=attempt_id, current_identity=current_identity, strict_same_source=strict_same_source)
