"""Cold ionic steady/time jobs on the existing supervisor, manifest and store."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import platform
from typing import TYPE_CHECKING, Any

from solarlab.config.identity import (
    BuildSource, ExecutionDescription, ExecutionEngine, ExecutionManifest, IdentityInputs, ModelInputs,
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
    from solarlab.experiments.ionic_transient import IonicTransientOptions
    from solarlab.experiments.transient import TransientProtocol
    from solarlab.io.run_store import RunStore
    from solarlab.solvers.nonlinear.ionic_1d import IonicSteadyOptions

__all__ = ["IonicTransientJob", "prepare_ionic_transient_job", "ionic_transient_worker_binding",
           "run_ionic_transient_worker", "read_ionic_transient_job"]
_ID = "ionic_transient_1d"
_SCHEMA = _document("solarlab.ionic-transient.worker-request.v1", {
    "engine": "solarlab", "fields": ["schema", "engine", "prepared_sha256", "protocol", "grid", "controls"]})
_REGISTRY = _document("solarlab.ionic-transient.worker-registry.v1", {
    "engine": "solarlab", "worker": _ID, "scope": "one_blocking_domain_classical_1d_ionic_time", "fallback": "forbidden"})
_RESULT = "solarlab.ionic-transient.worker-result.v1"
_PHYSICS = ("solarlab/device/compile_ionic_1d.py", "solarlab/equations/ionic_1d.py",
            "solarlab/solvers/time/ionic_backward_euler.py", "solarlab/experiments/ionic_transient.py")


def _identity_inputs(prepared: PreparedDevice, implementation: BuildSource) -> IdentityInputs:
    sources = tuple(next((s for s in implementation.files if s.id == name), None) for name in _PHYSICS)
    if any(source is None for source in sources):
        raise ValueError("actual ionic compiler, equation, time and driver sources must be bound")
    checked = tuple(source for source in sources if source is not None)
    ref = ModelRef("classical_ionic_1d_transient", 1)
    rule = CapabilityRule("source_bound_classical_ionic_transient", (1,), ("transient",),
        tuple(EvidenceRef(s.id, s.sha256, "source") for s in checked),
        backends=(_ID,), statistics=("maxwell_boltzmann",))
    definition = ModelDefinition(ref, "Source-bound blocking ionic time", "classical_ionic_transient", ("layer",), "node",
        ("mu_n", "mu_p", "tau_n", "tau_p", "alpha", "B_rad", "C_n", "C_p", "N_A", "N_D",
         "P0", "P0_neg", "P_lim", "P_lim_neg", "D_ion", "D_ion_neg", "E_a_ion"), (), (), rule,
        parameter_schema="full_layer")
    registry = ModelRegistry((definition,))
    defaults = {"layer": (ModelChoice("classical_ionic_transient", ref, parameter_schema="full_layer"),)}
    topology = device_topology(prepared)
    parameters = {Scope("layer", (layer.id,)): layer_parameter_source(layer)
                  for layer in prepared.layers if layer.id in topology.electrical_layer_ids}
    context = CapabilityContext(1, "transient", _ID, "maxwell_boltzmann", temperature_K=prepared.to_mapping()["settings"]["T"])
    selection = select_models(registry, topology, context, defaults, parameters, {})
    return IdentityInputs(prepared, selection, registry, ModelInputs.from_mappings(defaults=defaults, overrides={}),
                          (SemanticProjection(ref, "all_values_v1", checked),))


def _validate(prepared: PreparedDevice, protocol: TransientProtocol, intervals: tuple[int, ...], domain: str,
              endpoints: tuple[str, str], options: IonicTransientOptions, initial: IonicSteadyOptions) -> None:
    from solarlab.device.compile_1d import _unsupported
    from solarlab.experiments.ionic_transient import IonicTransientOptions
    from solarlab.experiments.transient import TransientProtocol
    from solarlab.solvers.nonlinear.ionic_1d import IonicSteadyOptions

    if type(prepared) is not PreparedDevice or type(protocol) is not TransientProtocol:
        raise ValueError("explicit prepared device and transient protocol required")
    if type(options) is not IonicTransientOptions or type(initial) is not IonicSteadyOptions:
        raise ValueError("explicit ionic steady and time controls required")
    layers = tuple(layer for layer in prepared.layers if layer.role != "substrate")
    if (len(intervals) != len(layers) or any(type(v) is not int or v < 2 for v in intervals)
            or domain not in {layer.id for layer in layers} or endpoints != ("blocking", "blocking")):
        raise ValueError("explicit layer intervals and one named blocking ion domain required")
    count = intervals[next(i for i, layer in enumerate(layers) if layer.id == domain)] + 1
    if (type(initial.inventory_row_node) is not int or not 0 <= initial.inventory_row_node < count
            or protocol.times_s.size - 1 > options.electronic.max_steps
            or initial.wall_s > 30 or options.electronic.wall_s > 30):
        raise ValueError("invalid ionic grid/step budget or numerical wall ceiling above30s")
    data = prepared.to_mapping()
    populations = ("P0", "P0_neg", "D_ion", "D_ion_neg")
    for layer in prepared.layers:
        parameters = dict(layer.parameters)
        if layer.id != domain and any(parameters[name] != 0.0 for name in populations):
            raise ValueError("outside-domain ions or multiple ionic domains are unsupported")
    p: dict[str, Any] = dict(next(layer for layer in layers if layer.id == domain).parameters)
    if not math.isfinite(float(p["E_a_ion"])) or p["E_a_ion"] < 0:
        raise ValueError("finite nonnegative ionic activation energy required")
    dual = p["P0_neg"] != 0 or p["D_ion_neg"] != 0
    for suffix in ("", "_neg") if dual else ("",):
        reference, capacity, diffusivity = (float(p[name + suffix]) for name in ("P0", "P_lim", "D_ion"))
        if not (all(math.isfinite(x) for x in (reference, capacity, diffusivity)) and 0 < reference < capacity and diffusivity > 0):
            raise ValueError("cold ionic steady requires a positive unsaturated connected species")
    if data["settings"]["ion_steric_shared_site"] and (not dual or not data["settings"]["ion_steric_diffusion_only"]
            or p["P_lim"] != p["P_lim_neg"] or p["P0"] + p["P0_neg"] >= p["P_lim"]):
        raise ValueError("shared sites require two unsaturated populations and one diffusion-only capacity")
    # Static electronic admission only; the original prepared input remains the
    # identity and the child compiles the full ionic device, never this view.
    screen = {**data, "settings": {**data["settings"], "ion_steric_shared_site": False, "ion_steric_diffusion_only": False}}
    screened_layers = tuple(replace(layer, parameters=tuple((key, 0.0 if key in populations else value)
                             for key, value in layer.parameters)) for layer in layers)
    _unsupported(prepared, screen, screened_layers)


def _body(prepared: PreparedDevice, protocol: TransientProtocol, intervals: tuple[int, ...], domain: str,
          endpoints: tuple[str, str], options: IonicTransientOptions, initial: IonicSteadyOptions) -> dict[str, Any]:
    return {"schema": _SCHEMA.id, "engine": "solarlab", "prepared_sha256": prepared.content_sha256,
        "protocol": {name: getattr(protocol, name).tolist() for name in ("times_s", "voltage_V", "generation_scale")},
        "grid": {"intervals_per_layer": list(intervals), "ion_domain_layer": domain, "ion_domain_endpoints": list(endpoints)},
        "controls": {"transient": asdict(options), "initial_steady": asdict(initial)}}


def _roles(prepared: PreparedDevice, body: dict[str, Any], implementation: BuildSource) -> dict[str, SourceDocument]:
    return {"protocol": _document("solarlab.ionic-transient.protocol.v1", body["protocol"]),
        "preparation": _document("solarlab.ionic-transient.preparation.v1", {
            "prepared_sha256": prepared.content_sha256, "resolved_experiment": prepared.to_mapping(),
            "source_experiment": [{"id": s.id, "sha256": s.sha256, "content_hex": s.content.hex()} for s in prepared.sources],
            "adapter_sources": _adapter_sources(implementation), "cold_compile_per_job": True,
            "source_scope": "committed_core_plus_explicit_active_adapters", "fallback": "forbidden"}),
        "initial_state": _document("solarlab.ionic-transient.initial-recipe.v1", {
            "recipe": "cold_solve_ionic_steady_state_in_worker", "prepared_sha256": prepared.content_sha256,
            "voltage_V": body["protocol"]["voltage_V"][0], "generation_scale": body["protocol"]["generation_scale"][0],
            "supplied_working_point": False, "cross_job_reuse": False}),
        "grid": _document("solarlab.ionic-transient.grid.v1", body["grid"]),
        "numeric_controls": _document("solarlab.ionic-transient.controls.v1", body["controls"])}


def _limits(limits: ResourceLimits) -> None:
    if limits.accepted_steps is not None or limits.calls is not None:
        raise ValueError("ionic cold/time adapter does not observe whole-job counter budgets")
    if limits.wall_seconds > 30:
        raise ValueError("ionic worker requires a whole-job wall ceiling <=30s")


@dataclass(frozen=True, slots=True)
class IonicTransientJob:
    request: WorkerRequest
    execution: ManifestBinding

    def submit(self, supervisor: ProcessSupervisor, run_id: str, limits: ResourceLimits) -> JobHandle:
        _limits(limits)
        return supervisor.submit(run_id, _ID, self.request, limits, execution=self.execution)


def prepare_ionic_transient_job(prepared: PreparedDevice, protocol: TransientProtocol, *,
    intervals_per_layer: int | tuple[int, ...], ion_domain_layer: str, ion_domain_endpoints: tuple[str, str],
    options: IonicTransientOptions, initial_options: IonicSteadyOptions,
    implementation: BuildSource, current_repository_head: str, engine: ExecutionEngine,
) -> IonicTransientJob:
    """Bind the complete explicit cold recipe without compiling or solving."""
    if type(engine) is not ExecutionEngine or engine.name != "solarlab" or type(implementation) is not BuildSource:
        raise ValueError("explicit solarlab engine and immutable core sources required")
    if type(prepared) is not PreparedDevice:
        raise ValueError("explicit PreparedDevice required")
    layers = tuple(layer for layer in prepared.layers if layer.role != "substrate")
    intervals = (intervals_per_layer,) * len(layers) if isinstance(intervals_per_layer, int) else tuple(intervals_per_layer)
    _validate(prepared, protocol, intervals, ion_domain_layer, ion_domain_endpoints, options, initial_options)
    inputs = _identity_inputs(prepared, implementation)
    body = _body(prepared, protocol, intervals, ion_domain_layer, ion_domain_endpoints, options, initial_options)
    settings = tuple(sorted((name, "1") for name in _THREADS))
    environment = RuntimeEnvironment(platform.python_implementation(), platform.python_version(), platform.system(), platform.machine(),
        tuple((name, importlib.metadata.version(name)) for name in _PACKAGES), tuple((name, json.dumps(value)) for name, value in settings))
    description = ExecutionDescription(**_roles(prepared, body, implementation), implementation=implementation,
        environment=environment, current_repository_head=current_repository_head)
    _verify_sources(description)
    manifest = execution_manifest(inputs, description, engine=engine)
    execution = ManifestBinding(manifest, inputs, description, engine, manifest.sha256, settings)
    identity = manifest.export()["document"]["execution_identity"]
    request = WorkerRequest(_document(_SCHEMA.id, body), _REGISTRY, _SCHEMA,
        _document("solarlab.ionic-transient.execution-identity.v1", identity), identity["payload"]["H_input"], identity["payload"]["H_physics"])
    execution.validate_request(request)
    return IonicTransientJob(request, execution)


def ionic_transient_worker_binding() -> WorkerBinding:
    return WorkerBinding.from_function(_ID, run_ionic_transient_worker, registry_sha256=_REGISTRY.sha256, schema_sha256=_SCHEMA.sha256)


def run_ionic_transient_worker(request: WorkerRequest, worker: WorkerContext) -> WorkerResult:
    worker.check_cancel()
    binding = worker.consume_manifest(request, engine="solarlab")
    _verify_sources(binding.execution)
    _limits(worker._limits)
    from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
    from solarlab.experiments.ionic_transient import IonicTransientInitial, IonicTransientOptions, solve_ionic_transient
    from solarlab.experiments.ionic_transient_results import INITIAL_ROLE, ionic_initial_document, ionic_transient_payload, result_digest
    from solarlab.experiments.transient import TransientOptions, TransientProtocol
    from solarlab.experiments.transient_results import _fingerprint
    from solarlab.solvers.nonlinear.ionic_1d import IonicSteadyOptions, solve_ionic_steady_state
    from solarlab.solvers.nonlinear.problem import SolveBudget
    from solarlab.solvers.nonlinear.steady_1d import SteadyOptions
    from threadpoolctl import threadpool_info, threadpool_limits

    body = json.loads(request.request.content)
    controls, grid = body["controls"], body["grid"]
    initial_data, time_data = controls["initial_steady"], controls["transient"]
    electronic = time_data["electronic"]
    options = IonicTransientOptions(**{**time_data, "electronic": TransientOptions(**{**electronic,
        "newton_budget": SolveBudget(**electronic["newton_budget"])})})
    initial_options = IonicSteadyOptions(**{**initial_data, "electronic": SteadyOptions(**initial_data["electronic"]),
                                          "budget": SolveBudget(**initial_data["budget"])})
    protocol = TransientProtocol(**body["protocol"])
    intervals, domain, endpoints = tuple(grid["intervals_per_layer"]), grid["ion_domain_layer"], tuple(grid["ion_domain_endpoints"])
    prepared = binding.inputs.prepared
    _validate(prepared, protocol, intervals, domain, endpoints, options, initial_options)
    inputs = _identity_inputs(prepared, binding.execution.implementation)
    if (body != _body(prepared, protocol, intervals, domain, endpoints, options, initial_options)
            or input_identity(inputs).sha256 != request.H_input or physics_identity(inputs).sha256 != request.H_physics
            or any(getattr(binding.execution, name) != value for name, value in _roles(prepared, body, binding.execution.implementation).items())):
        raise ValueError("ionic request/configuration/protocol/controls differ from manifest")
    _verify_sources(binding.execution)
    worker.check_cancel()
    capture: dict[str, Any] = {"scope": "cold_ionic_compile_steady_and_time_in_worker", "started_utc": datetime.now(timezone.utc).isoformat(),
        "prepared_sha256": prepared.content_sha256, "request_sha256": request.request.sha256,
        "manifest_sha256": binding.expected_sha256, "build_source_commit": binding.execution.implementation.build_source_commit}
    with threadpool_limits(limits=1):
        pools = threadpool_info()
        if any(pool["num_threads"] != 1 for pool in pools):
            raise ValueError("child thread limit was not installed")
        capture.update(runtime=asdict(observe_environment(_PACKAGES, _THREADS)), threadpools=pools)
        worker.progress({"stage": "ionic_cold_compile"})
        compiled = compile_ionic_device_1d(prepared, intervals_per_layer=intervals,
                                         ion_domain_layer=domain, ion_domain_endpoints=endpoints)
        worker.check_cancel()
        worker.progress({"stage": "ionic_initial_steady", "electrical_nodes": compiled.node_count, "ion_nodes": compiled.ion_node_count})
        initial = solve_ionic_steady_state(compiled, voltage_V=float(protocol.voltage_V[0]),
            generation_scale=float(protocol.generation_scale[0]), options=initial_options)
        initial_bytes = ionic_initial_document(compiled, initial, protocol)
        initial_sha256 = hashlib.sha256(initial_bytes).hexdigest()
        worker.check_cancel()
        history = None
        if initial.accepted:
            assert initial.state is not None
            admitted = IonicTransientInitial(initial.state, float(protocol.generation_scale[0]), initial_options,
                                              INITIAL_ROLE, initial_sha256, initial_bytes.decode())
            worker.progress({"stage": "ionic_time_steps", "requested_steps": protocol.times_s.size - 1})
            history = solve_ionic_transient(compiled, admitted, protocol, options=options)
        worker.check_cancel()
    _verify_sources(binding.execution)
    summary, payloads = ionic_transient_payload(compiled, initial, protocol, history, initial_bytes)
    sources = {source.id: source.sha256 for source in binding.execution.implementation.files}
    sources.update({item["id"]: item["sha256"] for item in json.loads(binding.execution.preparation.content)["adapter_sources"]})
    capture.update(finished_utc=datetime.now(timezone.utc).isoformat(), compiled_identity=compiled.identity,
        initial_artifact_sha256=initial_sha256, initial_state=None if initial.state is None else _fingerprint(initial.state.u),
        source_sha256=sources, result_sha256=result_digest(summary, payloads), time_solve_started=initial.accepted)
    capture = json.loads(_document("ionic-capture", capture).content)
    return WorkerResult.from_dict({"schema": _RESULT, "engine": "solarlab", "H_input": request.H_input,
        "H_physics": request.H_physics, "H_execution": request.execution_identity.sha256, "manifest_sha256": binding.expected_sha256,
        "prepared_sha256": prepared.content_sha256, "ionic_result": summary, "reason": summary["reason"],
        "capture": capture, "payloads": {name: {"sha256": p.sha256, "format": p.format, "metadata": json.loads(p.metadata_json)}
        for name, p in payloads.items()}, "scientific_qualification": False}, artifacts=tuple(payloads.items()),
        state="succeeded" if summary["outcome"] == "completed" else "failed")


def read_ionic_transient_job(store: RunStore, run_id: str, *, attempt_id: str | None = None) -> dict[str, Any]:
    """Verify committed completed/partial ionic history without solver imports."""
    from solarlab.experiments.ionic_transient_results import (
        SCHEMA, result_digest, storage_components, stored_units, units, unpack_ionic_arrays, validate_ionic_history,
    )
    from solarlab.experiments.transient_results import _decode, _fingerprint
    from solarlab.io.artifacts import ArtifactPayload, _json_document, _parse_document

    run = store.get_run(run_id)
    selected = run["current_attempt_id"] if attempt_id is None else attempt_id
    attempt = next((a for a in store.attempts(run_id) if a["attempt_id"] == selected), None)
    if attempt is None or attempt["state"] not in {"succeeded", "failed"} or attempt["terminal"] is None:
        raise ValueError("a committed ionic completed/failed attempt is required; cancelled has no invented history")
    result = (attempt["result"] or {}).get("worker_result", {})
    if result.get("schema") != _RESULT or result.get("engine") != "solarlab" or result.get("scientific_qualification") is not False:
        raise ValueError("attempt has no returned ionic history")
    identity = run["request"]["identity"]
    metadata = run["request"]["input_metadata"]["execution_manifest"]
    if any(result[name] != identity[name] for name in ("H_input", "H_physics", "H_execution")) or result["manifest_sha256"] != metadata["manifest_sha256"]:
        raise ValueError("ionic result identity differs from its attempt")
    summary = result["ionic_result"]
    species = tuple(summary["ion_species"])
    unit_map = stored_units()
    references = attempt["terminal"]["artifacts"]
    records = [store.artifact(ref["artifact_id"]) for ref in references]
    by_name = {item["name"]: item for item in records}
    expected = {*unit_map, "ionic_initial_steady", "ionic_transient_diagnostics"}
    if len(records) != len(expected) + 1 or set(by_name) != {*expected, "execution_manifest"} or set(result["payloads"]) != expected:
        raise ValueError("ionic terminal lacks the exact artifact roles")
    for reference, item in zip(references, records):
        if (item["phase"] != "committed" or item["run_id"] != run_id or item["attempt_id"] != selected
                or item["generation"] != attempt["generation"] or item["manifest_sha256"] != reference["manifest_sha256"]):
            raise ValueError("ionic artifact belongs to another committed attempt/reference")
    manifest = ExecutionManifest(store.read_artifact(by_name["execution_manifest"]["artifact_id"]), metadata["current_repository_head"])
    document = manifest.export()["document"]
    bound = document["execution_identity"]["payload"]
    if (manifest.sha256 != result["manifest_sha256"] or manifest.execution_sha256 != identity["H_execution"]
            or any(bound[name] != identity[name] for name in ("H_input", "H_physics")) or bound["engine"]["name"] != "solarlab"):
        raise ValueError("ionic execution manifest differs from request/result")
    stored, payloads = {}, {}
    for name, data in result["payloads"].items():
        item = by_name[name]
        if (data != {"sha256": item["sha256"], "format": item["format"], "metadata": item["metadata"]}
                or item["metadata"].get("schema") != SCHEMA or item["metadata"].get("compiled_identity") != summary["compiled_identity"]):
            raise ValueError("ionic payload metadata differs from result")
        payloads[name] = ArtifactPayload(store.read_artifact(item["artifact_id"]), item["format"], _json_document(item["metadata"]))
        if name in unit_map:
            if (item["metadata"].get("unit") != unit_map[name]
                    or item["metadata"].get("components") != list(storage_components(species).get(name, ()))):
                raise ValueError("unexpected ionic physical unit")
            stored[name] = store.read_array(item["artifact_id"])
    raw = _decode(_parse_document(payloads["ionic_transient_diagnostics"].data.decode()))
    arrays = unpack_ionic_arrays(summary, stored, raw)
    validate_ionic_history(summary, arrays, raw, payloads["ionic_initial_steady"].data)
    if attempt["state"] != ("succeeded" if summary["outcome"] == "completed" else "failed") or result["reason"] != summary["reason"]:
        raise ValueError("ionic terminal state differs from the retained numerical result")
    roles = {name: json.loads(bytes.fromhex(value["content_hex"])) for name, value in document["content_roles"].items()}
    transport = WorkerRequest.from_bytes(_json_document(run["request"]["input_metadata"]["transport"]).encode())
    submitted = json.loads(transport.request.content)
    if (transport.H_input != identity["H_input"] or transport.H_physics != identity["H_physics"]
            or transport.execution_identity.sha256 != identity["H_execution"]
            or submitted.get("schema") != _SCHEMA.id or submitted.get("engine") != "solarlab"
            or submitted["prepared_sha256"] != summary["prepared_sha256"]
            or submitted["protocol"] != roles["protocol"] or submitted["grid"] != roles["grid"]
            or submitted["controls"] != roles["numeric_controls"]):
        raise ValueError("ionic manifest/result differs from the exact submitted request")
    grid = roles["grid"]
    layers = [layer for layer in roles["preparation"]["resolved_experiment"]["layers"] if layer["role"] != "substrate"]
    domain = next((i for i, layer in enumerate(layers) if layer["id"] == grid["ion_domain_layer"]), None)
    if (domain is None or grid["ion_domain_endpoints"] != ["blocking", "blocking"]
            or raw["compiled_binding"]["ion_domain"]["layer"] != grid["ion_domain_layer"]
            or len(grid["intervals_per_layer"]) != len(layers)
            or len(arrays["nodes_m"]) != 1 + sum(grid["intervals_per_layer"])
            or len(arrays["ion_nodes_m"]) != 1 + grid["intervals_per_layer"][domain]):
        raise ValueError("compiled ionic grid differs from the explicit submitted domain recipe")
    if (raw["protocol"] is None or any(arrays["requested_" + name].tolist() != roles["protocol"][name]
            for name in ("times_s", "voltage_V", "generation_scale"))
            or raw["initial"]["solver_controls"] != roles["numeric_controls"]["initial_steady"]
            or roles["preparation"]["prepared_sha256"] != summary["prepared_sha256"]):
        raise ValueError("saved ionic protocol/initial controls differ from manifest")
    if raw["history"] is not None:
        saved_controls = raw["history"]["solver_controls"]
        if any(saved_controls[name] != value for name, value in roles["numeric_controls"]["transient"].items()):
            raise ValueError("ionic time controls differ from the executed request")
    capture = result["capture"]
    sources = {item["id"]: item["sha256"] for item in document["implementation_content"]}
    sources.update({item["id"]: item["sha256"] for item in roles["preparation"]["adapter_sources"]})
    initial_state = raw["initial"]["state"]
    if (capture["scope"] != "cold_ionic_compile_steady_and_time_in_worker"
            or capture["build_source_commit"] != bound["implementation"]["build_source_commit"]
            or capture["source_sha256"] != sources or capture["runtime"] != bound["environment"]
            or capture["manifest_sha256"] != manifest.sha256 or capture["compiled_identity"] != summary["compiled_identity"]
            or capture["request_sha256"] != transport.request.sha256
            or capture["prepared_sha256"] != summary["prepared_sha256"]
            or capture["initial_artifact_sha256"] != payloads["ionic_initial_steady"].sha256
            or capture["initial_state"] != (None if initial_state is None else _fingerprint(initial_state["u"]))
            or capture["time_solve_started"] != raw["initial"]["accepted"]
            or capture["result_sha256"] != result_digest(summary, payloads)):
        raise ValueError("ionic computation capture differs from actual stored source/inputs/result")
    return {"document": {**summary, "run_id": run_id, "attempt_id": selected, "generation": attempt["generation"],
        "arrays": {name: by_name[name]["artifact_id"] for name in unit_map}, "logical_array_units": units(species),
        "array_layout": {"packed_components": storage_components(species),
            "raw_diagnostic_fields": ["state_coordinates", "ion_nodes_m"],
            "cell_inventory_definition": "declared ion cell measure times committed species density"}},
        "arrays": arrays, "stored_arrays": stored, "diagnostics": raw,
        "worker_result": result, "manifest": manifest.export(), "run_state": attempt["state"]}
