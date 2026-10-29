"""Cold ion-free transient jobs and solver-free readback on the shared runtime.

The worker returns the accepted transient publisher's payloads. The supervisor
alone publishes them and commits success or numerical failure. Cancellation
and resource overrides discard late output through the existing owner.
"""
from __future__ import annotations

from solarlab.reproducibility.job_provenance import (
    JobProvenance, read_provenance, validate_recorded_sources, worker_result_provenance,
)

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import platform
from typing import TYPE_CHECKING, Any

from solarlab.config.identity import (
    BuildSource, ExecutionDescription, ExecutionEngine, ExecutionManifest, IdentityInputs, ModelInputs,
    RuntimeEnvironment, execution_manifest, input_identity, physics_identity,
)
from solarlab.config.model_parameters import device_topology, layer_parameter_source
from solarlab.device.resolved import PreparedDevice
from solarlab.experiments.jv.worker import (
    _PACKAGES, _THREADS, _adapter_sources, _document, _verify_sources, validate_steady_jv_scope,
)
from solarlab.experiments.worker import (
    ManifestBinding, ResourceLimits, WorkerBinding, WorkerContext, WorkerRequest, WorkerResult,
    observe_environment,
)
from solarlab.materials.source import SourceDocument
from solarlab.physics.registry import (
    CapabilityContext, CapabilityRule, EvidenceRef, ModelDefinition, ModelRef, ModelRegistry, Scope,
)
from solarlab.physics.selection import ModelChoice, select_models
from solarlab.physics.semantic_projection import SemanticProjection

if TYPE_CHECKING:
    from solarlab.experiments.execution import JobHandle, ProcessSupervisor
    from solarlab.experiments.transient import TransientOptions, TransientProtocol
    from solarlab.io.run_store import RunStore
    from solarlab.solvers.nonlinear.steady_1d import SteadyOptions

__all__ = ["TransientJob", "prepare_transient_job", "transient_worker_binding",
           "run_transient_worker", "read_transient_job", "transient_identity_inputs"]

_ID = "transient_1d"
_SCHEMA = _document("solarlab.transient.worker-request.v1", {
    "engine": "solarlab", "fields": ["schema", "engine", "prepared_sha256", "protocol",
        "intervals_per_layer", "transient_options", "initial_options"]})
_REGISTRY = _document("solarlab.transient.worker-registry.v1", {
    "engine": "solarlab", "worker": _ID, "scope": "classical_ion_free_1d_transient", "fallback": "forbidden"})
_RESULT = "solarlab.transient.worker-result.v1"
_PHYSICS_SOURCES = ("solarlab/device/compile_1d.py", "solarlab/equations/dc_1d.py",
                    "solarlab/solvers/time/backward_euler.py")


def _identity_inputs(prepared: PreparedDevice, implementation: BuildSource) -> IdentityInputs:
    sources = tuple(next((s for s in implementation.files if s.id == name), None) for name in _PHYSICS_SOURCES)
    if any(source is None for source in sources):
        raise ValueError("actual compiler, equation and conservative time sources must be bound")
    # This selection describes the implemented law; it grants no scientific certificate.
    checked = tuple(source for source in sources if source is not None)
    ref = ModelRef("classical_1d_transient", 1)
    rule = CapabilityRule("source_bound_classical_transient", (1,), ("transient",),
        tuple(EvidenceRef(s.id, s.sha256, "source") for s in checked),
        backends=(_ID,), statistics=("maxwell_boltzmann",))
    definition = ModelDefinition(ref, "Conservative ion-free 1D source binding", "classical_transient",
        ("layer",), "node", ("mu_n", "mu_p", "tau_n", "tau_p", "alpha", "B_rad", "C_n", "C_p", "N_A", "N_D"),
        (), (), rule, parameter_schema="full_layer")
    registry = ModelRegistry((definition,))
    defaults = {"layer": (ModelChoice("classical_transient", ref, parameter_schema="full_layer"),)}
    topology = device_topology(prepared)
    parameters = {Scope("layer", (layer.id,)): layer_parameter_source(layer)
                  for layer in prepared.layers if layer.id in topology.electrical_layer_ids}
    context = CapabilityContext(1, "transient", _ID, "maxwell_boltzmann",
                                temperature_K=prepared.to_mapping()["settings"]["T"])
    selection = select_models(registry, topology, context, defaults, parameters, {})
    return IdentityInputs(prepared, selection, registry, ModelInputs.from_mappings(defaults=defaults, overrides={}),
                          (SemanticProjection(ref, "all_values_v1", checked),))


def _body(prepared: PreparedDevice, protocol: TransientProtocol, intervals: int,
          options: TransientOptions, initial_options: SteadyOptions) -> dict[str, Any]:
    return {"schema": _SCHEMA.id, "engine": "solarlab", "prepared_sha256": prepared.content_sha256,
        "protocol": {name: getattr(protocol, name).tolist() for name in ("times_s", "voltage_V", "generation_scale")},
        "intervals_per_layer": intervals, "transient_options": asdict(options), "initial_options": asdict(initial_options)}


def _roles(prepared: PreparedDevice, body: dict[str, Any], implementation: BuildSource) -> dict[str, SourceDocument]:
    return {
        "protocol": _document("solarlab.transient.protocol.v1", body["protocol"]),
        "preparation": _document("solarlab.transient.preparation.v1", {
            "prepared_sha256": prepared.content_sha256, "resolved_experiment": prepared.to_mapping(),
            "source_experiment": [{"id": s.id, "sha256": s.sha256, "content_hex": s.content.hex()}
                                  for s in prepared.sources],
            "adapter_sources": _adapter_sources(implementation),
            "source_scope": "supplied_committed_core_plus_explicit_adapter_bytes",
            "cold_compile_per_job": True, "fallback": "forbidden"}),
        "initial_state": _document("solarlab.transient.initial-recipe.v1", {
            "prepared_sha256": prepared.content_sha256, "recipe": "cold_solve_steady_state_in_worker",
            "voltage_V": body["protocol"]["voltage_V"][0],
            "generation_scale": body["protocol"]["generation_scale"][0],
            "supplied_working_point": False, "cross_job_reuse": False}),
        "grid": _document("solarlab.transient.grid-recipe.v1", {
            "intervals_per_layer": body["intervals_per_layer"], "electrical_grid": prepared.to_mapping()["electrical_grid"]}),
        "numeric_controls": _document("solarlab.transient.controls.v1", {
            "transient": body["transient_options"], "initial_dc": body["initial_options"]}),
    }


def _validate_controls(protocol: TransientProtocol, intervals: int, options: TransientOptions,
                       initial_options: SteadyOptions) -> None:
    from solarlab.experiments.transient import TransientOptions, TransientProtocol
    from solarlab.solvers.nonlinear.steady_1d import SteadyOptions

    if type(protocol) is not TransientProtocol or type(options) is not TransientOptions or type(initial_options) is not SteadyOptions:
        raise ValueError("explicit transient protocol, time controls and initial DC controls are required")
    if type(intervals) is not int or intervals < 2:
        raise ValueError("explicit integer intervals_per_layer >= 2 required")
    if protocol.times_s.size - 1 > options.max_steps:
        raise ValueError("requested protocol exceeds its explicit step budget")


@dataclass(frozen=True, slots=True)
class TransientJob:
    request: WorkerRequest
    execution: ManifestBinding

    def submit(self, supervisor: ProcessSupervisor, run_id: str, limits: ResourceLimits) -> JobHandle:
        if limits.accepted_steps is not None or limits.calls is not None:
            raise ValueError("this cold DC/transient adapter does not observe whole-job counter budgets")
        return supervisor.submit(run_id, _ID, self.request, limits, execution=self.execution)


def prepare_transient_job(prepared: PreparedDevice, protocol: TransientProtocol, *,
    intervals_per_layer: int, options: TransientOptions, initial_options: SteadyOptions,
    implementation: BuildSource, current_repository_head: str, engine: ExecutionEngine,
) -> TransientJob:
    """Bind explicit experiment/protocol/controls and source bytes; perform no solve."""
    if type(engine) is not ExecutionEngine or engine.name != "solarlab" or type(implementation) is not BuildSource:
        raise ValueError("explicit solarlab engine and actual implementation sources required")
    validate_steady_jv_scope(prepared)
    _validate_controls(protocol, intervals_per_layer, options, initial_options)
    inputs = _identity_inputs(prepared, implementation)
    body = _body(prepared, protocol, intervals_per_layer, options, initial_options)
    settings = tuple(sorted((name, "1") for name in _THREADS))
    environment = RuntimeEnvironment(platform.python_implementation(), platform.python_version(),
        platform.system(), platform.machine(), tuple((n, importlib.metadata.version(n)) for n in _PACKAGES),
        tuple((name, json.dumps(value)) for name, value in settings))
    description = ExecutionDescription(**_roles(prepared, body, implementation), implementation=implementation,
        environment=environment, current_repository_head=current_repository_head)
    _verify_sources(description)
    manifest = execution_manifest(inputs, description, engine=engine)
    binding = ManifestBinding(manifest, inputs, description, engine, manifest.sha256, settings)
    identity = manifest.export()["document"]["execution_identity"]
    request = WorkerRequest(_document(_SCHEMA.id, body), _REGISTRY, _SCHEMA,
        _document("solarlab.transient.execution-identity.v1", identity),
        identity["payload"]["H_input"], identity["payload"]["H_physics"])
    binding.validate_request(request)
    return TransientJob(request, binding)


def transient_worker_binding() -> WorkerBinding:
    return WorkerBinding.from_function(_ID, run_transient_worker,
        registry_sha256=_REGISTRY.sha256, schema_sha256=_SCHEMA.sha256)


def run_transient_worker(request: WorkerRequest, worker: WorkerContext) -> WorkerResult:
    """Cold compile/DC followed by one time history, within the existing owner."""
    worker.check_cancel()
    binding = worker.consume_manifest(request, engine="solarlab")
    _verify_sources(binding.execution)
    body = json.loads(request.request.content)
    prepared = binding.inputs.prepared
    if body.get("engine") != "solarlab" or body.get("schema") != _SCHEMA.id:
        raise ValueError("request differs from the explicit transient invocation")
    validate_steady_jv_scope(prepared)
    from solarlab.device.compile_1d import compile_device_1d
    from solarlab.experiments.transient import TransientOptions, TransientProtocol
    from solarlab.experiments.transient_results import _payload, run_transient_for_publication
    from solarlab.solvers.nonlinear.problem import SolveBudget
    from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, solve_steady_state
    from threadpoolctl import threadpool_info, threadpool_limits

    protocol = TransientProtocol(**body["protocol"])
    options = TransientOptions(**{**body["transient_options"],
                                 "newton_budget": SolveBudget(**body["transient_options"]["newton_budget"])})
    initial_options = SteadyOptions(**body["initial_options"])
    intervals = body["intervals_per_layer"]
    _validate_controls(protocol, intervals, options, initial_options)
    expected = _identity_inputs(prepared, binding.execution.implementation)
    if (body != _body(prepared, protocol, intervals, options, initial_options)
            or input_identity(expected).sha256 != request.H_input or physics_identity(expected).sha256 != request.H_physics
            or any(getattr(binding.execution, name) != value for name, value in
                   _roles(prepared, body, binding.execution.implementation).items())):
        raise ValueError("request/input/model/protocol/controls differ from the transient manifest")
    if worker._limits.accepted_steps is not None or worker._limits.calls is not None:
        raise ValueError("whole-job counter budgets are not observed by this transient adapter")
    _verify_sources(binding.execution)
    worker.check_cancel()
    with worker.admit_provenance(request, operator="BackwardEuler1DStep"):
        started = datetime.now(timezone.utc).isoformat()
        with threadpool_limits(limits=1):
            pools = threadpool_info()
            if any(pool["num_threads"] != 1 for pool in pools):
                raise ValueError("child numerical thread limit was not installed")
            runtime = asdict(observe_environment(_PACKAGES, _THREADS))
            worker.progress({"stage": "transient_cold_compile", "requested_steps": protocol.times_s.size - 1})
            compiled = compile_device_1d(prepared, intervals_per_layer=intervals)
            worker.check_cancel()
            worker.progress({"stage": "transient_initial_dc", "nodes": compiled.node_count})
            initial = solve_steady_state(compiled, voltage_V=float(protocol.voltage_V[0]),
                generation_scale=float(protocol.generation_scale[0]), options=initial_options)
            worker.check_cancel()
            cold = {"scope": "cold_compile_and_initial_dc_in_worker", "started_utc": started,
                "finished_utc": datetime.now(timezone.utc).isoformat(), "prepared_sha256": prepared.content_sha256,
                "compiled_identity": compiled.identity, "manifest_sha256": binding.expected_sha256,
                "runtime": runtime, "threadpools": pools}
            worker.progress({"stage": "transient_time_steps", "nodes": compiled.node_count})
            captured = run_transient_for_publication(compiled, initial, protocol,
                                                   options=options, initial_options=initial_options)
        worker.check_cancel()
        summary, payloads = _payload(captured.result, captured.nodes_m)
        capture = json.loads(captured.capture_json)
        # Link the pre-transient cold execution to the wrapper's actual input words.
        cold["inputs"] = capture["description"]["inputs"]
        return WorkerResult.from_dict({"schema": _RESULT, "engine": "solarlab",
            "prepared_sha256": prepared.content_sha256, "H_input": request.H_input, "H_physics": request.H_physics,
            "H_execution": request.execution_identity.sha256, "manifest_sha256": binding.expected_sha256,
            "transient_result": summary, "reason": summary["reason"], "cold_start": json.loads(_document("cold", cold).content),
            "transient_capture": capture,
            "payloads": {name: {"sha256": p.sha256, "format": p.format, "metadata": json.loads(p.metadata_json)}
                         for name, p in payloads.items()},
            "scientific_qualification": False}, artifacts=tuple(payloads.items()),
            state="succeeded" if captured.result.completed else "failed")


def read_transient_job(store: RunStore, run_id: str, *, attempt_id: str | None = None,
    current_identity: JobProvenance | dict[str, Any] | None = None, strict_same_source: bool = False) -> dict[str, Any]:
    """Verify one committed complete/partial history without importing a solver."""
    import numpy as np
    from solarlab.experiments.transient_results import (
        _UNITS, _decode, _digest, _inputs, _result_digest, _validate,
    )
    from solarlab.io.artifacts import ArtifactPayload, _json_document, _parse_document

    run = store.get_run(run_id)
    selected = run["current_attempt_id"] if attempt_id is None else attempt_id
    attempt = next((a for a in store.attempts(run_id) if a["attempt_id"] == selected), None)
    if attempt is None or attempt["state"] not in {"succeeded", "failed"} or attempt["terminal"] is None:
        raise ValueError("a committed completed or partial transient attempt is required")
    result = (attempt["result"] or {}).get("worker_result", {})
    if (result.get("schema") != _RESULT or result.get("engine") != "solarlab"
            or result.get("scientific_qualification") is not False):
        raise ValueError("attempt has no returned transient history")
    identity = run["request"]["identity"]
    metadata = run["request"]["input_metadata"]["execution_manifest"]
    if any(result[name] != identity[name] for name in ("H_input", "H_physics", "H_execution")) or result["manifest_sha256"] != metadata["manifest_sha256"]:
        raise ValueError("transient result belongs to another execution identity/manifest")
    summary = result["transient_result"]
    references = attempt["terminal"]["artifacts"]
    records = [store.artifact(ref["artifact_id"]) for ref in references]
    by_name = {item["name"]: item for item in records}
    if (len(records) != 16 or len(by_name) != 16 or set(result["payloads"]) != {*_UNITS, "transient_diagnostics"}
            or set(by_name) != {*result["payloads"], "execution_manifest"}):
        raise ValueError("terminal lacks the exact transient payloads and execution manifest")
    for ref, item in zip(references, records):
        if (item["phase"] != "committed" or item["run_id"] != run_id or item["attempt_id"] != selected
                or item["generation"] != attempt["generation"] or item["manifest_sha256"] != ref["manifest_sha256"]):
            raise ValueError("transient artifact/terminal reference belongs to another committed attempt")
    raw = store.read_artifact(by_name["execution_manifest"]["artifact_id"])
    manifest = ExecutionManifest(raw, metadata["current_repository_head"])
    document = manifest.export()["document"]
    bound = document["execution_identity"]["payload"]
    if (manifest.sha256 != result["manifest_sha256"] or manifest.execution_sha256 != identity["H_execution"]
            or any(bound[name] != identity[name] for name in ("H_input", "H_physics"))
            or bound["engine"]["name"] != "solarlab"):
        raise ValueError("committed manifest differs from the transient execution binding")
    roles = {name: json.loads(bytes.fromhex(value["content_hex"])) for name, value in document["content_roles"].items()}
    arrays, payloads = {}, {}
    for name, item_binding in result["payloads"].items():
        item = by_name[name]
        if (item_binding != {"sha256": item["sha256"], "format": item["format"], "metadata": item["metadata"]}
                or item["metadata"].get("schema") != summary["schema"]
                or item["metadata"].get("compiled_identity") != summary["compiled_identity"]):
            raise ValueError("transient payload differs from its result identity/metadata")
        payloads[name] = ArtifactPayload(store.read_artifact(item["artifact_id"]), item["format"], _json_document(item["metadata"]))
        if name in _UNITS:
            if item["metadata"].get("unit") != _UNITS[name]:
                raise ValueError("unexpected transient physical unit")
            arrays[name] = store.read_array(item["artifact_id"])
    diagnostics = _decode(_parse_document(payloads["transient_diagnostics"].data.decode()))
    _validate(summary, arrays, diagnostics)
    if (attempt["state"] != ("succeeded" if summary["outcome"] == "completed" else "failed")
            or result["reason"] != summary["reason"]):
        raise ValueError("terminal state/reason differs from the actual transient outcome")
    protocol = {name: arrays["requested_" + name].tolist() for name in ("times_s", "voltage_V", "generation_scale")}
    controls = roles["numeric_controls"]
    expected_controls = dict(controls["transient"])
    if np.any(arrays["requested_voltage_V"] != arrays["requested_voltage_V"][0]):
        expected_controls["voltage_seed_policy"] = "harmonic_potential_preserve_densities_when_voltage_changes"
    if (protocol != roles["protocol"] or diagnostics["solver_controls"] != expected_controls
            or diagnostics["initial_controls"] != controls["initial_dc"]
            or result["prepared_sha256"] != roles["preparation"]["prepared_sha256"]):
        raise ValueError("saved history differs from the prepared protocol or numerical controls")
    capture, cold = result["transient_capture"], result["cold_start"]
    description = capture["description"]
    inputs = _inputs(summary["compiled_identity"], arrays, diagnostics["initial"], diagnostics["initial_controls"])
    expected_identity = {"H_input": _digest(inputs), "H_physics": summary["compiled_identity"], "H_execution": _digest(description)}
    if (capture["status"] != "captured_at_computation" or description["scope"] != "solve_transient_call"
            or any(description[n] is not None for n in ("compiled_device_history", "initial_dc_history"))
            or description["inputs"] != inputs or capture["identity"] != expected_identity
            or description["numerics"] != controls["transient"]
            or capture["result_sha256"] != _result_digest(summary, payloads)):
        raise ValueError("transient computation capture differs from saved inputs/results")
    sources = {s["id"]: s["sha256"] for s in document["implementation_content"]}
    sources.update({s["id"]: s["sha256"] for s in roles["preparation"]["adapter_sources"]})
    if description.get("job_provenance") is not None:
        if description["job_provenance"] != result.get("job_provenance"):
            raise ValueError("transient producer differs from admitted job provenance")
        validate_recorded_sources(description)
        if description["build_source_commit"] != read_provenance(description["job_provenance"]).export()["git_commit"]:
            raise ValueError("transient build commit differs from admitted job provenance")
    elif description["build_source_commit"] is not None or any(
            sources.get(name) != digest for name, digest in description["sources"].items()):
        raise ValueError("transient execution source differs from the request source binding")
    if (cold["scope"] != "cold_compile_and_initial_dc_in_worker" or cold["inputs"] != inputs
            or cold["compiled_identity"] != summary["compiled_identity"]
            or cold["prepared_sha256"] != result["prepared_sha256"] or cold["manifest_sha256"] != manifest.sha256
            or cold["runtime"] != bound["environment"]
            or any(pool["num_threads"] != 1 for pool in cold["threadpools"])):
        raise ValueError("cold compile/DC capture differs from the actual execution")
    return {"document": {**summary, "run_id": run_id, "attempt_id": selected, "generation": attempt["generation"],
                "arrays": {name: by_name[name]["artifact_id"] for name in _UNITS}},
            "arrays": arrays, "diagnostics": diagnostics, "worker_result": result,
            "run_state": attempt["state"], "manifest": manifest.export(), "provenance_comparison": worker_result_provenance(result, run, attempt, store_id=store.store_id,
            current=current_identity, strict_same_source=strict_same_source,
            provenance_ref=by_name["execution_manifest"]["metadata"].get("job_provenance_sha256"), execution_document=document)}

# Public entry point for trusted companion preparation.
transient_identity_inputs = _identity_inputs
