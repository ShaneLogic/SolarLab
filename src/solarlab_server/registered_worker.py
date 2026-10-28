"""Checked installed routes to the existing legacy and steady workers."""
from dataclasses import asdict, replace
import base64
import json

from solarlab.experiments.worker import WorkerContext, WorkerRequest, WorkerResult
from solarlab.materials.source import SourceDocument
from solarlab_server.legacy_engine import _run_legacy, legacy_worker_binding


def run_legacy(request: WorkerRequest, context: WorkerContext) -> WorkerResult:
    catalog = json.loads(request.registry.content)
    body = json.loads(request.request.content)
    if (set(body) != {"experiment_id", "version", "engine", "legacy_request_id", "legacy_request_base64"}
            or catalog["schema"] != "solarlab.execution-catalog.v1" or body["engine"] != "legacy"
            or catalog["job_schema_sha256"] != request.schema.sha256
            or catalog["legacy_facade_sha256"] != legacy_worker_binding().source_sha256):
        raise ValueError("worker registry/schema/facade differs from the admitted source")
    entry = next((r for r in catalog["bindings"] if (r["experiment_id"], r["version"], r["engine"]) ==
                  (body["experiment_id"], body["version"], body["engine"])), None)
    binding = context.consume_manifest(request, engine="legacy")
    if (entry is None or catalog["model_registry_sha256"] != binding.inputs.registry.content_sha256
            or entry["model_registry_sha256"] != binding.inputs.registry.content_sha256):
        raise ValueError("worker experiment/model registry mismatch")
    inner = replace(request, request=SourceDocument(body["legacy_request_id"],
        base64.b64decode(body["legacy_request_base64"], validate=True)))
    if json.loads(inner.request.content)["driver"] != entry["driver"]:
        raise ValueError("worker driver differs from the selected experiment")
    output = _run_legacy(inner, context)
    return WorkerResult.from_dict({**json.loads(output.result_json),
        "experiment_id": body["experiment_id"], "experiment_version": body["version"],
        "execution_catalog_sha256": request.registry.sha256}, artifacts=output.artifacts)


def run_steady(request: WorkerRequest, context: WorkerContext) -> WorkerResult:
    """Consume the installed source/device binding, then the real cold worker."""
    from solarlab.experiments.jv.worker import run_steady_jv_worker, steady_jv_worker_binding
    from solarlab_server.steady_preparation import _device_binding, _implementation_binding

    context.check_cancel()
    catalog = json.loads(request.registry.content)
    body = json.loads(request.request.content)
    if (set(body) != {"experiment_id", "version", "engine", "device_id", "steady_request_id", "steady_request_base64"}
            or catalog["schema"] != "solarlab.execution-catalog.v1" or body["engine"] != "solarlab"
            or catalog["job_schema_sha256"] != request.schema.sha256):
        raise ValueError("steady worker registry/schema differs from the installed route")
    entry = next((r for r in catalog["bindings"] if (r["experiment_id"], r["version"], r["engine"]) ==
                  (body["experiment_id"], body["version"], body["engine"])), None)
    if (entry is None or entry["scope"] != "static_classical_ion_free_1d_dc"
            or entry["driver"] != "steady_jv"
            or entry["worker_source_sha256"] != steady_jv_worker_binding().source_sha256):
        raise ValueError("steady experiment/worker source is not the installed binding")
    binding = context.consume_manifest(request, engine="solarlab")
    preparation = entry["preparation_context"]
    selected = next((item for item in preparation["devices"] if item["device_id"] == body["device_id"]), None)
    if (entry["model_registry_sha256"] != binding.inputs.registry.content_sha256
            or selected != _device_binding(body["device_id"], binding.inputs.prepared)
            or preparation["implementation"] != _implementation_binding(binding.execution.implementation)
            or preparation["current_repository_head"] != binding.execution.current_repository_head):
        raise ValueError("steady registry/device/source differs from the consumed manifest")
    inner = replace(request, request=SourceDocument(body["steady_request_id"],
        base64.b64decode(body["steady_request_base64"], validate=True)))
    output = run_steady_jv_worker(inner, context)
    return WorkerResult.from_dict({**json.loads(output.result_json),
        "experiment_id": body["experiment_id"], "experiment_version": body["version"],
        "registered_device_id": body["device_id"],
        "execution_catalog_sha256": request.registry.sha256}, artifacts=output.artifacts)


def run_transient(request: WorkerRequest, context: WorkerContext) -> WorkerResult:
    """Check the trusted transient route and retain the numerical outcome."""
    from solarlab.experiments.transient_worker import run_transient_worker, transient_worker_binding
    from solarlab_server.steady_preparation import _device_binding, _implementation_binding
    from solarlab_server.transient_preparation import TransientJobInput, _numerical_policy, _server_bindings, _server_runtime

    context.check_cancel()
    catalog = json.loads(request.registry.content)
    body = json.loads(request.request.content)
    if (set(body) != {"experiment_id", "version", "engine", "device_id", "transient_request_id", "transient_request_base64"}
            or catalog["schema"] != "solarlab.execution-catalog.v1" or body["engine"] != "solarlab"
            or catalog["job_schema_sha256"] != request.schema.sha256):
        raise ValueError("transient worker registry/schema differs from the installed route")
    entry = next((r for r in catalog["bindings"] if (r["experiment_id"], r["version"], r["engine"]) ==
                  (body["experiment_id"], body["version"], body["engine"])), None)
    if (entry is None or entry["scope"] != "classical_ion_free_1d_transient" or entry["driver"] != "transient_1d"
            or entry["worker_source_sha256"] != transient_worker_binding().source_sha256):
        raise ValueError("transient experiment/worker source is not the installed binding")
    binding = context.consume_manifest(request, engine="solarlab")
    preparation = entry["preparation_context"]
    selected = next((item for item in preparation["devices"] if item["device_id"] == body["device_id"]), None)
    if (entry["model_registry_sha256"] != binding.inputs.registry.content_sha256
            or selected != _device_binding(body["device_id"], binding.inputs.prepared)
            or preparation["implementation"] != _implementation_binding(binding.execution.implementation)
            or preparation["current_repository_head"] != binding.execution.current_repository_head
            or preparation["server_sources"] != _server_bindings()
            or preparation["server_runtime"] != _server_runtime()
            or preparation["fixed_numerical_policy"] != _numerical_policy()):
        raise ValueError("transient registry/device/source/policy differs from the consumed manifest")
    inner = replace(request, request=SourceDocument(body["transient_request_id"],
        base64.b64decode(body["transient_request_base64"], validate=True)))
    inner_body = json.loads(inner.request.content)
    declared = TransientJobInput.model_validate({"device_id": body["device_id"], **inner_body["protocol"],
        "intervals_per_layer": inner_body["intervals_per_layer"],
        "controls": {key: inner_body["transient_options"][key] for key in ("newton_budget", "max_steps", "wall_s")}})
    if (inner_body["transient_options"] != asdict(declared.options())
            or inner_body["initial_options"] != _numerical_policy()["initial_dc"]):
        raise ValueError("transient inner controls differ from the installed fixed numerical policy")
    output = run_transient_worker(inner, context)
    if preparation["server_sources"] != _server_bindings():
        raise ValueError("transient server source changed during the computation")
    return WorkerResult.from_dict({**json.loads(output.result_json),
        "experiment_id": body["experiment_id"], "experiment_version": body["version"],
        "registered_device_id": body["device_id"], "execution_catalog_sha256": request.registry.sha256,
        "server_sources": preparation["server_sources"], "server_runtime": preparation["server_runtime"]},
        artifacts=output.artifacts, state=output.state)
