"""Module-level registry check followed by the existing sole legacy facade."""
from dataclasses import replace
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
    if entry is None or catalog["model_registry_sha256"] != binding.inputs.registry.content_sha256:
        raise ValueError("worker experiment/model registry mismatch")
    inner = replace(request, request=SourceDocument(body["legacy_request_id"],
        base64.b64decode(body["legacy_request_base64"], validate=True)))
    if json.loads(inner.request.content)["driver"] != entry["driver"]:
        raise ValueError("worker driver differs from the selected experiment")
    output = _run_legacy(inner, context)
    return WorkerResult.from_dict({**json.loads(output.result_json),
        "experiment_id": body["experiment_id"], "experiment_version": body["version"],
        "execution_catalog_sha256": request.registry.sha256}, artifacts=output.artifacts)
