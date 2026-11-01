"""Source-bound short legacy input registration for real HTTP/worker tests."""
from typing import Literal
import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field

from solarlab.experiments.registry import ExperimentDefinition, ExperimentRegistry
from solarlab.experiments.worker import ResourceLimits
from solarlab.materials.source import SourceDocument
from solarlab.physics.registry import CapabilityRule, EvidenceRef
from solarlab_server.job_service import ExecutionCatalog, LegacyRegistration
from legacy_fixtures import LEGACY, ROOT, make_job


class Parameters(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    N_grid: int = Field(default=8, ge=4, le=8)
    n_points: int = Field(default=3, ge=1, le=3)
    V_max: float = Field(default=.02, ge=0., le=.03)
    v_rate: float = Field(default=1000., gt=0.)


class LegacyInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    device: None
    config_path: Literal["fixture:scaps"]
    params: Parameters


def prepare_jv(input: LegacyInput):
    # The input model permits only this explicit source and bounded test range.
    return make_job(arguments=input.params.model_dump())[0]


def catalog(*, denied=False, wall_seconds=30., ceiling=None):
    job = make_job()[0]
    documents = {}
    def reference(path, symbol=""):
        documents[path] = (ROOT / path).read_bytes()
        return EvidenceRef(path + ("#" + symbol if symbol else ""), hashlib.sha256(documents[path]).hexdigest(), "source")
    factory = reference("tests/runtime/job_route_fixtures.py", "prepare_jv")
    schema = json.dumps(LegacyInput.model_json_schema(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    documents["schema:legacy-http-fixture"] = schema
    schema_ref = EvidenceRef("schema:legacy-http-fixture", hashlib.sha256(schema).hexdigest(), "source")
    definition = ExperimentDefinition("jv", 1, "Short source-bound legacy J-V test", schema_ref,
        reference("src/solarlab/experiments/worker.py", "WorkerResult"), factory,
        reference("src/solarlab_server/legacy_engine/__init__.py", "_run_legacy"), "jv",
        reference("web/src/workstation/panes/main-plot-pane.ts", "renderJV"),
        CapabilityRule("legacy-http-fixture", (2,) if denied else (1,), ("transient",), (factory,), backends=("metadata_only",)))
    documents["physics/temperature.py"] = (LEGACY / "physics/temperature.py").read_bytes()
    registry = ExperimentRegistry((definition,))
    models = job.execution.inputs.registry
    registration = LegacyRegistration("jv", 1, "scaps_transient", LegacyInput, prepare_jv,
                                       legacy_kinds=("jv",), synchronous_paths=("/api/jv",))
    cap = ceiling or ResourceLimits(wall_seconds, 1024**3, 64 * 1024**2, None, None)
    return ExecutionCatalog(registry, models, tuple(SourceDocument(k, v) for k, v in documents.items()),
        (registration,), expected_registry_sha256=registry.content_sha256, expected_model_sha256=models.content_sha256, ceiling=cap), job


def request(run_id="http-case", **parameters):
    return {"schema_version": "solarlab.job-submission.v1", "run_id": run_id, "experiment_id": "jv", "version": 1,
        "engine": "legacy", "input": {"device": None, "config_path": "fixture:scaps", "params": parameters},
        "limits": {"wall_seconds": 30., "rss_bytes": 1024**3, "output_bytes": 64 * 1024**2,
                   "accepted_steps": None, "calls": None}}
