"""Explicit legacy execution registrations over the existing P03/P06 contracts.

Factories and source bindings are supplied by trusted application code, never
by HTTP. Core declaration compatibility does not authorize the new engine.
This first route adapter invokes only the accepted legacy facade.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
import base64
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from solarlab.experiments.execution import JobHandle, ProcessSupervisor
from solarlab.experiments.registry import ExperimentRegistry
from solarlab.experiments.worker import ManifestBinding, ResourceLimits, RunSpec, WorkerBinding, WorkerRequest
from solarlab.materials.source import SourceDocument
from solarlab.physics.registry import ModelRegistry, metadata_digest
from solarlab_server.legacy_engine import LegacyJob, legacy_worker_binding


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class BudgetInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    wall_seconds: float = Field(gt=0)
    rss_bytes: int = Field(gt=0)
    output_bytes: int = Field(ge=16384)
    accepted_steps: int | None = Field(ge=0, lt=2**63)
    calls: int | None = Field(ge=0, lt=2**63)


class JobSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["solarlab.job-submission.v1"]
    run_id: str = Field(pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}$")
    experiment_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    engine: Literal["legacy", "solarlab"]
    input: dict[str, Any]
    limits: BudgetInput


@dataclass(frozen=True, slots=True)
class LegacyRegistration:
    experiment_id: str
    version: int
    driver: str
    input_type: type[BaseModel]
    prepare: Callable[[Any], LegacyJob]
    legacy_kinds: tuple[str, ...] = ()
    synchronous_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.driver not in {"scaps_transient", "scaps_steady_state", "ion_hysteresis"}:
            raise ValueError("only explicitly supported legacy facade drivers may be registered")
        if (not inspect.isfunction(self.prepare) or self.prepare.__closure__ is not None
                or self.prepare.__qualname__ != self.prepare.__name__ or self.prepare.__module__ == "__main__"):
            raise ValueError("legacy preparation must be an explicit module-level source-bound factory")
        if not issubclass(self.input_type, BaseModel) or self.input_type.model_config.get("extra") != "forbid":
            raise ValueError("registered input models must refuse unknown fields")
        object.__setattr__(self, "legacy_kinds", tuple(self.legacy_kinds))
        object.__setattr__(self, "synchronous_paths", tuple(self.synchronous_paths))
        if any(not p.startswith("/api/") or "{" in p for p in self.synchronous_paths):
            raise ValueError("synchronous aliases require explicit fixed /api paths")


@dataclass(frozen=True, slots=True)
class ExecutionCatalog:
    """A source-checked table of explicit compatibility routes, not a new registry."""

    registry: ExperimentRegistry
    models: ModelRegistry
    sources: tuple[SourceDocument, ...]
    registrations: tuple[LegacyRegistration, ...]
    expected_registry_sha256: str
    expected_model_sha256: str
    ceiling: ResourceLimits
    schema: SourceDocument = field(init=False)
    document: SourceDocument = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "registrations", tuple(self.registrations))
        if type(self.ceiling) is not ResourceLimits or not self.registrations:
            raise ValueError("explicit route registrations and a resource ceiling are required")
        keys = [(r.experiment_id, r.version) for r in self.registrations]
        aliases = [a for r in self.registrations for a in (*r.legacy_kinds, *r.synchronous_paths)]
        if len(set(keys)) != len(keys) or len(set(aliases)) != len(aliases):
            raise ValueError("ambiguous execution registration or legacy alias")
        self.verify()
        object.__setattr__(self, "schema", SourceDocument("solarlab.job-submission.schema.v1", canonical(JobSubmission.model_json_schema())))
        object.__setattr__(self, "document", SourceDocument("solarlab.execution-catalog.v1", canonical({
            "schema": "solarlab.execution-catalog.v1", "registry": self.registry.export(),
            "registry_sha256": self.registry.content_sha256, "model_registry_sha256": self.models.content_sha256,
            "job_schema_sha256": self.schema.sha256, "resource_ceiling": asdict(self.ceiling),
            "legacy_facade_sha256": legacy_worker_binding().source_sha256,
            "bindings": [{"experiment_id": r.experiment_id, "version": r.version, "engine": "legacy",
                "driver": r.driver, "input_schema": r.input_type.model_json_schema(),
                "legacy_kinds": list(r.legacy_kinds), "synchronous_paths": list(r.synchronous_paths),
                "scope": "legacy_compatibility", "scientific_qualification": False} for r in self.registrations]})))

    def verify(self) -> None:
        self.registry.verify_startup(self.models, tuple((s.id, s.content) for s in self.sources),
            expected_sha256=self.expected_registry_sha256, expected_model_sha256=self.expected_model_sha256)
        for registration in self.registrations:
            definition = self.registry.get(registration.experiment_id, registration.version)
            source = Path(inspect.getfile(registration.prepare)).read_bytes()
            if (definition.protocol_builder.sha256 != hashlib.sha256(source).hexdigest()
                    or definition.protocol_builder.id.partition("#")[2] != registration.prepare.__name__
                    or definition.parameter_schema.sha256 != metadata_digest(registration.input_type.model_json_schema())):
                raise ValueError("loaded preparation/schema differs from the frozen registry")

    def worker_binding(self) -> WorkerBinding:
        from solarlab_server.registered_worker import run_legacy
        return WorkerBinding.from_function("registered_legacy", run_legacy,
            registry_sha256=self.document.sha256, schema_sha256=self.schema.sha256)

    def prepare(self, submission: JobSubmission) -> tuple[RunSpec, ManifestBinding]:
        self.verify()
        if submission.engine != "legacy":
            raise ValueError("new-engine execution is not admitted; legacy fallback is forbidden")
        registration = next((r for r in self.registrations if (r.experiment_id, r.version) ==
                             (submission.experiment_id, submission.version)), None)
        if registration is None:
            raise ValueError("experiment has no migrated legacy execution registration")
        limits = ResourceLimits(**submission.limits.model_dump())
        for key, maximum in asdict(self.ceiling).items():
            value = getattr(limits, key)
            if maximum is not None and (value is None or value > maximum):
                raise ValueError("requested budget exceeds the service ceiling: " + key)
        model = registration.input_type.model_validate(submission.input)
        job = registration.prepare(model)
        if type(job) is not LegacyJob or job.execution.engine.name != "legacy":
            raise ValueError("registered factory did not prepare the explicit legacy facade")
        behavior = job.execution.engine.behavior
        if behavior is None or getattr(behavior, "driver", None) != registration.driver:
            raise ValueError("registered driver differs from the complete execution manifest")
        if job.execution.inputs.registry.content_sha256 != self.expected_model_sha256:
            raise ValueError("prepared model registry differs from the server registry")
        preflight = self.registry.preflight(submission.experiment_id, submission.version,
            job.execution.inputs.selection, self.models)
        if not preflight.declared_compatible:
            raise ValueError("experiment capability refused: " + "; ".join(preflight.declared_problems))
        # This is an explicit legacy compatibility route. The core preflight's
        # can_execute=False and unavailable new engine remain unchanged.
        payload = SourceDocument("solarlab.registered-legacy-request.v1", canonical({
            "experiment_id": submission.experiment_id, "version": submission.version, "engine": "legacy",
            "legacy_request_id": job.request.request.id,
            "legacy_request_base64": base64.b64encode(job.request.request.content).decode()}))
        request = WorkerRequest(payload, self.document, self.schema, job.request.execution_identity,
                                job.request.H_input, job.request.H_physics)
        return RunSpec.from_binding("registered_legacy", request, limits, job.execution), job.execution


@dataclass(frozen=True, slots=True)
class JobService:
    catalog: ExecutionCatalog
    supervisor: ProcessSupervisor
    allow_legacy_routes: bool = False
    synchronous_wait_seconds: float = 5.0

    def __post_init__(self) -> None:
        if type(self.catalog) is not ExecutionCatalog or type(self.supervisor) is not ProcessSupervisor:
            raise ValueError("an explicit catalog and shared process supervisor are required")
        if type(self.allow_legacy_routes) is not bool:
            raise ValueError("legacy route activation must be explicit")
        if (type(self.synchronous_wait_seconds) not in {int, float}
                or not math.isfinite(self.synchronous_wait_seconds) or self.synchronous_wait_seconds <= 0):
            raise ValueError("synchronous compatibility wait must be positive and bounded")

    def submit(self, submission: JobSubmission) -> JobHandle:
        spec, execution = self.catalog.prepare(submission)
        return self.supervisor.submit(submission.run_id, spec.worker_id, spec.request, spec.limits, execution=execution)

def legacy_job_id(handle: JobHandle) -> str:
    return base64.urlsafe_b64encode(canonical([handle.run_id, handle.attempt_id])).decode().rstrip("=")


def legacy_job_target(value: str) -> tuple[str, str]:
    if len(value) > 512:
        raise ValueError("invalid legacy job identifier")
    try:
        raw = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
        pair = json.loads(raw)
        if type(pair) is not list or len(pair) != 2 or any(type(x) is not str or not x for x in pair):
            raise ValueError("invalid legacy job target")
        if base64.urlsafe_b64encode(canonical(pair)).decode().rstrip("=") != value:
            raise ValueError("noncanonical legacy job identifier")
        return pair[0], pair[1]
    except (ValueError, UnicodeError) as error:
        raise ValueError("invalid legacy job identifier") from error
