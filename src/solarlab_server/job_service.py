"""Explicit execution registrations over the existing P03/P06 contracts.

Factories and source bindings are supplied by trusted application code, never
by HTTP. Declaration compatibility alone never authorizes execution: legacy
drivers and the single classical steady adapter require separate installation.
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
from solarlab.experiments.jv.worker import SteadyJVJob, steady_jv_worker_binding
from solarlab.experiments.transient_worker import TransientJob, transient_worker_binding
from solarlab.experiments.registry import ExperimentRegistry
from solarlab.experiments.worker import ManifestBinding, ResourceLimits, RunSpec, WorkerBinding, WorkerRequest
from solarlab.materials.source import SourceDocument
from solarlab.physics.registry import ModelRegistry, metadata_digest
from solarlab_server.legacy_engine import LegacyJob, legacy_worker_binding
from solarlab_server.steady_preparation import SteadyJobInput, SteadyPreparationContext, prepare_steady_input
from solarlab_server.transient_preparation import TransientJobInput, TransientPreparationContext, prepare_transient_input


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
    engine: Literal["legacy"] = field(default="legacy", init=False)

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
class SteadyRegistration:
    """Only the installed steady adapter; no HTTP-selectable Python factory."""

    experiment_id: str
    version: int
    context: SteadyPreparationContext
    engine: Literal["solarlab"] = field(default="solarlab", init=False)
    input_type: type[SteadyJobInput] = field(default=SteadyJobInput, init=False)
    legacy_kinds: tuple[str, ...] = field(default=(), init=False)
    synchronous_paths: tuple[str, ...] = field(default=(), init=False)

    def __post_init__(self) -> None:
        if type(self.context) is not SteadyPreparationContext:
            raise ValueError("steady registration requires the actual trusted preparation context")


@dataclass(frozen=True, slots=True)
class TransientRegistration:
    """One installed transient adapter, with no caller-selectable code or state."""

    experiment_id: str
    version: int
    context: TransientPreparationContext
    engine: Literal["solarlab"] = field(default="solarlab", init=False)
    input_type: type[TransientJobInput] = field(default=TransientJobInput, init=False)
    legacy_kinds: tuple[str, ...] = field(default=(), init=False)
    synchronous_paths: tuple[str, ...] = field(default=(), init=False)

    def __post_init__(self) -> None:
        if type(self.context) is not TransientPreparationContext:
            raise ValueError("transient registration requires the actual trusted preparation context")


@dataclass(frozen=True, slots=True)
class ExecutionCatalog:
    """A source-checked table over the existing experiment/model declarations."""

    registry: ExperimentRegistry
    models: ModelRegistry
    sources: tuple[SourceDocument, ...]
    registrations: tuple[LegacyRegistration | SteadyRegistration | TransientRegistration, ...]
    expected_registry_sha256: str
    expected_model_sha256: str
    ceiling: ResourceLimits
    schema: SourceDocument = field(init=False)
    document: SourceDocument = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "registrations", tuple(self.registrations))
        if (type(self.ceiling) is not ResourceLimits or not self.registrations
                or any(type(r) not in {LegacyRegistration, SteadyRegistration, TransientRegistration} for r in self.registrations)):
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
            "bindings": [self._entry(r) for r in self.registrations]})))

    def _models(self, registration: LegacyRegistration | SteadyRegistration | TransientRegistration) -> ModelRegistry:
        return self.models if isinstance(registration, LegacyRegistration) else registration.context.models

    def _entry(self, registration: LegacyRegistration | SteadyRegistration | TransientRegistration) -> dict[str, Any]:
        entry = {"experiment_id": registration.experiment_id, "version": registration.version,
            "engine": registration.engine, "input_schema": registration.input_type.model_json_schema(),
            "model_registry_sha256": self._models(registration).content_sha256,
            "legacy_kinds": list(registration.legacy_kinds),
            "synchronous_paths": list(registration.synchronous_paths), "scientific_qualification": False}
        if isinstance(registration, LegacyRegistration):
            entry.update(driver=registration.driver, scope="legacy_compatibility")
        elif isinstance(registration, SteadyRegistration):
            entry.update(driver="steady_jv", scope="static_classical_ion_free_1d_dc",
                preparation_context=registration.context.export(),
                worker_source_sha256=steady_jv_worker_binding().source_sha256)
        else:
            entry.update(driver="transient_1d", scope="classical_ion_free_1d_transient",
                preparation_context=registration.context.export(),
                worker_source_sha256=transient_worker_binding().source_sha256,
                result_schema="solarlab.transient.worker-result.v1")
        return entry

    def verify(self) -> None:
        if self.registry.content_sha256 != self.expected_registry_sha256:
            raise ValueError("experiment registry changed from the expected frozen identity")
        sources = tuple((s.id, s.content) for s in self.sources)
        self.models.verify_sources(sources, expected_sha256=self.expected_model_sha256)
        by_key = {(r.experiment_id, r.version): r for r in self.registrations}
        # A mixed catalog must not relabel the steady worker's registry as the
        # legacy registry (or vice versa). Uninstalled declarations still get
        # the original startup verification against the catalog's base models.
        for definition in self.registry.definitions:
            registration = by_key.get((definition.id, definition.version))
            models = self.models if registration is None else self._models(registration)
            declared = ExperimentRegistry((definition,))
            declared.verify_startup(models, sources, expected_sha256=declared.content_sha256,
                                    expected_model_sha256=models.content_sha256)
        for registration in self.registrations:
            definition = self.registry.get(registration.experiment_id, registration.version)
            factory: Callable[..., Any]
            if isinstance(registration, LegacyRegistration):
                factory = registration.prepare
            elif isinstance(registration, SteadyRegistration):
                factory = prepare_steady_input
            else:
                registration.context.verify_sources()
                factory = prepare_transient_input
            source = Path(inspect.getfile(factory)).read_bytes()
            if (definition.protocol_builder.sha256 != hashlib.sha256(source).hexdigest()
                    or definition.protocol_builder.id.partition("#")[2] != factory.__name__
                    or definition.parameter_schema.sha256 != metadata_digest(registration.input_type.model_json_schema())):
                raise ValueError("loaded preparation/schema differs from the frozen registry")

    def worker_binding(self, engine: Literal["legacy", "solarlab"] = "legacy", *,
                       experiment_id: str | None = None) -> WorkerBinding:
        from solarlab_server.registered_worker import run_legacy, run_steady, run_transient
        registrations = [r for r in self.registrations if r.engine == engine
                         and (experiment_id is None or r.experiment_id == experiment_id)]
        if not registrations:
            raise ValueError("engine has no explicit installed execution registration")
        kinds = {"registered_legacy" if isinstance(r, LegacyRegistration) else
                 "registered_steady" if isinstance(r, SteadyRegistration) else "registered_transient" for r in registrations}
        if len(kinds) != 1:
            raise ValueError("multiple workers are installed for this engine; select an explicit experiment_id")
        id, = kinds
        function = {"registered_legacy": run_legacy, "registered_steady": run_steady, "registered_transient": run_transient}[id]
        return WorkerBinding.from_function(id, function,
            registry_sha256=self.document.sha256, schema_sha256=self.schema.sha256)

    def worker_bindings(self) -> tuple[WorkerBinding, ...]:
        """All explicitly installed worker kinds for one shared supervisor."""
        values = [self.worker_binding(r.engine, experiment_id=r.experiment_id) for r in self.registrations]
        return tuple({binding.id: binding for binding in values}.values())

    def prepare(self, submission: JobSubmission) -> tuple[RunSpec, ManifestBinding]:
        self.verify()
        registration = next((r for r in self.registrations if (r.experiment_id, r.version) ==
                             (submission.experiment_id, submission.version)), None)
        if registration is None:
            raise ValueError("experiment has no explicit installed execution registration")
        if submission.engine != registration.engine:
            raise ValueError("engine differs from the installed registration; fallback is forbidden")
        limits = ResourceLimits(**submission.limits.model_dump())
        for key, maximum in asdict(self.ceiling).items():
            value = getattr(limits, key)
            if maximum is not None and (value is None or value > maximum):
                raise ValueError("requested budget exceeds the service ceiling: " + key)
        job: LegacyJob | SteadyJVJob | TransientJob
        if isinstance(registration, LegacyRegistration):
            model = registration.input_type.model_validate(submission.input)
            job = registration.prepare(model)
            if type(job) is not LegacyJob or job.execution.engine.name != "legacy":
                raise ValueError("registered factory did not prepare the explicit legacy facade")
            behavior = job.execution.engine.behavior
            if behavior is None or getattr(behavior, "driver", None) != registration.driver:
                raise ValueError("registered driver differs from the complete execution manifest")
            worker_id, request_kind = "registered_legacy", "legacy"
            extra: dict[str, Any] = {}
        elif isinstance(registration, SteadyRegistration):
            if limits.accepted_steps is not None or limits.calls is not None:
                raise ValueError("steady worker counters are unobserved; accepted_steps/calls budgets are unsupported")
            steady_input = SteadyJobInput.model_validate(submission.input)
            job = prepare_steady_input(steady_input, registration.context)
            if type(job) is not SteadyJVJob or job.execution.engine.name != "solarlab":
                raise ValueError("registered preparation did not return the explicit steady worker")
            worker_id, request_kind = "registered_steady", "steady"
            extra = {"device_id": steady_input.device_id}
        else:
            if limits.accepted_steps is not None or limits.calls is not None:
                raise ValueError("transient worker counters are unobserved; accepted_steps/calls budgets are unsupported")
            transient_input = TransientJobInput.model_validate(submission.input)
            job = prepare_transient_input(transient_input, registration.context)
            if type(job) is not TransientJob or job.execution.engine.name != "solarlab":
                raise ValueError("registered preparation did not return the explicit transient worker")
            worker_id, request_kind = "registered_transient", "transient"
            extra = {"device_id": transient_input.device_id}
        models = self._models(registration)
        if job.execution.inputs.registry.content_sha256 != models.content_sha256:
            raise ValueError("prepared model registry differs from the server registry")
        preflight = self.registry.preflight(submission.experiment_id, submission.version,
            job.execution.inputs.selection, models)
        if not preflight.declared_compatible:
            raise ValueError("experiment capability refused: " + "; ".join(preflight.declared_problems))
        # The explicit installation admits this adapter only; declaration
        # can_execute and scientific qualification are not changed.
        payload = SourceDocument(f"solarlab.registered-{request_kind}-request.v1", canonical({
            "experiment_id": submission.experiment_id, "version": submission.version, "engine": submission.engine,
            f"{request_kind}_request_id": job.request.request.id,
            f"{request_kind}_request_base64": base64.b64encode(job.request.request.content).decode(), **extra}))
        request = WorkerRequest(payload, self.document, self.schema, job.request.execution_identity,
                                job.request.H_input, job.request.H_physics)
        return RunSpec.from_binding(worker_id, request, limits, job.execution), job.execution


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
