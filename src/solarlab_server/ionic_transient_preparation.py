"""Explicit dark, positive-diffusivity ionic jobs on the existing HTTP owner."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import inspect
import json
import math
from pathlib import Path
import re
from typing import Annotated, Any, TYPE_CHECKING, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from solarlab.config.identity import BuildSource, ExecutionEngine
from solarlab.device.resolved import PreparedDevice
from solarlab.experiments.ionic_transient_worker import (
    IonicTransientJob, ionic_transient_identity_inputs, prepare_ionic_transient_job, read_ionic_transient_job,
)
from solarlab.experiments.worker import ResourceLimits, WorkerRequest
from solarlab.materials.source import SourceDocument
from solarlab.physics.registry import ModelRegistry, metadata_digest
from solarlab_server.steady_preparation import _device_binding, _implementation_binding
from solarlab_server.transient_preparation import _server_runtime

if TYPE_CHECKING:
    from solarlab.experiments.ionic_transient import IonicTransientOptions
    from solarlab.experiments.transient import TransientProtocol
    from solarlab.io.run_store import RunStore
    from solarlab.solvers.nonlinear.ionic_1d import IonicSteadyOptions
    from solarlab_server.job_service import ExecutionCatalog

__all__ = ["IonicTransientJobInput", "IonicTransientPreparationContext", "ionic_transient_execution_catalog",
           "read_registered_ionic_transient_job"]
EXPERIMENT_ID = "ionic_transient_1d"
SCOPE = "classical_1d_dark_positive_diffusivity_ionic_transient"
_SERVER_PATHS = ("ionic_transient_preparation.py", "transient_preparation.py", "steady_preparation.py",
                 "job_service.py", "registered_worker.py", "app.py", "routes/jobs.py", "routes/results.py", "events.py")


class _NewtonBudgetInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    max_iterations: int = Field(ge=1, le=60)
    max_residual_evaluations: int = Field(ge=1, le=600)
    max_jacobian_evaluations: int = Field(ge=1, le=60)
    max_backtracks: int = Field(ge=0, le=24)


class _InitialControls(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    budget: _NewtonBudgetInput
    inventory_row_node: int = Field(ge=0, le=64)
    wall_s: float = Field(gt=0, le=30)


class _TimeControls(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    newton_budget: _NewtonBudgetInput
    max_steps: int = Field(ge=1, le=16)
    wall_s: float = Field(gt=0, le=30)

    @model_validator(mode="after")
    def bounded_time_budget(self) -> _TimeControls:
        b = self.newton_budget
        if b.max_iterations > 50 or b.max_residual_evaluations > 500 or b.max_jacobian_evaluations > 50:
            raise ValueError("time Newton work exceeds the installed 50/500/50 budget")
        return self


class _Controls(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    initial_steady: _InitialControls
    transient: _TimeControls


def _numerical_policy() -> dict[str, Any]:
    from solarlab.solvers.nonlinear.production import POISSON_CONTACT_BALANCE_V1
    from solarlab.solvers.nonlinear.steady_1d import SteadyOptions

    return {"initial_steady": {"electronic": asdict(SteadyOptions()), "inventory_relative_tolerance": 1e-12,
        "ion_balance_tolerance_m2_s": 1e6, "ion_flux_tolerance_m2_s": 1e6, "ion_logit_step": 1.},
        "transient": {"electronic": {"poisson_tolerance": 1e-11, "continuity_tolerance_A_m2": 1e-8,
            "contact_log_tolerance": 1e-11, "potential_step_V": .1, "quasi_fermi_step": 2.,
            "predictor": "previous", "convergence_policy": POISSON_CONTACT_BALANCE_V1},
            "ion_balance_tolerance_m2_s": 1e6, "ion_logit_step": 1.}}


class IonicTransientJobInput(BaseModel):
    """All samples/grid/work controls are explicit; scientific gates stay fixed."""
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    device_id: str = Field(pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}$")
    times_s: list[Annotated[float, Field(ge=0, le=1)]] = Field(min_length=2, max_length=17)
    voltage_V: list[Annotated[float, Field(ge=-.02, le=.02)]] = Field(min_length=2, max_length=17)
    generation_scale: list[Annotated[float, Field(ge=0, le=0)]] = Field(min_length=2, max_length=17)
    intervals_per_layer: list[Annotated[int, Field(ge=2, le=64)]] = Field(min_length=1, max_length=8)
    controls: _Controls

    @model_validator(mode="after")
    def check_protocol(self) -> IonicTransientJobInput:
        self.protocol()
        if len(self.times_s) - 1 > self.controls.transient.max_steps:
            raise ValueError("explicit samples exceed the requested step budget")
        return self

    def protocol(self) -> TransientProtocol:
        import numpy as np
        from solarlab.experiments.transient import TransientProtocol
        return TransientProtocol(np.asarray(self.times_s), np.asarray(self.voltage_V), np.asarray(self.generation_scale))

    def initial_options(self) -> IonicSteadyOptions:
        from solarlab.solvers.nonlinear.ionic_1d import IonicSteadyOptions
        from solarlab.solvers.nonlinear.problem import SolveBudget
        from solarlab.solvers.nonlinear.steady_1d import SteadyOptions
        policy = _numerical_policy()["initial_steady"]
        controls = self.controls.initial_steady
        return IonicSteadyOptions(**{**policy, "electronic": SteadyOptions(**policy["electronic"]),
            "budget": SolveBudget(**controls.budget.model_dump()), "inventory_row_node": controls.inventory_row_node,
            "wall_s": controls.wall_s})

    def options(self) -> IonicTransientOptions:
        from solarlab.experiments.ionic_transient import IonicTransientOptions
        from solarlab.experiments.transient import TransientOptions
        from solarlab.solvers.nonlinear.problem import SolveBudget
        policy = _numerical_policy()["transient"]
        controls = self.controls.transient
        return IonicTransientOptions(**{**policy, "electronic": TransientOptions(**policy["electronic"],
            newton_budget=SolveBudget(**controls.newton_budget.model_dump()), max_steps=controls.max_steps, wall_s=controls.wall_s)})


def _server_documents() -> tuple[SourceDocument, ...]:
    root = Path(__file__).resolve().parent
    return tuple(SourceDocument("solarlab_server/" + name, (root / name).read_bytes()) for name in _SERVER_PATHS)


def _server_bindings() -> list[dict[str, str]]:
    return [{"id": s.id, "sha256": s.sha256} for s in _server_documents()]


@dataclass(frozen=True, slots=True)
class IonicTransientPreparationContext:
    devices: tuple[tuple[str, PreparedDevice], ...]
    ion_domain_layer: str
    implementation: BuildSource
    current_repository_head: str
    models: ModelRegistry = field(init=False)
    server_sources: tuple[SourceDocument, ...] = field(init=False)

    def __post_init__(self) -> None:
        devices = tuple(tuple(item) for item in self.devices)
        if (not devices or any(len(item) != 2 or type(item[0]) is not str
                or re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}", item[0]) is None
                or type(item[1]) is not PreparedDevice for item in devices)
                or len({item[0] for item in devices}) != len(devices)
                or len({cast(PreparedDevice, item[1]).content_sha256 for item in devices}) != len(devices)):
            raise ValueError("ionic installation requires unique IDs and unique actual prepared-device identities")
        if (type(self.implementation) is not BuildSource or type(self.current_repository_head) is not str
                or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", self.current_repository_head) is None):
            raise ValueError("actual build sources and independently supplied maintenance HEAD required")
        object.__setattr__(self, "devices", tuple(sorted(devices, key=lambda item: item[0])))
        for _, prepared in self.devices:
            domain = [layer for layer in prepared.layers if layer.id == self.ion_domain_layer and layer.role != "substrate"]
            if len(domain) != 1:
                raise ValueError("one trusted electrical ion domain is required")
            parameters: dict[str, Any] = dict(domain[0].parameters)
            for suffix in ("", "_neg"):
                reference, diffusion = float(parameters["P0" + suffix]), float(parameters["D_ion" + suffix])
                if (not math.isfinite(reference) or not math.isfinite(diffusion)
                        or (suffix == "" or reference != 0 or diffusion != 0) and (reference <= 0 or diffusion <= 0)):
                    raise ValueError("this ionic HTTP installation requires positive connected diffusivities; D0 is unsupported")
            for layer in prepared.layers:
                if layer.id != self.ion_domain_layer and any(dict(layer.parameters)[name] != 0
                        for name in ("P0", "P0_neg", "D_ion", "D_ion_neg")):
                    raise ValueError("multiple or outside-domain ionic populations are unsupported")
        object.__setattr__(self, "models", ionic_transient_identity_inputs(self.devices[0][1], self.implementation).registry)
        object.__setattr__(self, "server_sources", _server_documents())

    def verify_sources(self) -> None:
        if self.server_sources != _server_documents():
            raise ValueError("active ionic server sources changed after installation")

    def device(self, id: str) -> PreparedDevice:
        for key, prepared in self.devices:
            if key == id:
                return prepared
        raise ValueError("device is not installed in this trusted ionic registration")

    def export(self) -> dict[str, Any]:
        self.verify_sources()
        return {"devices": [_device_binding(id, p) for id, p in self.devices],
            "ion_domain_layer": self.ion_domain_layer, "ion_domain_endpoints": ["blocking", "blocking"],
            "implementation": _implementation_binding(self.implementation), "current_repository_head": self.current_repository_head,
            "server_sources": [{"id": s.id, "sha256": s.sha256} for s in self.server_sources],
            "server_runtime": _server_runtime(), "fixed_numerical_policy": _numerical_policy(),
            "request_transport": "unchanged_ionic_worker_request_with_unique_registered_prepared_identity"}


def prepare_ionic_transient_input(input: IonicTransientJobInput, context: IonicTransientPreparationContext) -> IonicTransientJob:
    context.verify_sources()
    return prepare_ionic_transient_job(context.device(input.device_id), input.protocol(),
        intervals_per_layer=tuple(input.intervals_per_layer), ion_domain_layer=context.ion_domain_layer,
        ion_domain_endpoints=("blocking", "blocking"), options=input.options(), initial_options=input.initial_options(),
        implementation=context.implementation, current_repository_head=context.current_repository_head, engine=ExecutionEngine("solarlab"))


def ionic_transient_execution_catalog(context: IonicTransientPreparationContext, *, ceiling: ResourceLimits) -> ExecutionCatalog:
    from solarlab.experiments.ionic_transient_results import ionic_transient_payload
    from solarlab.experiments.registry import ExperimentDefinition, ExperimentRegistry
    from solarlab.physics.registry import CapabilityRule, EvidenceRef
    from solarlab_server.job_service import ExecutionCatalog, IonicTransientRegistration, canonical
    from solarlab_server.registered_worker import run_ionic_transient
    if type(context) is not IonicTransientPreparationContext or ceiling.wall_seconds > 30:
        raise ValueError("explicit ionic installation and a <=30s job ceiling required")
    context.verify_sources()
    documents = {s.id: s for s in context.server_sources}

    def reference(id: str, function: Any) -> EvidenceRef:
        source = SourceDocument(id, Path(inspect.getfile(function)).read_bytes())
        documents[id] = source
        return EvidenceRef(id + "#" + function.__name__, source.sha256, "source")

    schema = SourceDocument("solarlab.ionic-transient-job-input.schema.v1", canonical(IonicTransientJobInput.model_json_schema()))
    documents[schema.id] = schema
    factory = reference("solarlab_server/ionic_transient_preparation.py", prepare_ionic_transient_input)
    result = reference("solarlab/experiments/ionic_transient_results.py", ionic_transient_payload)
    executor = reference("solarlab_server/registered_worker.py", run_ionic_transient)
    reader = reference("solarlab_server/ionic_transient_preparation.py", read_registered_ionic_transient_job)
    for source in context.implementation.files:
        if source.id in {"solarlab/device/compile_ionic_1d.py", "solarlab/equations/ionic_1d.py",
                         "solarlab/solvers/time/ionic_backward_euler.py", "solarlab/experiments/ionic_transient.py"}:
            documents[source.id] = source
    definition = ExperimentDefinition(EXPERIMENT_ID, 1, "Dark positive-D blocking 1D ionic transient",
        EvidenceRef(schema.id, schema.sha256, "source"), result, factory, executor, "ionic_transient_history_arrays", reader,
        CapabilityRule("installed_dark_positive_d_ionic_transient", (1,), ("transient",), (factory,),
            backends=(EXPERIMENT_ID,), statistics=("maxwell_boltzmann",)), tuple(m.ref for m in context.models.definitions))
    registry = ExperimentRegistry((definition,))
    return ExecutionCatalog(registry, context.models, tuple(documents.values()),
        (IonicTransientRegistration(EXPERIMENT_ID, 1, context),), registry.content_sha256, context.models.content_sha256, ceiling)


def read_registered_ionic_transient_job(store: RunStore, run_id: str, *, attempt_id: str | None = None) -> dict[str, Any]:
    """Verify saved HTTP bindings as well as the unchanged solver-free reader."""
    from solarlab_server.job_service import canonical
    loaded = read_ionic_transient_job(store, run_id, attempt_id=attempt_id)
    transport = WorkerRequest.from_bytes(canonical(store.get_run(run_id)["request"]["input_metadata"]["transport"]))
    catalog, result = json.loads(transport.registry.content), loaded["worker_result"]
    entries = [r for r in catalog["bindings"] if r["driver"] == EXPERIMENT_ID]
    if len(entries) != 1:
        raise ValueError("saved ionic catalog must retain its unique installed route")
    entry = entries[0]
    preparation = entry["preparation_context"]
    selected = [d for d in preparation["devices"] if d["prepared_sha256"] == loaded["document"]["prepared_sha256"]]
    if (catalog["schema"] != "solarlab.execution-catalog.v1" or catalog["job_schema_sha256"] != transport.schema.sha256
            or result["execution_catalog_sha256"] != transport.registry.sha256
            or entry["experiment_id"] != EXPERIMENT_ID or entry["version"] != 1 or entry["engine"] != "solarlab"
            or entry["scope"] != SCOPE or len(selected) != 1 or result["registered_device_id"] != selected[0]["device_id"]
            or result["experiment_id"] != EXPERIMENT_ID or result["experiment_version"] != 1
            or result["registered_model_registry_sha256"] != entry["model_registry_sha256"]
            or result["server_sources"] != preparation["server_sources"] or result["server_runtime"] != preparation["server_runtime"]
            or result["registered_numerical_policy"] != preparation["fixed_numerical_policy"]
            or result["registered_input_schema_sha256"] != metadata_digest(entry["input_schema"])):
        raise ValueError("saved ionic HTTP catalog/schema/device/source binding differs from the result")
    return loaded
