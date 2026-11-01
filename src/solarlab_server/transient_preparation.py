"""Trusted, bounded ion-free transient installation on the existing job service."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib.metadata
import inspect
from pathlib import Path
import re
from typing import Annotated, Any, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, model_validator

from solarlab.config.identity import BuildSource, ExecutionEngine
from solarlab.device.resolved import PreparedDevice
from solarlab.experiments.jv.worker import validate_steady_jv_scope
from solarlab.experiments.transient_worker import TransientJob, prepare_transient_job, transient_identity_inputs
from solarlab.experiments.worker import ResourceLimits
from solarlab.materials.source import SourceDocument
from solarlab.physics.registry import ModelRegistry
from solarlab_server.steady_preparation import _device_binding, _implementation_binding

if TYPE_CHECKING:
    from solarlab.experiments.transient import TransientOptions, TransientProtocol
    from solarlab_server.job_service import ExecutionCatalog

__all__ = ["TransientJobInput", "TransientPreparationContext", "transient_execution_catalog"]

_SERVER_PATHS = ("transient_preparation.py", "steady_preparation.py", "job_service.py", "registered_worker.py",
                 "app.py", "routes/jobs.py", "routes/results.py", "events.py")


class _NewtonBudgetInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    max_iterations: int = Field(ge=1, le=50)
    max_residual_evaluations: int = Field(ge=1, le=1000)
    max_jacobian_evaluations: int = Field(ge=1, le=50)
    max_backtracks: int = Field(ge=0, le=24)


class _TimeControls(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    newton_budget: _NewtonBudgetInput
    max_steps: int = Field(ge=1, le=32)
    wall_s: float = Field(gt=0, le=30)


def _numerical_policy() -> dict[str, Any]:
    from solarlab.solvers.nonlinear.production import POISSON_CONTACT_BALANCE_V1
    from solarlab.solvers.nonlinear.steady_1d import SteadyOptions

    return {"transient": {"poisson_tolerance": 1e-11, "continuity_tolerance_A_m2": 1e-8,
        "contact_log_tolerance": 1e-11, "potential_step_V": .1, "quasi_fermi_step": 2.,
        "predictor": "previous", "convergence_policy": POISSON_CONTACT_BALANCE_V1},
        "initial_dc": asdict(SteadyOptions())}


class TransientJobInput(BaseModel):
    """Explicit driving samples and work budgets; scientific gates are server policy."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    device_id: str = Field(pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}$")
    times_s: list[Annotated[float, Field(ge=0, le=1)]] = Field(min_length=2, max_length=33)
    voltage_V: list[Annotated[float, Field(ge=-2, le=2)]] = Field(min_length=2, max_length=33)
    generation_scale: list[Annotated[float, Field(ge=0, le=10)]] = Field(min_length=2, max_length=33)
    intervals_per_layer: int = Field(ge=2, le=256)
    controls: _TimeControls

    @model_validator(mode="after")
    def check_protocol(self) -> TransientJobInput:
        self.protocol()
        if len(self.times_s) - 1 > self.controls.max_steps:
            raise ValueError("explicit samples exceed the requested max_steps budget")
        return self

    def protocol(self) -> TransientProtocol:
        import numpy as np
        from solarlab.experiments.transient import TransientProtocol
        return TransientProtocol(np.asarray(self.times_s), np.asarray(self.voltage_V), np.asarray(self.generation_scale))

    def options(self) -> TransientOptions:
        from solarlab.experiments.transient import TransientOptions
        from solarlab.solvers.nonlinear.problem import SolveBudget
        return TransientOptions(**_numerical_policy()["transient"],
            newton_budget=SolveBudget(**self.controls.newton_budget.model_dump()),
            max_steps=self.controls.max_steps, wall_s=self.controls.wall_s)


def _server_documents() -> tuple[SourceDocument, ...]:
    root = Path(__file__).resolve().parent
    return tuple(SourceDocument("solarlab_server/" + name, (root / name).read_bytes()) for name in _SERVER_PATHS)


def _server_bindings() -> list[dict[str, str]]:
    return [{"id": source.id, "sha256": source.sha256} for source in _server_documents()]


def _server_runtime() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in ("fastapi", "starlette", "uvicorn", "pydantic")}


@dataclass(frozen=True, slots=True)
class TransientPreparationContext:
    devices: tuple[tuple[str, PreparedDevice], ...]
    implementation: BuildSource
    current_repository_head: str
    models: ModelRegistry = field(init=False)
    server_sources: tuple[SourceDocument, ...] = field(init=False)

    def __post_init__(self) -> None:
        devices = tuple(tuple(item) for item in self.devices)
        if (not devices or any(len(item) != 2 or type(item[0]) is not str
                or re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}", item[0]) is None
                or type(item[1]) is not PreparedDevice for item in devices)
                or len({item[0] for item in devices}) != len(devices)):
            raise ValueError("transient installation requires unique registered IDs and actual PreparedDevice data")
        if type(self.implementation) is not BuildSource:
            raise ValueError("transient installation requires actual trusted build sources")
        if (type(self.current_repository_head) is not str
                or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", self.current_repository_head) is None):
            raise ValueError("maintenance HEAD must be supplied independently of build-source identity")
        object.__setattr__(self, "devices", tuple(sorted(devices, key=lambda item: item[0])))
        for _, prepared in self.devices:
            validate_steady_jv_scope(prepared)
        object.__setattr__(self, "models", transient_identity_inputs(self.devices[0][1], self.implementation).registry)
        object.__setattr__(self, "server_sources", _server_documents())

    def verify_sources(self) -> None:
        if self.server_sources != _server_documents():
            raise ValueError("active transient server sources changed after trusted installation")

    def device(self, id: str) -> PreparedDevice:
        for key, prepared in self.devices:
            if key == id:
                return prepared
        raise ValueError("device is not installed in this trusted transient registration")

    def export(self) -> dict[str, Any]:
        self.verify_sources()
        return {"devices": [_device_binding(id, prepared) for id, prepared in self.devices],
                "implementation": _implementation_binding(self.implementation),
                "current_repository_head": self.current_repository_head,
                "server_sources": [{"id": s.id, "sha256": s.sha256} for s in self.server_sources],
                "server_runtime": _server_runtime(), "fixed_numerical_policy": _numerical_policy()}


def prepare_transient_input(input: TransientJobInput, context: TransientPreparationContext) -> TransientJob:
    from solarlab.solvers.nonlinear.steady_1d import SteadyOptions

    context.verify_sources()
    return prepare_transient_job(context.device(input.device_id), input.protocol(),
        intervals_per_layer=input.intervals_per_layer, options=input.options(), initial_options=SteadyOptions(),
        implementation=context.implementation, current_repository_head=context.current_repository_head,
        engine=ExecutionEngine("solarlab"))


def transient_execution_catalog(context: TransientPreparationContext, *, ceiling: ResourceLimits) -> ExecutionCatalog:
    """Install this one source-checked adapter; no declaration grants broad execution."""
    from solarlab.experiments.registry import ExperimentDefinition, ExperimentRegistry
    from solarlab.experiments.transient_results import transient_result_payload
    from solarlab.experiments.transient_worker import read_transient_job
    from solarlab.physics.registry import CapabilityRule, EvidenceRef
    from solarlab_server.job_service import ExecutionCatalog, TransientRegistration, canonical
    from solarlab_server.registered_worker import run_transient

    if type(context) is not TransientPreparationContext:
        raise ValueError("an explicit trusted transient preparation context is required")
    context.verify_sources()
    documents = {source.id: source for source in context.server_sources}

    def reference(id: str, function: Any) -> EvidenceRef:
        source = SourceDocument(id, Path(inspect.getfile(function)).read_bytes())
        documents[id] = source
        return EvidenceRef(id + "#" + function.__name__, source.sha256, "source")

    schema = SourceDocument("solarlab.transient-job-input.schema.v1", canonical(TransientJobInput.model_json_schema()))
    documents[schema.id] = schema
    factory = reference("solarlab_server/transient_preparation.py", prepare_transient_input)
    result = reference("solarlab/experiments/transient_results.py", transient_result_payload)
    executor = reference("solarlab_server/registered_worker.py", run_transient)
    reader = reference("solarlab/experiments/transient_worker.py", read_transient_job)
    for source in context.implementation.files:
        if source.id in {"solarlab/device/compile_1d.py", "solarlab/equations/dc_1d.py", "solarlab/solvers/time/backward_euler.py"}:
            documents[source.id] = source
    definition = ExperimentDefinition("transient_1d", 1, "Classical ion-free 1D transient",
        EvidenceRef(schema.id, schema.sha256, "source"), result, factory, executor,
        "transient_history_arrays", reader,
        CapabilityRule("installed_classical_1d_transient", (1,), ("transient",), (factory,),
                       backends=("transient_1d",), statistics=("maxwell_boltzmann",)),
        tuple(model.ref for model in context.models.definitions))
    registry = ExperimentRegistry((definition,))
    return ExecutionCatalog(registry, context.models, tuple(documents.values()),
        (TransientRegistration("transient_1d", 1, context),), registry.content_sha256,
        context.models.content_sha256, ceiling)
