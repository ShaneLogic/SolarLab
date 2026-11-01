"""Trusted installation data for the single classical 1D steady HTTP adapter.

Only a registered device ID and a bounded numerical protocol cross HTTP. The
application supplies prepared devices, their defaults/resources and actual
build sources. Installation does not discover files or run a numerical model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from pathlib import Path
import re
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, model_validator

from solarlab.config.identity import BuildSource, ExecutionEngine
from solarlab.device.resolved import PreparedDevice
from solarlab.experiments.jv.worker import (
    SteadyJVJob, SteadyJVProtocol, prepare_steady_jv_job,
    steady_jv_identity_inputs, validate_steady_jv_scope,
)
from solarlab.experiments.worker import ResourceLimits
from solarlab.materials.source import SourceDocument
from solarlab.physics.registry import ModelRegistry, metadata_digest

if TYPE_CHECKING:
    from solarlab_server.job_service import ExecutionCatalog

__all__ = ["SteadyJobInput", "SteadyPreparationContext", "steady_execution_catalog"]


class SteadyJobInput(BaseModel):
    """All protocol fields are explicit; no caller-selected code or source."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    device_id: str = Field(pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}$")
    voltage_V: list[float] = Field(min_length=2, max_length=101)
    intervals_per_layer: int = Field(ge=2, le=256)
    generation_scale: float = Field(ge=0)
    incident_power_W_m2: float | None = Field(gt=0)

    @model_validator(mode="after")
    def check_protocol(self) -> SteadyJobInput:
        self.protocol()
        return self

    def protocol(self) -> SteadyJVProtocol:
        return SteadyJVProtocol(tuple(self.voltage_V), self.intervals_per_layer,
                                self.generation_scale, self.incident_power_W_m2)


def _device_binding(id: str, prepared: PreparedDevice) -> dict[str, Any]:
    return {"device_id": id, "prepared_sha256": prepared.content_sha256,
            "defaults_sha256": prepared.defaults.content_sha256,
            "resources_sha256": prepared.resources.content_sha256,
            "sources": [{"id": source.id, "sha256": source.sha256} for source in prepared.sources]}


def _implementation_binding(implementation: BuildSource) -> dict[str, str]:
    return {"build_source_commit": implementation.build_source_commit,
            "source_manifest_sha256": metadata_digest(
                [{"id": source.id, "sha256": source.sha256} for source in implementation.files])}


@dataclass(frozen=True, slots=True)
class SteadyPreparationContext:
    devices: tuple[tuple[str, PreparedDevice], ...]
    implementation: BuildSource
    current_repository_head: str
    models: ModelRegistry = field(init=False)

    def __post_init__(self) -> None:
        devices = tuple(tuple(item) for item in self.devices)
        if (not devices or any(len(item) != 2 or type(item[0]) is not str
                or re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}", item[0]) is None
                or type(item[1]) is not PreparedDevice for item in devices)
                or len({item[0] for item in devices}) != len(devices)):
            raise ValueError("steady installation requires unique registered IDs and actual PreparedDevice data")
        if type(self.implementation) is not BuildSource:
            raise ValueError("steady installation requires actual trusted build sources")
        if (type(self.current_repository_head) is not str
                or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", self.current_repository_head) is None):
            raise ValueError("maintenance HEAD must be supplied independently of build-source identity")
        object.__setattr__(self, "devices", tuple(sorted(devices, key=lambda item: item[0])))
        # Reuse the worker's static scope and model binding. These helpers do
        # not compile operators or call a solver; do not duplicate their policy.
        for _, prepared in self.devices:
            validate_steady_jv_scope(prepared)
        object.__setattr__(self, "models", steady_jv_identity_inputs(self.devices[0][1], self.implementation).registry)

    def device(self, id: str) -> PreparedDevice:
        for key, prepared in self.devices:
            if key == id:
                return prepared
        raise ValueError("device is not installed in this trusted steady registration")

    def export(self) -> dict[str, Any]:
        return {"devices": [_device_binding(id, prepared) for id, prepared in self.devices],
                "implementation": _implementation_binding(self.implementation),
                "current_repository_head": self.current_repository_head}


def prepare_steady_input(input: SteadyJobInput, context: SteadyPreparationContext) -> SteadyJVJob:
    """Prepare the existing cold worker; all trusted records stay server-owned."""
    return prepare_steady_jv_job(context.device(input.device_id), input.protocol(),
        implementation=context.implementation, current_repository_head=context.current_repository_head,
        engine=ExecutionEngine("solarlab"))


def steady_execution_catalog(context: SteadyPreparationContext, *, ceiling: ResourceLimits) -> ExecutionCatalog:
    """Explicitly install the one supported steady adapter with actual sources.

    The renderer reference is the existing array reader, not a claim that a
    frontend has been installed. Global declaration/qualification flags stay
    unchanged. A caller may combine this registration with explicit legacy
    registrations; each retains its own model-registry binding.
    """
    from solarlab.experiments.jv.steady import steady_jv_payload
    from solarlab.experiments.jv.worker import read_steady_jv_job
    from solarlab.experiments.registry import ExperimentDefinition, ExperimentRegistry
    from solarlab.physics.registry import CapabilityRule, EvidenceRef
    from solarlab_server.job_service import ExecutionCatalog, SteadyRegistration, canonical
    from solarlab_server.registered_worker import run_steady

    if type(context) is not SteadyPreparationContext:
        raise ValueError("an explicit trusted steady preparation context is required")
    documents: dict[str, SourceDocument] = {}

    def reference(id: str, function: Any) -> EvidenceRef:
        source = SourceDocument(id, Path(inspect.getfile(function)).read_bytes())
        documents[id] = source
        return EvidenceRef(id + "#" + function.__name__, source.sha256, "source")

    schema = SourceDocument("solarlab.steady-job-input.schema.v1", canonical(SteadyJobInput.model_json_schema()))
    documents[schema.id] = schema
    factory = reference("solarlab_server/steady_preparation.py", prepare_steady_input)
    result = reference("solarlab/experiments/jv/steady.py", steady_jv_payload)
    executor = reference("solarlab_server/registered_worker.py", run_steady)
    reader = reference("solarlab/experiments/jv/worker.py", read_steady_jv_job)
    compiler = next(source for source in context.implementation.files if source.id == "solarlab/device/compile_1d.py")
    documents[compiler.id] = compiler
    definition = ExperimentDefinition("steady_jv", 1, "Classical ion-free 1D steady J-V",
        EvidenceRef(schema.id, schema.sha256, "source"), result, factory, executor,
        "steady_jv_arrays", reader, CapabilityRule("installed_classical_1d_dc", (1,), ("dc",), (factory,),
            backends=("steady_jv_worker",), statistics=("maxwell_boltzmann",)),
        tuple(model.ref for model in context.models.definitions))
    registry = ExperimentRegistry((definition,))
    return ExecutionCatalog(registry, context.models, tuple(documents.values()),
        (SteadyRegistration("steady_jv", 1, context),), registry.content_sha256,
        context.models.content_sha256, ceiling)
