"""Explicit ordered coordinates and stable parameter-instance targets."""
from __future__ import annotations

from typing import Literal
from pydantic import Field, StrictBool, model_validator, field_validator

from solarlab.device.inputs import DeviceInput
from solarlab.experiments.inputs import ExperimentInput, invalid
from solarlab.experiments.jv.inputs import JVExperimentInput
from solarlab.materials.full_parameters import StableId
from solarlab.materials.parameters import QuantityInput


class SweepTargetInput(ExperimentInput):
    family: Literal["layer_parameter", "material_parameter", "setting", "bulk_defect", "interface_defect", "jv", "cbo"]
    owner_id: StableId | None = None
    local_id: StableId | None = None
    reference_id: StableId | None = None
    parameter: tuple[StableId, ...] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def _identity(self) -> SweepTargetInput:
        required = {"owner_id"} if self.family not in {"setting", "jv"} else set()
        if self.family in {"bulk_defect", "interface_defect"}:
            required.add("local_id")
        if self.family == "cbo":
            required.add("reference_id")
        for name in ("owner_id", "local_id", "reference_id"):
            if (getattr(self, name) is not None) != (name in required):
                invalid(type(self).__name__, (name,), f"{self.family} target requires exactly {sorted(required)}", getattr(self, name))
        return self


class SweepCoordinateInput(ExperimentInput):
    _nullable_fields = frozenset({"value"})
    kind: Literal["value", "omit"]
    value: StrictBool | QuantityInput | None = None

    @model_validator(mode="after")
    def _presence(self) -> SweepCoordinateInput:
        if ("value" in self.model_fields_set) != (self.kind == "value"):
            invalid(type(self).__name__, ("value",), "value coordinates require an explicit value (including null); omission coordinates carry no value", self.editing_data())
        return self


class SweepAxisInput(ExperimentInput):
    id: StableId
    target: SweepTargetInput
    coordinates: tuple[SweepCoordinateInput, ...] = Field(min_length=1)


class SweepInput(ExperimentInput):
    _nullable_fields = frozenset({"reference_id"})
    schema_version: Literal["solarlab.sweep-preparation.v1"]
    id: StableId
    base: DeviceInput | JVExperimentInput
    axes: tuple[SweepAxisInput, ...] = Field(min_length=1, max_length=2)
    reference_id: StableId | None = None

    @field_validator("base", mode="before")
    @classmethod
    def _base(cls, value: object) -> object:
        if isinstance(value, (DeviceInput, JVExperimentInput)):
            value = value.editing_data()
        models = {"solarlab.device-preparation.v1": DeviceInput, "solarlab.experiment-preparation.v1": JVExperimentInput}
        if not isinstance(value, dict) or value.get("schema_version") not in models:
            invalid(cls.__name__, ("schema_version",), "sweep preparation requires a supplied DeviceInput or one-dimensional JVExperimentInput", value)
        return models[value["schema_version"]].model_validate(value)

    @model_validator(mode="after")
    def _axes(self) -> SweepInput:
        seen = set()
        for index, axis in enumerate(self.axes):
            if axis.id in seen:
                invalid(type(self).__name__, ("axes", index, "id"), "duplicate sweep-axis ID", axis.id)
            seen.add(axis.id)
        return self
