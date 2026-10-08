"""Editable 1D experiment envelope; pure declarations have no device dependency."""
from __future__ import annotations

from typing import Literal
from pydantic import field_validator

from solarlab.device.inputs import DeviceInput
from solarlab.experiments.inputs import ExperimentInput as SpatialInput, invalid
from solarlab.materials.full_parameters import StableId
from solarlab.experiments.jv.declarations import (
    JVWaveformInput as JVWaveformInput,
    JVWaveformControlsInput as JVWaveformControlsInput,
    JVNumericalDefaults as JVNumericalDefaults,
    JVHistoryDefaults as JVHistoryDefaults,
    JVIlluminationStepInput as JVIlluminationStepInput,
    JVScanInput as JVScanInput,
    JVSamplingInput as JVSamplingInput,
    JVDCSettleInput as JVDCSettleInput,
    JVProtocolInput as JVProtocolInput,
    JVInput as JVInput,
    DarkJVInput as DarkJVInput,
    JV_DEFAULT_FIELDS as JV_DEFAULT_FIELDS,
)


class JVExperimentInput(SpatialInput):
    schema_version: Literal["solarlab.experiment-preparation.v1"]
    id: StableId
    device: DeviceInput
    experiment: JVInput | DarkJVInput

    @field_validator("experiment", mode="before")
    @classmethod
    def _branch(cls, value: object) -> object:
        if isinstance(value, (JVInput, DarkJVInput)):
            value = value.editing_data()
        if not isinstance(value, dict) or value.get("kind") not in {"jv", "dark_jv"}:
            invalid(cls.__name__, ("kind",), "choose jv or dark_jv", value)
        assert isinstance(value, dict)
        return (JVInput if value["kind"] == "jv" else DarkJVInput).model_validate(value)
