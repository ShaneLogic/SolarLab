"""Editable 1D J-V declarations; source defaults are injected separately."""
from __future__ import annotations

from typing import Literal
from pydantic import Field, field_validator, model_validator

from solarlab.device.inputs import DeviceInput
from solarlab.experiments.two_dimensional.inputs import SpatialInput, invalid, vector
from solarlab.materials.full_parameters import StableId, quantity_field as q
from solarlab.materials.parameters import QuantityInput
from solarlab.units import normalize_quantity


class JVWaveformInput(SpatialInput):
    """All eight fields of JVWaveform.from_dict are explicit, including null."""
    _nullable_fields = frozenset({"uniform_generation_rate_m3_s"})
    schema_version: Literal[1]
    start_voltage_V: QuantityInput = q("V", signed=True, required=True)
    dark_seed_s: QuantityInput = q("s", required=True)
    dark_prep_s: QuantityInput = q("s", required=True)
    branch_dwell_s: QuantityInput = q("s", positive=True, required=True)
    turnaround_s: QuantityInput = q("s", required=True)
    turnaround_dark: bool
    uniform_generation_rate_m3_s: QuantityInput | None = q("m^-3/s", nullable=True, required=True)

    @field_validator("schema_version", mode="before")
    @classmethod
    def _version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1")
        return value

    @model_validator(mode="after")
    def _start(self) -> JVWaveformInput:
        if normalize_quantity(self.start_voltage_V, "V") > 0:
            invalid(type(self).__name__, ("start_voltage_V",), "start voltage must be <= 0 for the existing Jsc extraction", self.start_voltage_V)
        return self


class JVWaveformControlsInput(SpatialInput):
    rtol: QuantityInput = q("1", positive=True, required=True)
    atol_m3: QuantityInput = q("m^-3", positive=True, required=True)


class JVNumericalDefaults(SpatialInput):
    rtol: QuantityInput = q("1", positive=True, required=True)
    atol: QuantityInput = q("m^-3", positive=True, required=True)


class JVHistoryDefaults(SpatialInput):
    illuminated_soak_s: QuantityInput = q("s", positive=True, required=True)


class JVIlluminationStepInput(SpatialInput):
    _nullable_fields = frozenset({"duration_s", "intensity_suns", "photon_flux_m2_s", "relative_generation_change", "source_reference"})
    phase: str = Field(min_length=1)
    condition: Literal["dark", "baseline", "scaled", "monochromatic", "pulse"]
    duration_s: QuantityInput | None = q("s", nullable=True, required=True)
    intensity_suns: QuantityInput | None = q("1", nullable=True, required=True)
    photon_flux_m2_s: QuantityInput | None = q("m^-2/s", positive=True, nullable=True, required=True)
    relative_generation_change: QuantityInput | None = q("1", signed=True, nullable=True, required=True)
    source_reference: str | None = Field(min_length=1)


class JVScanInput(SpatialInput):
    axis: Literal["voltage_V"]
    direction: Literal["ascending", "descending", "ascending_then_descending", "declared_order", "forward_time"]
    start: QuantityInput = q("V", signed=True, required=True)
    stop: QuantityInput = q("V", signed=True, required=True)
    rate_V_s: QuantityInput = q("V/s", positive=True, required=True)


class JVSamplingInput(SpatialInput):
    axis: Literal["voltage_V"]
    mode: Literal["linear", "log", "declared", "piecewise_linear"]
    values: tuple[QuantityInput, ...] = vector("V", minimum=1)


class JVDCSettleInput(SpatialInput):
    _nullable_fields = frozenset({"duration_s", "max_carrier_area_rate_A_m2", "max_ion_area_rate_A_m2", "max_ionic_face_current_A_m2", "max_face_current_spread_A_m2"})
    kind: Literal["finite_time", "residual_certified", "finite_time_with_certificate", "not_applicable"]
    duration_s: QuantityInput | None = q("s", nullable=True, positive=True, required=True)
    max_carrier_area_rate_A_m2: QuantityInput | None = q("A/m^2", nullable=True, positive=True, required=True)
    max_ion_area_rate_A_m2: QuantityInput | None = q("A/m^2", nullable=True, positive=True, required=True)
    max_ionic_face_current_A_m2: QuantityInput | None = q("A/m^2", nullable=True, positive=True, required=True)
    max_face_current_spread_A_m2: QuantityInput | None = q("A/m^2", nullable=True, positive=True, required=True)

    @model_validator(mode="after")
    def _policy(self) -> JVDCSettleInput:
        if self.kind == "finite_time" and self.duration_s is None:
            invalid(type(self).__name__, ("duration_s",), "finite_time requires a duration", None)
        if self.kind == "not_applicable":
            for name in self._nullable_fields:
                if getattr(self, name) is not None:
                    invalid(type(self).__name__, (name,), "not_applicable cannot carry thresholds", getattr(self, name))
        return self


class JVProtocolInput(SpatialInput):
    """The existing ExperimentProtocol fields for its J-V branch only."""
    _nullable_fields = frozenset({"pre_bias_V", "soak_duration_s", "dwell_duration_s", "ac_excitation", "voc_search"})
    experiment: Literal["jv_hysteresis"]
    initial_state_source: Literal["dark_equilibrium", "dark_equilibrium_each_sample", "finite_time_illuminated_preconditioned", "finite_time_dc_preconditioned", "qf_dc_candidate", "user_supplied_state"]
    pre_bias_V: QuantityInput | None = q("V", signed=True, nullable=True, required=True)
    soak_duration_s: QuantityInput | None = q("s", nullable=True, required=True)
    dwell_duration_s: QuantityInput | None = q("s", nullable=True, required=True)
    illumination_history: tuple[JVIlluminationStepInput, ...] = Field(min_length=1)
    temperature_K: QuantityInput = q("K", positive=True, required=True)
    scan: JVScanInput
    ac_excitation: None
    dc_settle: JVDCSettleInput
    sampling: JVSamplingInput
    voc_search: None
    implicit_legacy_protocol: bool
    schema_version: Literal[1]

    @field_validator("schema_version", mode="before")
    @classmethod
    def _version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1")
        return value


class JVInput(SpatialInput):
    _nullable_fields = frozenset({"V_max", "waveform", "waveform_controls", "experiment_protocol"})
    kind: Literal["jv"]
    # Transport provenance chooses actual existing endpoint defaults. It is
    # not a physical default or an executor identifier.
    request_api: Literal["jobs", "jv_endpoint"] = "jobs"
    N_grid: int | None = Field(default=None, ge=3, title="Requested electrical grid allocation")
    n_points: int | None = Field(default=None, ge=2, title="Samples per branch")
    v_rate: QuantityInput | None = q("V/s", positive=True)
    V_max: QuantityInput | None = q("V", signed=True, nullable=True)
    illuminated: bool | None = None
    solver: Literal["transient", "steady_state", "quasi_fermi"] | None = None
    iface_states: bool | None = None
    interface_boundary: bool | None = None
    interface_transport_model: str | None = Field(default=None, min_length=1)
    protocol_mode: Literal["compatibility", "research_strict"] | None = None
    waveform: JVWaveformInput | None = None
    waveform_controls: JVWaveformControlsInput | None = None
    experiment_protocol: JVProtocolInput | None = None


class DarkJVInput(SpatialInput):
    kind: Literal["dark_jv"]
    N_grid: int | None = Field(default=None, ge=3, title="Requested electrical grid allocation")
    n_points: int | None = Field(default=None, ge=2, title="Samples per branch")
    v_rate: QuantityInput | None = q("V/s", positive=True)
    V_max: QuantityInput | None = q("V", signed=True)


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
        return (JVInput if value["kind"] == "jv" else DarkJVInput).model_validate(value)


JV_DEFAULT_FIELDS = {
    "jv_jobs": ("N_grid", "n_points", "v_rate", "V_max", "solver", "iface_states", "interface_boundary", "interface_transport_model", "protocol_mode", "illuminated"),
    "jv_endpoint": ("N_grid", "n_points", "v_rate", "V_max", "solver", "iface_states", "interface_boundary", "interface_transport_model", "protocol_mode"),
    "dark_jv": ("N_grid", "n_points", "v_rate", "V_max"),
    "jv_numerical": tuple(JVNumericalDefaults.model_fields),
    "dark_jv_numerical": tuple(JVNumericalDefaults.model_fields),
    "jv_waveform_controls": tuple(JVWaveformControlsInput.model_fields),
    "jv_history": tuple(JVHistoryDefaults.model_fields),
}
