"""Explicit editable preparation schemas, separate from the old wire inputs."""

from __future__ import annotations

from typing import Annotated, Any, Literal, get_origin

from pydantic import Field, ValidationInfo, field_validator, model_validator

from solarlab.materials.full_parameters import FullParameterInput, ParameterFieldsInput, quantity_field as q
from solarlab.materials.parameters import QuantityInput
from solarlab.device.settings import DeviceSettingsInput

StableId = Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.:-]*$")]
Nonempty = Annotated[str, Field(min_length=1)]


class StructuredInput(ParameterFieldsInput):
    @field_validator("*", mode="before")
    @classmethod
    def _explicit_sequences(cls, value: object, info: ValidationInfo) -> object:
        assert info.field_name is not None
        if get_origin(cls.model_fields[info.field_name].annotation) is tuple:
            if not isinstance(value, (tuple, list)):
                raise ValueError("expected an explicit sequence")
            return tuple(value)
        return value

    def normalized_data(self) -> dict[str, Any]:
        checked = type(self).model_validate(self)
        own = dict(checked.normalized_items())
        def item(value: Any) -> Any:
            if isinstance(value, StructuredInput):
                return value.normalized_data()
            if isinstance(value, ParameterFieldsInput):
                return dict(value.normalized_items())
            if isinstance(value, tuple):
                return [item(part) for part in value]
            return value
        return {name: item(value) for name, value in own.items()}


class KineticsInput(StructuredInput):
    sigma_n_m2: QuantityInput = q("m^2", required=True)
    sigma_p_m2: QuantityInput = q("m^2", required=True)
    thermal_velocity_n_m_s: QuantityInput = q("m/s", positive=True, required=True)
    thermal_velocity_p_m_s: QuantityInput = q("m/s", positive=True, required=True)


class DistributionInput(StructuredInput):
    kind: Literal["single_level", "gaussian", "uniform", "conduction_band_tail", "valence_band_tail"]
    normalization: Literal["integrated_total"]
    total_density_m3: QuantityInput = q("m^-3", positive=True, required=True)
    center_eV_above_vb: QuantityInput | None = q("eV", nullable=True)
    width_eV: QuantityInput | None = q("eV", positive=True, nullable=True)
    width_convention: Literal["not_applicable", "gaussian_standard_deviation", "scaps_characteristic_energy", "uniform_full_width", "unresolved"] | None = None
    energy_reference: Literal["above_valence_band"] | None = None
    support_width_multiplier: QuantityInput | None = q("1", positive=True, nullable=True)

    @model_validator(mode="after")
    def _distribution(self) -> DistributionInput:
        if self.kind == "single_level":
            if self.width_eV is not None or self.support_width_multiplier is not None or self.width_convention not in {None, "not_applicable"}:
                raise ValueError("single_level cannot carry width or support fields")
        elif self.kind == "uniform":
            if self.width_eV is None or self.width_convention != "uniform_full_width" or self.support_width_multiplier is not None:
                raise ValueError("uniform requires a complete full width and no support multiplier")
        elif self.kind in {"conduction_band_tail", "valence_band_tail"}:
            if self.width_eV is None or self.width_convention != "scaps_characteristic_energy" or self.support_width_multiplier is None:
                raise ValueError("band tail requires characteristic width and finite support multiplier")
        elif self.width_convention == "unresolved":
            if self.width_eV is not None or self.support_width_multiplier is not None:
                raise ValueError("unresolved Gaussian cannot carry resolved width or support")
        elif self.width_eV is None or self.width_convention not in {"gaussian_standard_deviation", "scaps_characteristic_energy"}:
            raise ValueError("Gaussian requires an explicit width convention or unresolved metadata")
        return self


class EnergyLevelInput(StructuredInput):
    reference: Literal["below_conduction_band", "above_valence_band"]
    value_eV: QuantityInput = q("eV", required=True)


class SpatialKnotInput(StructuredInput):
    position_fraction: QuantityInput = q("1", required=True)
    density_multiplier: QuantityInput = q("1", required=True)


class SpatialProfileInput(StructuredInput):
    coordinate: Literal["normalized_layer_coordinate"]
    interpolation: Literal["piecewise_linear"]
    density_normalization: Literal["layer_average_unity"]
    knots: tuple[SpatialKnotInput, ...]


class BulkDefectInput(StructuredInput):
    id: StableId
    name: Nonempty | None = None
    distribution: DistributionInput
    charge_transition: Literal["neutral", "acceptor", "donor", "unresolved"]
    neutral_reference: Literal["all_occupancies", "empty", "filled", "unresolved"]
    kinetics: KineticsInput
    degeneracy: QuantityInput = q("1", positive=True, required=True)
    spatial_profile: SpatialProfileInput | None = None
    energy_level: EnergyLevelInput | None = None

    @model_validator(mode="after")
    def _charge_reference(self) -> BulkDefectInput:
        expected = {"neutral": "all_occupancies", "acceptor": "empty", "donor": "filled", "unresolved": "unresolved"}
        if self.neutral_reference != expected[self.charge_transition]:
            raise ValueError("defect charge transition and neutral reference disagree")
        if (self.energy_level is None) == (self.distribution.center_eV_above_vb is None):
            raise ValueError("declare exactly one defect energy authority: center or referenced energy_level")
        return self


class ScapsDefectMetadataInput(StructuredInput):
    """Preserved partner metadata; a Gaussian note is not a qualified closure."""
    distribution: Literal["single", "gaussian"]
    E_char_eV: QuantityInput | None = q("eV", positive=True, nullable=True)
    N_peak_m3: QuantityInput | None = q("m^-3", nullable=True)
    energy_reference: Literal["below_conduction_band", "above_valence_band"]
    trap_depth_eV: QuantityInput = q("eV", required=True)
    defect_id: StableId


class InterfaceDefectInput(StructuredInput):
    id: StableId
    trap_depth_eV: QuantityInput = q("eV", required=True)
    energy_reference: Literal["below_conduction_band", "above_valence_band"]
    total_density_m2: QuantityInput = q("m^-2", required=True)
    kinetics: KineticsInput
    calibration_factor: QuantityInput = q("1", required=True)
    iface_state_calibration_factor: QuantityInput = q("1", required=True)
    partner_metadata: ScapsDefectMetadataInput | None = None


class InterfaceInput(StructuredInput):
    id: StableId
    left: StableId
    right: StableId
    v_n: QuantityInput | None = q("m/s", nullable=True)
    v_p: QuantityInput | None = q("m/s", nullable=True)
    defect: InterfaceDefectInput | None = None


class ContactInput(StructuredInput):
    id: StableId
    side: Literal["left", "right"]
    layer: StableId
    S_n: QuantityInput | None = q("m/s", nullable=True)
    S_p: QuantityInput | None = q("m/s", nullable=True)


class GrainBoundaryInput(StructuredInput):
    id: StableId
    layer_ids: tuple[StableId, ...]
    x_position: QuantityInput = q("m", required=True)
    width: QuantityInput = q("m", positive=True, required=True)
    tau_n: QuantityInput = q("s", positive=True, required=True)
    tau_p: QuantityInput = q("s", positive=True, required=True)
    source_layer_role: str | None = None


class GridLayerInput(StructuredInput):
    layer: StableId
    interval_weight: QuantityInput = q("1", positive=True, required=True)
    alpha: QuantityInput = q("1", positive=True, required=True)


class FullLayerInput(StructuredInput):
    parameterization: Literal["standard", "scaps"] = "standard"
    id: StableId
    name: Nonempty
    role: Nonempty
    thickness: QuantityInput = q("m", positive=True, required=True)
    material: StableId | None = None
    parameters: FullParameterInput = Field(default_factory=FullParameterInput)
    defect_schema_version: str | None = None
    defect_model: Literal["effective_lifetime", "explicit_quasi_steady"] | None = None
    bulk_defects: tuple[BulkDefectInput, ...] = ()
    scaps_defect_metadata: tuple[ScapsDefectMetadataInput, ...] = ()


class NamedMaterialInput(StructuredInput):
    id: StableId
    name: Nonempty
    parameters: FullParameterInput


class SimulationHintsInput(StructuredInput):
    min_N_grid: int | None = Field(default=None, gt=0)
    notes: str | None = None


class DeviceInput(StructuredInput):
    schema_version: Literal["solarlab.device-preparation.v1"]
    id: StableId
    name: str | None = None
    description: str | None = None
    source_schema_version: str | int | None = None
    source_format: Literal["standard", "scaps", "canonical"]
    settings: DeviceSettingsInput = Field(default_factory=DeviceSettingsInput)
    materials: tuple[NamedMaterialInput, ...] = ()
    layers: tuple[FullLayerInput, ...]
    interfaces: tuple[InterfaceInput, ...] = ()
    contacts: tuple[ContactInput, ...] = ()
    grain_boundaries: tuple[GrainBoundaryInput, ...] = ()
    electrical_grid: tuple[GridLayerInput, ...] = ()
    simulation_hints: SimulationHintsInput | None = None
    spectrum: str | None = None
    fixed_generation: str | None = None


class OpticalLayerInput(StructuredInput):
    id: StableId
    name: Nonempty
    thickness: QuantityInput = q("m", positive=True, required=True)
    optical_material: Nonempty
    incoherent: bool


class BenchmarkInput(StructuredInput):
    reference: Nonempty
    target_pce_fraction: QuantityInput = q("1", required=True)
    target_jsc_A_m2: QuantityInput = q("A/m^2", required=True)
    target_voc_V: QuantityInput = q("V", required=True)
    target_ff_fraction: QuantityInput = q("1", required=True)
    tolerance_fraction: QuantityInput = q("1", required=True)


class TandemInput(StructuredInput):
    schema_version: Literal["solarlab.tandem-preparation.v1"]
    id: StableId
    source_schema_version: Literal[1]
    device_type: Literal["tandem_2T_monolithic"]
    top_cell: DeviceInput
    bottom_cell: DeviceInput
    top_cell_reference: Nonempty
    bottom_cell_reference: Nonempty
    junction_model: Literal["ideal_ohmic"]
    light_direction: Literal["top_first"]
    junction_stack: tuple[OpticalLayerInput, ...]
    back_reflector: OpticalLayerInput | None = None
    benchmark: BenchmarkInput | None = None

    @field_validator("source_schema_version", mode="before")
    @classmethod
    def _version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("tandem schema version must be the integer 1")
        return value
