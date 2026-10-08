"""Editable declarations from the existing 2D request/protocol contracts.

Geometry belongs to the experiment, not DeviceInput. Source defaults are
injected separately; optional None defaults represent absent fields only.
"""
from __future__ import annotations

from typing import Any, ClassVar, Literal, NoReturn

from pydantic import Field, ValidationError, ValidationInfo, field_validator, model_validator

from solarlab.device.inputs import DeviceInput, GrainBoundaryInput
from solarlab.materials.full_parameters import StableId, StructuredInput, quantity_field as q
from solarlab.materials.parameters import QuantityInput
from solarlab.units import normalize_quantity


def vector(unit: str, *, required: bool = True, minimum: int = 0,
           input_unit: str | None = None, positive: bool = False) -> Any:
    return Field(default=... if required else None, min_length=minimum,
                 json_schema_extra={"item_unit": unit, "item_input_unit": input_unit,
                                    "item_positive": positive})


def invalid(name: str, path: tuple[str | int, ...], message: str, value: object) -> NoReturn:
    raise ValidationError.from_exception_data(name, [{"type": "value_error", "loc": path,
        "input": value, "ctx": {"error": ValueError(message)}}])


class SpatialInput(StructuredInput):
    _nullable_fields: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def _allows_null(cls, name: str) -> bool:
        return name in cls._nullable_fields

    @field_validator("*")
    @classmethod
    def _nulls(cls, value: object, info: ValidationInfo) -> object:
        if value is None and not cls._allows_null(str(info.field_name)):
            raise ValueError("omit to inherit; explicit null is unsupported")
        return value

    @field_validator("*", mode="before")
    @classmethod
    def _vectors(cls, value: object, info: ValidationInfo) -> object:
        extra = cls.model_fields[str(info.field_name)].json_schema_extra
        if not isinstance(extra, dict) or "item_unit" not in extra or value is None:
            return value
        if not isinstance(value, (tuple, list)):
            raise ValueError("supply an explicit ordered array")
        for index, item in enumerate(value):
            try:
                number = normalize_quantity(item, str(extra["item_unit"]), input_unit=extra.get("item_input_unit"))
                if extra.get("item_positive") and number <= 0:
                    raise ValueError("must be positive; values are never silently dropped")
            except ValueError as error:
                invalid(cls.__name__, (index,), str(error), item)
        return tuple(value)

    def normalized_data(self) -> dict[str, Any]:
        data = super().normalized_data()
        for name in self.model_fields_set:
            extra = type(self).model_fields[name].json_schema_extra
            value = getattr(self, name)
            if value is not None and isinstance(extra, dict) and "item_unit" in extra:
                data[name] = [normalize_quantity(item, str(extra["item_unit"]), input_unit=extra.get("item_input_unit")) for item in value]
        return data


class ComponentwiseAtolInput(SpatialInput):
    carrier_fraction: QuantityInput = q("1", positive=True, required=True)
    ion_fraction: QuantityInput = q("1", positive=True, required=True)
    interface_fraction: QuantityInput = q("1", positive=True, required=True)
    minimum_atol: QuantityInput = q("m^-3", positive=True, required=True)
    refinement_factor: QuantityInput = q("1", positive=True, required=True)


class JV2DAtolInput(SpatialInput):
    _nullable_fields = frozenset({"scalar_atol", "carrier_fraction", "ion_fraction", "interface_fraction", "minimum_atol", "refinement_factor"})
    mode: Literal["scalar", "componentwise"]
    scalar_atol: QuantityInput | None = q("m^-3", positive=True, nullable=True, required=True)
    carrier_fraction: QuantityInput | None = q("1", positive=True, nullable=True, required=True)
    ion_fraction: QuantityInput | None = q("1", positive=True, nullable=True, required=True)
    interface_fraction: QuantityInput | None = q("1", positive=True, nullable=True, required=True)
    minimum_atol: QuantityInput | None = q("m^-3", positive=True, nullable=True, required=True)
    refinement_factor: QuantityInput | None = q("1", positive=True, nullable=True, required=True)

    @model_validator(mode="after")
    def _variant(self) -> JV2DAtolInput:
        for name in self._nullable_fields:
            expected = name == "scalar_atol" if self.mode == "scalar" else name != "scalar_atol"
            if (getattr(self, name) is not None) != expected:
                invalid(type(self).__name__, (name,), f"{self.mode} mode has inconsistent tolerance fields", getattr(self, name))
        return self


class JV2DGrainProtocolInput(SpatialInput):
    x_position_m: QuantityInput = q("m", signed=True, required=True)
    width_m: QuantityInput = q("m", positive=True, required=True)
    tau_n_s: QuantityInput = q("s", positive=True, required=True)
    tau_p_s: QuantityInput = q("s", positive=True, required=True)
    layer_role: str = Field(min_length=1)

    @field_validator("layer_role")
    @classmethod
    def _role(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("layer_role must be nonempty")
        return value


class JV2DProtocolInput(SpatialInput):
    """Supplied execution-protocol declaration, not a generated or bound mesh."""
    _nullable_fields = frozenset({"illumination_source", "initial_state_settle_s"})
    temperature_K: QuantityInput = q("K", positive=True, required=True)
    illuminated: bool
    illumination_source: str | None
    initial_state_source: Literal["one_dimensional_illuminated_finite_time", "one_dimensional_dark_equilibrium"]
    initial_state_voltage_V: QuantityInput = q("V", signed=True, required=True)
    initial_state_settle_s: QuantityInput | None = q("s", positive=True, nullable=True, required=True)
    voltage_values_V: tuple[QuantityInput, ...] = vector("V", minimum=1)
    dwell_time_per_voltage_s: QuantityInput = q("s", positive=True, required=True)
    state_topology: Literal["frozen_ion_background", "single_positive_mobile_ion"]
    ion_boundary_condition: Literal["frozen", "blocking"]
    carrier_boundary_condition: Literal["ohmic", "selective_robin"]
    interface_srh: Literal["off", "two_sided_cross_node"]
    lateral_bc: Literal["periodic", "neumann"]
    x_coordinates_m: tuple[QuantityInput, ...] = vector("m", minimum=2)
    y_coordinates_m: tuple[QuantityInput, ...] = vector("m", minimum=3)
    grain_boundaries: tuple[JV2DGrainProtocolInput, ...]
    current_composition: Literal["electron_hole_conduction", "electron_hole_positive_ion_displacement"]
    current_sampling: Literal["instantaneous_dwell_endpoint"]
    applied_voltage_rate_at_sampling_V_s: QuantityInput = q("V/s", signed=True, required=True)
    solver_method: Literal["Radau"]
    solver_rtol: QuantityInput = q("1", positive=True, required=True)
    solver_atol: JV2DAtolInput
    solver_max_step_divisor: int = Field(gt=0)
    max_nfev_per_solve: int = Field(gt=0)
    max_bisect: int = Field(ge=0)
    ion_inventory_rtol: QuantityInput = q("1", required=True)
    save_snapshots: bool
    implicit_legacy_protocol: bool
    schema_version: Literal["jv-2d-execution-protocol-v1"]

    @model_validator(mode="after")
    def _relations(self) -> JV2DProtocolInput:
        def fail(field: str, message: str) -> None:
            invalid(type(self).__name__, (field,), message, getattr(self, field))
        if self.illuminated:
            if not self.illumination_source or not self.illumination_source.strip():
                fail("illumination_source", "illuminated 2D J-V requires an illumination source")
            if self.initial_state_source != "one_dimensional_illuminated_finite_time":
                fail("initial_state_source", "illuminated 2D J-V requires illuminated 1D initial state")
            if self.initial_state_settle_s is None:
                fail("initial_state_settle_s", "illuminated 1D initial state requires settle time")
        else:
            if self.illumination_source is not None:
                fail("illumination_source", "dark 2D J-V cannot carry illumination source")
            if self.initial_state_source != "one_dimensional_dark_equilibrium":
                fail("initial_state_source", "dark 2D J-V requires dark-equilibrium initial state")
            if self.initial_state_settle_s is not None:
                fail("initial_state_settle_s", "dark-equilibrium initial state has no settle time")
        if normalize_quantity(self.initial_state_voltage_V, "V") != 0:
            fail("initial_state_voltage_V", "2D J-V v1 initial voltage must be zero")
        if normalize_quantity(self.applied_voltage_rate_at_sampling_V_s, "V/s") != 0:
            fail("applied_voltage_rate_at_sampling_V_s", "fixed-voltage dwell endpoint requires zero voltage rate")
        for field, unit in (("voltage_values_V", "V"), ("x_coordinates_m", "m"), ("y_coordinates_m", "m")):
            values = [normalize_quantity(value, unit) for value in getattr(self, field)]
            if field == "voltage_values_V" and values[0] != 0:
                fail(field, "voltage values must start at zero")
            if any(right <= left for left, right in zip(values, values[1:])):
                fail(field, "values must be strictly increasing; input is never sorted")
        mobile = self.state_topology == "single_positive_mobile_ion"
        if (self.grain_boundaries or mobile or self.interface_srh != "off") and self.lateral_bc != "neumann":
            fail("lateral_bc", "grain, mobile-ion or interface-SRH declarations require Neumann-x topology")
        if (mobile or self.interface_srh != "off") and self.carrier_boundary_condition != "ohmic":
            fail("carrier_boundary_condition", "mobile-ion/interface-SRH 2D J-V requires ohmic contacts")
        if self.ion_boundary_condition != ("blocking" if mobile else "frozen"):
            fail("ion_boundary_condition", "ion boundary label disagrees with state topology")
        if self.current_composition != ("electron_hole_positive_ion_displacement" if mobile else "electron_hole_conduction"):
            fail("current_composition", "current composition disagrees with state topology")
        return self


class MicrostructureInput(SpatialInput):
    grain_boundaries: tuple[GrainBoundaryInput, ...] = ()


class JV2DInput(SpatialInput):
    _nullable_fields = frozenset({"microstructure", "lateral_bc", "componentwise_atol", "jv_2d_protocol"})
    kind: Literal["jv_2d"]
    lateral_length: QuantityInput | None = q("m", positive=True)
    Nx: int | None = Field(default=None, gt=0, title="Lateral intervals (Nx)")
    Ny_per_layer: int | None = Field(default=None, gt=0, title="Vertical intervals per electrical layer")
    lateral_bc: Literal["periodic", "neumann"] | None = None
    microstructure: MicrostructureInput | None = None
    V_max: QuantityInput | None = q("V")
    V_step: QuantityInput | None = q("V", positive=True)
    illuminated: bool | None = None
    save_snapshots: bool | None = None
    settle_t: QuantityInput | None = q("s", positive=True)
    rtol: QuantityInput | None = q("1", positive=True)
    atol: QuantityInput | None = q("m^-3", positive=True)
    componentwise_atol: ComponentwiseAtolInput | None = None
    max_nfev_per_solve: int | None = Field(default=None, gt=0)
    max_bisect: int | None = Field(default=None, ge=0)
    ion_inventory_rtol: QuantityInput | None = q("1")
    initial_state_settle_s: QuantityInput | None = q("s", positive=True)
    ion_dynamics: Literal["frozen", "single_mobile"] | None = None
    interface_srh: Literal["off", "two_sided_cross_node"] | None = None
    protocol_mode: Literal["compatibility", "research_strict"] | None = None
    jv_2d_protocol: JV2DProtocolInput | None = None

    @model_validator(mode="after")
    def _tolerances(self) -> JV2DInput:
        if self.componentwise_atol is not None and "atol" in self.model_fields_set:
            invalid(type(self).__name__, ("componentwise_atol",), "supply atol or componentwise_atol, not both", self.componentwise_atol.editing_data())
        return self


class GrainSweepInput(SpatialInput):
    _nullable_fields = frozenset({"grain_sizes_nm", "grain_sizes"})
    kind: Literal["voc_grain_sweep"]
    grain_sizes_nm: tuple[QuantityInput, ...] | None = vector("m", required=False, input_unit="nm", positive=True)
    grain_sizes: tuple[QuantityInput, ...] | None = vector("m", required=False, input_unit="nm", positive=True)
    tau_gb_n: QuantityInput | None = q("s", positive=True)
    tau_gb_p: QuantityInput | None = q("s", positive=True)
    gb_width: QuantityInput | None = q("m", positive=True)
    Nx: int | None = Field(default=None, gt=0, title="Lateral intervals (Nx)")
    Ny_per_layer: int | None = Field(default=None, gt=0, title="Vertical intervals per electrical layer")
    V_max: QuantityInput | None = q("V")
    V_step: QuantityInput | None = q("V", positive=True)
    illuminated: bool | None = None
    settle_t: QuantityInput | None = q("s", positive=True)


class SpatialExperimentInput(SpatialInput):
    schema_version: Literal["solarlab.experiment-preparation.v1"]
    id: StableId
    device: DeviceInput
    experiment: JV2DInput | GrainSweepInput

    @field_validator("experiment", mode="before")
    @classmethod
    def _experiment(cls, value: object) -> object:
        # The shared input primitives validate every field before parsing, so
        # a Pydantic field discriminator cannot wrap their kind validators.
        # Validate the selected branch first for precise, non-duplicated paths.
        if isinstance(value, (JV2DInput, GrainSweepInput)):
            value = value.editing_data()
        if not isinstance(value, dict) or value.get("kind") not in {"jv_2d", "voc_grain_sweep"}:
            invalid(cls.__name__, ("kind",), "choose jv_2d or voc_grain_sweep", value)
        model = JV2DInput if value["kind"] == "jv_2d" else GrainSweepInput
        return model.model_validate(value)


# Names only: values are read from the actual backend/tolerance source bytes.
EXPERIMENT_DEFAULT_FIELDS = {
    "jv_2d": tuple(name for name in JV2DInput.model_fields if name not in
                   {"kind", "lateral_bc", "microstructure", "componentwise_atol", "jv_2d_protocol"}),
    "voc_grain_sweep": tuple(name for name in GrainSweepInput.model_fields if name not in
                            {"kind", "grain_sizes_nm", "grain_sizes"}),
    "jv_2d_componentwise_atol": tuple(ComponentwiseAtolInput.model_fields),
}
