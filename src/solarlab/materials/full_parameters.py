"""Full declared scalar layer parameters; physical defaults are supplied data.

The smaller LayerValuesInput/template contract remains unchanged. This schema
adds actual MaterialParams fields without copying their physical defaults.
Layer-owned doping and trap-profile values are separated during resolution.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Annotated, Any, Literal, get_origin

from pydantic import Field, GetJsonSchemaHandler, ValidationInfo, field_validator
from pydantic.fields import FieldInfo
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema

from solarlab.materials.parameters import EditableInput, LayerValuesInput, QuantityInput, Scalar
from solarlab.units import normalize_quantity

__all__ = ["ParameterFieldsInput", "StructuredInput", "StableId", "Nonempty", "FullParameterInput", "quantity_field", "LAYER_OWNED_PARAMETERS", "full_parameter_items"]

StableId = Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.:-]*$")]
Nonempty = Annotated[str, Field(min_length=1)]


def quantity_field(unit: str, *, positive: bool = False, signed: bool = False,
                   nullable: bool = False, required: bool = False,
                   input_unit: str | None = None) -> Any:
    # None is an absent editing placeholder, never a physical default.
    return Field(default=... if required else None, json_schema_extra={
        "unit": unit, "positive": positive, "nonnegative": not signed,
        "nullable": nullable,
        "input_unit": input_unit,
    })


def _scalar(name: str) -> FieldInfo:
    return deepcopy(LayerValuesInput.model_fields[name])


class ParameterFieldsInput(EditableInput):
    """Strict unit fields shared by these explicit preparation schemas."""

    @classmethod
    def _allows_null(cls, name: str) -> bool:
        extra = cls.model_fields[name].json_schema_extra
        return not isinstance(extra, dict) or "unit" not in extra or bool(extra.get("nullable", False))

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        schema = handler(core_schema)
        for name, definition in schema.get("properties", {}).items():
            if definition.get("default") is None:
                definition.pop("default", None)
            if not cls._allows_null(name) and "anyOf" in definition:
                definition["anyOf"] = [choice for choice in definition["anyOf"] if choice.get("type") != "null"]
        return schema

    @field_validator("*", mode="before")
    @classmethod
    def _check_parameter(cls, value: object, info: ValidationInfo) -> object:
        # Pydantic revalidates nested instances through its core schema, not
        # the public model_validate override. Snapshot only supplied fields
        # before that path can reinterpret absent placeholders as nulls.
        if isinstance(value, EditableInput):
            return type(value).model_validate(value).editing_data()
        if isinstance(value, (tuple, list)):
            return tuple(type(item).model_validate(item).editing_data() if isinstance(item, EditableInput) else item for item in value)
        name = info.field_name
        assert name is not None
        extra = cls.model_fields[name].json_schema_extra
        if isinstance(extra, dict) and "unit" in extra:
            if value is None:
                if extra.get("nullable", False):
                    return value
                raise ValueError(f"{name}: omit to inherit; null is not a number")
            number = normalize_quantity(value, str(extra["unit"]), path=name, input_unit=extra.get("input_unit"))
            if extra.get("positive", False) and number <= 0:
                raise ValueError(f"{name}: must be positive")
            if extra.get("nonnegative", True) and number < 0:
                raise ValueError(f"{name}: must be nonnegative")
        return value

    def normalized_items(self) -> tuple[tuple[str, Scalar], ...]:
        checked = type(self).model_validate(self)
        values = []
        for name in sorted(checked.model_fields_set):
            value = getattr(checked, name)
            extra = type(self).model_fields[name].json_schema_extra
            if value is not None and isinstance(extra, dict) and "unit" in extra:
                value = normalize_quantity(value, str(extra["unit"]), path=name, input_unit=extra.get("input_unit"))
            values.append((name, value))
        return tuple(values)


class StructuredInput(ParameterFieldsInput):
    """Named structured input shared by material and device declarations."""

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


class FullParameterInput(ParameterFieldsInput):
    eps_r: QuantityInput | None = _scalar("eps_r")
    mu_n: QuantityInput | None = _scalar("mu_n")
    mu_p: QuantityInput | None = _scalar("mu_p")
    D_ion: QuantityInput | None = _scalar("D_ion")
    P_lim: QuantityInput | None = _scalar("P_lim")
    P0: QuantityInput | None = _scalar("P0")
    ni: QuantityInput | None = _scalar("ni")
    tau_n: QuantityInput | None = _scalar("tau_n")
    tau_p: QuantityInput | None = _scalar("tau_p")
    n1: QuantityInput | None = _scalar("n1")
    p1: QuantityInput | None = _scalar("p1")
    B_rad: QuantityInput | None = _scalar("B_rad")
    C_n: QuantityInput | None = _scalar("C_n")
    C_p: QuantityInput | None = _scalar("C_p")
    alpha: QuantityInput | None = _scalar("alpha")
    N_A: QuantityInput | None = _scalar("N_A")
    N_D: QuantityInput | None = _scalar("N_D")
    chi: QuantityInput | None = _scalar("chi")
    Eg: QuantityInput | None = _scalar("Eg")
    A_star_n: QuantityInput | None = quantity_field("A/(m^2 K^2)")
    A_star_p: QuantityInput | None = quantity_field("A/(m^2 K^2)")
    v_th: QuantityInput | None = quantity_field("m/s", positive=True)
    D_ion_neg: QuantityInput | None = quantity_field("m^2/s")
    P0_neg: QuantityInput | None = quantity_field("m^-3")
    P_lim_neg: QuantityInput | None = quantity_field("m^-3")
    Nc300: QuantityInput | None = quantity_field("m^-3", positive=True, nullable=True)
    Nv300: QuantityInput | None = quantity_field("m^-3", positive=True, nullable=True)
    carrier_statistics: Literal["maxwell_boltzmann", "fermi_dirac"] | None = None
    dopant_ionization_model: Literal["fully_ionized", "discrete_level"] | None = None
    donor_binding_energy_eV: QuantityInput | None = quantity_field("eV", nullable=True)
    acceptor_binding_energy_eV: QuantityInput | None = quantity_field("eV", nullable=True)
    donor_degeneracy: QuantityInput | None = quantity_field("1", positive=True)
    acceptor_degeneracy: QuantityInput | None = quantity_field("1", positive=True)
    band_gap_narrowing_model: Literal["off", "slotboom"] | None = None
    bgn_reference_energy_eV: QuantityInput | None = quantity_field("eV")
    bgn_reference_density_m3: QuantityInput | None = quantity_field("m^-3", positive=True)
    bgn_log_shape: QuantityInput | None = quantity_field("1", positive=True)
    bgn_conduction_band_fraction: QuantityInput | None = quantity_field("1")
    mu_T_gamma: QuantityInput | None = quantity_field("1", signed=True)
    E_a_ion: QuantityInput | None = quantity_field("eV")
    B_rad_T_gamma: QuantityInput | None = quantity_field("1", signed=True)
    varshni_alpha: QuantityInput | None = quantity_field("eV/K", signed=True)
    varshni_beta: QuantityInput | None = quantity_field("K")
    trap_N_t_interface: QuantityInput | None = quantity_field("m^-3", nullable=True)
    trap_N_t_bulk: QuantityInput | None = quantity_field("m^-3", nullable=True)
    trap_decay_length: QuantityInput | None = quantity_field("m", nullable=True)
    trap_profile_shape: Literal["exponential", "gaussian"] | None = None
    trap_edge: Literal["both", "left", "right"] | None = None
    optical_material: str | None = None
    n_optical: QuantityInput | None = quantity_field("1", positive=True, nullable=True)
    incoherent: bool | None = None
    v_sat_n: QuantityInput | None = quantity_field("m/s")
    v_sat_p: QuantityInput | None = quantity_field("m/s")
    ct_beta_n: QuantityInput | None = quantity_field("1")
    ct_beta_p: QuantityInput | None = quantity_field("1")
    pf_gamma_n: QuantityInput | None = quantity_field("(m/V)^0.5")
    pf_gamma_p: QuantityInput | None = quantity_field("(m/V)^0.5")
    Eg_back: QuantityInput | None = quantity_field("eV", nullable=True)
    chi_back: QuantityInput | None = quantity_field("eV", signed=True, nullable=True)
    grading_profile: Literal["linear", "parabolic", "exponential"] | None = None
    grading_direction: Literal["front_to_back", "back_to_front"] | None = None
    grading_bowing: QuantityInput | None = quantity_field("eV", signed=True)
    grading_char_length: QuantityInput | None = quantity_field("m", positive=True, nullable=True)
    grading_N_mult: int | None = Field(default=None, gt=0)
    N_A_bulk: QuantityInput | None = quantity_field("m^-3", nullable=True)
    N_D_bulk: QuantityInput | None = quantity_field("m^-3", nullable=True)
    doping_profile_shape: Literal["gaussian"] | None = None
    doping_decay_length: QuantityInput | None = quantity_field("m", positive=True, nullable=True)
    doping_edge: Literal["front", "back"] | None = None

    @classmethod
    def _allows_null(cls, name: str) -> bool:
        extra = cls.model_fields[name].json_schema_extra
        return bool(isinstance(extra, dict) and extra.get("nullable", False)) or name in {"optical_material", "doping_profile_shape"}

    @field_validator("*")
    @classmethod
    def _check_null(cls, value: object, info: ValidationInfo) -> object:
        name = info.field_name
        assert name is not None
        if value is None and not cls._allows_null(name):
            raise ValueError(f"{name}: omit to inherit; null is not supported")
        if name == "optical_material" and value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError("optical_material requires a nonempty name or null")
        return value


LAYER_OWNED_PARAMETERS = frozenset({
    "N_A", "N_D", "N_A_bulk", "N_D_bulk", "doping_profile_shape", "doping_decay_length", "doping_edge",
    "tau_n", "tau_p", "n1", "p1", "trap_N_t_interface", "trap_N_t_bulk",
    "trap_decay_length", "trap_profile_shape", "trap_edge",
})


def full_parameter_items(items: tuple[tuple[str, Scalar], ...], *, complete: bool = False) -> tuple[tuple[str, Scalar], ...]:
    if len({name for name, _ in items}) != len(items):
        raise ValueError("duplicate full parameter name")
    checked = FullParameterInput.model_validate(dict(items))
    values: dict[str, Any] = dict(checked.normalized_items())
    if complete and set(values) != set(FullParameterInput.model_fields):
        raise ValueError(f"missing full parameter values: {sorted(set(FullParameterInput.model_fields) - set(values))}")
    for density, capacity in (("P0", "P_lim"), ("P0_neg", "P_lim_neg")):
        if {density, capacity} <= values.keys() and values[density] > values[capacity]:
            raise ValueError("ion density exceeds declared site capacity")
    if "bgn_conduction_band_fraction" in values and not 0 <= values["bgn_conduction_band_fraction"] <= 1:
        raise ValueError("BGN conduction fraction must be within [0,1]")
    if values.get("carrier_statistics") == "fermi_dirac" and {"Eg", "Nc300", "Nv300"} <= values.keys() and any(values[name] is None or values[name] <= 0 for name in ("Eg", "Nc300", "Nv300")):
        raise ValueError("Fermi-Dirac data require positive Eg/Nc300/Nv300")
    profile_fields = {"N_A_bulk", "N_D_bulk", "doping_profile_shape", "doping_decay_length"}
    profile = values.get("N_A_bulk") is not None or values.get("N_D_bulk") is not None
    if profile_fields <= values.keys() and profile and (values["doping_profile_shape"] != "gaussian" or values["doping_decay_length"] is None):
        raise ValueError("spatial doping requires gaussian shape and positive decay length")
    if profile_fields <= values.keys() and not profile and (values["doping_profile_shape"] is not None or values["doping_decay_length"] is not None):
        raise ValueError("orphaned spatial doping profile")
    if values.get("dopant_ionization_model") == "discrete_level":
        for carrier, density in (("donor", "N_D"), ("acceptor", "N_A")):
            required = {density, density + "_bulk", carrier + "_binding_energy_eV"}
            if required <= values.keys() and max(values[density], values[density + "_bulk"] or 0) > 0 and values[carrier + "_binding_energy_eV"] is None:
                raise ValueError(f"active discrete {carrier} requires binding energy")
    return tuple(sorted(values.items()))
