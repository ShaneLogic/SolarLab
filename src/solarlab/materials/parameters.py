"""Editable scalar fields consumed by the existing layer-template library.

None defaults here mark an absent override, not a physical default. Actual
defaults remain in the supplied template document. This schema does not claim
to resolve the full legacy MaterialParams or nested defect/2D configurations.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Any, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, GetJsonSchemaHandler, ValidationInfo, field_validator
from pydantic.fields import FieldInfo
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema

from solarlab.units import normalize_quantity

__all__ = ["EditableInput", "LayerValuesInput", "Scalar", "parameter_units"]

Scalar: TypeAlias = float | bool | str | None
QuantityInput: TypeAlias = int | float | Decimal | str


class EditableInput(BaseModel):
    """Input editing always reconstructs and validates complete values."""

    model_config = ConfigDict(
        strict=True, extra="forbid", allow_inf_nan=False,
        validate_assignment=True, revalidate_instances="always",
    )

    @classmethod
    def model_validate(cls, obj: Any, **kwargs: Any) -> Self:
        # Revalidating an editable instance must not turn absent placeholders
        # into explicit nulls. Preserve exactly its supplied editing fields.
        if isinstance(obj, cls):
            if set(obj.__dict__) - set(cls.model_fields):
                raise ValueError("unknown input fields")
            obj = obj.editing_data()
        return super().model_validate(obj, **kwargs)

    def editing_data(self) -> dict[str, Any]:
        """Keep child edits even if the containing object began as a default."""
        def edited(value: Any) -> Any:
            if isinstance(value, EditableInput):
                return value.editing_data()
            if isinstance(value, (tuple, list)):
                return tuple(edited(item) for item in value)
            return value

        data = self.model_dump(exclude_unset=True, round_trip=True)
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, EditableInput):
                nested = value.editing_data()
                if nested or name in self.model_fields_set:
                    data[name] = nested
            elif isinstance(value, (tuple, list)) and name in self.model_fields_set:
                data[name] = edited(value)
        return data

    def validated_update(self, updates: Mapping[str, object]) -> Self:
        data = self.editing_data()
        data.update(updates)
        return type(self).model_validate(data)

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False,
    ) -> Self:
        """Unlike BaseModel.model_copy, this editing path revalidates updates."""
        return self.validated_update(update or {})


def _quantity(
    unit: str, *, positive: bool = False, nonnegative: bool = True,
) -> FieldInfo:
    return Field(default=None, json_schema_extra={
        "unit": unit, "positive": positive, "nonnegative": nonnegative,
    })


class LayerValuesInput(EditableInput):
    """Explicit scalar overrides; absence inherits, only optical null clears."""

    thickness: QuantityInput | None = _quantity("m", positive=True)
    eps_r: QuantityInput | None = _quantity("1", positive=True)
    mu_n: QuantityInput | None = _quantity("m^2/(V s)")
    mu_p: QuantityInput | None = _quantity("m^2/(V s)")
    D_ion: QuantityInput | None = _quantity("m^2/s")
    P_lim: QuantityInput | None = _quantity("m^-3")
    P0: QuantityInput | None = _quantity("m^-3")
    ni: QuantityInput | None = _quantity("m^-3")
    tau_n: QuantityInput | None = _quantity("s", positive=True)
    tau_p: QuantityInput | None = _quantity("s", positive=True)
    n1: QuantityInput | None = _quantity("m^-3")
    p1: QuantityInput | None = _quantity("m^-3")
    B_rad: QuantityInput | None = _quantity("m^3/s")
    C_n: QuantityInput | None = _quantity("m^6/s")
    C_p: QuantityInput | None = _quantity("m^6/s")
    alpha: QuantityInput | None = _quantity("m^-1")
    N_A: QuantityInput | None = _quantity("m^-3")
    N_D: QuantityInput | None = _quantity("m^-3")
    chi: QuantityInput | None = _quantity("eV", nonnegative=False)
    Eg: QuantityInput | None = _quantity("eV")
    incoherent: bool | None = None
    optical_material: str | None = None

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        schema = handler(core_schema)
        for name, definition in schema.get("properties", {}).items():
            # Internal absent placeholders are not schema defaults. Only the
            # explicitly nullable optical source accepts a JSON null value.
            definition.pop("default", None)
            if name != "optical_material" and "anyOf" in definition:
                definition["anyOf"] = [choice for choice in definition["anyOf"] if choice.get("type") != "null"]
        return schema

    @field_validator("*", mode="before")
    @classmethod
    def _validate_value(cls, value: object, info: ValidationInfo) -> object:
        name = info.field_name
        assert name is not None
        if name == "optical_material":
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError("optical_material must be a nonempty string or null")
            return value
        if value is None:
            raise ValueError(f"{name}: null is not a numeric/boolean value; omit it to inherit")
        if name == "incoherent":
            if type(value) is not bool:
                raise ValueError("incoherent requires a boolean")
            return value
        extra = cls.model_fields[name].json_schema_extra
        assert isinstance(extra, dict)
        normalized = normalize_quantity(value, str(extra["unit"]), path=name)
        if extra["positive"] and normalized <= 0:
            raise ValueError(f"{name}: must be positive")
        if extra["nonnegative"] and normalized < 0:
            raise ValueError(f"{name}: must be nonnegative")
        return value

    def normalized_items(self, *, complete: bool = False) -> tuple[tuple[str, Scalar], ...]:
        """Return owned scalar data after checking even constructed instances."""
        unexpected = set(self.__dict__) - set(type(self).model_fields)
        if unexpected:
            raise ValueError(f"unknown parameter keys: {sorted(unexpected)}")
        data = self.model_dump(exclude_unset=True, round_trip=True)
        checked = type(self).model_validate(data)
        if complete:
            missing = set(type(self).model_fields) - checked.model_fields_set
            if missing:
                raise ValueError(f"missing template parameters: {sorted(missing)}")
        result: list[tuple[str, Scalar]] = []
        for name in sorted(checked.model_fields_set):
            value = getattr(checked, name)
            if name not in {"optical_material", "incoherent"}:
                extra = type(self).model_fields[name].json_schema_extra
                assert isinstance(extra, dict)
                value = normalize_quantity(value, str(extra["unit"]), path=name)
            result.append((name, value))
        return tuple(result)


def parameter_units() -> tuple[tuple[str, str], ...]:
    """Units derive from the same field declarations used by input validation."""
    return tuple(
        (name, str(field.json_schema_extra["unit"]))
        for name, field in sorted(LayerValuesInput.model_fields.items())
        if isinstance(field.json_schema_extra, dict) and "unit" in field.json_schema_extra
    )
