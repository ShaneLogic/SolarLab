"""Four independent tunnelling declarations, with no device execution claim.

The current resolved-device contract covers only an isolated electron intraband
barrier. A valid declaration of the other channels does not qualify coupling.
"""

from __future__ import annotations

from typing import Any, Final, Literal, Mapping

from pydantic import Field, ValidationInfo, field_validator, model_validator

from solarlab.materials.full_parameters import StructuredInput, quantity_field as q
from solarlab.materials.parameters import QuantityInput, Scalar
from solarlab.units import normalize_quantity

TUNNELLING_VERSION: Final = "solarlab-wkb-tunnelling-channels-v1"


class ChannelInput(StructuredInput):
    enabled: bool | None = None

    @classmethod
    def _allows_null(cls, name: str) -> bool:
        return False

    @field_validator("*")
    @classmethod
    def _no_null(cls, value: object, info: ValidationInfo) -> object:
        if value is None:
            raise ValueError(f"{info.field_name}: omit to inherit; null is unsupported")
        return value


class BandToBandInput(ChannelInput):
    reduced_effective_mass_rel: QuantityInput | None = q("1", positive=True)
    energy_quadrature_order: int | None = Field(default=None, ge=4)
    minimum_field_V_m: QuantityInput | None = q("V/m", positive=True)


class IntrabandInput(ChannelInput):
    electron_effective_mass_rel: QuantityInput | None = q("1", positive=True)
    hole_effective_mass_rel: QuantityInput | None = q("1", positive=True)
    carrier: Literal["electron", "hole", "both"] | None = None
    energy_quadrature_order: int | None = Field(default=None, ge=4)


class DefectAssistedInput(ChannelInput):
    electron_effective_mass_rel: QuantityInput | None = q("1", positive=True)
    hole_effective_mass_rel: QuantityInput | None = q("1", positive=True)
    requires_explicit_occupancy: Literal[True] | None = None

    @field_validator("requires_explicit_occupancy", mode="before")
    @classmethod
    def _explicit_occupancy(cls, value: object) -> object:
        if value is not True:
            raise ValueError("requires_explicit_occupancy must be boolean true")
        return value


class ContactTunnellingInput(ChannelInput):
    electron_effective_mass_rel: QuantityInput | None = q("1", positive=True)
    hole_effective_mass_rel: QuantityInput | None = q("1", positive=True)
    barrier_height_eV: QuantityInput | None = q("eV")
    side: Literal["left", "right", "both"] | None = None
    energy_quadrature_order: int | None = Field(default=None, ge=4)

    @model_validator(mode="after")
    def _barrier(self) -> ContactTunnellingInput:
        if self.enabled and self.barrier_height_eV is not None and normalize_quantity(self.barrier_height_eV, "eV") <= 0:
            raise ValueError("enabled contact tunnelling requires a positive barrier")
        return self


CHANNEL_TYPES = {
    "band_to_band": BandToBandInput,
    "intraband": IntrabandInput,
    "interface_defect_assisted": DefectAssistedInput,
    "contact": ContactTunnellingInput,
}


class TunnellingInput(StructuredInput):
    schema_version: Literal["solarlab-wkb-tunnelling-channels-v1"] = TUNNELLING_VERSION
    band_to_band: BandToBandInput | None = None
    intraband: IntrabandInput | None = None
    interface_defect_assisted: DefectAssistedInput | None = None
    contact: ContactTunnellingInput | None = None

    def resolved_data(self, defaults: Mapping[str, Mapping[str, Scalar]]) -> dict[str, Any]:
        checked = type(self).model_validate(self)
        result: dict[str, Any] = {"schema_version": self.schema_version}
        for name, model in CHANNEL_TYPES.items():
            if name not in defaults:
                raise ValueError(f"missing supplied tunnelling default data: {name}")
            channel = getattr(checked, name)
            data = {**defaults[name], **(channel.normalized_data() if channel is not None else {})}
            if set(data) != set(model.model_fields):
                raise ValueError(f"incomplete or unknown resolved channel fields: {name}")
            result[name] = model.model_validate(data).normalized_data()
        return result


def validate_resolved_tunnelling(data: dict[str, Any]) -> dict[str, Any]:
    """Public copied/resolved payloads cannot bypass complete-field validation."""
    checked = TunnellingInput.model_validate(data)
    if set(data) != {"schema_version", *CHANNEL_TYPES}:
        raise ValueError("resolved tunnelling document requires all four channels")
    for name, model in CHANNEL_TYPES.items():
        channel = getattr(checked, name)
        if channel is None or channel.model_fields_set != set(model.model_fields):
            raise ValueError(f"resolved tunnelling channel is incomplete: {name}")
    return checked.normalized_data()
