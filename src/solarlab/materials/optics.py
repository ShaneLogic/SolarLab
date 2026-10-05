"""Composition optical declarations; no dielectric or optical calculation."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, ValidationInfo, field_validator, model_validator

from solarlab.materials.full_parameters import StructuredInput, quantity_field as q
from solarlab.materials.parameters import QuantityInput
from solarlab.units import normalize_quantity


class CigsOpticsInput(StructuredInput):
    """GGI endpoints follow the layer's electrical grading coordinate.

Absent model, slice and quadrature values are supplied by the injected catalog,
compiled from the existing model declaration. None is never a numeric default.
"""

    ggi_front: QuantityInput = q("1", required=True)
    ggi_back: QuantityInput = q("1", required=True)
    cgi: QuantityInput = q("1", required=True)
    model: Literal["minoura_2015"] | None = None
    slices: int | None = Field(default=None, ge=1, le=512)
    kk_quadrature_order: int | None = Field(default=None, ge=48, le=2048)

    @classmethod
    def _allows_null(cls, name: str) -> bool:
        return False

    @field_validator("model", "slices", "kk_quadrature_order")
    @classmethod
    def _no_null(cls, value: object, info: ValidationInfo) -> object:
        if value is None:
            raise ValueError(f"{info.field_name}: omit to inherit; null is unsupported")
        return value

    @model_validator(mode="after")
    def _composition(self) -> CigsOpticsInput:
        if any(normalize_quantity(value, "1") > 1 for value in (self.ggi_front, self.ggi_back, self.cgi)):
            raise ValueError("CIGS composition fractions cannot exceed 1")
        if normalize_quantity(self.cgi, "1") < 0.75:
            raise ValueError("cgi must lie in [0.75,1]")
        return self
