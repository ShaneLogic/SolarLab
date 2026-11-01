"""The editable layer-template document used by configuration preparation."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field, field_validator

from solarlab.materials.parameters import EditableInput, LayerValuesInput

__all__ = ["LayerInput"]

_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_.:-]*\Z")


class LayerInput(EditableInput):
    """A named layer with a stable ID and explicitly supplied overrides."""

    schema_version: Literal[2] = 2
    kind: Literal["layer_template"] = "layer_template"
    id: str
    name: str
    template: str
    overrides: LayerValuesInput = Field(default_factory=LayerValuesInput)

    @field_validator("overrides", mode="before")
    @classmethod
    def _typed_overrides(cls, value: object) -> object:
        if isinstance(value, LayerValuesInput):
            checked = LayerValuesInput.model_validate(value)
            return checked.model_dump(exclude_unset=True, round_trip=True)
        return value

    @field_validator("schema_version", mode="before")
    @classmethod
    def _version(cls, value: object) -> object:
        if type(value) is not int or value != 2:
            raise ValueError("schema_version requires the integer 2")
        return value

    @field_validator("id")
    @classmethod
    def _identifier(cls, value: str) -> str:
        if _ID.fullmatch(value) is None:
            raise ValueError("id must be a stable nonempty identifier")
        return value

    @field_validator("name", "template")
    @classmethod
    def _nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a nonempty string")
        return value
