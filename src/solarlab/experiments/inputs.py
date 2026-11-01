"""Strict editable experiment primitives shared by declared preparation routes."""
from __future__ import annotations

import json
from typing import Any, ClassVar, NoReturn
from pydantic import Field, ValidationError, ValidationInfo, field_validator
from solarlab.materials.full_parameters import StructuredInput
from solarlab.units import normalize_quantity


def vector(unit: str, *, required: bool = True, minimum: int = 0,
           input_unit: str | None = None, positive: bool = False) -> Any:
    return Field(default=... if required else None, min_length=minimum,
                 json_schema_extra={"item_unit": unit, "item_input_unit": input_unit,
                                    "item_positive": positive})


def invalid(name: str, path: tuple[str | int, ...], message: str, value: object) -> NoReturn:
    raise ValidationError.from_exception_data(name, [{"type": "value_error", "loc": path,
        "input": value, "ctx": {"error": ValueError(message)}}])


class ExperimentInput(StructuredInput):
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


def input_document(raw: bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        if len({key for key, _ in items}) != len(items):
            raise ValueError("duplicate experiment JSON declaration field")
        return dict(items)
    return json.loads(raw, object_pairs_hook=pairs, parse_int=lambda word: -0.0 if word == "-0" else int(word))
