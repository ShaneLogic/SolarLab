"""The two actual scalar parameter schemas accepted by metadata selection."""

from __future__ import annotations

from typing import Any

from solarlab.materials.full_parameters import FullParameterInput, full_parameter_items
from solarlab.materials.library import validated_parameters
from solarlab.materials.parameters import LayerValuesInput, Scalar

__all__ = ["parameter_names", "parameter_schema", "parameter_values"]


def _model(name: str) -> Any:
    if name == "scalar_layer":
        return LayerValuesInput
    if name == "full_layer":
        return FullParameterInput
    raise ValueError(f"unsupported parameter schema: {name!r}")


def parameter_names(name: str) -> frozenset[str]:
    return frozenset(_model(name).model_fields)


def parameter_schema(name: str) -> dict[str, Any]:
    return _model(name).model_json_schema()


def parameter_values(values: tuple[tuple[str, Scalar], ...], schema: str) -> tuple[tuple[str, Scalar], ...]:
    _model(schema)
    if schema == "scalar_layer":
        return validated_parameters(values)
    return full_parameter_items(values)
