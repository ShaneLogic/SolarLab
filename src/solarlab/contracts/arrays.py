"""Owned immutable arrays and the explicit physical-array precision boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.contracts.errors import ContractError

__all__ = [
    "FloatArray", "ImmutableArrays", "PhysicalArray", "Vector",
    "frozen_array", "integer_indices",
]

Vector: TypeAlias = NDArray[np.float64]


def frozen_array(value: ArrayLike) -> Vector:
    """Own immutable bytes, including against setflags(write=True) by callers."""
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ContractError("nonfinite_array")
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


def integer_indices(value: ArrayLike) -> NDArray[np.intp]:
    """Reject fractional/overflowing connectivity before converting its dtype."""
    raw = np.asarray(value)
    if raw.dtype.kind not in {"i", "u", "f"} or not np.isfinite(raw).all():
        raise ContractError("invalid_index")
    if raw.size:
        limits = np.iinfo(np.intp)
        if raw.dtype.kind == "f":
            invalid = (raw != np.floor(raw)) | (raw < limits.min) | (raw >= -limits.min)
            if invalid.any():
                raise ContractError("invalid_index")
        elif int(raw.min()) < limits.min or int(raw.max()) > limits.max:
            raise ContractError("invalid_index")
    return np.asarray(raw, dtype=np.intp)


class ImmutableArrays:
    """Expose fresh array views so callers cannot mutate owned shape or dtype.

    Frozen bytes protect values; a separate view header also protects metadata.
    Subclasses freeze their stored arrays during construction.
    """

    def __getattribute__(self, name: str) -> Any:
        value = object.__getattribute__(self, name)
        return value.view() if isinstance(value, np.ndarray) else value


class PhysicalArray(Protocol):
    """Explicit precision boundary; no implicit conversion to a float array."""

    @property
    def shape(self) -> tuple[int, ...]: ...
    def immutable_copy(self) -> PhysicalArray: ...
    def difference(self, other: PhysicalArray) -> PhysicalArray: ...
    def take(self, indices: ArrayLike) -> PhysicalArray: ...
    def log_ratio(self, other: PhysicalArray) -> PhysicalArray: ...
    def is_finite(self) -> bool: ...
    def identity_bytes(self) -> bytes: ...


@dataclass(frozen=True)
class FloatArray(ImmutableArrays):
    """The binary64 physical array; richer precision lives behind the Protocol."""

    values: Vector

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", frozen_array(self.values))

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.values.shape)

    def immutable_copy(self) -> FloatArray:
        return FloatArray(self.values)

    def difference(self, other: PhysicalArray) -> FloatArray:
        if not isinstance(other, FloatArray) or other.shape != self.shape:
            raise ContractError("precision_or_shape_mismatch")
        return FloatArray(self.values - other.values)

    def take(self, indices: ArrayLike) -> FloatArray:
        selected = integer_indices(indices)
        if np.any(selected < 0) or np.any(selected >= self.shape[0]):
            raise ContractError("index_outside_support")
        return FloatArray(self.values[selected])

    def log_ratio(self, other: PhysicalArray) -> FloatArray:
        if not isinstance(other, FloatArray) or other.shape != self.shape:
            raise ContractError("precision_or_shape_mismatch")
        if np.any(self.values <= 0) or np.any(other.values <= 0):
            raise ContractError("log_ratio_requires_positive_fields")
        relative = (self.values - other.values) / other.values
        close = np.abs(relative) < 0.5
        result = np.log(self.values) - np.log(other.values)
        result[close] = np.log1p(relative[close])
        return FloatArray(result)

    def is_finite(self) -> bool:
        return bool(np.isfinite(self.values).all())

    def identity_bytes(self) -> bytes:
        return b"float64:" + repr(self.shape).encode() + self.values.tobytes()
