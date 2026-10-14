"""Array operations supplied by a precision backend, with no physics imports."""

from __future__ import annotations

from typing import Any, Iterable, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.contracts.arrays import FloatArray, PhysicalArray, Vector, frozen_array
from solarlab.contracts.errors import ContractError

__all__ = ["AssemblyArithmetic", "FloatArithmetic"]


class AssemblyArithmetic(Protocol):
    """Array operations preserving one representation.

    ``array`` checks finite values; ``freeze`` returns owned immutable storage.
    Geometry and support validation remain the responsibility of TermSink.
    The precision linear-form hooks of the P02-02 prototype are pending G2-R
    (see :mod:`solarlab.contracts.pending`); a richer backend may implement a
    structural superset of this protocol.
    """

    def array(self, value: ArrayLike | PhysicalArray) -> Any: ...
    def zeros(self, shape: tuple[int, ...]) -> Any: ...
    def weighted(self, value: Any, weights: Vector) -> Any: ...
    def add(self, left: Any, right: Any) -> Any: ...
    def subtract(self, left: Any, right: Any) -> Any: ...
    def divide(self, left: Any, right: Any) -> Any: ...
    def matmul(self, left: Any, right: Any) -> Any: ...
    def concatenate(self, values: Iterable[Any], axis: int = 0) -> Any: ...
    def reshape(self, value: Any, shape: tuple[int, ...]) -> Any: ...
    def scatter_add(self, target: Any, indices: NDArray[np.intp], value: Any) -> Any: ...
    def freeze(self, value: Any) -> Any: ...


class FloatArithmetic:
    """The product binary64 implementation of the assembly arithmetic contract."""

    def array(self, value: ArrayLike | PhysicalArray) -> Vector:
        if isinstance(value, FloatArray):
            value = value.values
        return frozen_array(value)  # type: ignore[arg-type]

    def zeros(self, shape: tuple[int, ...]) -> Vector:
        return np.zeros(shape)

    def weighted(self, value: Vector, weights: Vector) -> Vector:
        if value.shape != weights.shape:
            raise ContractError("arithmetic_shape_mismatch")
        return value * weights

    def add(self, left: Vector, right: Vector) -> Vector:
        if left.shape != right.shape:
            raise ContractError("arithmetic_shape_mismatch")
        return left + right

    def subtract(self, left: Vector, right: Vector) -> Vector:
        if left.shape != right.shape:
            raise ContractError("arithmetic_shape_mismatch")
        return left - right

    def divide(self, left: Vector, right: Vector) -> Vector:
        if left.shape != right.shape or np.any(right == 0):
            raise ContractError("arithmetic_shape_or_divisor")
        return left / right

    def matmul(self, left: Vector, right: Vector) -> Vector:
        if left.ndim != 2 or right.ndim not in {1, 2} or left.shape[1] != right.shape[0]:
            raise ContractError("arithmetic_shape_mismatch")
        return left @ right

    def concatenate(self, values: Iterable[Vector], axis: int = 0) -> Vector:
        return np.concatenate(tuple(values), axis=axis)

    def reshape(self, value: Vector, shape: tuple[int, ...]) -> Vector:
        return value.reshape(shape)

    def scatter_add(self, target: Vector, indices: NDArray[np.intp], value: Vector) -> Vector:
        np.add.at(target, indices, value)
        return target

    def freeze(self, value: Vector) -> Vector:
        return frozen_array(value)
