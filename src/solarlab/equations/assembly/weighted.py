"""Contribution expressions that carry the geometric measure symbolically.

A Term never sees control-volume measures. It writes each residual value or
local tangent as an ordinary arithmetic expression in which the symbol
:data:`MEASURE` stands for the measure of the cell, face or port it is
assembled on, for example ``-Q * MEASURE * (n + p) / thermal_voltage``. The
Assembler substitutes the actual measure array and evaluates the expression
tree in the order Python built it, so the rounding sequence of an existing
expression is reproduced exactly while the measure value stays owned by the
Assembler.

:func:`measure_count` lets the Assembler require exactly one measure factor
in every weighted contribution and none in a constraint. Sums, differences
and measures in a denominator are rejected when the expression is built:
inside a contribution the measure may only scale.
"""

from __future__ import annotations

from typing import Any, TypeAlias, Union

import numpy as np
from numpy.typing import NDArray

from solarlab.contracts.errors import ContractError

__all__ = ["MEASURE", "Weighted", "evaluate", "measure_count"]


class Weighted:
    """Node of a product/quotient expression tree; NumPy defers to it."""

    __array_ufunc__ = None
    __slots__ = ()

    def __mul__(self, other: Any) -> Weighted:
        return _Binary("*", self, _leaf(other))

    def __rmul__(self, other: Any) -> Weighted:
        return _Binary("*", _leaf(other), self)

    def __truediv__(self, other: Any) -> Weighted:
        divisor = _leaf(other)
        if measure_count(divisor):
            raise ContractError("measure_in_denominator")
        return _Binary("/", self, divisor)

    def __rtruediv__(self, other: Any) -> Weighted:
        raise ContractError("measure_in_denominator")

    def __neg__(self) -> Weighted:
        return _Negative(self)

    def __add__(self, other: Any) -> Weighted:
        raise ContractError("weighted_sum_not_allowed")

    __radd__ = __sub__ = __rsub__ = __add__

    def __bool__(self) -> bool:
        raise ContractError("weighted_has_no_truth_value")


class _Measure(Weighted):
    __slots__ = ()

    def __repr__(self) -> str:
        return "MEASURE"


class _Leaf(Weighted):
    __slots__ = ("value",)

    def __init__(self, value: Any):
        array = np.asarray(value)
        if array.dtype.kind not in "fiu" or array.ndim > 1:
            raise ContractError("weighted_operand_must_be_real_scalar_or_vector")
        self.value = value

    def __repr__(self) -> str:
        shape = np.shape(self.value)
        return f"<{shape}>" if shape else repr(self.value)


class _Binary(Weighted):
    __slots__ = ("operator", "left", "right")

    def __init__(self, operator: str, left: Weighted, right: Weighted):
        self.operator = operator
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"({self.left!r} {self.operator} {self.right!r})"


class _Negative(Weighted):
    __slots__ = ("operand",)

    def __init__(self, operand: Weighted):
        self.operand = operand

    def __repr__(self) -> str:
        return f"-{self.operand!r}"


#: The control-volume measure of the row (cell) or face being assembled.
MEASURE: Weighted = _Measure()

Expression: TypeAlias = Union[Weighted, float, NDArray[np.float64]]


def _leaf(value: Any) -> Weighted:
    return value if isinstance(value, Weighted) else _Leaf(value)


def measure_count(expression: Expression) -> int:
    """Number of measure factors in an expression (plain values have none)."""
    if isinstance(expression, _Measure):
        return 1
    if isinstance(expression, _Binary):
        return measure_count(expression.left) + measure_count(expression.right)
    if isinstance(expression, _Negative):
        return measure_count(expression.operand)
    return 0


def evaluate(expression: Expression, measure: NDArray[np.float64]) -> NDArray[np.float64]:
    """Values of ``expression`` with :data:`MEASURE` bound to ``measure``.

    The result has the measure's shape; operands must broadcast to it.
    """
    def walk(node: Any) -> Any:
        if isinstance(node, _Measure):
            return measure
        if isinstance(node, _Leaf):
            return node.value
        if isinstance(node, _Binary):
            left, right = walk(node.left), walk(node.right)
            return left * right if node.operator == "*" else left / right
        if isinstance(node, _Negative):
            return -walk(node.operand)
        return node

    with np.errstate(all="ignore"):
        value = np.asarray(walk(expression), dtype=float)
    if value.shape != measure.shape:
        value = np.broadcast_to(value, measure.shape)
    return value
