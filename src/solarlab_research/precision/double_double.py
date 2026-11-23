"""Immutable double-double (DD) compensated arithmetic and public contract adapters.

Retains ~106 bits of significand precision via normalized non-overlapping
IEEE-754 binary64 pairs ``(hi, lo)`` satisfying ``fl(hi + lo) == hi``.
Injects into ``solarlab.contracts`` exclusively through ``PhysicalArray`` and
``AssemblyArithmetic`` without registering physics models.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
import operator
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.contracts import (
    AssemblyArithmetic,
    ContractError,
    FieldProjection,
    FloatArray,
    PhysicalArray,
    Vector,
    frozen_array,
    integer_indices,
)

__all__ = [
    "CompensatedArithmetic",
    "DD",
    "DDArray",
    "exp",
    "expm1",
    "log",
    "log1p",
    "sum",
]

_SPLIT_MAX = float(np.ldexp(1.0, 996))
_SPLIT_SCALE_DOWN = float(np.ldexp(1.0, -28))
_SPLIT_SCALE_UP = float(np.ldexp(1.0, 28))
_SAFE_MAX = float(np.ldexp(1.0, 1020))
_SPLITTER = float(np.ldexp(1.0, 27) + 1.0)
_SERIES_EPS = float(np.ldexp(1.0, -110))


def _readonly_float64(value: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


def _require_supported_input(value: object) -> NDArray[np.float64]:
    if isinstance(value, DD):
        raise TypeError("nested DD inputs are ambiguous; add or copy explicitly")
    raw = np.asarray(value)
    if raw.dtype.kind == "c" or raw.dtype == object:
        raise TypeError("DD requires finite real binary64-compatible inputs")
    array = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("DD requires finite inputs")
    if raw.dtype.kind in ("i", "u") and raw.size:
        exact = [int(item) for item in raw.flat]
        back = [int(item) for item in array.flat]
        if exact != back:
            raise TypeError("integer exceeds exact IEEE-754 binary64 precision")
    if np.any(np.abs(array) > _SAFE_MAX):
        raise OverflowError("DD input exceeds the safe compensated range")
    return array


def _two_sum(
    left: NDArray[np.float64], right: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        high = np.asarray(left + right, dtype=np.float64)
        virtual = high - left
        low = np.asarray((left - (high - virtual)) + (right - virtual), dtype=np.float64)
    return high, low


def _quick_two_sum(
    left: NDArray[np.float64], right: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        high = np.asarray(left + right, dtype=np.float64)
        low = np.asarray(right - (high - left), dtype=np.float64)
    return high, low


def _split(value: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    large = np.abs(value) > _SPLIT_MAX
    work = np.where(large, value * _SPLIT_SCALE_DOWN, value)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        temp = _SPLITTER * work
        high = temp - (temp - work)
        low = work - high
    if np.any(large):
        high = np.where(large, high * _SPLIT_SCALE_UP, high)
        low = np.where(large, low * _SPLIT_SCALE_UP, low)
    return np.asarray(high, dtype=np.float64), np.asarray(low, dtype=np.float64)


def _two_product(
    left: NDArray[np.float64], right: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    left_arr, right_arr = np.broadcast_arrays(left, right)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        product = np.asarray(left_arr * right_arr, dtype=np.float64)
    if np.any((left_arr != 0.0) & (right_arr != 0.0) & (product == 0.0)):
        raise ArithmeticError("DD product underflow")
    lh, ll = _split(left_arr)
    rh, rl = _split(right_arr)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        error = np.asarray(
            ((lh * rh - product) + lh * rl + ll * rh) + ll * rl,
            dtype=np.float64,
        )
    return product, error


class DD:
    """Immutable compensated double-double scalar or ndarray."""

    __slots__ = ("_hi", "_lo")
    __array_ufunc__ = None

    _hi: NDArray[np.float64]
    _lo: NDArray[np.float64]

    def __init__(self, hi: object, lo: object | None = None) -> None:
        high = _require_supported_input(hi)
        low = np.zeros_like(high) if lo is None else _require_supported_input(lo)
        high_b, low_b = np.broadcast_arrays(high, low)
        norm_hi, norm_lo = _two_sum(high_b, low_b)
        norm_hi, norm_lo = _quick_two_sum(norm_hi, norm_lo)
        if not (np.all(np.isfinite(norm_hi)) and np.all(np.isfinite(norm_lo))):
            raise ArithmeticError("DD normalization produced nonfinite words")
        if np.any(np.abs(norm_hi) > _SAFE_MAX):
            raise OverflowError("DD normalized value exceeds safe compensated range")
        object.__setattr__(self, "_hi", _readonly_float64(norm_hi))
        object.__setattr__(self, "_lo", _readonly_float64(norm_lo))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("DD instances are immutable")

    @classmethod
    def from_normalized_parts(
        cls, hi: NDArray[np.float64], lo: NDArray[np.float64]
    ) -> DD:
        """Wrap already validated, normalized finite parts without re-rounding."""
        instance = object.__new__(cls)
        object.__setattr__(instance, "_hi", _readonly_float64(hi))
        object.__setattr__(instance, "_lo", _readonly_float64(lo))
        return instance

    @classmethod
    def _normalized(cls, hi: NDArray[np.float64], lo: NDArray[np.float64]) -> DD:
        norm_hi, norm_lo = _two_sum(hi, lo)
        norm_hi, norm_lo = _quick_two_sum(norm_hi, norm_lo)
        if not (np.all(np.isfinite(norm_hi)) and np.all(np.isfinite(norm_lo))):
            raise ArithmeticError("DD operation produced nonfinite words")
        if np.any(np.abs(norm_hi) > _SAFE_MAX):
            raise OverflowError("DD operation exceeded safe compensated range")
        return cls.from_normalized_parts(norm_hi, norm_lo)

    @property
    def hi(self) -> NDArray[np.float64]:
        return self._hi.view()

    @property
    def lo(self) -> NDArray[np.float64]:
        return self._lo.view()

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._hi.shape)

    @property
    def ndim(self) -> int:
        return int(self._hi.ndim)

    @property
    def size(self) -> int:
        return int(self._hi.size)

    def copy(self) -> DD:
        return self

    def __deepcopy__(self, memo: object) -> DD:
        return self

    def __len__(self) -> int:
        return len(self._hi)

    def __getitem__(self, index: Any) -> DD:
        return self.from_normalized_parts(
            np.asarray(self._hi[index], dtype=np.float64),
            np.asarray(self._lo[index], dtype=np.float64),
        )

    def reshape(self, *shape: Any) -> DD:
        target = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        return self.from_normalized_parts(self._hi.reshape(target), self._lo.reshape(target))

    def ravel(self) -> DD:
        return self.from_normalized_parts(self._hi.ravel(), self._lo.ravel())

    def to_float(self) -> float | NDArray[np.float64]:
        if self._hi.ndim == 0:
            return float(self._hi.item())
        return _readonly_float64(self._hi)

    def __float__(self) -> float:
        if self._hi.size != 1:
            raise TypeError("only scalar DD instances convert to Python float")
        return float(self._hi.item())

    def to_pair_dict(self) -> dict[str, Any]:
        if self._hi.ndim == 0:
            return {"hi": float(self._hi.item()), "lo": float(self._lo.item())}
        return {"hi": self._hi.tolist(), "lo": self._lo.tolist()}

    @classmethod
    def from_pair_dict(cls, pair: Mapping[str, Any], *, expected_shape: tuple[int, ...] | None = None) -> DD:
        if set(pair) != {"hi", "lo"}:
            raise ValueError("DD pair dictionary must contain exact keys {'hi', 'lo'}")
        hi_arr = _require_supported_input(pair["hi"])
        lo_arr = _require_supported_input(pair["lo"])
        if hi_arr.shape != lo_arr.shape:
            raise ValueError("DD pair high and low shapes disagree")
        if expected_shape is not None and hi_arr.shape != expected_shape:
            raise ValueError(f"DD pair shape {hi_arr.shape} differs from expected {expected_shape}")
        normalized = cls(hi_arr, lo_arr)
        if not (np.array_equal(normalized.hi, hi_arr) and np.array_equal(normalized.lo, lo_arr)):
            raise ValueError("serialized DD pair words are not normalized")
        return cls.from_normalized_parts(hi_arr, lo_arr)

    def __repr__(self) -> str:
        return f"DD(hi={self._hi!r}, lo={self._lo!r})"

    def __bool__(self) -> bool:
        if self._hi.size != 1:
            raise ValueError("The truth value of a multi-element DD is ambiguous")
        return bool(self._hi.item() != 0.0 or self._lo.item() != 0.0)

    def __neg__(self) -> DD:
        return self.from_normalized_parts(-self._hi, -self._lo)

    def __pos__(self) -> DD:
        return self

    def __abs__(self) -> DD:
        negative = self._hi < 0.0
        return self.from_normalized_parts(
            np.where(negative, -self._hi, self._hi),
            np.where(negative, -self._lo, self._lo),
        )

    def __add__(self, other: object) -> DD:
        rhs = _as_dd(other)
        high, low = _two_sum(self._hi, rhs._hi)
        tail, tail_error = _two_sum(self._lo, rhs._lo)
        high, low = _two_sum(high, low + tail)
        high, low = _two_sum(high, low + tail_error)
        return self._normalized(high, low)

    __radd__ = __add__

    def __sub__(self, other: object) -> DD:
        return self + (-_as_dd(other))

    def __rsub__(self, other: object) -> DD:
        return _as_dd(other) + (-self)

    def __mul__(self, other: object) -> DD:
        rhs = _as_dd(other)
        high, low = _two_product(self._hi, rhs._hi)
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            cross = self._hi * rhs._lo + self._lo * rhs._hi
            low = low + cross
            high, low = _two_sum(high, low)
            high, low = _two_sum(high, low + self._lo * rhs._lo)
        return self._normalized(high, low)

    __rmul__ = __mul__

    def __truediv__(self, other: object) -> DD:
        rhs = _as_dd(other)
        if np.any(rhs._hi == 0.0):
            raise ZeroDivisionError("DD division by zero")
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            q1 = self._hi / rhs._hi
        if np.any((self._hi != 0.0) & (q1 == 0.0)):
            raise ArithmeticError("DD quotient underflow")
        first = DD(q1)
        remainder = self - rhs * first
        q2 = remainder._hi / rhs._hi
        second = DD(q2)
        remainder = remainder - rhs * second
        q3 = remainder._hi / rhs._hi
        return (first + second) + DD(q3)

    def __rtruediv__(self, other: object) -> DD:
        return _as_dd(other) / self

    def __eq__(self, other: object) -> Any:  # type: ignore[override]
        try:
            rhs = _as_dd(other)
        except (TypeError, ValueError, OverflowError):
            return False
        return (self._hi == rhs._hi) & (self._lo == rhs._lo)

    def __ne__(self, other: object) -> Any:  # type: ignore[override]
        eq = self.__eq__(other)
        return ~eq if isinstance(eq, np.ndarray) else not eq

    def __lt__(self, other: object) -> NDArray[np.bool_]:
        rhs = _as_dd(other)
        return np.asarray(
            (self._hi < rhs._hi) | ((self._hi == rhs._hi) & (self._lo < rhs._lo)),
            dtype=np.bool_,
        )

    def __le__(self, other: object) -> NDArray[np.bool_]:
        rhs = _as_dd(other)
        return np.asarray((self < rhs) | ((self._hi == rhs._hi) & (self._lo == rhs._lo)), dtype=np.bool_)

    def __gt__(self, other: object) -> NDArray[np.bool_]:
        return _as_dd(other) < self

    def __ge__(self, other: object) -> NDArray[np.bool_]:
        return _as_dd(other) <= self

    def exp(self) -> DD:
        return exp(self)

    def expm1(self) -> DD:
        return expm1(self)

    def log(self) -> DD:
        return log(self)

    def log1p(self) -> DD:
        return log1p(self)

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> DD:
        return sum(self, axis=axis, keepdims=keepdims)


def _as_dd(value: object) -> DD:
    if isinstance(value, DD):
        return value
    if type(value) in (float, int):
        return _scalar_constant(value)
    return DD(value)


@lru_cache(maxsize=128, typed=True)
def _scalar_constant(value: float | int) -> DD:
    return DD(value)


_LN2 = DD(0.6931471805599453, 2.3190468138462996e-17)
_LN2_TAIL = 5.707708438416212e-34
_SERIES_RECIPROCALS = tuple(DD(1) / n for n in range(1, 53))
_LOCAL_EXPM1_MAX = float(np.ldexp(1.0, -10))


def _scale_two(value: DD, exponent: NDArray[np.int64] | int) -> DD:
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        hi = np.asarray(np.ldexp(value._hi, exponent), dtype=np.float64)
        lo = np.asarray(np.ldexp(value._lo, exponent), dtype=np.float64)
    if np.any((value._hi != 0.0) & (hi == 0.0)):
        raise ArithmeticError("DD power-of-two scaling underflow")
    return DD._normalized(hi, lo)


def _small_expm1(value: DD) -> DD:
    result, term = value, value
    for n in range(2, 13):
        active = (np.abs(term._hi) > np.abs(result._hi) * _SERIES_EPS) & (
            np.abs(value._hi) > _SERIES_EPS
        )
        if not np.any(active):
            return result
        next_term = (term[active] * value[active]) * _SERIES_RECIPROCALS[n - 1]
        hi: NDArray[np.float64] = np.zeros(value.shape, dtype=np.float64)
        lo: NDArray[np.float64] = np.zeros(value.shape, dtype=np.float64)
        hi[active], lo[active] = next_term._hi, next_term._lo
        term = DD._normalized(hi, lo)
        result = result + term
    raise ArithmeticError("DD exponential Taylor series did not converge")


def _general_reduced_expm1(value: DD) -> tuple[DD, NDArray[np.int64]]:
    m = np.rint(value._hi / _LN2._hi).astype(np.int64)
    product_hi, product_lo = _two_product(_LN2._hi, m.astype(np.float64))
    residual = (value - DD(product_hi)) - DD(product_lo)
    residual = residual - DD(_LN2._lo) * m
    residual = residual - DD(_LN2_TAIL) * m
    reduced = _scale_two(residual, -9)
    small = _small_expm1(reduced)
    for _ in range(9):
        active = np.abs(small._hi) > _SERIES_EPS
        hi: NDArray[np.float64] = np.zeros(small.shape, dtype=np.float64)
        lo: NDArray[np.float64] = np.zeros(small.shape, dtype=np.float64)
        if np.any(active):
            square = small[active] * small[active]
            hi[active], lo[active] = square._hi, square._lo
        small = _scale_two(small, 1) + DD._normalized(hi, lo)
    return small, m


def _reduced_expm1(value: DD) -> tuple[DD, NDArray[np.int64]]:
    local = np.abs(value._hi) <= _LOCAL_EXPM1_MAX
    if np.all(local):
        return _small_expm1(value), np.zeros(value.shape, dtype=np.int64)
    if np.any(local):
        hi: NDArray[np.float64] = np.empty(value.shape, dtype=np.float64)
        lo: NDArray[np.float64] = np.empty(value.shape, dtype=np.float64)
        exponent: NDArray[np.int64] = np.zeros(value.shape, dtype=np.int64)
        small = _small_expm1(value[local])
        other, exponent[~local] = _general_reduced_expm1(value[~local])
        hi[local], lo[local] = small._hi, small._lo
        hi[~local], lo[~local] = other._hi, other._lo
        return DD._normalized(hi, lo), exponent
    return _general_reduced_expm1(value)


def exp(value: object) -> DD:
    """Compensated exponential ``exp(x)``."""
    dd = _as_dd(value)
    if np.any(np.abs(dd._hi) > 672.0):
        raise ArithmeticError("DD exponential is outside the supported precision range")
    reduced, exponent = _reduced_expm1(dd)
    return _scale_two(1.0 + reduced, exponent)


def expm1(value: object) -> DD:
    """Compensated ``exp(x) - 1`` without sub-ULP cancellation."""
    dd = _as_dd(value)
    if np.any(np.abs(dd._hi) > 672.0):
        raise ArithmeticError("DD exponential is outside the supported precision range")
    reduced, exponent = _reduced_expm1(dd)
    if not np.any(exponent):
        return reduced
    full = _scale_two(1.0 + reduced, exponent) - 1.0
    return DD._normalized(
        np.where(exponent == 0, reduced._hi, full._hi),
        np.where(exponent == 0, reduced._lo, full._lo),
    )


def _small_log1p(value: DD) -> DD:
    z = value / (2.0 + value)
    result, term = z, z
    active = np.abs(z._hi) > _SERIES_EPS
    hi: NDArray[np.float64] = np.zeros(z.shape, dtype=np.float64)
    lo: NDArray[np.float64] = np.zeros(z.shape, dtype=np.float64)
    if np.any(active):
        square = z[active] * z[active]
        hi[active], lo[active] = square._hi, square._lo
    z2 = DD._normalized(hi, lo)
    for odd in range(3, 53, 2):
        active = active & (np.abs(term._hi) > np.abs(result._hi) * _SERIES_EPS)
        if not np.any(active):
            return 2.0 * result
        next_term = term[active] * z2[active]
        hi = np.zeros(z.shape, dtype=np.float64)
        lo = np.zeros(z.shape, dtype=np.float64)
        hi[active], lo[active] = next_term._hi, next_term._lo
        term = DD._normalized(hi, lo)
        result = result + term * _SERIES_RECIPROCALS[odd - 1]
    raise ArithmeticError("DD logarithm Taylor series did not converge")


def log(value: object) -> DD:
    """Compensated natural logarithm with cancellation-safe branch near 1."""
    dd = _as_dd(value)
    if np.any(dd._hi <= 0.0):
        raise ValueError("DD logarithm requires a positive argument")
    near = (dd > 0.75) & (dd < 1.25)
    hi: NDArray[np.float64] = np.zeros(dd.shape, dtype=np.float64)
    lo: NDArray[np.float64] = np.zeros(dd.shape, dtype=np.float64)
    if np.any(near):
        local = _small_log1p(dd[near] - 1.0)
        hi[near], lo[near] = local._hi, local._lo
    if np.any(~near):
        a = dd[~near]
        guess = DD(np.log(a._hi))
        for _ in range(2):
            guess = guess + (a * exp(-guess) - 1.0)
        hi[~near], lo[~near] = guess._hi, guess._lo
    return DD._normalized(hi, lo)


def log1p(value: object) -> DD:
    """Compensated ``log(1 + x)``."""
    dd = _as_dd(value)
    if np.any(dd <= -1.0):
        raise ValueError("DD log1p requires x > -1")
    near = (dd >= -0.25) & (dd <= 0.25)
    hi: NDArray[np.float64] = np.zeros(dd.shape, dtype=np.float64)
    lo: NDArray[np.float64] = np.zeros(dd.shape, dtype=np.float64)
    if np.any(near):
        local = _small_log1p(dd[near])
        hi[near], lo[near] = local._hi, local._lo
    if np.any(~near):
        local = log(1.0 + dd[~near])
        hi[~near], lo[~near] = local._hi, local._lo
    return DD._normalized(hi, lo)


def sum(
    value: object,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> DD:
    """Compensated ordered reduction preserving both words."""
    dd = _as_dd(value)
    if axis is None:
        result = sum(dd.ravel(), axis=0)
        return result.reshape((1,) * dd.ndim) if keepdims else result
    raw_axes = (axis,) if isinstance(axis, int) else tuple(axis)
    axes = tuple(operator.index(a) for a in raw_axes)
    if any(a < -dd.ndim or a >= dd.ndim for a in axes):
        raise ValueError("DD reduction axis is out of bounds")
    norm_axes = tuple(a % dd.ndim for a in axes)
    if len(set(norm_axes)) != len(norm_axes):
        raise ValueError("DD reduction axes must be distinct")
    result = dd
    for current in sorted(norm_axes, reverse=True):
        high = np.moveaxis(result._hi, current, 0)
        low = np.moveaxis(result._lo, current, 0)
        reduced = DD(np.zeros(high.shape[1:], dtype=np.float64))
        for idx in range(high.shape[0]):
            reduced = reduced + DD.from_normalized_parts(high[idx], low[idx])
        result = reduced
    if keepdims:
        kept = tuple(1 if i in norm_axes else n for i, n in enumerate(dd.shape))
        result = result.reshape(kept)
    return result


@dataclass(frozen=True)
class DDArray:
    """Compensated ``PhysicalArray`` implementation backed by normalized ``DD``."""

    values: DD

    def __post_init__(self) -> None:
        if not isinstance(self.values, DD):
            raise ContractError("dd_array_requires_dd_values")

    @classmethod
    def from_parts(cls, hi: ArrayLike, lo: ArrayLike | None = None) -> DDArray:
        return cls(DD(hi, lo))

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    def immutable_copy(self) -> DDArray:
        return DDArray(self.values)

    def difference(self, other: PhysicalArray) -> DDArray:
        if not isinstance(other, DDArray) or other.shape != self.shape:
            raise ContractError("precision_or_shape_mismatch")
        return DDArray(self.values - other.values)

    def take(self, indices: ArrayLike) -> DDArray:
        selected = integer_indices(indices)
        if not self.shape or np.any(selected < 0) or np.any(selected >= self.shape[0]):
            raise ContractError("index_outside_support")
        return DDArray(self.values[selected])

    def log_ratio(self, other: PhysicalArray) -> DDArray:
        if not isinstance(other, DDArray) or other.shape != self.shape:
            raise ContractError("precision_or_shape_mismatch")
        if np.any(self.values <= 0.0) or np.any(other.values <= 0.0):
            raise ContractError("log_ratio_requires_positive_fields")
        relative = (self.values - other.values) / other.values
        close = abs(relative) <= 0.25
        hi: NDArray[np.float64] = np.zeros(self.shape, dtype=np.float64)
        lo: NDArray[np.float64] = np.zeros(self.shape, dtype=np.float64)
        if np.any(close):
            near = log1p(relative[close])
            hi[close], lo[close] = near.hi, near.lo
        if np.any(~close):
            far = log(self.values[~close] / other.values[~close])
            hi[~close], lo[~close] = far.hi, far.lo
        return DDArray(DD.from_normalized_parts(hi, lo))

    def is_finite(self) -> bool:
        return bool(np.isfinite(self.values.hi).all() and np.isfinite(self.values.lo).all())

    def identity_bytes(self) -> bytes:
        return (
            b"float64-pair-v1:"
            + repr(self.shape).encode()
            + self.values.hi.tobytes()
            + self.values.lo.tobytes()
        )

    def project_to_float(self, *, authority_identity: str) -> FieldProjection:
        """Project to binary64 with explicit compensated truncation error bound."""
        hi = frozen_array(self.values.hi)
        # Projecting DD(hi, lo) -> hi discards lo plus at most 0.5 ULP of DD tail.
        bound = np.abs(self.values.lo) + 0.5 * np.spacing(np.abs(hi)) * (2.0**-52)
        return FieldProjection(
            value=FloatArray(hi),
            authority_identity=authority_identity,
            evaluator_identity="dd-high-word-projection-v1",
            absolute_error_bound=FloatArray(bound),
        )


def _unwrap_dd(value: Any) -> DD:
    if isinstance(value, DD):
        return value
    if isinstance(value, DDArray):
        return value.values
    if isinstance(value, FloatArray):
        return DD(value.values)
    try:
        return DD(value)
    except (TypeError, ValueError, OverflowError, ArithmeticError) as exc:
        raise ContractError("invalid_compensated_operand") from exc


class CompensatedArithmetic:
    """``AssemblyArithmetic`` implementation preserving ``DD`` representation."""

    def array(self, value: ArrayLike | PhysicalArray) -> DD:
        return _unwrap_dd(value)

    def zeros(self, shape: tuple[int, ...]) -> DD:
        return DD(np.zeros(shape, dtype=np.float64))

    def weighted(self, value: Any, weights: Vector) -> DD:
        dd = _unwrap_dd(value)
        w = frozen_array(weights)
        if dd.shape != w.shape:
            raise ContractError("arithmetic_shape_mismatch")
        return dd * DD(w)

    def add(self, left: Any, right: Any) -> DD:
        lhs, rhs = _unwrap_dd(left), _unwrap_dd(right)
        if lhs.shape != rhs.shape:
            raise ContractError("arithmetic_shape_mismatch")
        return lhs + rhs

    def subtract(self, left: Any, right: Any) -> DD:
        lhs, rhs = _unwrap_dd(left), _unwrap_dd(right)
        if lhs.shape != rhs.shape:
            raise ContractError("arithmetic_shape_mismatch")
        return lhs - rhs

    def divide(self, left: Any, right: Any) -> DD:
        lhs, rhs = _unwrap_dd(left), _unwrap_dd(right)
        if lhs.shape != rhs.shape or np.any(rhs.hi == 0.0):
            raise ContractError("arithmetic_shape_or_divisor")
        return lhs / rhs

    def matmul(self, left: Any, right: Any) -> DD:
        lhs, rhs = _unwrap_dd(left), _unwrap_dd(right)
        if lhs.ndim != 2 or rhs.ndim not in {1, 2} or lhs.shape[1] != rhs.shape[0]:
            raise ContractError("arithmetic_shape_mismatch")
        if rhs.ndim == 1:
            products = lhs * rhs.reshape((1, rhs.shape[0]))
            return sum(products, axis=1)
        cols = [sum(lhs * rhs[:, col].reshape((1, lhs.shape[1])), axis=1) for col in range(rhs.shape[1])]
        hi = np.stack([col.hi for col in cols], axis=1)
        lo = np.stack([col.lo for col in cols], axis=1)
        return DD.from_normalized_parts(hi, lo)

    def concatenate(self, values: Iterable[Any], axis: int = 0) -> DD:
        items = tuple(_unwrap_dd(v) for v in values)
        if not items:
            raise ContractError("empty_concatenation")
        hi = np.concatenate([item.hi for item in items], axis=axis)
        lo = np.concatenate([item.lo for item in items], axis=axis)
        return DD.from_normalized_parts(hi, lo)

    def reshape(self, value: Any, shape: tuple[int, ...]) -> DD:
        return _unwrap_dd(value).reshape(shape)

    def scatter_add(self, target: Any, indices: NDArray[np.intp], value: Any) -> DD:
        acc = _unwrap_dd(target)
        inc = _unwrap_dd(value)
        idx = integer_indices(indices)
        if idx.shape != inc.shape or acc.ndim != 1:
            raise ContractError("arithmetic_shape_mismatch")
        if np.any(idx < 0) or np.any(idx >= acc.shape[0]):
            raise ContractError("index_outside_support")
        hi = acc.hi.copy()
        lo = acc.lo.copy()
        for pos, k in enumerate(idx.tolist()):
            updated = DD.from_normalized_parts(np.asarray(hi[k]), np.asarray(lo[k])) + inc[pos]
            hi[k] = float(updated.hi.item())
            lo[k] = float(updated.lo.item())
        return DD.from_normalized_parts(hi, lo)

    def freeze(self, value: Any) -> DD:
        return _unwrap_dd(value)


# Verify protocol structural conformance at import time for static type checkers.
_ARITHMETIC_CONFORMANCE: AssemblyArithmetic = CompensatedArithmetic()
