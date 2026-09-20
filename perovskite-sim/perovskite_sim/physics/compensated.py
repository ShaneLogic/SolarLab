"""Vectorized, explicit two-float arithmetic for persistent physical states.

``DD(hi, lo)`` represents the real sum of two binary64 arrays.  Arithmetic
normalizes the pair; converting to a single float is deliberately explicit.
The components are read-only, and broadcasting and slicing follow NumPy.

Algorithms: Knuth's TwoSum and Dekker's split product; accurate compensated
addition/division and exponential argument reduction as described by Hida,
Li and Bailey, *Algorithms for Quad-Double Precision Floating Point
Arithmetic* (2001), and the authors' QD implementation (``inline.h``,
``dd_inline.h``, ``dd_real.cpp``).  The logarithm near one uses the convergent
atanh series so that a tiny log is not obtained by subtracting two O(1) values.
Primary reference: https://github.com/BL-highprecision/QD

This is finite real arithmetic, not an IEEE exceptional-value implementation.
Non-finite inputs, division by zero and invalid logarithms raise.  Nonzero
normalized high parts below 2**-970 or above 2**970 are explicitly unsupported
to leave room for the low component.  Operations that cannot preserve that
range raise rather than silently returning a binary64 fallback.  Accuracy
tests cover the simulator's mixed physical scales, not arbitrary conditioning
or an accumulated error bound for a scientific trajectory.
"""

from __future__ import annotations

import operator

import numpy as np

__all__ = ["DD", "exp", "expm1", "log", "log1p", "sum"]

_SPLITTER = 134217729.0  # 2**27 + 1
_MIN_HIGH = np.ldexp(1.0, -970)
_MAX_HIGH = np.ldexp(1.0, 970)
_SERIES_EPS = np.ldexp(1.0, -108)


def _two_sum(a, b):
    with np.errstate(over="raise", invalid="raise"):
        s = a + b
        z = s - a
        return s, (a - (s - z)) + (b - z)


def _two_product(a, b):
    """Scale the Dekker split to avoid overflow of the splitter product."""
    am, ae = np.frexp(a)
    bm, be = np.frexp(b)
    ca, cb = _SPLITTER * am, _SPLITTER * bm
    ah, bh = ca - (ca - am), cb - (cb - bm)
    al, bl = am - ah, bm - bh
    p = am * bm
    error = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        high = np.ldexp(p, ae + be)
        low = np.ldexp(error, ae + be)
    if np.any((a != 0) & (b != 0) & (high == 0)):
        raise ArithmeticError("DD product underflow")
    return high, low


def _validate(hi, lo):
    if not np.all(np.isfinite(hi)) or not np.all(np.isfinite(lo)):
        raise ArithmeticError("DD requires finite real components")
    magnitude = np.abs(hi)
    if np.any((magnitude != 0) & (magnitude < _MIN_HIGH)):
        raise ArithmeticError("DD result is below the supported precision range")
    if np.any(magnitude > _MAX_HIGH):
        raise ArithmeticError("DD result is above the supported precision range")


class DD:
    """A normalized, immutable pair of broadcast binary64 arrays.

    Use ``value.hi`` and ``value.lo`` for serialization, ``value[index]`` for
    slicing, and ``value.to_float()`` only at an explicitly rounded boundary.
    Plain real scalars/arrays are accepted in arithmetic and interpreted as
    their exact binary64 values.  Decimal/string inputs are rejected.
    """

    __slots__ = ("_hi", "_lo")
    __array_priority__ = 1000

    def __init__(self, hi, lo=0.0):
        if isinstance(hi, DD):
            if np.any(np.asarray(lo) != 0):
                raise TypeError("use DD arithmetic to add a low part to a DD")
            hi, lo = hi.hi, hi.lo
        a, b = np.asarray(hi), np.asarray(lo)
        if a.dtype.kind not in "biuf" or b.dtype.kind not in "biuf":
            raise TypeError("DD inputs must be real binary64-compatible numbers")
        if any(v.dtype.kind == "f" and v.dtype.itemsize > 8 for v in (a, b)):
            raise TypeError("split wider floating inputs explicitly into DD components")
        for v in (a, b):
            if v.dtype.kind in "iu":
                # Integer input is convenient for coefficients, but must not
                # silently lose bits before it reaches the compensated pair.
                as_float = v.astype(float)
                if any(int(i) != int(f) for i, f in zip(v.flat, as_float.flat)):
                    raise ValueError("integer input is not exact in one binary64 component")
        a, b = np.broadcast_arrays(a.astype(float), b.astype(float))
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise ArithmeticError("DD requires finite real components")
        high, low = _two_sum(a, b)
        _validate(high, low)
        self._hi, self._lo = np.array(high, copy=True), np.array(low, copy=True)
        self._hi.flags.writeable = self._lo.flags.writeable = False

    @classmethod
    def _normalized(cls, hi, lo):
        _validate(hi, lo)
        result = object.__new__(cls)
        result._hi, result._lo = np.asarray(hi), np.asarray(lo)
        result._hi.flags.writeable = result._lo.flags.writeable = False
        return result

    @classmethod
    def from_float(cls, value):
        """Treat a binary64 value/array as exact, with a zero low part."""
        return cls(value)

    @property
    def hi(self):
        return self._hi

    @property
    def lo(self):
        return self._lo

    @property
    def shape(self):
        return self.hi.shape

    @property
    def ndim(self):
        return self.hi.ndim

    @property
    def size(self):
        return self.hi.size

    def __len__(self):
        return len(self.hi)

    def __getitem__(self, index):
        return self._normalized(self.hi[index], self.lo[index])

    def reshape(self, *shape):
        return self._normalized(self.hi.reshape(*shape), self.lo.reshape(*shape))

    def ravel(self):
        return self.reshape(-1)

    def copy(self):
        return DD(self.hi, self.lo)

    def to_float(self):
        """Explicitly round the represented value to one binary64 array."""
        return self.hi + self.lo

    def __array__(self, dtype=None, copy=None):
        raise TypeError("implicit DD rounding is forbidden; use .to_float()")

    def __float__(self):
        raise TypeError("implicit DD rounding is forbidden; use .to_float()")

    def __bool__(self):
        if self.ndim:
            raise ValueError("the truth value of a DD array is ambiguous")
        return bool(self.hi != 0)

    def __repr__(self):
        return f"DD(hi={self.hi!r}, lo={self.lo!r})"

    def __neg__(self):
        return self._normalized(-self.hi, -self.lo)

    def __pos__(self):
        return self

    def __abs__(self):
        return self._normalized(
            np.where(self.hi < 0, -self.hi, self.hi),
            np.where(self.hi < 0, -self.lo, self.lo),
        )

    def __add__(self, other):
        other = _as_dd(other)
        high, low = _two_sum(self.hi, other.hi)
        tail, tail_error = _two_sum(self.lo, other.lo)
        high, low = _two_sum(high, low + tail)
        high, low = _two_sum(high, low + tail_error)
        return self._normalized(high, low)

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-_as_dd(other))

    def __rsub__(self, other):
        return _as_dd(other) + (-self)

    def __mul__(self, other):
        other = _as_dd(other)
        high, low = _two_product(self.hi, other.hi)
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            cross = self.hi * other.lo + self.lo * other.hi
            low = low + cross
            high, low = _two_sum(high, low)
            high, low = _two_sum(high, low + self.lo * other.lo)
        return self._normalized(high, low)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = _as_dd(other)
        if np.any(other.hi == 0):
            raise ZeroDivisionError("DD division by zero")
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            q1 = self.hi / other.hi
        if np.any((self.hi != 0) & (q1 == 0)):
            raise ArithmeticError("DD quotient underflow")
        first = DD(q1)
        remainder = self - other * first
        q2 = remainder.hi / other.hi
        second = DD(q2)
        remainder = remainder - other * second
        q3 = remainder.hi / other.hi
        return (first + second) + DD(q3)

    def __rtruediv__(self, other):
        return _as_dd(other) / self

    def __eq__(self, other):
        other = _as_dd(other)
        return (self.hi == other.hi) & (self.lo == other.lo)

    def __ne__(self, other):
        return ~(self == other)

    def __lt__(self, other):
        other = _as_dd(other)
        return (self.hi < other.hi) | ((self.hi == other.hi) & (self.lo < other.lo))

    def __le__(self, other):
        return (self < other) | (self == other)

    def __gt__(self, other):
        return _as_dd(other) < self

    def __ge__(self, other):
        return _as_dd(other) <= self

    def exp(self):
        return exp(self)

    def expm1(self):
        return expm1(self)

    def log(self):
        return log(self)

    def log1p(self):
        return log1p(self)

    def sum(self, axis=None, keepdims=False):
        return sum(self, axis=axis, keepdims=keepdims)


def _as_dd(value):
    return value if isinstance(value, DD) else DD(value)


_LN2 = DD(0.6931471805599453, 2.3190468138462996e-17)
# A third constant word is used only during range reduction.  Forming m*ln2
# first and then subtracting x would round an O(m) DD before cancellation.
_LN2_TAIL = 5.707708438416212e-34


def _scale_two(value, exponent):
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        hi = np.ldexp(value.hi, exponent)
        lo = np.ldexp(value.lo, exponent)
    if np.any((value.hi != 0) & (hi == 0)):
        raise ArithmeticError("DD power-of-two scaling underflow")
    return DD._normalized(hi, lo)


def _small_expm1(value):
    """Taylor series on |x| <= log(2)/1024, preserving tiny x directly."""
    result, term = value, value
    for n in range(2, 13):
        active = (np.abs(term.hi) > np.abs(result.hi) * _SERIES_EPS) & (
            np.abs(value.hi) > _SERIES_EPS
        )
        if not np.any(active):
            return result
        next_term = term[active] * value[active] / float(n)
        increment = DD(np.zeros(value.shape))
        hi, lo = increment.hi.copy(), increment.lo.copy()
        hi[active], lo[active] = next_term.hi, next_term.lo
        term = DD._normalized(hi, lo)
        result = result + term
    raise ArithmeticError("DD exponential Taylor series did not converge")


def _reduced_expm1(value):
    m = np.rint(value.hi / _LN2.hi).astype(np.int64)
    product_hi, product_lo = _two_product(_LN2.hi, m.astype(float))
    residual = (value - DD(product_hi)) - DD(product_lo)
    residual = residual - DD(_LN2.lo) * m
    residual = residual - DD(_LN2_TAIL) * m
    reduced = _scale_two(residual, -9)
    small = _small_expm1(reduced)
    for _ in range(9):
        active = np.abs(small.hi) > _SERIES_EPS
        hi, lo = np.zeros(small.shape), np.zeros(small.shape)
        if np.any(active):
            square = small[active] * small[active]
            hi[active], lo[active] = square.hi, square.lo
        small = _scale_two(small, 1) + DD._normalized(hi, lo)
    return small, m


def exp(value):
    """Exponential with compensated argument reduction and reconstruction."""
    value = _as_dd(value)
    if np.any(np.abs(value.hi) > 672.0):
        raise ArithmeticError("DD exponential is outside the supported precision range")
    reduced, exponent = _reduced_expm1(value)
    return _scale_two(1.0 + reduced, exponent)


def expm1(value):
    """Compute exp(x)-1 without discarding a sub-ULP driving increment."""
    value = _as_dd(value)
    if np.any(np.abs(value.hi) > 672.0):
        raise ArithmeticError("DD exponential is outside the supported precision range")
    reduced, exponent = _reduced_expm1(value)
    full = _scale_two(1.0 + reduced, exponent) - 1.0
    return DD._normalized(
        np.where(exponent == 0, reduced.hi, full.hi),
        np.where(exponent == 0, reduced.lo, full.lo),
    )


def _small_log1p(value):
    """log(1+x)=2*atanh(x/(2+x)); |x| <= 1/4."""
    z = value / (2.0 + value)
    result, term = z, z
    active = np.abs(z.hi) > _SERIES_EPS
    hi, lo = np.zeros(z.shape), np.zeros(z.shape)
    if np.any(active):
        square = z[active] * z[active]
        hi[active], lo[active] = square.hi, square.lo
    z2 = DD._normalized(hi, lo)
    for odd in range(3, 53, 2):
        active &= np.abs(term.hi) > np.abs(result.hi) * _SERIES_EPS
        if not np.any(active):
            return 2.0 * result
        next_term = term[active] * z2[active]
        hi, lo = np.zeros(z.shape), np.zeros(z.shape)
        hi[active], lo[active] = next_term.hi, next_term.lo
        term = DD._normalized(hi, lo)
        result = result + term / float(odd)
    raise ArithmeticError("DD logarithm Taylor series did not converge")


def log(value):
    """Natural logarithm, with a cancellation-safe path close to one."""
    value = _as_dd(value)
    if np.any(value.hi <= 0):
        raise ValueError("DD logarithm requires a positive argument")
    near = (value > 0.75) & (value < 1.25)
    hi, lo = np.zeros(value.shape), np.zeros(value.shape)
    if np.any(near):
        local = _small_log1p(value[near] - 1.0)
        hi[near], lo[near] = local.hi, local.lo
    if np.any(~near):
        a = value[~near]
        guess = DD(np.log(a.hi))
        for _ in range(2):
            guess = guess + (a * exp(-guess) - 1.0)
        hi[~near], lo[~near] = guess.hi, guess.lo
    return DD._normalized(hi, lo)


def log1p(value):
    """Natural log of 1+x, including a low part beside x=-1."""
    value = _as_dd(value)
    if np.any(value <= -1.0):
        raise ValueError("DD log1p requires x > -1")
    near = (value >= -0.25) & (value <= 0.25)
    hi, lo = np.zeros(value.shape), np.zeros(value.shape)
    if np.any(near):
        local = _small_log1p(value[near])
        hi[near], lo[near] = local.hi, local.lo
    if np.any(~near):
        local = log(1.0 + value[~near])
        hi[~near], lo[~near] = local.hi, local.lo
    return DD._normalized(hi, lo)


def sum(value, axis=None, keepdims=False):
    """Compensated ordered reduction; never reduce high and low separately."""
    value = _as_dd(value)
    if axis is None:
        result = sum(value.ravel(), axis=0)
        return result.reshape((1,) * value.ndim) if keepdims else result
    axes = (axis,) if np.isscalar(axis) else tuple(axis)
    axes = tuple(operator.index(a) for a in axes)
    if any(a < -value.ndim or a >= value.ndim for a in axes):
        raise ValueError("DD reduction axis is out of bounds")
    axes = tuple(a % value.ndim for a in axes)
    if len(set(axes)) != len(axes):
        raise ValueError("DD reduction axes must be distinct")
    result = value
    for current in sorted(axes, reverse=True):
        high = np.moveaxis(result.hi, current, 0)
        low = np.moveaxis(result.lo, current, 0)
        reduced = DD(np.zeros(high.shape[1:]))
        for i in range(high.shape[0]):
            reduced = reduced + DD._normalized(high[i], low[i])
        result = reduced
    if keepdims:
        result = result.reshape(tuple(1 if i in axes else n for i, n in enumerate(value.shape)))
    return result
