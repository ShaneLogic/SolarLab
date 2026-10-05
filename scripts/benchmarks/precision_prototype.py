"""P02-08 precision provider injected into the isolated public contracts.

Only the existing stateless DD arithmetic is reused. No old device, solver,
private R1 system or experiment is imported. The arithmetic implementation is
to move to solarlab_research when the package migration is admitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import json
from math import fsum
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike

from perovskite_sim.physics.compensated import DD
from scripts.benchmarks.contract_prototype import (
    ContractError, FieldProjection, FloatArray, Layout, PhysicalArray, Point,
    PhysicalLinearForm, RateView, ResolvedAuthority, StateIncrement, StateView, Vector,
    _apply_linear_form, frozen_array, integer_indices,
)


@dataclass(frozen=True, init=False)
class DoubleArray:
    """Own both binary64 words, including their serialized zero signs."""

    _high: Vector
    _low: Vector

    def __init__(self, high: ArrayLike, low: ArrayLike | None = None):
        # Validate the raw values before binary64 conversion, preserving DD's
        # rejection of inexact integers, strings and wider floating inputs.
        raw_high = np.asarray(high)
        raw_low = np.zeros(raw_high.shape) if low is None else np.asarray(low)
        if raw_high.shape != raw_low.shape:
            raise ContractError("precision_word_shape_mismatch")
        normalized = DD(raw_high, raw_low)
        high = frozen_array(high)
        low = frozen_array(raw_low)
        if not np.array_equal(normalized.hi, high) or not np.array_equal(normalized.lo, low):
            raise ContractError("unnormalized_precision_words")
        object.__setattr__(self, "_high", high)
        object.__setattr__(self, "_low", low)

    @property
    def high(self) -> Vector:
        return self._high.view()

    @property
    def low(self) -> Vector:
        return self._low.view()

    @property
    def words(self) -> tuple[Vector, Vector]:
        """Public complete words for explicit rate-input binding, without rounding."""
        return self.high, self.low

    @classmethod
    def from_dd(cls, value: DD) -> DoubleArray:
        return cls(value.hi, value.lo)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.high.shape

    def as_dd(self) -> DD:
        return DD(self.high, self.low)

    def immutable_copy(self) -> DoubleArray:
        return DoubleArray(self.high, self.low)

    def difference(self, other: PhysicalArray) -> DoubleArray:
        if not isinstance(other, DoubleArray) or self.shape != other.shape:
            raise ContractError("precision_or_shape_mismatch")
        return DoubleArray.from_dd(self.as_dd() - other.as_dd())

    def take(self, indices: ArrayLike) -> DoubleArray:
        indices = integer_indices(indices)
        if np.any(indices < 0) or np.any(indices >= self.shape[0]):
            raise ContractError("index_outside_support")
        return DoubleArray(self.high[indices], self.low[indices])

    def log_ratio(self, other: PhysicalArray) -> DoubleArray:
        if not isinstance(other, DoubleArray) or other.shape != self.shape:
            raise ContractError("precision_or_shape_mismatch")
        return DoubleArray.from_dd(_log_ratio(self.as_dd(), other.as_dd()))

    def is_finite(self) -> bool:
        return bool(np.isfinite(self.high).all() and np.isfinite(self.low).all())

    def identity_bytes(self) -> bytes:
        return b"double-word:" + repr(self.shape).encode() + self.high.tobytes() + self.low.tobytes()

    def __array__(self, dtype=None, copy=None):
        raise TypeError("implicit precision rounding is forbidden")

    def __float__(self):
        raise TypeError("implicit precision rounding is forbidden")


class DoubleArithmetic:
    """A precision implementation of the public assembly operations."""

    def array(self, value: ArrayLike | PhysicalArray) -> DD:
        if isinstance(value, DD):
            return value.copy()
        if isinstance(value, DoubleArray):
            return value.as_dd()
        if isinstance(value, FloatArray):
            return DD(value.values)
        return DD(value)

    def zeros(self, shape: tuple[int, ...]) -> DD:
        return DD(np.zeros(shape))

    def weighted(self, value: DD, weights: Vector) -> DD:
        if value.shape != weights.shape:
            raise ContractError("arithmetic_shape_mismatch")
        return value * DD(weights)

    def add(self, left: DD, right: DD) -> DD:
        if left.shape != right.shape:
            raise ContractError("arithmetic_shape_mismatch")
        return left + right

    def subtract(self, left: DD, right: DD) -> DD:
        if left.shape != right.shape:
            raise ContractError("arithmetic_shape_mismatch")
        return left - right

    def divide(self, left: DD, right: DD) -> DD:
        if left.shape != right.shape or np.any(right == 0):
            raise ContractError("arithmetic_shape_or_divisor")
        return left / right

    def matmul(self, left: DD, right: DD) -> DD:
        if left.ndim != 2 or right.ndim not in {1, 2} or left.shape[1] != right.shape[0]:
            raise ContractError("arithmetic_shape_mismatch")
        if right.ndim == 1:
            return (left * right.reshape(1, right.size)).sum(axis=1)
        if not right.shape[1]:
            return self.zeros((left.shape[0], 0))
        columns = [self.matmul(left, right[:, index]).reshape(left.shape[0], 1)
                   for index in range(right.shape[1])]
        return self.concatenate(columns, axis=1)

    def concatenate(self, values, axis: int = 0) -> DD:
        values = tuple(values)
        return DD(np.concatenate([value.hi for value in values], axis=axis),
                  np.concatenate([value.lo for value in values], axis=axis))

    def reshape(self, value: DD, shape: tuple[int, ...]) -> DD:
        return value.reshape(shape)

    def scatter_add(self, target: DD, indices: np.ndarray, value: DD) -> DD:
        indices = integer_indices(indices)
        if indices.shape != value.shape or target.ndim != 1 or indices.ndim != 1:
            raise ContractError("arithmetic_shape_mismatch")
        if np.any(indices < 0) or np.any(indices >= target.size):
            raise ContractError("index_outside_support")
        high, low = target.hi.copy(), target.lo.copy()
        # This bounded prototype preserves DD sums for repeated node indices.
        # Product binary64 assembly keeps its vectorized np.add.at path.
        for index in np.unique(indices):
            total = target[index] + value[indices == index].sum()
            high[index], low[index] = total.hi, total.lo
        return DD(high, low)

    def freeze(self, value: DD) -> DoubleArray:
        return DoubleArray.from_dd(value)

    def validate_rate(self, value: RateView) -> None:
        _linear_affine_authority(value.point.state)
        _double_linear_words(value.values)

    def linear_form(self, form: PhysicalLinearForm, operand: Any, *,
                    point=None, left=None, right=None, sources=()):
        if isinstance(operand, RateView):
            self.validate_rate(operand)
        return _apply_linear_form(
            form, operand, point=point, left=left, right=right, sources=sources,
            array_words=_double_linear_words, state_words=_double_linear_state_words,
            increment_words=_double_linear_increment_words, output_words=2,
            finish=DoubleArray)


def _double_linear_words(value) -> tuple[Vector, ...]:
    if type(value) is FloatArray:
        return (value.values,)
    if type(value) is DoubleArray:
        return value.words
    if type(value) is PrimitiveExpansion:
        return value.words
    raise ContractError("linear_unsupported_word_source")


def _linear_affine_authority(state):
    authority = state.authority
    if type(authority) is ResolvedAuthority:
        return authority
    if (type(authority) is not MappedAuthority or authority.map_name != "relative-fields"
            or any(mode != "linear" for mode in authority.modes.values())):
        raise ContractError("linear_action_requires_affine_physical_si")
    return authority


def _double_linear_state_words(state, name) -> tuple[Vector, ...]:
    """The provider owns map interpretation; physical consumers see only actions."""
    authority = _linear_affine_authority(state)
    if type(authority) is ResolvedAuthority:
        return _double_linear_words(authority.field(name))
    return authority.anchor[name].words + authority.primitives[name].words


def _double_linear_increment_words(left, right, name) -> tuple[Vector, ...]:
    authority = _linear_affine_authority(right)
    _linear_affine_authority(left)
    if type(authority) is ResolvedAuthority:
        return (_double_linear_words(authority.field(name))
                + tuple(-word for word in _double_linear_words(left.authority.field(name))))
    # The common wrapper already validated the actual Point predecessor and
    # canonical increment. This checks map/root/primitive compatibility again
    # and keeps the bounded original operands; projected delta fields are not
    # authoritative inputs to a cross-species or spatial contraction.
    local = authority._temporal_primitives(left)[name]
    if type(local) is PrimitiveDifference:
        return local.current.words + tuple(-word for word in local.previous.words)
    return local.words


def _log_ratio(numerator: DD, denominator: DD) -> DD:
    if numerator.shape != denominator.shape or np.any(numerator <= 0) or np.any(denominator <= 0):
        raise ContractError("log_ratio_requires_positive_fields")
    relative = (numerator-denominator) / denominator
    close = (relative > -0.5) & (relative < 0.5)
    high, low = np.zeros(numerator.shape), np.zeros(numerator.shape)
    if np.any(close):
        value = relative[close].log1p()
        high[close], low[close] = value.hi, value.lo
    if np.any(~close):
        value = numerator[~close].log()-denominator[~close].log()
        high[~close], low[~close] = value.hi, value.lo
    return DD(high, low)


def _binary_sum_is_zero(values) -> bool:
    """Exact dyadic equality for the bounded input-capacity check only."""
    total, exponent = 0, 0
    for value in values:
        numerator, denominator = value.as_integer_ratio()
        if not numerator:
            continue
        power = denominator.bit_length() - 1
        if power > exponent:
            total <<= power - exponent
            exponent = power
        total += numerator << (exponent - power)
    return total == 0


def _primitive_words(components: tuple[Vector, ...]) -> tuple[Vector, ...]:
    """Reduce at most eight input words to a fixed four-word exact expansion.

    Four bounded fsum reductions propose the words. Exact dyadic integers only
    verify that no remainder was discarded; they do not evaluate physics.
    This verification also fails closed on platform summation differences.
    """
    if not components or len(components) > 8 or any(v.shape != components[0].shape for v in components):
        raise ContractError("primitive_shape_or_arity")
    shape = components[0].shape
    output = [np.zeros(int(np.prod(shape))) for _ in range(4)]
    for index in range(output[0].size):
        source = [float(value.ravel()[index]) for value in components]
        words = []
        try:
            for _ in range(4):
                words.append(fsum([*source, *(-value for value in words)]))
        except OverflowError as error:
            raise ContractError("primitive_expansion_sum_range") from error
        if not _binary_sum_is_zero([*source, *(-value for value in words)]):
            raise ContractError("primitive_expansion_capacity_exceeded")
        for target, value in zip(output, words):
            target[index] = value
    return tuple(value.reshape(shape) for value in output)


@dataclass(frozen=True, init=False)
class PrimitiveExpansion:
    """Four binary64 words for input primitives only; physical arithmetic is DD.

    The capacity is fixed, not adaptive. ``as_dd`` rejects a nonzero third or
    fourth word instead of silently rounding an authoritative coordinate.
    Map contractions consume ``words`` before producing physical DD outputs.
    """

    _words: tuple[Vector, ...]

    def __init__(self, words):
        words = tuple(words)
        if len(words) != 4:
            raise ContractError("primitive_word_count")
        owned = []
        for value in words:
            raw = np.asarray(value)
            if raw.dtype.kind not in {"i", "u", "f"} or raw.dtype.kind == "f" and raw.dtype.itemsize > 8:
                raise ContractError("primitive_word_dtype")
            converted = frozen_array(raw)
            if raw.dtype.kind in {"i", "u"} and any(int(a) != int(b) for a, b in zip(raw.ravel(), converted.ravel())):
                raise ContractError("inexact_primitive_integer")
            owned.append(converted)
        normalized = _primitive_words(tuple(owned))
        if any(not np.array_equal(a, b) for a, b in zip(owned, normalized)):
            raise ContractError("unnormalized_primitive_words")
        object.__setattr__(self, "_words", tuple(owned))

    @classmethod
    def _owned(cls, words) -> PrimitiveExpansion:
        value = object.__new__(cls)
        object.__setattr__(value, "_words", tuple(frozen_array(word) for word in words))
        return value

    @classmethod
    def from_value(cls, value) -> PrimitiveExpansion:
        if type(value) is cls:
            return value
        value = value if type(value) is DoubleArray else DoubleArray(value)
        return cls._owned((value.high, value.low, np.zeros(value.shape), np.zeros(value.shape)))

    @property
    def words(self) -> tuple[Vector, ...]:
        return tuple(word.view() for word in self._words)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._words[0].shape

    @property
    def high(self) -> Vector:
        """Explicit first-word projection used by the host coordinate vector."""
        return self._words[0].view()

    def immutable_copy(self) -> PrimitiveExpansion:
        return self

    def add(self, other: PrimitiveExpansion | DoubleArray | ArrayLike) -> PrimitiveExpansion:
        """Exactly add input primitives within the fixed four-word capacity."""
        return _exact_primitive_sum(self, other)

    def take_flat(self, indices: ArrayLike) -> PrimitiveExpansion:
        indices = integer_indices(indices)
        if np.any(indices < 0) or np.any(indices >= self._words[0].size):
            raise ContractError("index_outside_support")
        return self._owned(tuple(word.ravel()[indices] for word in self._words))

    def reshape(self, shape: tuple[int, ...]) -> PrimitiveExpansion:
        return self._owned(tuple(word.reshape(shape) for word in self._words))

    def is_zero(self) -> bool:
        return all(not np.any(word) for word in self._words)

    def is_finite(self) -> bool:
        """Public source-value protocol; inspect all words without projection."""
        return all(bool(np.isfinite(word).all()) for word in self._words)

    def as_dd(self) -> DD:
        if any(np.any(word) for word in self._words[2:]):
            raise ContractError("primitive_projection_would_discard_remainder")
        return DD(self._words[0], self._words[1])

    def identity_bytes(self) -> bytes:
        return b"input-expansion4-v1:"+repr(self.shape).encode()+b"".join(word.tobytes() for word in self._words)

    def __array__(self, dtype=None, copy=None):
        raise TypeError("implicit primitive projection is forbidden")

    def __float__(self):
        raise TypeError("implicit primitive projection is forbidden")


@dataclass(frozen=True)
class PrimitiveDifference:
    """Exact local difference of two four-word endpoints, without reduction.

    There are exactly two operands and at most 2 x 4 signed source words per
    component. Operands cannot themselves be differences, so neither memory
    nor arithmetic arity grows with the number of accepted steps. Consumers
    contract these words together before the existing physical DD projection.
    """

    current: PrimitiveExpansion
    previous: PrimitiveExpansion

    def __post_init__(self):
        if (type(self.current) is not PrimitiveExpansion
                or type(self.previous) is not PrimitiveExpansion
                or self.current.shape != self.previous.shape):
            raise ContractError("primitive_difference_requires_two_endpoints")

    @property
    def shape(self) -> tuple[int, ...]:
        return self.current.shape

    def immutable_copy(self) -> PrimitiveDifference:
        return self

    def take_flat(self, indices: ArrayLike) -> PrimitiveDifference:
        return PrimitiveDifference(self.current.take_flat(indices), self.previous.take_flat(indices))

    def is_zero(self) -> bool:
        components = self.current.words + tuple(-word for word in self.previous.words)
        return all(_binary_sum_is_zero([float(word.ravel()[i]) for word in components])
                   for i in range(self.current.high.size))

    def identity_bytes(self) -> bytes:
        return (b"paired-endpoint-difference-v1:"+self.current.identity_bytes()
                +b":minus:"+self.previous.identity_bytes())

    def __array__(self, dtype=None, copy=None):
        raise TypeError("implicit primitive difference projection is forbidden")

    def __float__(self):
        raise TypeError("implicit primitive difference projection is forbidden")


def _exact_primitive_sum(left, right) -> PrimitiveExpansion:
    left, right = PrimitiveExpansion.from_value(left), PrimitiveExpansion.from_value(right)
    if left.shape != right.shape:
        raise ContractError("primitive_shape_mismatch")
    return PrimitiveExpansion._owned(_primitive_words((*left.words, *right.words)))


def _exact_primitive_difference(left, right) -> PrimitiveExpansion:
    right = PrimitiveExpansion.from_value(right)
    return _exact_primitive_sum(left, PrimitiveExpansion._owned(tuple(-word for word in right.words)))


def _linear_map_sum(terms: tuple[tuple[float, DD | PrimitiveExpansion | PrimitiveDifference], ...]) -> DD:
    """One bounded map contraction, with DD products preserved before summing.

    Each coefficient is an exact binary64 map parameter. Multiplying its
    source words separately retains all product words until the short linear
    form is reduced to DD; this is not compensation added to a final current.
    No arbitrary-precision physics or growing expression/history is involved.
    """
    if not terms or len(terms) > 8:
        raise ContractError("unsupported_linear_map_arity")
    shape = terms[0][1].shape
    components = []
    for coefficient, value in terms:
        if not np.isfinite(coefficient) or value.shape != shape:
            raise ContractError("linear_map_shape_or_coefficient")
        if coefficient == 0.0:
            continue
        multiplier = None if coefficient in (-1.0, 1.0) else DD(coefficient)
        # A local difference remains two bounded endpoint operands until this
        # final contraction; do not normalize it into a four-word primitive.
        words = (value.current.words+tuple(-word for word in value.previous.words)
                 if isinstance(value, PrimitiveDifference) else value.words
                 if isinstance(value, PrimitiveExpansion) else (value.hi, value.lo))
        for word in words:
            if not np.any(word):
                continue
            # Multiplication by +/-1 is exact for every finite source word.
            # Retain the words themselves until the same fsum contraction.
            if coefficient == 1.0:
                components.append(word.ravel())
            elif coefficient == -1.0:
                components.append((-word).ravel())
            else:
                product = multiplier*DD(word)
                components.extend((product.hi.ravel(), product.lo.ravel()))
    high, low = np.zeros(int(np.prod(shape))), np.zeros(int(np.prod(shape)))
    for index in range(len(high)):
        words = [float(component[index]) for component in components]
        high[index] = fsum(words)
        low[index] = fsum([*words, -high[index]])
    return DD(high.reshape(shape), low.reshape(shape))


def _map_identity(layout: Layout, name: str, modes: Mapping[str, str], parameters: Mapping[str, float]) -> str:
    return _digest({"layout": layout.identity, "name": name, "version": 1,
                    "modes": dict(sorted(modes.items())),
                    "parameters": {key: float(value).hex() for key, value in sorted(parameters.items())}})


@dataclass(frozen=True)
class MappedAuthority:
    """Bounded root plus primitive map; projected words never replace the root.

    Only the relative linear/log/logit maps and the named R1 lift are supported.
    Endpoint composition must be exact in four words; v5 affine transitions
    retain two endpoints when their local difference is wider. Transcendental physical
    evaluations retain the existing DD domain and require observable-specific
    qualification; absolute projections carry no generic sign certificate.
    """

    layout: Layout
    anchor: Mapping[str, DoubleArray]
    modes: Mapping[str, str]
    primitives: Mapping[str, PrimitiveExpansion]
    time: float
    inputs: Vector
    coordinates: Vector
    reference: str
    map_name: str = "relative-fields"
    parameters: Mapping[str, float] = field(default_factory=dict)
    previous_primitives: Mapping[str, PrimitiveExpansion] | None = None
    local_primitives: Mapping[str, PrimitiveExpansion | PrimitiveDifference] | None = None
    previous_authority_identity: str | None = None
    previous_point_identity: str | None = None
    coordinate_kind: str = "local"
    fixed_reference: Point | None = None
    identity: str = field(init=False)
    root_identity: str = field(init=False)
    map_identity: str = field(init=False)
    _projection_cache: Mapping[str, DD] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        names = {spec.id for spec in self.layout.variables}
        if set(self.anchor) != names or set(self.modes) != names:
            raise ContractError("incomplete_authority_anchor_or_modes")
        if self.map_name not in {"relative-fields", "r1-input-lift"}:
            raise ContractError("unsupported_authority_map")
        modes = dict(sorted(self.modes.items()))
        allowed = {"linear", "log", "inactive", "logit"} if self.map_name == "relative-fields" else {
            "linear", "log_zero", "logit", "fixed"}
        if any(mode not in allowed for mode in modes.values()):
            raise ContractError("unsupported_authority_mode")
        if self.map_name == "r1-input-lift" and modes != _r1_modes(self.layout):
            raise ContractError("authority_primitive_roles_mismatch")
        expected = self._primitive_shapes()
        if set(self.primitives) != set(expected):
            raise ContractError("authority_primitive_roles_mismatch")
        for name in ("primitives", "previous_primitives", "local_primitives"):
            values = getattr(self, name)
            if values is not None:
                object.__setattr__(self, name, MappingProxyType({key: value if name == "local_primitives"
                                                               and type(value) is PrimitiveDifference
                                                               else PrimitiveExpansion.from_value(value)
                                                               for key, value in sorted(values.items())}))
        parameters = dict(self.parameters)
        if self.map_name == "relative-fields":
            if parameters:
                raise ContractError("unexpected_authority_parameters")
        elif (set(parameters) != {"thermal_voltage_V"}
              or not np.isfinite(parameters["thermal_voltage_V"]) or parameters["thermal_voltage_V"] <= 0):
            raise ContractError("invalid_thermal_voltage")
        for name, shape in expected.items():
            if type(self.primitives[name]) is not PrimitiveExpansion or self.primitives[name].shape != shape:
                raise ContractError("authority_primitive_shape_or_type")
        for spec in self.layout.variables:
            value = self.anchor[spec.id]
            if type(value) is not DoubleArray or value.shape != spec.shape:
                raise ContractError("authority_anchor_shape_or_type")
            mode, physical = modes[spec.id], value.as_dd()
            if mode == "log" and np.any(physical <= 0):
                raise ContractError("log_coordinate_requires_positive_inventory")
            if mode == "log_zero" and np.any(physical < 0):
                raise ContractError("negative_physical_population")
            if mode == "inactive" and np.any(physical != 0):
                raise ContractError("inactive_inventory_not_zero")
            if mode == "logit" and (np.any(physical < 0) or np.any(physical > 1)):
                raise ContractError("invalid_trap_occupancy")
        for name in ("anchor", "primitives"):
            object.__setattr__(self, name, MappingProxyType({key: value.immutable_copy()
                                                           for key, value in sorted(getattr(self, name).items())}))
        object.__setattr__(self, "modes", MappingProxyType(modes))
        object.__setattr__(self, "parameters", MappingProxyType(parameters))
        object.__setattr__(self, "inputs", frozen_array(self.inputs))
        object.__setattr__(self, "coordinates", frozen_array(self.coordinates))
        if not np.isfinite(self.time) or not self.reference or self.inputs.ndim != 1 or self.coordinates.ndim != 1:
            raise ContractError("invalid_authority_binding")
        object.__setattr__(self, "root_identity", ResolvedAuthority(self.layout.identity, self.anchor).identity)
        object.__setattr__(self, "map_identity", _map_identity(self.layout, self.map_name, modes, parameters))
        if self.coordinate_kind not in {"local", "fixed-reference"}:
            raise ContractError("unknown_coordinate_meaning")
        if self.coordinate_kind == "local" and self.fixed_reference is not None:
            raise ContractError("unexpected_fixed_reference")
        if self.coordinate_kind == "fixed-reference" and (
                self.map_name != "relative-fields" or any(mode != "linear" for mode in modes.values())
                or not isinstance(self.fixed_reference, Point)):
            raise ContractError("affine_trial_requires_linear_reference")
        if (self.previous_primitives is None) != (self.local_primitives is None):
            raise ContractError("incomplete_authority_transition")
        if self.local_primitives is not None:
            if (set(self.local_primitives) != set(expected) or set(self.previous_primitives) != set(expected)
                    or not self.previous_authority_identity or not self.previous_point_identity):
                raise ContractError("incomplete_authority_transition")
            paired = [type(value) is PrimitiveDifference for value in self.local_primitives.values()]
            if any(paired) and (not all(paired) or self.coordinate_kind != "fixed-reference"):
                raise ContractError("paired_transition_requires_affine_endpoints")
            for key in expected:
                local = self.local_primitives[key]
                if isinstance(local, PrimitiveDifference):
                    if (local.shape != expected[key]
                            or local.current.identity_bytes() != self.primitives[key].identity_bytes()
                            or local.previous.identity_bytes() != self.previous_primitives[key].identity_bytes()):
                        raise ContractError("authority_transition_mismatch")
                else:
                    # Preserve the exact historical v3/v4 normalization and
                    # identity check when decoding their four-word deltas.
                    composed = _exact_primitive_sum(self.previous_primitives[key], local)
                    if composed.identity_bytes() != self.primitives[key].identity_bytes():
                        raise ContractError("authority_transition_mismatch")
            for name in ("previous_primitives", "local_primitives"):
                object.__setattr__(self, name, MappingProxyType({key: value.immutable_copy()
                                                               for key, value in sorted(getattr(self, name).items())}))
        elif any(not value.is_zero() for value in self.primitives.values()):
            raise ContractError("authority_transition_required")
        order = [spec.id for spec in self.layout.variables] if self.map_name == "relative-fields" else sorted(expected)
        coordinate_primitives = self.local_primitives
        if self.coordinate_kind == "fixed-reference":
            if self.local_primitives is None:
                raise ContractError("fixed_reference_transition_required")
            base = self._fixed_reference_primitives()
            coordinate_primitives = {key: _exact_primitive_difference(self.primitives[key], base[key])
                                     for key in expected}
        expected_coordinates = (np.concatenate([coordinate_primitives[key].high.ravel() for key in order])
                                if coordinate_primitives is not None and order else np.zeros(sum(int(np.prod(expected[key])) for key in order)))
        if self.coordinates.shape != expected_coordinates.shape or self.coordinates.tobytes() != expected_coordinates.tobytes():
            raise ContractError("authority_coordinate_projection_mismatch")
        if self.coordinate_kind == "fixed-reference":
            if self.reference != f"affine-trial:{self.map_identity}:{self.fixed_reference.identity}":
                raise ContractError("authority_fixed_reference_mismatch")
        else:
            prefix = f"relative:{self.map_identity}:" if self.map_name == "relative-fields" else "physical-input-lift:"
            if not self.reference.startswith(prefix):
                raise ContractError("authority_reference_mismatch")
            if self.local_primitives is not None and self.reference != prefix + self.previous_point_identity:
                raise ContractError("authority_predecessor_reference_mismatch")
        object.__setattr__(self, "identity", _digest(self.payload()))
        projections = {}
        for spec in self.layout.variables:
            value = self._project(spec.id)
            if (spec.lower is not None and np.any(value < spec.lower)
                    or spec.upper is not None and np.any(value > spec.upper)):
                raise ContractError("physical_state_outside_domain")
            projections[spec.id] = value
        # The full source, map and arrays have already been made immutable.
        # Reuse this validated projection without changing its word values.
        object.__setattr__(self, "_projection_cache", MappingProxyType(projections))

    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        return value.view() if isinstance(value, np.ndarray) else value

    @property
    def layout_identity(self) -> str:
        return self.layout.identity

    def _fixed_reference_primitives(self) -> Mapping[str, PrimitiveExpansion]:
        reference = self.fixed_reference
        if not isinstance(reference, Point) or reference.state.layout.identity != self.layout.identity:
            raise ContractError("affine_trial_reference_layout")
        authority = reference.state.authority
        if isinstance(authority, MappedAuthority):
            if authority.coordinate_kind != "local":
                raise ContractError("affine_trial_reference_rebase_forbidden")
            self._compatible(authority)
            return authority.primitives
        if not isinstance(authority, ResolvedAuthority) or authority.identity != self.root_identity:
            raise ContractError("affine_trial_reference_root_mismatch")
        return {key: PrimitiveExpansion.from_value(np.zeros(value.shape)) for key, value in self.anchor.items()}

    def _primitive_shapes(self) -> dict[str, tuple[int, ...]]:
        shapes = {spec.id: spec.shape for spec in self.layout.variables}
        if self.map_name == "relative-fields":
            return shapes
        if not {"n_m3", "p_m3", "phi_V"} <= shapes.keys() or len({shapes[key] for key in ("n_m3", "p_m3", "phi_V")}) != 1:
            raise ContractError("input_lift_bulk_fields_mismatch")
        roles = {"electron": "n_m3", "hole": "p_m3", "potential": "phi_V",
                 "positive": "positive_m3", "occupancy": "occupancy", "trace_density": "trace_state_m3",
                 "trace_potential": "trace_potential_V", "voltage_lift": "phi_V", "trace_voltage_lift": "trace_potential_V"}
        return {key: shapes[name] for key, name in roles.items() if name in shapes}

    def _drive(self, name: str) -> DD:
        shape = self.anchor[name].shape
        indices = np.arange(int(np.prod(shape)), dtype=np.intp).reshape(shape)
        terms = self._drive_terms(name, indices)
        return _linear_map_sum(terms) if terms else DD(np.zeros(shape))

    def _drive_terms(self, name: str, selection: np.ndarray) -> tuple[tuple[float, PrimitiveExpansion], ...]:
        """The same named map as _drive, before reducing its primitive terms."""
        return tuple((coefficient, self.primitives[role].take_flat(selection))
                     for coefficient, role in self._drive_coefficients(name))

    def _drive_coefficients(self, name: str) -> tuple[tuple[float, str], ...]:
        if self.map_name == "relative-fields":
            return ((1.0, name),)
        vt = self.parameters["thermal_voltage_V"]
        return {
            "n_m3": ((1.0, "electron"), (1.0, "potential")),
            "p_m3": ((1.0, "hole"), (-1.0, "potential")),
            "phi_V": ((vt, "potential"), (1.0, "voltage_lift")),
            "dqfn_V": ((vt, "electron"), (-1.0, "voltage_lift")),
            "dqfp_V": ((vt, "hole"), (1.0, "voltage_lift")),
            "trace_potential_V": ((vt, "trace_potential"), (1.0, "trace_voltage_lift")),
            "positive_m3": ((1.0, "positive"),), "occupancy": ((1.0, "occupancy"),),
            "trace_state_m3": ((1.0, "trace_density"),),
        }.get(name, ())

    def _drive_difference(self, other: MappedAuthority, name: str,
                          right_selection: np.ndarray, left_selection: np.ndarray) -> DD:
        self._compatible(other)
        terms = self._drive_terms(name, right_selection) + tuple(
            (-coefficient, value) for coefficient, value in other._drive_terms(name, left_selection))
        return _linear_map_sum(terms) if terms else DD(np.zeros(right_selection.shape))

    def _project(self, name: str) -> DD:
        cached = getattr(self, "_projection_cache", None)
        if cached is not None:
            return cached[name]
        base = self.anchor[name].as_dd()
        if self.coordinate_kind == "fixed-reference":
            selection = np.arange(int(np.prod(base.shape)), dtype=np.intp).reshape(base.shape)
            # V4 affine trials contract the physical anchor and every source
            # primitive together. Projecting the drive first could erase a
            # small positive or negative remainder at complete depletion.
            return _linear_map_sum(((1.0, base),)+self._drive_terms(name, selection))
        # Preserve the declared projection semantics of historical v3 maps.
        drive, mode = self._drive(name), self.modes[name]
        if mode in {"log", "log_zero"}:
            return base*drive.exp()
        if mode == "logit":
            positive = drive >= 0
            hi, lo = np.zeros(base.shape), np.zeros(base.shape)
            if np.any(positive):
                f, d = base[positive], drive[positive]
                value = f/(f+(1-f)*(-d).exp())
                hi[positive], lo[positive] = value.hi, value.lo
            if np.any(~positive):
                f, e = base[~positive], drive[~positive].exp()
                value = f*e/((1-f)+f*e)
                hi[~positive], lo[~positive] = value.hi, value.lo
            return DD(hi, lo)
        if mode in {"inactive", "fixed"}:
            if np.any(drive != 0): raise ContractError("inactive_coordinate_change")
            return base
        return base+drive

    def field(self, name: str) -> MappedArray:
        return MappedArray(self, name)

    def validate_binding(self, time, y, inputs, reference):
        if (time != self.time or reference != self.reference or y.tobytes() != self.coordinates.tobytes()
                or y.shape != self.coordinates.shape or inputs.shape != self.inputs.shape
                or inputs.tobytes() != self.inputs.tobytes()):
            raise ContractError("point_authority_binding_mismatch")

    def validate_transition(self, left: Point, right: Point) -> None:
        if right.state.authority.identity != self.identity:
            raise ContractError("incompatible_state_authority")
        if left.identity != right.identity and left.identity != self.previous_point_identity:
            raise ContractError("increment_predecessor_point_mismatch")
        if self.coordinate_kind == "fixed-reference" and left.identity != self.fixed_reference.identity:
            authority = left.state.authority
            if (not isinstance(authority, MappedAuthority) or authority.coordinate_kind != "fixed-reference"
                    or authority.fixed_reference.identity != self.fixed_reference.identity):
                raise ContractError("trial_predecessor_reference_mismatch")
            self._compatible(authority)

    def _temporal_primitives(self, left: StateView) -> Mapping[str, PrimitiveExpansion | PrimitiveDifference]:
        if left.authority.identity == self.identity:
            return {key: PrimitiveExpansion.from_value(np.zeros(value.shape))
                    for key, value in self.primitives.items()}
        if left.authority.identity != self.previous_authority_identity or self.local_primitives is None:
            raise ContractError("increment_transition_anchor_mismatch")
        if isinstance(left.authority, MappedAuthority):
            self._compatible(left.authority)
            if any(left.authority.primitives[key].identity_bytes()
                   != self.previous_primitives[key].identity_bytes() for key in self.primitives):
                raise ContractError("increment_transition_primitive_mismatch")
        elif (not isinstance(left.authority, ResolvedAuthority)
              or left.authority.identity != self.root_identity
              or any(not value.is_zero() for value in self.previous_primitives.values())):
            raise ContractError("incompatible_state_authority")
        return self.local_primitives

    def temporal_log_ratio(self, left: StateView, variable_id: str,
                           indices: ArrayLike | None) -> DoubleArray:
        changes = self._temporal_primitives(left)
        before, after = left.field(variable_id), self.field(variable_id)
        selection = np.arange(int(np.prod(after.shape)), dtype=np.intp).reshape(after.shape)
        if indices is not None:
            before, after = before.take(indices), after.take(indices)
            selection = selection[integer_indices(indices)]
        old, new = before.as_dd(), after.as_dd()
        if np.any(old <= 0) or np.any(new <= 0):
            raise ContractError("log_ratio_requires_positive_fields")
        if self.modes[variable_id] in {"log", "log_zero"}:
            terms = tuple((coefficient, changes[role].take_flat(selection))
                          for coefficient, role in self._drive_coefficients(variable_id))
            return DoubleArray.from_dd(_linear_map_sum(terms) if terms else DD(np.zeros(after.shape)))
        change = self.delta_from(left)[variable_id]
        if indices is not None:
            change = change.take(indices)
        relative = change.as_dd()/old
        close = (relative > -0.5) & (relative < 0.5)
        high, low = np.zeros(after.shape), np.zeros(after.shape)
        if np.any(close):
            value = relative[close].log1p(); high[close], low[close] = value.hi, value.lo
        if np.any(~close):
            value = _log_ratio(new[~close], old[~close]); high[~close], low[~close] = value.hi, value.lo
        return DoubleArray(high, low)

    def _temporal_face_form(self, left: StateView, pairs: ArrayLike,
                            coefficients: tuple[tuple[float, str], ...]) -> DoubleArray:
        changes = self._temporal_primitives(left)
        pairs = integer_indices(pairs)
        # Only identical named terms with exactly opposite coefficients cancel.
        # Other coefficients/products remain separate through the DD reduction.
        remaining: list[tuple[float, str]] = []
        for coefficient, role in coefficients:
            for index, (previous, name) in enumerate(remaining):
                if name == role and previous == -coefficient:
                    remaining.pop(index)
                    break
            else:
                remaining.append((coefficient, role))
        terms = tuple(term for coefficient, role in remaining for term in (
            (coefficient, changes[role].take_flat(pairs[:, 1])),
            (-coefficient, changes[role].take_flat(pairs[:, 0]))))
        return DoubleArray.from_dd(_linear_map_sum(terms) if terms else DD(np.zeros(len(pairs))))

    def face_delta(self, left: StateView, variable_id: str, pairs: ArrayLike) -> DoubleArray:
        if self.modes[variable_id] not in {"linear", "inactive", "fixed"}:
            raise ContractError("face_delta_requires_affine_map")
        return self._temporal_face_form(left, pairs, self._drive_coefficients(variable_id))

    def _affine_log_face_delta(self, left: StateView, density_id: str, pairs: np.ndarray) -> DD:
        before, after = left.field(density_id), self.field(density_id)
        i, j = pairs[:, 0], pairs[:, 1]
        n0l, n0r = before.take(i).as_dd(), before.take(j).as_dd()
        n1l = after.take(i).as_dd()
        initial_drop = before.take(j).difference(before.take(i)).as_dd()
        drop_change = self.face_delta(left, density_id, pairs).as_dd()
        change_left = self.delta_from(left)[density_id].take(i).as_dd()
        # Change of the spatial density ratio, before taking its logarithm:
        # (n1R*n0L)/(n1L*n0R)-1 =
        # [n0L*delta(nR-nL) - (n0R-n0L)*delta(nL)]/(n0R*n1L).
        # Source-bound spatial differences retain an existing tiny gradient
        # when both populations receive the same large affine increment.
        ratio_change = (n0l*drop_change-initial_drop*change_left)/(n0r*n1l)
        close = (ratio_change > -0.5) & (ratio_change < 0.5)
        high, low = np.zeros(len(pairs)), np.zeros(len(pairs))
        if np.any(close):
            value = ratio_change[close].log1p()
            high[close], low[close] = value.hi, value.lo
        if np.any(~close):
            # A large ratio change has no near-zero temporal face signal;
            # individual finite logs avoid log1p rounding at depletion.
            value = (self.temporal_log_ratio(left, density_id, j[~close]).as_dd()
                     - self.temporal_log_ratio(left, density_id, i[~close]).as_dd())
            high[~close], low[~close] = value.hi, value.lo
        return DD(high, low)

    def electrochemical_delta(self, left: StateView, density_id: str, potential_id: str,
                              pairs: ArrayLike, thermal_voltage: float,
                              potential_sign: int) -> DoubleArray | None:
        if self.modes[potential_id] not in {"linear", "inactive", "fixed"}:
            raise ContractError("temporal_activity_requires_affine_potential")
        pairs = integer_indices(pairs)
        for value in (left.field(density_id), self.field(density_id)):
            if np.any(value.take(pairs.ravel()).as_dd() <= 0):
                raise ContractError("log_ratio_requires_positive_fields")
        if self.modes[density_id] in {"linear", "inactive", "fixed"}:
            density = self._affine_log_face_delta(left, density_id, pairs)
            potential = self.face_delta(left, potential_id, pairs).as_dd()
            return DoubleArray.from_dd(density+potential_sign*potential/thermal_voltage)
        if self.modes[density_id] not in {"log", "log_zero"}:
            return None
        coefficients = tuple((thermal_voltage*coefficient, role)
                             for coefficient, role in self._drive_coefficients(density_id)) + tuple(
            (potential_sign*coefficient, role) for coefficient, role in self._drive_coefficients(potential_id))
        return DoubleArray.from_dd(self._temporal_face_form(left, pairs, coefficients).as_dd()/thermal_voltage)

    def project(self, name: str) -> FieldProjection:
        value = DoubleArray.from_dd(self._project(name))
        exact_root = all(v.is_zero() for v in self.primitives.values())
        bound = FloatArray(np.zeros(value.shape)) if exact_root else None
        evaluator = ("stateless-dd-affine-joint-v1" if self.coordinate_kind == "fixed-reference"
                     else "stateless-dd-map-v1")
        return FieldProjection(value, self.identity, evaluator, bound)

    def electrochemical_difference(self, density_id: str, potential_id: str, pairs: ArrayLike,
                                   thermal_voltage: float, potential_sign: int) -> DoubleArray | None:
        if self.map_name != "r1-input-lift" or potential_id != "phi_V" or density_id not in {"n_m3", "p_m3"}:
            return None
        pairs = integer_indices(pairs)
        density, potential = self.anchor[density_id].as_dd(), self.anchor[potential_id].as_dd()
        if (pairs.ndim != 2 or pairs.shape[1] != 2 or density.ndim != 1 or density.shape != potential.shape
                or np.any(pairs < 0) or np.any(pairs >= density.size)):
            raise ContractError("face_outside_support")
        left, right = pairs[:, 0], pairs[:, 1]
        base = _log_ratio(density[right], density[left]) + potential_sign*(potential[right]-potential[left])/thermal_voltage
        role, coupled_sign = ("electron", 1) if density_id == "n_m3" else ("hole", -1)
        primary, lift = self.primitives[role], self.primitives["voltage_lift"]
        terms = [(thermal_voltage, primary.take_flat(right)), (-thermal_voltage, primary.take_flat(left)),
                 (float(potential_sign), lift.take_flat(right)), (-float(potential_sign), lift.take_flat(left))]
        # With equal VT, the same named potential primitive has coefficient
        # c+s=0 for electron/hole activities. Cancel that identity before a
        # coordinate difference or VT product can round away the weak drive.
        mapped_vt = self.parameters["thermal_voltage_V"]
        if mapped_vt != thermal_voltage or coupled_sign + potential_sign != 0:
            zphi = self.primitives["potential"]
            terms.extend(((coupled_sign*thermal_voltage, zphi.take_flat(right)),
                          (-coupled_sign*thermal_voltage, zphi.take_flat(left)),
                          (potential_sign*mapped_vt, zphi.take_flat(right)),
                          (-potential_sign*mapped_vt, zphi.take_flat(left))))
        return DoubleArray.from_dd(base + _linear_map_sum(tuple(terms))/thermal_voltage)

    def _compatible(self, other: MappedAuthority) -> None:
        if self.map_identity != other.map_identity or self.root_identity != other.root_identity:
            raise ContractError("incompatible_state_authority")

    def delta_from(self, left: StateView) -> Mapping[str, PhysicalArray]:
        if left.authority.identity != self.previous_authority_identity and left.authority.identity != self.identity:
            raise ContractError("increment_transition_anchor_mismatch")
        if isinstance(left.authority, MappedAuthority):
            self._compatible(left.authority)
            if left.authority.identity != self.identity and any(
                    left.authority.primitives[key].identity_bytes() != self.previous_primitives[key].identity_bytes()
                    for key in self.primitives):
                raise ContractError("increment_transition_primitive_mismatch")
            return {name: self.field(name).difference(left.field(name)) for name in self.anchor}
        if left.authority.identity != self.root_identity:
            raise ContractError("incompatible_state_authority")
        if any(not value.is_zero() for value in self.previous_primitives.values()):
            raise ContractError("increment_transition_primitive_mismatch")
        # Root-relative finite change uses the same stable map, not a projected
        # endpoint subtraction. This also permits the first named R1 lift.
        return {name: DoubleArray.from_dd(self._root_delta(name)) for name in self.anchor}

    def _root_delta(self, name: str) -> DD:
        base, drive, mode = self.anchor[name].as_dd(), self._drive(name), self.modes[name]
        if mode in {"log", "log_zero"}: return base*drive.expm1()
        if mode == "logit":
            active = (base > 0) & (base < 1)
            high, low = np.zeros(base.shape), np.zeros(base.shape)
            if np.any(active):
                f, e = base[active], drive[active].expm1()
                value = f*(1-f)*e/(1+f*e)
                high[active], low[active] = value.hi, value.lo
            return DD(high, low)
        return drive

    def payload(self, *, include_transition: bool = True) -> dict[str, Any]:
        payload = {"schema": "solarlab.state-authority.v3", "layout": self.layout.identity,
                   "map": self.map_name, "version": 1, "modes": dict(self.modes),
                   "parameters": {key: float(value).hex() for key, value in self.parameters.items()},
                   "anchor": {key: _word_payload(value) for key, value in self.anchor.items()},
                   "primitives": {key: _primitive_payload(value) for key, value in self.primitives.items()},
                   "time": float(self.time).hex(), "inputs": [float(v).hex() for v in self.inputs],
                   "coordinates": [float(v).hex() for v in self.coordinates], "reference": self.reference,
                   "composition_policy": "exact-four-word-primitives-v1"}
        if self.coordinate_kind == "fixed-reference":
            payload["schema"] = "solarlab.state-authority.v4"
            # Empty layouts have no paired operands: retain their canonical
            # legacy v4 identity even when trial's requested policy is paired.
            if self.local_primitives and all(type(value) is PrimitiveDifference
                                             for value in self.local_primitives.values()):
                payload["schema"] = "solarlab.state-authority.v5"
                payload["composition_policy"] = "four-word-endpoints-paired-transition-v1"
            payload["coordinate_contract"] = {"kind": "fixed-reference-affine-v1",
                                               "solver_projection": "first-word",
                                               "reference": encode_point(self.fixed_reference)}
        if include_transition:
            payload["transition"] = None if self.local_primitives is None else {
                "previous_authority": self.previous_authority_identity,
                "previous_point": self.previous_point_identity,
                "previous_primitives": {key: _primitive_payload(value) for key, value in self.previous_primitives.items()},
                "local_primitives": {key: _difference_payload(value) if isinstance(value, PrimitiveDifference)
                                     else _primitive_payload(value) for key, value in self.local_primitives.items()}}
        return payload


@dataclass(frozen=True, init=False)
class MappedArray(DoubleArray):
    """PhysicalArray selection retaining its generating authority and map.

    as_dd() is an explicit bounded materialization for nonsingular products.
    Differences and log ratios must use these public methods before projection.
    """

    authority: MappedAuthority
    variable: str
    _selection: np.ndarray

    def __init__(self, authority: MappedAuthority, variable: str, selection=None):
        shape = authority.anchor[variable].shape
        indices = np.arange(int(np.prod(shape)), dtype=np.intp).reshape(shape) if selection is None else integer_indices(selection)
        projected = authority._project(variable).ravel()[indices]
        super().__init__(projected.hi, projected.lo)
        object.__setattr__(self, "authority", authority)
        object.__setattr__(self, "variable", variable)
        object.__setattr__(self, "_selection", np.frombuffer(indices.tobytes(), dtype=np.intp).reshape(indices.shape))

    def immutable_copy(self) -> MappedArray:
        return MappedArray(self.authority, self.variable, self._selection)

    def take(self, indices: ArrayLike) -> MappedArray:
        indices = integer_indices(indices)
        if np.any(indices < 0) or np.any(indices >= self.shape[0]):
            raise ContractError("index_outside_support")
        return MappedArray(self.authority, self.variable, self._selection[indices])

    def identity_bytes(self) -> bytes:
        return (b"authority-field-v1:" + self.authority.identity.encode() + self.variable.encode()
                + repr(self.shape).encode() + self._selection.tobytes())

    def _pair(self, other: PhysicalArray, *, need_change: bool = True):
        if not isinstance(other, MappedArray) or self.variable != other.variable or self.shape != other.shape:
            raise ContractError("authority_field_mismatch")
        self.authority._compatible(other.authority)
        base = self.authority.anchor[self.variable].as_dd().ravel()
        left, right = base[other._selection], base[self._selection]
        change = (self.authority._drive_difference(other.authority, self.variable,
                                                  self._selection, other._selection)
                  if need_change else None)
        return left, right, change

    def log_ratio(self, other: PhysicalArray) -> DoubleArray:
        mode = self.authority.modes[self.variable]
        left, right, change = self._pair(other, need_change=mode in {"log", "log_zero"})
        if mode in {"log", "log_zero"}:
            return DoubleArray.from_dd(_log_ratio(right, left)+change)
        difference = self.difference(other).as_dd()
        actual_left, actual_right = other.as_dd(), self.as_dd()
        if np.any(actual_left <= 0) or np.any(actual_right <= 0):
            raise ContractError("log_ratio_requires_positive_fields")
        relative = difference/actual_left
        close = (relative > -0.5) & (relative < 0.5)
        high, low = np.zeros(self.shape), np.zeros(self.shape)
        if np.any(close):
            value = relative[close].log1p(); high[close], low[close] = value.hi, value.lo
        if np.any(~close):
            value = _log_ratio(actual_right[~close], actual_left[~close])
            high[~close], low[~close] = value.hi, value.lo
        return DoubleArray(high, low)

    def difference(self, other: PhysicalArray) -> DoubleArray:
        left, right, _ = self._pair(other, need_change=False)
        mode = self.authority.modes[self.variable]
        if mode in {"linear", "inactive", "fixed"}:
            if "fixed-reference" in (self.authority.coordinate_kind, other.authority.coordinate_kind):
                # V4 paired observables use the same joint affine source as
                # field projection. Reducing the two drives first can erase
                # the entire surviving gradient after anchor cancellation.
                terms = ((1.0, right), (-1.0, left)) + self.authority._drive_terms(
                    self.variable, self._selection) + tuple(
                    (-coefficient, value) for coefficient, value in other.authority._drive_terms(
                        self.variable, other._selection))
                return DoubleArray.from_dd(_linear_map_sum(terms))
        change = self.authority._drive_difference(other.authority, self.variable,
                                                  self._selection, other._selection)
        if mode in {"linear", "inactive", "fixed"}:
            return DoubleArray.from_dd((right-left)+change)
        actual_left, actual_right = other.as_dd(), self.as_dd()
        if mode in {"log", "log_zero"}:
            active = (left > 0) & (right > 0)
            output = actual_right-actual_left
            hi, lo = output.hi.copy(), output.lo.copy()
            if np.any(active):
                ratio = _log_ratio(right[active], left[active])+change[active]
                positive = ratio >= 0
                vh, vl = np.zeros(ratio.shape), np.zeros(ratio.shape)
                if np.any(positive):
                    value = actual_right[active][positive]*(-(-ratio[positive]).expm1())
                    vh[positive], vl[positive] = value.hi, value.lo
                if np.any(~positive):
                    value = actual_left[active][~positive]*ratio[~positive].expm1()
                    vh[~positive], vl[~positive] = value.hi, value.lo
                hi[active], lo[active] = vh, vl
            return DoubleArray(hi, lo)
        # Logit finite differences share the same odds authority. Endpoint
        # values are evaluated separately, so depletion never becomes zero by
        # recombining an almost -1 finite increment with the root.
        active = (left > 0) & (left < 1) & (right > 0) & (right < 1)
        output = actual_right-actual_left
        hi, lo = output.hi.copy(), output.lo.copy()
        if np.any(active):
            odds = (_log_ratio(right[active], left[active])
                    - _log_ratio(1-right[active], 1-left[active]) + change[active])
            f, e = actual_left[active], odds.expm1()
            value = f*(1-f)*e/(1+f*e)
            hi[active], lo[active] = value.hi, value.lo
        return DoubleArray(hi, lo)


@dataclass(frozen=True)
class RelativeCoordinates:
    """Keep a resolved root and compose named physical primitive coordinates.

    Trials share the accepted authority. The local transition and complete map
    are retained on acceptance; an evaluated endpoint is never a new root.
    """

    layout: Layout
    modes: Mapping[str, str]

    def __post_init__(self) -> None:
        modes = dict(self.modes)
        if set(modes) != {spec.id for spec in self.layout.variables}:
            raise ContractError("incomplete_coordinate_modes")
        if any(mode not in {"linear", "log", "logit", "inactive"} for mode in modes.values()):
            raise ContractError("unknown_coordinate_mode")
        object.__setattr__(self, "modes", MappingProxyType(modes))

    @property
    def identity(self) -> str:
        return _map_identity(self.layout, "relative-fields", self.modes, {})

    def initial(self, state: StateView, time: float = 0.0, inputs: ArrayLike = ()) -> Point:
        if state.layout.identity != self.layout.identity:
            raise ContractError("coordinate_layout_mismatch")
        if not isinstance(state.authority, ResolvedAuthority):
            raise ContractError("explicit_authority_rebase_required")
        if any(type(value) is not DoubleArray for value in state.fields.values()):
            raise ContractError("precision_coordinate_requires_double_words")
        y, inputs = np.zeros(self.layout.size), frozen_array(inputs)
        reference = f"relative:{self.identity}:initial"
        authority = MappedAuthority(self.layout, state.fields, self.modes,
                                    {spec.id: DoubleArray(np.zeros(spec.shape)) for spec in self.layout.variables},
                                    time, inputs, y, reference)
        return Point(time, y, inputs, StateView(self.layout, authority=authority), reference)

    def advance(self, anchor: Point, increment: ArrayLike, time: float,
                inputs: ArrayLike = ()) -> tuple[Point, StateIncrement]:
        dz = PrimitiveExpansion.from_value(increment)
        if dz.shape != (self.layout.size,) or anchor.state.layout.identity != self.layout.identity:
            raise ContractError("coordinate_layout_mismatch")
        if not anchor.coordinate_reference.startswith(f"relative:{self.identity}:"):
            raise ContractError("coordinate_reference_mismatch")
        previous = anchor.state.authority
        if not isinstance(previous, MappedAuthority) or previous.map_identity != self.identity:
            raise ContractError("coordinate_authority_mismatch")
        local, accumulated = {}, {}
        for spec in self.layout.variables:
            offset = self.layout.offsets[spec.id]
            local[spec.id] = dz.take_flat(np.arange(offset.start, offset.stop, dtype=np.intp)).reshape(spec.shape)
            accumulated[spec.id] = _exact_primitive_sum(previous.primitives[spec.id], local[spec.id])
        y, inputs = dz.high, frozen_array(inputs)
        reference = f"relative:{self.identity}:{anchor.identity}"
        authority = MappedAuthority(self.layout, previous.anchor, self.modes, accumulated,
                                    time, inputs, y, reference, previous_primitives=previous.primitives,
                                    local_primitives=local, previous_authority_identity=previous.identity,
                                    previous_point_identity=anchor.identity)
        right = Point(time, y, inputs, StateView(self.layout, authority=authority), reference)
        return right, StateIncrement.from_points(anchor, right)

    def trial(self, reference: Point, cumulative_increment: ArrayLike | DoubleArray | PrimitiveExpansion,
              time: float, inputs: ArrayLike = (), *, predecessor: Point | None = None,
              transition_representation: str = "paired-endpoints-v1"
              ) -> tuple[Point, StateIncrement]:
        """Affine physical trial at a fixed reference, with an actual predecessor.

        Point.y is the first-word projection of the cumulative SI remainder;
        complete input words remain authoritative and are serialized. Local
        predecessor differences are distinct from these fixed coordinates.
        Existing advance() continues to expose local increments as Point.y.
        The explicit four-word mode is for faithful historical v4 restoration;
        it retains the original capacity rejection rather than rewriting it.
        """
        if transition_representation not in {"paired-endpoints-v1", "four-word-v1"}:
            raise ContractError("unknown_affine_transition_representation")
        if any(mode != "linear" for mode in self.modes.values()):
            raise ContractError("affine_trial_requires_linear_reference")
        if not isinstance(reference, Point) or reference.state.layout.identity != self.layout.identity:
            raise ContractError("affine_trial_reference_layout")
        cumulative = PrimitiveExpansion.from_value(cumulative_increment)
        if cumulative.shape != (self.layout.size,):
            raise ContractError("coordinate_layout_mismatch")
        source = reference.state.authority
        if isinstance(source, ResolvedAuthority):
            if any(type(value) is not DoubleArray for value in reference.state.fields.values()):
                raise ContractError("precision_coordinate_requires_double_words")
            anchor = reference.state.fields
            base = {spec.id: PrimitiveExpansion.from_value(np.zeros(spec.shape)) for spec in self.layout.variables}
        elif isinstance(source, MappedAuthority) and source.map_identity == self.identity:
            if source.coordinate_kind != "local":
                raise ContractError("affine_trial_reference_rebase_forbidden")
            anchor, base = source.anchor, source.primitives
        else:
            raise ContractError("coordinate_authority_mismatch")
        previous = reference if predecessor is None else predecessor
        if not isinstance(previous, Point) or previous.state.layout.identity != self.layout.identity:
            raise ContractError("trial_predecessor_reference_mismatch")
        previous_authority = previous.state.authority
        if previous.identity == reference.identity:
            old = base
        elif (isinstance(previous_authority, MappedAuthority)
              and previous_authority.coordinate_kind == "fixed-reference"
              and previous_authority.fixed_reference.identity == reference.identity
              and previous_authority.map_identity == self.identity):
            old = previous_authority.primitives
        else:
            raise ContractError("trial_predecessor_reference_mismatch")
        current, local = {}, {}
        for spec in self.layout.variables:
            offset = self.layout.offsets[spec.id]
            value = cumulative.take_flat(np.arange(offset.start, offset.stop, dtype=np.intp)).reshape(spec.shape)
            current[spec.id] = _exact_primitive_sum(base[spec.id], value)
            local[spec.id] = (PrimitiveDifference(current[spec.id], old[spec.id])
                              if transition_representation == "paired-endpoints-v1" else
                              _exact_primitive_difference(current[spec.id], old[spec.id]))
        # Use the authority's canonical cumulative words for the solver view.
        # In particular, a valid input -0 has canonical physical remainder +0;
        # the exact byte binding below remains strict rather than using allclose.
        coordinate_words = [_exact_primitive_difference(current[spec.id], base[spec.id])
                            for spec in self.layout.variables]
        y = np.concatenate([value.high.ravel() for value in coordinate_words]) if coordinate_words else np.empty(0)
        inputs = frozen_array(inputs)
        binding = f"affine-trial:{self.identity}:{reference.identity}"
        authority = MappedAuthority(self.layout, anchor, self.modes, current, time, inputs, y, binding,
                                    previous_primitives=old, local_primitives=local,
                                    previous_authority_identity=previous_authority.identity,
                                    previous_point_identity=previous.identity,
                                    coordinate_kind="fixed-reference", fixed_reference=reference)
        right = Point(time, y, inputs, StateView(self.layout, authority=authority), binding)
        return right, StateIncrement.from_points(previous, right)

    def rebase(self, point: Point, maximum_projection_error: Mapping[str, ArrayLike]) -> Point:
        """Only a certified projection can become an explicitly declared root.

        Nontrivial transcendental maps have no generic sign certificate here;
        they stay rooted until a consumer-specific rebase qualification exists.
        """
        if set(maximum_projection_error) != set(point.state.fields):
            raise ContractError("incomplete_rebase_error_budget")
        fields = []
        for name in point.state.fields:
            projection = point.state.project_field(name)
            projection.require_error(maximum_projection_error[name])
            fields.append((name, projection.value))
        return self.initial(StateView(self.layout, fields), point.time, point.inputs)


def _r1_modes(layout: Layout) -> dict[str, str]:
    logarithmic = {"n_m3", "p_m3", "positive_m3", "trace_state_m3"}
    linear = {"phi_V", "dqfn_V", "dqfp_V", "trace_potential_V"}
    return {spec.id: ("log_zero" if spec.id in logarithmic else "linear" if spec.id in linear
                      else "logit" if spec.id == "occupancy" else "fixed") for spec in layout.variables}


@dataclass(frozen=True)
class InputLiftCoordinates:
    """The named R1 physical input map, without a private Newton/solver state."""

    layout: Layout
    thermal_voltage_V: float

    def advance(self, anchor: Point, increments: Mapping[str, ArrayLike | DoubleArray | PrimitiveExpansion], time: float,
                voltage_lift_V: ArrayLike | DoubleArray | PrimitiveExpansion | None = None,
                trace_voltage_lift_V: ArrayLike | DoubleArray | PrimitiveExpansion | None = None,
                inputs: ArrayLike | None = None) -> tuple[Point, StateIncrement]:
        if anchor.state.layout.identity != self.layout.identity:
            raise ContractError("coordinate_layout_mismatch")
        params, modes = {"thermal_voltage_V": self.thermal_voltage_V}, _r1_modes(self.layout)
        previous = anchor.state.authority
        if isinstance(previous, MappedAuthority):
            if previous.map_identity != _map_identity(self.layout, "r1-input-lift", modes, params):
                raise ContractError("incompatible_state_authority")
            root, old = previous.anchor, previous.primitives
            shapes = previous._primitive_shapes()
        else:
            if not isinstance(previous, ResolvedAuthority):
                raise ContractError("explicit_authority_rebase_required")
            root = anchor.state.fields
            field_shapes = {spec.id: spec.shape for spec in self.layout.variables}
            roles = {"electron": "n_m3", "hole": "p_m3", "potential": "phi_V", "positive": "positive_m3",
                     "occupancy": "occupancy", "trace_density": "trace_state_m3", "trace_potential": "trace_potential_V",
                     "voltage_lift": "phi_V", "trace_voltage_lift": "trace_potential_V"}
            shapes = {key: field_shapes[name] for key, name in roles.items() if name in field_shapes}
            old = {key: DoubleArray(np.zeros(shape)) for key, shape in shapes.items()}
        if not set(increments) <= set(shapes)-{"voltage_lift", "trace_voltage_lift"}:
            raise ContractError("unknown_input_lift_coordinate")
        raw = dict(increments)
        if voltage_lift_V is not None: raw["voltage_lift"] = voltage_lift_V
        if trace_voltage_lift_V is not None: raw["trace_voltage_lift"] = trace_voltage_lift_V
        if not set(raw) <= shapes.keys():
            raise ContractError("input_lift_field_missing")
        local, accumulated = {}, {}
        for key, shape in sorted(shapes.items()):
            value = raw.get(key, np.zeros(shape))
            local[key] = PrimitiveExpansion.from_value(value)
            if local[key].shape != shape:
                raise ContractError("input_lift_coordinate_shape_mismatch")
            accumulated[key] = _exact_primitive_sum(old[key], local[key])
        y = np.concatenate([value.high.ravel() for value in local.values()])
        inputs = anchor.inputs if inputs is None else frozen_array(inputs)
        reference = f"physical-input-lift:{anchor.identity}"
        authority = MappedAuthority(self.layout, root, modes, accumulated, time, inputs, y, reference,
                                    "r1-input-lift", params, old, local, previous.identity, anchor.identity)
        right = Point(time, y, inputs, StateView(self.layout, authority=authority), reference)
        return right, StateIncrement.from_points(anchor, right)


def bernoulli(value: DoubleArray) -> DoubleArray:
    """Stable x/(exp(x)-1) with an exact zero branch and explicit DD range."""
    x = value.as_dd()
    high, low = np.ones(x.shape), np.zeros(x.shape)
    positive, negative = x > 0, x < 0
    if np.any(positive):
        z = x[positive]
        result = z * (-z).exp() / (-(-z).expm1())
        high[positive], low[positive] = result.hi, result.lo
    if np.any(negative):
        z = x[negative]
        result = -z / (-z.expm1())
        high[negative], low[negative] = result.hi, result.lo
    return DoubleArray(high, low)


def _word_payload(value: DoubleArray) -> dict[str, Any]:
    return {"shape": list(value.shape), "high": [float(x).hex() for x in value.high.ravel()],
            "low": [float(x).hex() for x in value.low.ravel()]}


def _primitive_payload(value: PrimitiveExpansion) -> dict[str, Any]:
    return {"shape": list(value.shape), "precision": "input-expansion4-v1",
            "words": [[float(x).hex() for x in word.ravel()] for word in value.words]}


def _difference_payload(value: PrimitiveDifference) -> dict[str, Any]:
    return {"shape": list(value.shape), "precision": "paired-endpoint-difference-v1",
            "current": _primitive_payload(value.current), "previous": _primitive_payload(value.previous)}


def _decode_difference(payload: Mapping[str, Any], shape: tuple[int, ...]) -> PrimitiveDifference:
    if (set(payload) != {"shape", "precision", "current", "previous"}
            or payload["shape"] != list(shape)
            or payload["precision"] != "paired-endpoint-difference-v1"):
        raise ContractError("precision_codec_primitive_difference")
    return PrimitiveDifference(_decode_primitive(payload["current"], shape),
                               _decode_primitive(payload["previous"], shape))


def _decode_primitive(payload: Mapping[str, Any], shape: tuple[int, ...]) -> PrimitiveExpansion:
    if (set(payload) != {"shape", "precision", "words"} or payload["shape"] != list(shape)
            or payload["precision"] != "input-expansion4-v1" or len(payload["words"]) != 4
            or any(len(word) != int(np.prod(shape)) for word in payload["words"])):
        raise ContractError("precision_codec_primitive_shape")
    return PrimitiveExpansion(tuple(np.asarray([float.fromhex(x) for x in word]).reshape(shape)
                                    for word in payload["words"]))


def _decode_words(payload: Mapping[str, Any], shape: tuple[int, ...]) -> DoubleArray:
    if payload.get("shape") != list(shape) or set(payload) != {"shape", "high", "low"}:
        raise ContractError("precision_codec_shape_mismatch")
    count = int(np.prod(shape))
    if len(payload["high"]) != count or len(payload["low"]) != count:
        raise ContractError("precision_codec_word_count")
    high = np.asarray([float.fromhex(word) for word in payload["high"]]).reshape(shape)
    low = np.asarray([float.fromhex(word) for word in payload["low"]]).reshape(shape)
    return DoubleArray(high, low)


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def encode_point(point: Point) -> dict[str, Any]:
    if any(not isinstance(value, DoubleArray) for value in point.state.fields.values()):
        raise ContractError("precision_codec_requires_double_words")
    payload = {
        "schema": "solarlab.precision-point.v2", "layout": point.state.layout.identity,
        "coordinate_reference": point.coordinate_reference, "identity": point.identity,
        "time": float(point.time).hex(), "y": [float(x).hex() for x in point.y],
        "inputs": [float(x).hex() for x in point.inputs],
        "fields": {key: _word_payload(value) for key, value in point.state.fields.items()},
        "authority_identity": point.state.authority.identity,
        "authority": point.state.authority.payload() if isinstance(point.state.authority, MappedAuthority) else None,
        "field_role": "projection" if isinstance(point.state.authority, MappedAuthority) else "resolved-input",
    }
    return {"payload": payload, "sha256": _digest(payload)}


def decode_point(record: Mapping[str, Any], layout: Layout) -> Point:
    payload = record["payload"]
    if _digest(payload) != record.get("sha256"):
        raise ContractError("precision_codec_digest_mismatch")
    if payload.get("schema") not in {"solarlab.precision-point.v1", "solarlab.precision-point.v2"} or payload.get("layout") != layout.identity:
        raise ContractError("precision_codec_layout_mismatch")
    if set(payload["fields"]) != {spec.id for spec in layout.variables}:
        raise ContractError("precision_codec_fields_mismatch")
    fields = [(spec.id, _decode_words(payload["fields"][spec.id], spec.shape)) for spec in layout.variables]
    if payload.get("authority") is not None:
        if payload.get("field_role") != "projection":
            raise ContractError("precision_codec_authority_role")
        authority = _decode_authority(payload["authority"], layout)
        state = StateView(layout, authority=authority)
        for name, value in fields:
            if value.identity_bytes() != authority.project(name).value.identity_bytes():
                raise ContractError("precision_codec_projection_mismatch")
    else:
        if (payload.get("field_role", "resolved-input") != "resolved-input"
                or payload["coordinate_reference"].startswith(("relative:", "physical-input-lift:", "affine-trial:"))):
            raise ContractError("precision_codec_unresolved_authority")
        state = StateView(layout, fields)
    if payload.get("schema") == "solarlab.precision-point.v2" and payload.get("authority_identity") != state.authority.identity:
        raise ContractError("precision_codec_authority_identity")
    point = Point(float.fromhex(payload["time"]),
                  np.asarray([float.fromhex(x) for x in payload["y"]]),
                  np.asarray([float.fromhex(x) for x in payload["inputs"]]), state,
                  payload["coordinate_reference"])
    if point.identity != payload["identity"]:
        raise ContractError("precision_codec_identity_mismatch")
    return point


def _decode_authority(payload: Mapping[str, Any], layout: Layout) -> MappedAuthority:
    keys = {"schema", "layout", "map", "version", "modes", "parameters", "anchor", "primitives",
            "time", "inputs", "coordinates", "reference", "composition_policy", "transition"}
    fixed_reference = None
    kind = "local"
    paired = payload.get("schema") == "solarlab.state-authority.v5"
    if payload.get("schema") in {"solarlab.state-authority.v4", "solarlab.state-authority.v5"}:
        keys.add("coordinate_contract")
        contract = payload.get("coordinate_contract", {})
        if (set(contract) != {"kind", "solver_projection", "reference"}
                or contract["kind"] != "fixed-reference-affine-v1"
                or contract["solver_projection"] != "first-word"):
            raise ContractError("precision_codec_coordinate_meaning")
        fixed_reference = decode_point(contract["reference"], layout)
        kind = "fixed-reference"
    policy = "four-word-endpoints-paired-transition-v1" if paired else "exact-four-word-primitives-v1"
    if (set(payload) != keys or payload["schema"] not in {"solarlab.state-authority.v3", "solarlab.state-authority.v4", "solarlab.state-authority.v5"}
            or payload["layout"] != layout.identity or payload["version"] != 1
            or payload["composition_policy"] != policy):
        raise ContractError("precision_codec_authority_schema")
    shapes = {spec.id: spec.shape for spec in layout.variables}
    if set(payload["anchor"]) != set(shapes) or set(payload["modes"]) != set(shapes):
        raise ContractError("precision_codec_anchor_closure")
    anchor = {key: _decode_words(payload["anchor"][key], shape) for key, shape in shapes.items()}
    if payload["map"] == "relative-fields":
        primitive_shapes = shapes
    elif payload["map"] == "r1-input-lift":
        roles = {"electron": "n_m3", "hole": "p_m3", "potential": "phi_V", "positive": "positive_m3",
                 "occupancy": "occupancy", "trace_density": "trace_state_m3", "trace_potential": "trace_potential_V",
                 "voltage_lift": "phi_V", "trace_voltage_lift": "trace_potential_V"}
        primitive_shapes = {key: shapes[name] for key, name in roles.items() if name in shapes}
    else:
        raise ContractError("unsupported_authority_map")
    def decode_primitives(values):
        if set(values) != set(primitive_shapes):
            raise ContractError("authority_primitive_roles_mismatch")
        return {key: _decode_primitive(values[key], shape) for key, shape in primitive_shapes.items()}
    primitives = decode_primitives(payload["primitives"])
    if paired and not primitives:
        raise ContractError("precision_codec_empty_paired_transition")
    previous, local, previous_identity, previous_point = None, None, None, None
    if payload["transition"] is not None:
        transition = payload["transition"]
        if set(transition) != {"previous_authority", "previous_point", "previous_primitives", "local_primitives"}:
            raise ContractError("incomplete_authority_transition")
        previous = decode_primitives(transition["previous_primitives"])
        if paired:
            if set(transition["local_primitives"]) != set(primitive_shapes):
                raise ContractError("authority_primitive_roles_mismatch")
            local = {key: _decode_difference(transition["local_primitives"][key], shape)
                     for key, shape in primitive_shapes.items()}
        else:
            local = decode_primitives(transition["local_primitives"])
        previous_identity = transition["previous_authority"]
        previous_point = transition["previous_point"]
    return MappedAuthority(layout, anchor, payload["modes"], primitives, float.fromhex(payload["time"]),
                           np.asarray([float.fromhex(x) for x in payload["inputs"]]),
                           np.asarray([float.fromhex(x) for x in payload["coordinates"]]), payload["reference"],
                           payload["map"], {key: float.fromhex(value) for key, value in payload["parameters"].items()},
                           previous, local, previous_identity, previous_point, kind, fixed_reference)
