"""P02-08 precision provider injected into the isolated public contracts.

Only the existing stateless DD arithmetic is reused. No old device, solver,
private R1 system or experiment is imported. The arithmetic implementation is
to move to solarlab_research when the package migration is admitted.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike

from perovskite_sim.physics.compensated import DD
from scripts.benchmarks.contract_prototype import (
    ContractError, FloatArray, Layout, PhysicalArray, Point, StateIncrement,
    StateView, Vector, frozen_array, integer_indices,
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

    def identity_bytes(self) -> bytes:
        return b"double-word:" + repr(self.shape).encode() + self.high.tobytes() + self.low.tobytes()

    def __array__(self, dtype=None, copy=None):
        raise TypeError("implicit precision rounding is forbidden")

    def __float__(self):
        raise TypeError("implicit precision rounding is forbidden")


class DoubleArithmetic:
    """A precision implementation of the public assembly operations."""

    def array(self, value: ArrayLike | PhysicalArray) -> DD:
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


@dataclass(frozen=True)
class RelativeCoordinates:
    """Explicitly use the supplied physical point as each trial's reference.

    Newton trials use the same left anchor. Advancing an accepted point creates
    a new reference identity while carrying every physical word. ``Point.y`` is
    the increment relative to that named anchor, not a silently rounded sum of
    earlier coordinate increments.
    """

    layout: Layout
    modes: Mapping[str, str]

    def __post_init__(self) -> None:
        modes = dict(self.modes)
        if set(modes) != {spec.id for spec in self.layout.variables}:
            raise ContractError("incomplete_coordinate_modes")
        if any(mode not in {"linear", "log", "inactive"} for mode in modes.values()):
            raise ContractError("unknown_coordinate_mode")
        object.__setattr__(self, "modes", MappingProxyType(modes))

    @property
    def identity(self) -> str:
        return hashlib.sha256(repr((self.layout.identity, sorted(self.modes.items()))).encode()).hexdigest()

    def initial(self, state: StateView, time: float = 0.0, inputs: ArrayLike = ()) -> Point:
        if state.layout.identity != self.layout.identity:
            raise ContractError("coordinate_layout_mismatch")
        for spec in self.layout.variables:
            value = state.field(spec.id)
            if not isinstance(value, DoubleArray):
                raise ContractError("precision_coordinate_requires_double_words")
            if (spec.lower is not None and np.any(value.as_dd() < spec.lower)
                    or spec.upper is not None and np.any(value.as_dd() > spec.upper)):
                raise ContractError("physical_state_outside_domain")
            if self.modes[spec.id] == "log" and np.any(value.as_dd() <= 0):
                raise ContractError("log_coordinate_requires_positive_inventory")
            if self.modes[spec.id] == "inactive" and np.any(value.as_dd() != 0):
                raise ContractError("inactive_inventory_not_zero")
        return Point(time, np.zeros(self.layout.size), frozen_array(inputs), state,
                     f"relative:{self.identity}:initial")

    def advance(self, anchor: Point, increment: ArrayLike, time: float,
                inputs: ArrayLike = ()) -> tuple[Point, StateIncrement]:
        dz = frozen_array(increment)
        if dz.shape != (self.layout.size,) or anchor.state.layout.identity != self.layout.identity:
            raise ContractError("coordinate_layout_mismatch")
        if not anchor.coordinate_reference.startswith(f"relative:{self.identity}:"):
            raise ContractError("coordinate_reference_mismatch")
        fields, deltas = [], {}
        for spec in self.layout.variables:
            previous = anchor.state.field(spec.id)
            if not isinstance(previous, DoubleArray):
                raise ContractError("precision_coordinate_requires_double_words")
            change = DD(dz[self.layout.offsets[spec.id]].reshape(spec.shape))
            mode = self.modes[spec.id]
            if mode == "log":
                delta = previous.as_dd() * change.expm1()
                next_value = previous.as_dd() * change.exp()
            elif mode == "inactive":
                if np.any(change != 0) or np.any(previous.as_dd() != 0):
                    raise ContractError("inactive_coordinate_change")
                delta = DD(np.zeros(spec.shape))
                next_value = previous.as_dd()
            else:
                delta = change
                next_value = previous.as_dd() + delta
            if (spec.lower is not None and np.any(next_value < spec.lower)
                    or spec.upper is not None and np.any(next_value > spec.upper)):
                raise ContractError("physical_state_outside_domain")
            fields.append((spec.id, DoubleArray.from_dd(next_value)))
            deltas[spec.id] = DoubleArray.from_dd(delta)
        right = Point(time, dz, frozen_array(inputs), StateView(self.layout, fields),
                      f"relative:{self.identity}:{anchor.identity}")
        change = StateIncrement(anchor.identity, right.identity, deltas)
        change.validate(anchor, right)
        return right, change


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
        "schema": "solarlab.precision-point.v1", "layout": point.state.layout.identity,
        "coordinate_reference": point.coordinate_reference, "identity": point.identity,
        "time": float(point.time).hex(), "y": [float(x).hex() for x in point.y],
        "inputs": [float(x).hex() for x in point.inputs],
        "fields": {key: _word_payload(value) for key, value in point.state.fields.items()},
    }
    return {"payload": payload, "sha256": _digest(payload)}


def decode_point(record: Mapping[str, Any], layout: Layout) -> Point:
    payload = record["payload"]
    if _digest(payload) != record.get("sha256"):
        raise ContractError("precision_codec_digest_mismatch")
    if payload.get("schema") != "solarlab.precision-point.v1" or payload.get("layout") != layout.identity:
        raise ContractError("precision_codec_layout_mismatch")
    if set(payload["fields"]) != {spec.id for spec in layout.variables}:
        raise ContractError("precision_codec_fields_mismatch")
    state = StateView(layout, [(spec.id, _decode_words(payload["fields"][spec.id], spec.shape))
                               for spec in layout.variables])
    point = Point(float.fromhex(payload["time"]),
                  np.asarray([float.fromhex(x) for x in payload["y"]]),
                  np.asarray([float.fromhex(x) for x in payload["inputs"]]), state,
                  payload["coordinate_reference"])
    if point.identity != payload["identity"]:
        raise ContractError("precision_codec_identity_mismatch")
    return point
