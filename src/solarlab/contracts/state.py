"""Physical state views and increments that do not expose solver coordinates.

Models read named fields, stable face differences, log ratios and
electrochemical differences. A precision representation can specialize these
through its ``StateAuthority`` without the model importing it. The production
coordinate map/codec belongs to the G2-R route decision; ``LinearCoordinates``
is only the identity map for already resolved physical SI states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.contracts.arithmetic import AssemblyArithmetic, FloatArithmetic
from solarlab.contracts.arrays import (
    FloatArray, ImmutableArrays, PhysicalArray, Vector, frozen_array, integer_indices,
)
from solarlab.contracts.errors import ContractError
from solarlab.contracts.layout import Layout
from solarlab.contracts.units import VOLT

__all__ = [
    "FieldProjection", "LinearCoordinates", "Point", "ResolvedAuthority",
    "StateAuthority", "StateIncrement", "StateView",
]


@dataclass(frozen=True)
class FieldProjection:
    """An explicit materialization, never an implicit new accepted anchor.

    ``absolute_error_bound=None`` means display-only validity; physical use or
    rebasing needs an independently qualified observable error certificate.
    """

    value: PhysicalArray
    authority_identity: str
    evaluator_identity: str
    absolute_error_bound: FloatArray | None

    def require_error(self, maximum: ArrayLike) -> None:
        limit = frozen_array(maximum)
        if limit.shape != self.value.shape or np.any(limit < 0):
            raise ContractError("projection_budget_shape_or_sign")
        if self.absolute_error_bound is None:
            raise ContractError("projection_not_certified_for_physical_use")
        if np.any(self.absolute_error_bound.values > limit):
            raise ContractError("projection_error_budget_exceeded")


class StateAuthority(Protocol):
    """A resolved physical input meaning, implemented outside the core if needed."""

    @property
    def layout_identity(self) -> str: ...
    @property
    def identity(self) -> str: ...

    def field(self, name: str) -> PhysicalArray: ...
    def delta_from(self, left: StateView) -> Mapping[str, PhysicalArray]: ...
    def project(self, name: str) -> FieldProjection: ...
    def validate_binding(self, time: float, y: Vector, inputs: Vector, reference: str) -> None: ...
    def validate_transition(self, left: Point, right: Point) -> None: ...
    def electrochemical_difference(self, density_id: str, potential_id: str, pairs: ArrayLike,
                                   thermal_voltage: float,
                                   potential_sign: int) -> PhysicalArray | None: ...
    def temporal_log_ratio(self, left: StateView, variable_id: str,
                           indices: ArrayLike | None) -> PhysicalArray | None: ...
    def face_delta(self, left: StateView, variable_id: str,
                   pairs: ArrayLike) -> PhysicalArray | None: ...
    def electrochemical_delta(self, left: StateView, density_id: str, potential_id: str,
                              pairs: ArrayLike, thermal_voltage: float,
                              potential_sign: int) -> PhysicalArray | None: ...


@dataclass(frozen=True)
class ResolvedAuthority:
    """Explicit absolute input words; finite changes mean endpoint differences."""

    layout_identity: str
    values: Mapping[str, PhysicalArray]
    identity: str = field(init=False)

    def __post_init__(self) -> None:
        owned = {name: value.immutable_copy() for name, value in self.values.items()}
        # An authority-backed field needs its generating map. Call project()
        # explicitly to declare different external absolute input semantics.
        if any(getattr(value, "authority", None) is not None for value in owned.values()):
            raise ContractError("mapped_field_requires_its_authority")
        object.__setattr__(self, "values", MappingProxyType(owned))
        digest = sha256(b"resolved-physical-words-v1:" + self.layout_identity.encode())
        for name in sorted(owned):
            digest.update(name.encode() + owned[name].identity_bytes())
        object.__setattr__(self, "identity", digest.hexdigest())

    def field(self, name: str) -> PhysicalArray:
        return self.values[name]

    def delta_from(self, left: StateView) -> Mapping[str, PhysicalArray]:
        if not isinstance(left.authority, ResolvedAuthority):
            raise ContractError("authority_rebase_required")
        return {name: value.difference(left.field(name)) for name, value in self.values.items()}

    def project(self, name: str) -> FieldProjection:
        value = self.field(name).immutable_copy()
        return FieldProjection(value, self.identity, "resolved-words-v1",
                               FloatArray(np.zeros(value.shape)))

    def validate_binding(self, time: float, y: Vector, inputs: Vector, reference: str) -> None:
        # A resolved state is an explicit external physical input. Point still
        # binds its coordinates, time and prescribed inputs into its identity.
        return None

    def validate_transition(self, left: Point, right: Point) -> None:
        return None

    def electrochemical_difference(self, density_id: str, potential_id: str, pairs: ArrayLike,
                                   thermal_voltage: float,
                                   potential_sign: int) -> PhysicalArray | None:
        return None

    def temporal_log_ratio(self, left: StateView, variable_id: str,
                           indices: ArrayLike | None) -> PhysicalArray | None:
        return None

    def face_delta(self, left: StateView, variable_id: str,
                   pairs: ArrayLike) -> PhysicalArray | None:
        return None

    def electrochemical_delta(self, left: StateView, density_id: str, potential_id: str,
                              pairs: ArrayLike, thermal_voltage: float,
                              potential_sign: int) -> PhysicalArray | None:
        return None


def _face_pairs(pairs: ArrayLike) -> NDArray[np.intp]:
    indices = integer_indices(pairs)
    if indices.ndim != 2 or indices.shape[1] != 2:
        raise ContractError("invalid_face_pairs")
    return indices


@dataclass(frozen=True, init=False)
class StateView:
    layout: Layout
    fields: Mapping[str, PhysicalArray]
    authority: StateAuthority

    def __init__(self, layout: Layout, fields: Iterable[tuple[str, PhysicalArray]] = (), *,
                 authority: StateAuthority | None = None):
        owned: dict[str, PhysicalArray] = {}
        specs = {x.id: x for x in layout.variables}
        entries = tuple(fields)
        if authority is not None:
            if entries or authority.layout_identity != layout.identity:
                raise ContractError("state_authority_layout_or_fields_mismatch")
            entries = tuple((name, authority.field(name)) for name in specs)
        for key, value in entries:
            if key in owned:
                raise ContractError("duplicate_state_write")
            if key not in specs:
                raise ContractError("unknown_variable")
            if value.shape != specs[key].shape:
                raise ContractError("state_shape_mismatch")
            if not value.is_finite():
                raise ContractError("nonfinite_state")
            owned[key] = value.immutable_copy()
        if set(owned) != set(specs):
            raise ContractError("incomplete_state")
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "fields", MappingProxyType(owned))
        object.__setattr__(self, "authority", authority if authority is not None
                           else ResolvedAuthority(layout.identity, owned))

    def field(self, variable_id: str) -> PhysicalArray:
        return self.fields[variable_id]

    def face_difference(self, variable_id: str, pairs: ArrayLike) -> PhysicalArray:
        indices = _face_pairs(pairs)
        value = self.field(variable_id)
        if np.any(indices < 0) or np.any(indices >= value.shape[0]):
            raise ContractError("face_outside_support")
        return value.take(indices[:, 1]).difference(value.take(indices[:, 0]))

    def log_ratio(self, variable_id: str, pairs: ArrayLike) -> PhysicalArray:
        indices = _face_pairs(pairs)
        value = self.field(variable_id)
        if len(value.shape) != 1 or np.any(indices < 0) or np.any(indices >= value.shape[0]):
            raise ContractError("face_outside_support")
        return value.take(indices[:, 1]).log_ratio(value.take(indices[:, 0]))

    def electrochemical_difference(self, density_id: str, potential_id: str, pairs: ArrayLike,
                                   thermal_voltage: float, *, potential_sign: int,
                                   arithmetic: AssemblyArithmetic | None = None) -> Any:
        """``log(c_j/c_i) + s*(phi_j-phi_i)/VT`` on each declared face pair."""
        if not np.isfinite(thermal_voltage) or thermal_voltage <= 0 or potential_sign not in {-1, 1}:
            raise ContractError("invalid_activity_parameters")
        specs = {spec.id: spec for spec in self.layout.variables}
        if specs[potential_id].unit != VOLT:
            raise ContractError("activity_potential_unit_mismatch")
        a = arithmetic if arithmetic is not None else FloatArithmetic()
        specialized = self.authority.electrochemical_difference(
            density_id, potential_id, pairs, thermal_voltage, potential_sign)
        if specialized is not None:
            return a.freeze(a.array(specialized))
        density = a.array(self.log_ratio(density_id, pairs))
        potential = a.array(self.face_difference(potential_id, pairs))
        drop = a.divide(potential, a.array(np.full(potential.shape, thermal_voltage)))
        return a.freeze(a.add(density, a.weighted(drop, np.full(drop.shape, potential_sign))))

    def project_field(self, variable_id: str) -> FieldProjection:
        return self.authority.project(variable_id)


@dataclass(frozen=True)
class Point(ImmutableArrays):
    """A state bound to time, prescribed inputs and an opaque coordinate reference."""

    time: float
    y: Vector
    inputs: Vector
    state: StateView
    coordinate_reference: str
    identity: str = field(init=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.time) or not self.coordinate_reference:
            raise ContractError("invalid_point")
        object.__setattr__(self, "time", float(self.time))
        object.__setattr__(self, "y", frozen_array(self.y))
        object.__setattr__(self, "inputs", frozen_array(self.inputs))
        if self.y.ndim != 1 or self.inputs.ndim != 1:
            raise ContractError("invalid_coordinate_shape")
        self.state.authority.validate_binding(self.time, self.y, self.inputs,
                                              self.coordinate_reference)
        digest = sha256(repr((self.time, self.coordinate_reference,
                              self.state.layout.identity)).encode())
        if not isinstance(self.state.authority, ResolvedAuthority):
            digest.update(self.state.authority.identity.encode())
        digest.update(self.y.tobytes())
        digest.update(self.inputs.tobytes())
        for spec in self.state.layout.variables:
            digest.update(spec.id.encode())
            digest.update(self.state.field(spec.id).identity_bytes())
        object.__setattr__(self, "identity", digest.hexdigest())


@dataclass(frozen=True)
class StateIncrement:
    """Endpoint-bound physical changes; values are checked against the authority."""

    left_identity: str
    right_identity: str
    fields: Mapping[str, PhysicalArray]

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", MappingProxyType({
            key: value.immutable_copy() for key, value in self.fields.items()
        }))

    def field(self, variable_id: str) -> PhysicalArray:
        return self.fields[variable_id]

    @classmethod
    def from_points(cls, left: Point, right: Point) -> StateIncrement:
        if left.state.layout.identity != right.state.layout.identity:
            raise ContractError("increment_layout_mismatch")
        right.state.authority.validate_transition(left, right)
        result = cls(left.identity, right.identity, right.state.authority.delta_from(left.state))
        result.validate(left, right)
        return result

    def validate(self, left: Point, right: Point) -> None:
        if (left.identity, right.identity) != (self.left_identity, self.right_identity):
            raise ContractError("increment_endpoint_mismatch")
        if left.state.layout.identity != right.state.layout.identity:
            raise ContractError("increment_layout_mismatch")
        if set(self.fields) != set(left.state.fields):
            raise ContractError("incomplete_increment")
        if any(self.fields[key].shape != left.state.field(key).shape for key in self.fields):
            raise ContractError("increment_shape_mismatch")
        right.state.authority.validate_transition(left, right)
        canonical = right.state.authority.delta_from(left.state)
        if set(canonical) != set(self.fields) or any(
                canonical[key].shape != self.fields[key].shape
                or not canonical[key].is_finite() for key in self.fields):
            raise ContractError("invalid_canonical_increment")
        if any(self.fields[key].identity_bytes() != canonical[key].identity_bytes()
               for key in self.fields):
            raise ContractError("increment_authority_mismatch")

    def _log_ratio(self, left: Point, right: Point, variable_id: str,
                   indices: ArrayLike | None) -> PhysicalArray:
        if variable_id not in self.fields:
            raise ContractError("unknown_variable")
        before, after = left.state.field(variable_id), right.state.field(variable_id)
        if indices is not None:
            indices = integer_indices(indices)
            if not before.shape:
                raise ContractError("index_outside_support")
            before, after = before.take(indices), after.take(indices)
        value = right.state.authority.temporal_log_ratio(left.state, variable_id, indices)
        if value is None:
            value = after.log_ratio(before)
        if value.shape != after.shape or not value.is_finite():
            raise ContractError("invalid_temporal_log_ratio")
        return value.immutable_copy()

    def log_ratio(self, left: Point, right: Point, variable_id: str, *,
                  indices: ArrayLike | None = None) -> PhysicalArray:
        """Finite temporal log(n_right/n_left) on selected positive populations."""
        self.validate(left, right)
        return self._log_ratio(left, right, variable_id, indices)

    def _pairs(self, variable_id: str, pairs: ArrayLike) -> NDArray[np.intp]:
        if variable_id not in self.fields:
            raise ContractError("unknown_variable")
        indices = integer_indices(pairs)
        shape = self.fields[variable_id].shape
        if (len(shape) != 1 or indices.ndim != 2 or indices.shape[1] != 2
                or np.any(indices < 0) or np.any(indices >= shape[0])):
            raise ContractError("face_outside_support")
        return indices

    def _face_delta(self, left: Point, right: Point, variable_id: str,
                    pairs: NDArray[np.intp]) -> PhysicalArray:
        value = right.state.authority.face_delta(left.state, variable_id, pairs)
        if value is None:
            change = self.field(variable_id)
            value = change.take(pairs[:, 1]).difference(change.take(pairs[:, 0]))
        if value.shape != (len(pairs),) or not value.is_finite():
            raise ContractError("invalid_temporal_face_delta")
        return value

    def face_delta(self, left: Point, right: Point, variable_id: str, pairs: ArrayLike, *,
                   arithmetic: AssemblyArithmetic | None = None) -> Any:
        """Finite change of a spatial difference between these endpoints."""
        self.validate(left, right)
        indices = self._pairs(variable_id, pairs)
        a = arithmetic if arithmetic is not None else FloatArithmetic()
        return a.freeze(a.array(self._face_delta(left, right, variable_id, indices)))

    def electrochemical_delta(self, left: Point, right: Point, density_id: str,
                              potential_id: str, pairs: ArrayLike, thermal_voltage: float, *,
                              potential_sign: int,
                              arithmetic: AssemblyArithmetic | None = None) -> Any:
        """Change of log(n_j/n_i)+s*(phi_j-phi_i)/VT across these endpoints."""
        self.validate(left, right)
        if not np.isfinite(thermal_voltage) or thermal_voltage <= 0 or potential_sign not in {-1, 1}:
            raise ContractError("invalid_activity_parameters")
        indices = self._pairs(density_id, pairs)
        self._pairs(potential_id, indices)
        specs = {spec.id: spec for spec in right.state.layout.variables}
        if specs[potential_id].unit != VOLT:
            raise ContractError("activity_potential_unit_mismatch")
        if self.field(density_id).shape != self.field(potential_id).shape:
            raise ContractError("activity_support_shape_mismatch")
        a = arithmetic if arithmetic is not None else FloatArithmetic()
        value = right.state.authority.electrochemical_delta(
            left.state, density_id, potential_id, indices, thermal_voltage, potential_sign)
        if value is not None:
            if value.shape != (len(indices),) or not value.is_finite():
                raise ContractError("invalid_temporal_electrochemical_delta")
            return a.freeze(a.array(value))
        temporal = self._log_ratio(left, right, density_id, indices.ravel())
        even, odd = np.arange(0, 2 * len(indices), 2), np.arange(1, 2 * len(indices), 2)
        density = a.array(temporal.take(odd).difference(temporal.take(even)))
        potential = a.array(self._face_delta(left, right, potential_id, indices))
        drop = a.divide(potential, a.array(np.full(potential.shape, thermal_voltage)))
        return a.freeze(a.add(density, a.weighted(drop, np.full(drop.shape, potential_sign))))


@dataclass(frozen=True)
class LinearCoordinates:
    """Identity coordinates for resolved physical SI states (not a route choice)."""

    layout: Layout
    reference: str = "physical-si-v1"

    def point(self, y: ArrayLike, time: float = 0.0, inputs: ArrayLike = ()) -> Point:
        words = frozen_array(y)
        if words.shape != (self.layout.size,):
            raise ContractError("coordinate_layout_mismatch")
        fields = [(spec.id, FloatArray(words[self.layout.offsets[spec.id]].reshape(spec.shape)))
                  for spec in self.layout.variables]
        return Point(time, words, frozen_array(inputs),
                     StateView(self.layout, fields), self.reference)

    def advance(self, left: Point, dy: ArrayLike, time: float,
                inputs: ArrayLike = ()) -> tuple[Point, StateIncrement]:
        step = frozen_array(dy)
        if step.shape != left.y.shape or left.coordinate_reference != self.reference:
            raise ContractError("coordinate_reference_mismatch")
        right = self.point(left.y + step, time, inputs)
        increment = StateIncrement(left.identity, right.identity, {
            # Binary64 endpoints; a precision provider must retain a richer
            # right endpoint to preserve a sub-ULP update.
            spec.id: right.state.field(spec.id).difference(left.state.field(spec.id))
            for spec in self.layout.variables
        })
        increment.validate(left, right)
        return right, increment
