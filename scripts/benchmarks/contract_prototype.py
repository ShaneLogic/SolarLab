"""Isolated P02 numerical contracts; no production entry point imports this module.

The independent references live in tests/numerical_contracts.  This module owns
interfaces and candidate arithmetic, never reference answers or acceptance gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csc_matrix, isspmatrix_csc

Vector = NDArray[np.float64]


class ContractError(ValueError):
    """An input or assembly contract failed, with a machine-readable reason."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


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
    Subclasses freeze their stored arrays during construction. Returning a view
    preserves normal NumPy access and dataclass construction/replace semantics.
    """

    def __getattribute__(self, name: str) -> Any:
        value = object.__getattribute__(self, name)
        return value.view() if isinstance(value, np.ndarray) else value


@dataclass(frozen=True)
class Unit:
    """Dimensions (length, time, charge, number, mass, temperature); no scale."""

    powers: tuple[int, int, int, int, int, int]

    def __post_init__(self) -> None:
        if len(self.powers) != 6 or any(type(x) is not int for x in self.powers):
            raise ContractError("invalid_unit")

    def __mul__(self, other: Unit) -> Unit:
        return Unit(tuple(a + b for a, b in zip(self.powers, other.powers)))

    def __truediv__(self, other: Unit) -> Unit:
        return Unit(tuple(a - b for a, b in zip(self.powers, other.powers)))


ONE = Unit((0, 0, 0, 0, 0, 0))
SECOND = Unit((0, 1, 0, 0, 0, 0))
LENGTH = Unit((1, 0, 0, 0, 0, 0))
VOLUME = Unit((3, 0, 0, 0, 0, 0))
AREA = Unit((2, 0, 0, 0, 0, 0))
PARTICLE = Unit((0, 0, 0, 1, 0, 0))
COULOMB = Unit((0, 0, 1, 0, 0, 0))
VOLT = Unit((2, -2, -1, 0, 1, 0))
KELVIN = Unit((0, 0, 0, 0, 0, 1))


@dataclass(frozen=True)
class Support:
    id: str
    location: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.id or self.location not in {"cell", "face", "global"}:
            raise ContractError("invalid_support")
        if not self.shape or any(type(n) is not int or n < 0 for n in self.shape):
            raise ContractError("invalid_shape")


@dataclass(frozen=True)
class VariableSpec:
    id: str
    owner: str
    support: str
    shape: tuple[int, ...]
    unit: Unit
    role: str = "storage"
    orientation: int = 0
    lower: float | None = None
    upper: float | None = None


@dataclass(frozen=True)
class EquationSpec:
    id: str
    owner: str
    support: str
    shape: tuple[int, ...]
    unit: Unit
    role: str = "storage"
    orientation: int = 0
    derivative_support: tuple[str, ...] = ()


@dataclass(frozen=True)
class Layout:
    supports: tuple[Support, ...]
    variables: tuple[VariableSpec, ...]
    equations: tuple[EquationSpec, ...]
    offsets: Mapping[str, slice] = field(init=False)
    size: int = field(init=False)
    identity: str = field(init=False)

    def __post_init__(self) -> None:
        supports = {x.id: x for x in self.supports}
        if len(supports) != len(self.supports):
            raise ContractError("duplicate_support")
        variables = tuple(sorted(self.variables, key=lambda x: x.id))
        equations = tuple(sorted(self.equations, key=lambda x: x.id))
        for specs in (variables, equations):
            if len({x.id for x in specs}) != len(specs):
                raise ContractError("duplicate_owner")
            for spec in specs:
                if not spec.id or not spec.owner:
                    raise ContractError("missing_instance_owner")
                if spec.support not in supports:
                    raise ContractError("unknown_support")
                if spec.shape != supports[spec.support].shape:
                    raise ContractError("support_shape_mismatch")
                if spec.orientation not in {-1, 0, 1}:
                    raise ContractError("invalid_orientation")
                if spec.role not in {"storage", "constraint", "prescribed"}:
                    raise ContractError("invalid_role")
        ids = {x.id for x in variables}
        for equation in equations:
            if not set(equation.derivative_support) <= ids:
                raise ContractError("unknown_derivative_support")
        for variable in variables:
            if (variable.lower is not None and variable.upper is not None
                    and variable.lower > variable.upper):
                raise ContractError("invalid_feasible_domain")
        offsets: dict[str, slice] = {}
        size = 0
        for variable in variables:
            end = size + int(np.prod(variable.shape))
            offsets[variable.id] = slice(size, end)
            size = end
        object.__setattr__(self, "supports", tuple(sorted(self.supports, key=lambda x: x.id)))
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "equations", equations)
        object.__setattr__(self, "offsets", MappingProxyType(offsets))
        object.__setattr__(self, "size", size)
        identity = repr((self.supports, variables, equations)).encode()
        object.__setattr__(self, "identity", sha256(identity).hexdigest())


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
    values: Vector

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", frozen_array(self.values))

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    def immutable_copy(self) -> FloatArray:
        return FloatArray(self.values)

    def difference(self, other: PhysicalArray) -> FloatArray:
        if not isinstance(other, FloatArray) or other.shape != self.shape:
            raise ContractError("precision_or_shape_mismatch")
        return FloatArray(self.values - other.values)

    def take(self, indices: ArrayLike) -> FloatArray:
        indices = integer_indices(indices)
        if np.any(indices < 0) or np.any(indices >= self.shape[0]):
            raise ContractError("index_outside_support")
        return FloatArray(self.values[indices])

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
        maximum = frozen_array(maximum)
        if maximum.shape != self.value.shape or np.any(maximum < 0):
            raise ContractError("projection_budget_shape_or_sign")
        if self.absolute_error_bound is None:
            raise ContractError("projection_not_certified_for_physical_use")
        if np.any(self.absolute_error_bound.values > maximum):
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
                                   thermal_voltage: float, potential_sign: int) -> PhysicalArray | None: ...
    def temporal_log_ratio(self, left: StateView, variable_id: str,
                           indices: ArrayLike | None) -> PhysicalArray | None: ...
    def face_delta(self, left: StateView, variable_id: str, pairs: ArrayLike) -> PhysicalArray | None: ...
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
        return FieldProjection(value, self.identity, "resolved-words-v1", FloatArray(np.zeros(value.shape)))

    def validate_binding(self, time: float, y: Vector, inputs: Vector, reference: str) -> None:
        # A resolved state is an explicit external physical input. Point still
        # binds its coordinates, time and prescribed inputs into its identity.
        return None

    def electrochemical_difference(self, density_id, potential_id, pairs, thermal_voltage, potential_sign):
        return None

    def validate_transition(self, left: Point, right: Point) -> None:
        return None

    def temporal_log_ratio(self, left, variable_id, indices):
        return None

    def face_delta(self, left, variable_id, pairs):
        return None

    def electrochemical_delta(self, left, density_id, potential_id, pairs, thermal_voltage, potential_sign):
        return None


@dataclass(frozen=True, init=False)
class StateView:
    layout: Layout
    fields: Mapping[str, PhysicalArray]
    authority: StateAuthority

    def __init__(self, layout: Layout, fields: Iterable[tuple[str, PhysicalArray]] = (), *,
                 authority: StateAuthority | None = None):
        owned: dict[str, PhysicalArray] = {}
        specs = {x.id: x for x in layout.variables}
        fields = tuple(fields)
        if authority is not None:
            if fields or authority.layout_identity != layout.identity:
                raise ContractError("state_authority_layout_or_fields_mismatch")
            fields = tuple((name, authority.field(name)) for name in specs)
        for key, value in fields:
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
        pairs = integer_indices(pairs)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ContractError("invalid_face_pairs")
        value = self.field(variable_id)
        if np.any(pairs < 0) or np.any(pairs >= value.shape[0]):
            raise ContractError("face_outside_support")
        return value.take(pairs[:, 1]).difference(value.take(pairs[:, 0]))

    def log_ratio(self, variable_id: str, pairs: ArrayLike) -> PhysicalArray:
        pairs = integer_indices(pairs)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ContractError("invalid_face_pairs")
        value = self.field(variable_id)
        if len(value.shape) != 1 or np.any(pairs < 0) or np.any(pairs >= value.shape[0]):
            raise ContractError("face_outside_support")
        return value.take(pairs[:, 1]).log_ratio(value.take(pairs[:, 0]))

    def electrochemical_difference(self, density_id: str, potential_id: str, pairs: ArrayLike,
                                   thermal_voltage: float, *, potential_sign: int,
                                   arithmetic: AssemblyArithmetic | None = None) -> Any:
        if not np.isfinite(thermal_voltage) or thermal_voltage <= 0 or potential_sign not in {-1, 1}:
            raise ContractError("invalid_activity_parameters")
        specs = {spec.id: spec for spec in self.layout.variables}
        if specs[potential_id].unit != VOLT:
            raise ContractError("activity_potential_unit_mismatch")
        a = arithmetic if arithmetic is not None else FloatArithmetic()
        specialized = self.authority.electrochemical_difference(density_id, potential_id, pairs,
                                                               thermal_voltage, potential_sign)
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
        self.state.authority.validate_binding(self.time, self.y, self.inputs, self.coordinate_reference)
        digest = sha256(repr((self.time, self.coordinate_reference, self.state.layout.identity)).encode())
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
        if set(canonical) != set(self.fields) or any(canonical[key].shape != self.fields[key].shape
                                                   or not canonical[key].is_finite() for key in self.fields):
            raise ContractError("invalid_canonical_increment")
        if any(self.fields[key].identity_bytes() != canonical[key].identity_bytes() for key in self.fields):
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
        """Finite temporal log(n_right/n_left) on selected positive populations.

        Selection follows PhysicalArray.take and occurs before logarithms, so
        inactive zero populations outside the selection need not be evaluated.
        Mapped authorities retain their generating primitives through this step.
        """
        self.validate(left, right)
        return self._log_ratio(left, right, variable_id, indices)

    def _pairs(self, variable_id: str, pairs: ArrayLike) -> NDArray[np.intp]:
        if variable_id not in self.fields:
            raise ContractError("unknown_variable")
        pairs = integer_indices(pairs)
        shape = self.fields[variable_id].shape
        if (len(shape) != 1 or pairs.ndim != 2 or pairs.shape[1] != 2
                or np.any(pairs < 0) or np.any(pairs >= shape[0])):
            raise ContractError("face_outside_support")
        return pairs

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
        """Finite change of a spatial difference for an affine mapped field.

        A mapped provider combines temporal and spatial primitive terms before
        projection. Explicitly resolved fields use their declared physical words.
        """
        self.validate(left, right)
        pairs = self._pairs(variable_id, pairs)
        a = arithmetic if arithmetic is not None else FloatArithmetic()
        return a.freeze(a.array(self._face_delta(left, right, variable_id, pairs)))

    def electrochemical_delta(self, left: Point, right: Point, density_id: str,
                              potential_id: str, pairs: ArrayLike, thermal_voltage: float, *,
                              potential_sign: int, arithmetic: AssemblyArithmetic | None = None) -> Any:
        """Change of log(n_j/n_i)+s*(phi_j-phi_i)/VT across these endpoints.

        Shared named primitives cancel before projection; subtracting two
        separately projected spatial affinities is not the mapped definition.
        """
        self.validate(left, right)
        if not np.isfinite(thermal_voltage) or thermal_voltage <= 0 or potential_sign not in {-1, 1}:
            raise ContractError("invalid_activity_parameters")
        pairs = self._pairs(density_id, pairs)
        self._pairs(potential_id, pairs)
        specs = {spec.id: spec for spec in right.state.layout.variables}
        if specs[potential_id].unit != VOLT:
            raise ContractError("activity_potential_unit_mismatch")
        if self.field(density_id).shape != self.field(potential_id).shape:
            raise ContractError("activity_support_shape_mismatch")
        a = arithmetic if arithmetic is not None else FloatArithmetic()
        value = right.state.authority.electrochemical_delta(
            left.state, density_id, potential_id, pairs, thermal_voltage, potential_sign)
        if value is not None:
            if value.shape != (len(pairs),) or not value.is_finite():
                raise ContractError("invalid_temporal_electrochemical_delta")
            return a.freeze(a.array(value))
        temporal = self._log_ratio(left, right, density_id, pairs.ravel())
        even, odd = np.arange(0, 2*len(pairs), 2), np.arange(1, 2*len(pairs), 2)
        density = a.array(temporal.take(odd).difference(temporal.take(even)))
        potential = a.array(self._face_delta(left, right, potential_id, pairs))
        drop = a.divide(potential, a.array(np.full(potential.shape, thermal_voltage)))
        return a.freeze(a.add(density, a.weighted(drop, np.full(drop.shape, potential_sign))))


@dataclass(frozen=True)
class LinearCoordinates:
    layout: Layout
    reference: str = "physical-si-v1"

    def point(self, y: ArrayLike, time: float = 0.0, inputs: ArrayLike = ()) -> Point:
        y = frozen_array(y)
        if y.shape != (self.layout.size,):
            raise ContractError("coordinate_layout_mismatch")
        fields = [(spec.id, FloatArray(y[self.layout.offsets[spec.id]].reshape(spec.shape)))
                  for spec in self.layout.variables]
        return Point(time, y, frozen_array(inputs), StateView(self.layout, fields), self.reference)

    def advance(self, left: Point, dy: ArrayLike, time: float,
                inputs: ArrayLike = ()) -> tuple[Point, StateIncrement]:
        dy = frozen_array(dy)
        if dy.shape != left.y.shape or left.coordinate_reference != self.reference:
            raise ContractError("coordinate_reference_mismatch")
        right = self.point(left.y + dy, time, inputs)
        increment = StateIncrement(left.identity, right.identity, {
            # This representation owns binary64 endpoints. A precision provider
            # must retain a richer right endpoint to preserve a sub-ULP update.
            spec.id: right.state.field(spec.id).difference(left.state.field(spec.id))
            for spec in self.layout.variables
        })
        increment.validate(left, right)
        return right, increment


@dataclass(frozen=True)
class Geometry(ImmutableArrays):
    volumes: Vector
    pairs: NDArray[np.int64]
    face_measures: Vector
    volume_unit: Unit = VOLUME
    face_unit: Unit = AREA
    cell_support: str = "cells"
    face_support: str = "faces"

    def __post_init__(self) -> None:
        object.__setattr__(self, "volumes", frozen_array(self.volumes))
        object.__setattr__(self, "face_measures", frozen_array(self.face_measures))
        pairs = integer_indices(self.pairs)
        immutable_pairs = np.frombuffer(pairs.tobytes(), dtype=np.int64).reshape(pairs.shape)
        object.__setattr__(self, "pairs", immutable_pairs)
        if self.volumes.ndim != 1 or np.any(self.volumes <= 0):
            raise ContractError("invalid_volumes")
        if pairs.ndim != 2 or pairs.shape != (self.face_measures.size, 2):
            raise ContractError("invalid_face_pairs")
        if self.face_measures.ndim != 1 or np.any(self.face_measures <= 0):
            raise ContractError("invalid_face_measures")
        if np.any(pairs < 0) or np.any(pairs >= self.volumes.size) or np.any(pairs[:, 0] == pairs[:, 1]):
            raise ContractError("face_outside_support")
        if not self.cell_support or not self.face_support or self.cell_support == self.face_support:
            raise ContractError("invalid_geometry_support")


@dataclass(frozen=True)
class Contribution:
    id: str
    support: str
    values: ArrayLike | PhysicalArray
    unit: Unit


@dataclass(frozen=True)
class CellSource(Contribution):
    pass


@dataclass(frozen=True)
class FaceFlux(Contribution):
    pass


@dataclass(frozen=True)
class AlgebraicTerm(Contribution):
    pass


@dataclass(frozen=True)
class SurfaceCharge(Contribution):
    pass


class AssemblyArithmetic(Protocol):
    """Array operations supplied by a precision backend, with no physics imports.

    ``array`` checks finite values; every operation preserves the representation.
    ``freeze`` returns owned immutable storage. Geometry and support validation
    remain the responsibility of TermSink rather than the injected backend.
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
        return frozen_array(value)

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


class TermSink:
    """Typed unweighted contributions; geometry is applied here exactly once."""

    def __init__(self, layout: Layout, equation_id: str, geometry: Geometry,
                 arithmetic: AssemblyArithmetic | None = None):
        equations = {x.id: x for x in layout.equations}
        if equation_id not in equations:
            raise ContractError("unknown_equation")
        self.equation = equations[equation_id]
        self.supports = {x.id: x for x in layout.supports}
        self.geometry = geometry
        self.arithmetic = arithmetic if arithmetic is not None else FloatArithmetic()
        for support_id, location, shape in (
            (geometry.cell_support, "cell", geometry.volumes.shape),
            (geometry.face_support, "face", geometry.face_measures.shape),
        ):
            support = self.supports.get(support_id)
            if support is None or support.location != location or support.shape != shape:
                raise ContractError("geometry_support_mismatch")
        self._seen: set[str] = set()
        self._value = self.arithmetic.zeros(self.equation.shape)

    def add(self, term: Contribution) -> None:
        if not term.id or term.id in self._seen:
            raise ContractError("duplicate_contribution")
        if term.support not in self.supports:
            raise ContractError("unknown_support")
        support = self.supports[term.support]
        target = self.supports[self.equation.support]
        arithmetic = self.arithmetic
        values = arithmetic.array(term.values)
        if values.shape != support.shape:
            raise ContractError("term_shape_mismatch")
        geometry = self.geometry
        if isinstance(term, CellSource):
            if support != target or support.location != "cell" or values.shape != geometry.volumes.shape:
                raise ContractError("wrong_source_support")
            if support.id != geometry.cell_support:
                raise ContractError("geometry_support_mismatch")
            expected = term.unit * geometry.volume_unit
            contribution = arithmetic.weighted(values, geometry.volumes)
        elif isinstance(term, FaceFlux):
            if (support.location != "face" or target.location != "cell"
                    or values.shape != geometry.face_measures.shape
                    or target.shape != geometry.volumes.shape):
                raise ContractError("wrong_flux_support")
            if support.id != geometry.face_support or target.id != geometry.cell_support:
                raise ContractError("geometry_support_mismatch")
            expected = term.unit * geometry.face_unit
            contribution = arithmetic.zeros(self.equation.shape)
            weighted = arithmetic.weighted(values, geometry.face_measures)
            contribution = arithmetic.scatter_add(
                contribution, geometry.pairs[:, 0], arithmetic.weighted(weighted, -np.ones(values.shape)),
            )
            contribution = arithmetic.scatter_add(contribution, geometry.pairs[:, 1], weighted)
        elif isinstance(term, AlgebraicTerm):
            if support != target or self.equation.role != "constraint":
                raise ContractError("wrong_algebraic_support")
            expected, contribution = term.unit, values
        elif isinstance(term, SurfaceCharge):
            if support != target or support.location != "face" or values.shape != geometry.face_measures.shape:
                raise ContractError("wrong_surface_charge_support")
            if support.id != geometry.face_support:
                raise ContractError("geometry_support_mismatch")
            expected = term.unit * geometry.face_unit
            contribution = arithmetic.weighted(values, geometry.face_measures)
        else:
            raise ContractError("untyped_contribution")
        if expected != self.equation.unit:
            raise ContractError("unit_mismatch")
        self._value = arithmetic.add(self._value, contribution)
        self._seen.add(term.id)

    def value(self) -> Any:
        return self.arithmetic.freeze(self._value)


@dataclass(frozen=True)
class StoragePartials:
    y: Vector
    inputs: Vector
    time: Vector


class StorageLaw(Protocol):
    def value(self, point: Point) -> Vector | PhysicalArray: ...
    def delta(self, left: Point, right: Point, increment: StateIncrement) -> Vector | PhysicalArray: ...
    def first(self, point: Point) -> StoragePartials: ...
    def rate_jvp(self, point: Point, ydot: Vector, adot: Vector,
                 dy: Vector, da: Vector, dt: float) -> Vector: ...


@dataclass(frozen=True)
class LinearStorage(ImmutableArrays):
    matrix: Vector
    coordinate_reference: str = "physical-si-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "matrix", frozen_array(self.matrix))
        if self.matrix.ndim != 2:
            raise ContractError("invalid_storage_matrix")

    def _check_reference(self, point: Point) -> None:
        if point.coordinate_reference != self.coordinate_reference:
            raise ContractError("linear_storage_coordinate_mismatch")
        if self.matrix.shape[1] != point.y.size:
            raise ContractError("storage_coordinate_shape_mismatch")

    def value(self, point: Point) -> Vector:
        self._check_reference(point)
        return self.matrix @ point.y

    def delta(self, left: Point, right: Point, increment: StateIncrement) -> Vector:
        increment.validate(left, right)
        self._check_reference(left)
        self._check_reference(right)
        physical = []
        for variable in left.state.layout.variables:
            value = increment.field(variable.id)
            if not isinstance(value, FloatArray):
                raise ContractError("linear_storage_requires_explicit_precision_adapter")
            physical.extend(value.values.ravel())
        return self.matrix @ np.asarray(physical)

    def first(self, point: Point) -> StoragePartials:
        self._check_reference(point)
        return StoragePartials(self.matrix, np.zeros((self.matrix.shape[0], point.inputs.size)),
                               np.zeros(self.matrix.shape[0]))

    def rate_jvp(self, point: Point, ydot: Vector, adot: Vector,
                 dy: Vector, da: Vector, dt: float) -> Vector:
        self._check_reference(point)
        return np.zeros(self.matrix.shape[0])


@dataclass(frozen=True)
class BalanceTerms:
    rate: Vector
    algebraic: Vector
    Ry: Vector
    Ra: Vector
    Rt: Vector
    gy: Vector
    ga: Vector
    gt: Vector


@dataclass(frozen=True)
class Linearization:
    y: Vector
    ydot: Vector
    inputs: Vector
    input_rate: Vector
    time: Vector
    arithmetic: AssemblyArithmetic | None = field(default=None, repr=False, compare=False)

    def ida_matrix(self, cj: float) -> Any:
        if not np.isfinite(cj):
            raise ContractError("nonfinite_cj")
        a = self.arithmetic if self.arithmetic is not None else FloatArithmetic()
        y, ydot = a.array(self.y), a.array(self.ydot)
        return a.freeze(a.add(y, a.weighted(ydot, np.full(ydot.shape, cj))))


@dataclass(frozen=True)
class ImplicitSystem:
    """Dense convenience for small analytic problems; precision is explicit.

    Large models use ``ImplicitProblem``/``ValidatedProblem`` below, whose
    residual path never requests derivative assembly or identity-column JVPs.
    """

    storage: StorageLaw
    terms: Callable[[Point], BalanceTerms]
    arithmetic: AssemblyArithmetic | None = None

    def _arithmetic(self) -> AssemblyArithmetic:
        return self.arithmetic if self.arithmetic is not None else FloatArithmetic()

    def evaluate(self, point: Point) -> tuple[StoragePartials, BalanceTerms]:
        first, terms = self.storage.first(point), self.terms(point)
        n, m = point.y.size, point.inputs.size
        if len(terms.rate.shape) != 1 or len(terms.algebraic.shape) != 1:
            raise ContractError("invalid_equation_shape")
        k, l = terms.rate.shape[0], terms.algebraic.shape[0]
        arrays = (
            (first.y, (k, n)), (first.inputs, (k, m)), (first.time, (k,)),
            (terms.Ry, (k, n)), (terms.Ra, (k, m)), (terms.Rt, (k,)),
            (terms.gy, (l, n)), (terms.ga, (l, m)), (terms.gt, (l,)),
        )
        if k + l != n or any(array.shape != shape for array, shape in arrays):
            raise ContractError("equation_derivative_shape_mismatch")
        a = self._arithmetic()
        try:
            converted = [a.array(array) for array, _ in arrays]
        except (ContractError, ValueError) as error:
            raise ContractError("nonfinite_derivative") from error
        try:
            rate, algebraic = a.array(terms.rate), a.array(terms.algebraic)
        except (ContractError, ValueError) as error:
            raise ContractError("nonfinite_residual") from error
        return StoragePartials(*converted[:3]), BalanceTerms(rate, algebraic, *converted[3:])

    def _check_rates(self, point: Point, ydot: Any, adot: Any) -> None:
        if ydot.shape != point.y.shape or adot.shape != point.inputs.shape:
            raise ContractError("rate_shape_mismatch")
        try:
            self._arithmetic().array(ydot)
            self._arithmetic().array(adot)
        except (ContractError, ValueError) as error:
            raise ContractError("nonfinite_rate") from error

    def _storage_vector(self, value: Any, count: int) -> Any:
        try:
            array = self._arithmetic().array(value)
        except (ContractError, ValueError) as error:
            raise ContractError("nonfinite_storage_callback") from error
        if array.shape != (count,):
            raise ContractError("storage_callback_shape_mismatch")
        return array

    def residual(self, point: Point, ydot: Vector, adot: Vector) -> Vector:
        self._check_rates(point, ydot, adot)
        first, terms = self.evaluate(point)
        a = self._arithmetic()
        rate = a.add(a.add(a.matmul(first.y, a.array(ydot)),
                          a.matmul(first.inputs, a.array(adot))), first.time)
        return a.freeze(a.concatenate((a.subtract(rate, terms.rate), terms.algebraic)))

    def linearize(self, point: Point, ydot: Vector, adot: Vector) -> Linearization:
        self._check_rates(point, ydot, adot)
        first, terms = self.evaluate(point)
        n, m = point.y.size, point.inputs.size
        k, l = terms.rate.shape[0], terms.algebraic.shape[0]
        a = self._arithmetic()
        zero_y, zero_a = np.zeros(n), np.zeros(m)
        def contraction(dy: Vector, da: Vector, dt: float) -> Vector:
            return self._storage_vector(
                self.storage.rate_jvp(point, ydot, adot, dy, da, dt), k,
            )
        rate_y = (a.concatenate([
            a.reshape(contraction(direction, zero_a, 0.0), (k, 1))
            for direction in np.eye(n)
        ], axis=1) if n else a.zeros((k, 0)))
        rate_a = (a.concatenate([
            a.reshape(contraction(zero_y, direction, 0.0), (k, 1))
            for direction in np.eye(m)
        ], axis=1) if m else a.zeros((k, 0)))
        rate_t = contraction(zero_y, zero_a, 1.0)
        return Linearization(
            a.freeze(a.concatenate((a.subtract(rate_y, terms.Ry), terms.gy))),
            a.freeze(a.concatenate((first.y, a.zeros((l, n))))),
            a.freeze(a.concatenate((a.subtract(rate_a, terms.Ra), terms.ga))),
            a.freeze(a.concatenate((first.inputs, a.zeros((l, m))))),
            a.freeze(a.concatenate((a.subtract(rate_t, terms.Rt), terms.gt))),
            self.arithmetic,
        )

    def conservative_residual(self, left: Point, right: Point,
                              increment: StateIncrement) -> Vector:
        h = right.time - left.time
        if h <= 0:
            raise ContractError("nonpositive_step")
        increment.validate(left, right)
        _, terms = self.evaluate(right)
        a = self._arithmetic()
        delta = self._storage_vector(self.storage.delta(left, right, increment), terms.rate.shape[0])
        return a.freeze(a.concatenate((a.subtract(delta, a.weighted(terms.rate, np.full(terms.rate.shape, h))),
                                       terms.algebraic)))

    def conservative_jacobian(self, left: Point, right: Point) -> Vector:
        h = right.time - left.time
        if h <= 0:
            raise ContractError("nonpositive_step")
        first, terms = self.evaluate(right)
        a = self._arithmetic()
        return a.freeze(a.concatenate((a.subtract(first.y, a.weighted(terms.Ry, np.full(terms.Ry.shape, h))),
                                       terms.gy)))


class StorageValues(Protocol):
    """Physical Q and finite delta Q, without a dense derivative requirement."""

    def value(self, point: Point) -> Any: ...
    def delta(self, left: Point, right: Point, increment: StateIncrement) -> Any: ...


@dataclass(frozen=True)
class PhysicalStorage:
    """Validate value callbacks in the same units, source and arithmetic lane."""

    value_callback: Callable[[Point], Any]
    delta_callback: Callable[[Point, Point, StateIncrement], Any]
    units: tuple[Unit, ...]
    source_identity: str
    arithmetic: AssemblyArithmetic | None = None

    def __post_init__(self) -> None:
        if not self.source_identity or not all(isinstance(unit, Unit) for unit in self.units):
            raise ContractError("invalid_storage_identity_or_units")
        object.__setattr__(self, "units", tuple(self.units))

    def _value(self, value: Any) -> Any:
        a = self.arithmetic if self.arithmetic is not None else FloatArithmetic()
        array = a.array(value)
        if array.shape != (len(self.units),):
            raise ContractError("storage_callback_shape_mismatch")
        return a.freeze(array)

    def value(self, point: Point) -> Any:
        return self._value(self.value_callback(point))

    def delta(self, left: Point, right: Point, increment: StateIncrement) -> Any:
        increment.validate(left, right)
        return self._value(self.delta_callback(left, right, increment))


@dataclass(frozen=True)
class SparseStructure(ImmutableArrays):
    """A frozen CSC slot contract copied from a declared structural graph.

    The graph owns named equation/variable/term support and unit checks. This
    adapter binds that declaration identity to exact slots, including zeros;
    a numerical nonzero mask is never used to construct or repair the pattern.
    Rectangular structures are permitted for storage/input partials.
    """

    shape: tuple[int, int]
    indices: NDArray[np.intp]
    indptr: NDArray[np.intp]
    declaration_identity: str
    layout_identity: str
    identity: str = field(init=False)

    def __post_init__(self) -> None:
        if (len(self.shape) != 2 or any(type(n) is not int or n < 0 for n in self.shape)
                or not self.declaration_identity or not self.layout_identity):
            raise ContractError("invalid_sparse_structure")
        indices, indptr = integer_indices(self.indices), integer_indices(self.indptr)
        if (indices.ndim != 1 or indptr.shape != (self.shape[1] + 1,)
                or indptr[0] != 0 or indptr[-1] != len(indices)
                or np.any(np.diff(indptr) < 0) or np.any(indices < 0)
                or np.any(indices >= self.shape[0])):
            raise ContractError("invalid_sparse_structure")
        for start, end in zip(indptr[:-1], indptr[1:]):
            if np.any(np.diff(indices[start:end]) <= 0):
                raise ContractError("noncanonical_sparse_structure")
        for name, array in (("indices", indices), ("indptr", indptr)):
            object.__setattr__(self, name, np.frombuffer(array.tobytes(), dtype=np.intp))
        data = repr((self.shape, self.declaration_identity, self.layout_identity)).encode()
        object.__setattr__(self, "identity", sha256(data + indices.tobytes() + indptr.tobytes()).hexdigest())

    @classmethod
    def from_graph(cls, graph: Any) -> SparseStructure:
        return cls(graph.shape, graph.indices, graph.indptr, graph.identity, graph.layout.identity)

    def matrix(self, matrix: csc_matrix) -> csc_matrix:
        if not isspmatrix_csc(matrix) or matrix.shape != self.shape:
            raise ContractError("sparse_matrix_shape_or_format")
        if (not np.array_equal(integer_indices(matrix.indices), self.indices)
                or not np.array_equal(integer_indices(matrix.indptr), self.indptr)):
            raise ContractError("structural_recompile_required")
        if matrix.data.dtype.kind not in {"i", "u", "f"}:
            raise ContractError("unsupported_sparse_dtype")
        data = frozen_array(matrix.data)
        if data.shape != self.indices.shape:
            raise ContractError("sparse_data_shape_mismatch")
        return csc_matrix((data.copy(), self.indices.copy(), self.indptr.copy()), shape=self.shape)

    def filled(self, values: ArrayLike) -> csc_matrix:
        values = frozen_array(values)
        if values.shape != self.indices.shape:
            raise ContractError("sparse_data_shape_mismatch")
        return csc_matrix((values.copy(), self.indices.copy(), self.indptr.copy()), shape=self.shape)


@dataclass(frozen=True)
class SparseLinearization(ImmutableArrays):
    """Complete analytic partials; Fy includes the storage-rate chain rule."""

    y: csc_matrix
    ydot: csc_matrix
    inputs: Vector
    input_rate: Vector
    time: Vector
    structure: SparseStructure
    source_identity: str

    def __getattribute__(self, name: str) -> Any:
        value = super().__getattribute__(name)
        return value.copy() if isspmatrix_csc(value) else value

    def __post_init__(self) -> None:
        if not self.source_identity or self.structure.shape[0] != self.structure.shape[1]:
            raise ContractError("invalid_linearization_identity")
        n = self.structure.shape[0]
        for name in ("y", "ydot"):
            object.__setattr__(self, name, self.structure.matrix(getattr(self, name)))
        for name in ("inputs", "input_rate", "time"):
            object.__setattr__(self, name, frozen_array(getattr(self, name)))
        if (self.inputs.ndim != 2 or self.inputs.shape[0] != n
                or self.input_rate.shape != self.inputs.shape or self.time.shape != (n,)):
            raise ContractError("equation_derivative_shape_mismatch")

    def ida_matrix(self, cj: float) -> csc_matrix:
        if not np.isfinite(cj):
            raise ContractError("nonfinite_cj")
        # Adding sparse matrices can discard exact-zero slots. Add the aligned
        # data instead, retaining the declared structure for every cj.
        return self.structure.filled(self.y.data + cj * self.ydot.data)


class ImplicitProblem(Protocol):
    """Solver-facing contract; no Device, registry, server or precision import."""

    @property
    def layout(self) -> Layout: ...
    @property
    def source_identity(self) -> str: ...
    @property
    def storage(self) -> StorageValues: ...

    def residual(self, point: Point, ydot: Vector, adot: Vector) -> Any: ...
    def linearize(self, point: Point, ydot: Vector, adot: Vector) -> Any: ...
    def conservative_residual(self, left: Point, right: Point, increment: StateIncrement) -> Any: ...
    def conservative_jacobian(self, left: Point, right: Point) -> Any: ...


@dataclass(frozen=True)
class ValidatedProblem:
    """Bind separately implemented analytic callbacks to one public problem.

    The first ``storage_count`` equation rows are differential. The remaining
    rows are constraints, unscaled by dt in the conservative residual. Runtime
    validation checks representation/contracts; independent physics and joint
    derivative tests remain mandatory, and cannot be replaced by these checks.
    """

    layout: Layout
    input_count: int
    storage_count: int
    storage: PhysicalStorage
    structure: SparseStructure
    source_identity: str
    residual_values: Callable[[Point, Vector, Vector], Any]
    analytic_linearization: Callable[[Point, Vector, Vector], SparseLinearization]
    conservative_values: Callable[[Point, Point, StateIncrement], Any]
    conservative_derivative: Callable[[Point, Point], csc_matrix]
    point_validator: Callable[[Point], None] | None = None
    arithmetic: AssemblyArithmetic | None = None

    def __post_init__(self) -> None:
        if (type(self.input_count) is not int or self.input_count < 0
                or type(self.storage_count) is not int or self.storage_count < 0):
            raise ContractError("invalid_problem_counts")
        if (self.structure.shape != (self.layout.size, self.layout.size)
                or self.structure.layout_identity != self.layout.identity):
            raise ContractError("problem_structure_layout_mismatch")
        units, roles = [], []
        for equation in self.layout.equations:
            units.extend([equation.unit] * int(np.prod(equation.shape)))
            roles.extend([equation.role] * int(np.prod(equation.shape)))
        if (len(units) != self.layout.size or self.storage_count > len(units)
                or any(role != "storage" for role in roles[:self.storage_count])
                or any(role == "storage" for role in roles[self.storage_count:])):
            raise ContractError("problem_equation_order_mismatch")
        if tuple(unit * SECOND for unit in units[:self.storage_count]) != self.storage.units:
            raise ContractError("storage_unit_mismatch")
        if not self.source_identity or self.storage.source_identity != self.source_identity:
            raise ContractError("problem_source_mismatch")
        if self.storage.arithmetic is not self.arithmetic:
            raise ContractError("problem_arithmetic_mismatch")

    def _point(self, point: Point) -> None:
        if (point.state.layout.identity != self.layout.identity or point.y.shape != (self.layout.size,)
                or point.inputs.shape != (self.input_count,)):
            raise ContractError("problem_point_mismatch")
        if self.point_validator is not None:
            self.point_validator(point)

    def _rates(self, point: Point, ydot: Vector, adot: Vector) -> tuple[Vector, Vector]:
        self._point(point)
        ydot, adot = frozen_array(ydot), frozen_array(adot)
        if ydot.shape != point.y.shape or adot.shape != point.inputs.shape:
            raise ContractError("rate_shape_mismatch")
        return ydot, adot

    def _values(self, values: Any) -> Any:
        a = self.arithmetic if self.arithmetic is not None else FloatArithmetic()
        values = a.array(values)
        if values.shape != (self.layout.size,):
            raise ContractError("residual_shape_mismatch")
        return a.freeze(values)

    def residual(self, point: Point, ydot: Vector, adot: Vector) -> Any:
        ydot, adot = self._rates(point, ydot, adot)
        return self._values(self.residual_values(point, ydot, adot))

    def linearize(self, point: Point, ydot: Vector, adot: Vector) -> SparseLinearization:
        ydot, adot = self._rates(point, ydot, adot)
        result = self.analytic_linearization(point, ydot, adot)
        if (not isinstance(result, SparseLinearization) or result.source_identity != self.source_identity
                or result.structure.identity != self.structure.identity):
            raise ContractError("linearization_source_or_structure_mismatch")
        if result.inputs.shape != (self.layout.size, self.input_count):
            raise ContractError("equation_derivative_shape_mismatch")
        return result

    def _interval(self, left: Point, right: Point) -> None:
        self._point(left)
        self._point(right)
        if right.time <= left.time:
            raise ContractError("nonpositive_step")

    def conservative_residual(self, left: Point, right: Point, increment: StateIncrement) -> Any:
        self._interval(left, right)
        increment.validate(left, right)
        return self._values(self.conservative_values(left, right, increment))

    def conservative_jacobian(self, left: Point, right: Point) -> csc_matrix:
        self._interval(left, right)
        return self.structure.matrix(self.conservative_derivative(left, right))


@dataclass(frozen=True)
class PhysicalScaling(ImmutableArrays):
    """Explicit solver scaling; physical arithmetic is retained before rounding."""

    rows: Vector
    columns: Vector

    def __post_init__(self) -> None:
        for name in ("rows", "columns"):
            value = frozen_array(getattr(self, name))
            if value.ndim != 1 or np.any(value <= 0):
                raise ContractError("invalid_solver_scale")
            object.__setattr__(self, name, value)

    def residual(self, value: Any, *, arithmetic: AssemblyArithmetic | None = None) -> Any:
        a = arithmetic if arithmetic is not None else FloatArithmetic()
        value = a.array(value)
        if value.shape != self.rows.shape:
            raise ContractError("scale_shape_mismatch")
        return a.freeze(a.divide(value, a.array(self.rows)))

    def jacobian(self, value: Any, *, arithmetic: AssemblyArithmetic | None = None) -> Any:
        shape = (len(self.rows), len(self.columns))
        if value.shape != shape:
            raise ContractError("scale_shape_mismatch")
        if isspmatrix_csc(value):
            if arithmetic is not None:
                raise ContractError("sparse_precision_requires_explicit_values_adapter")
            result = value.copy()
            cols = np.repeat(np.arange(shape[1]), np.diff(result.indptr))
            result.data = frozen_array(result.data * self.columns[cols] / self.rows[result.indices]).copy()
            return result
        a = arithmetic if arithmetic is not None else FloatArithmetic()
        result = a.weighted(a.array(value), np.broadcast_to(self.columns, shape))
        return a.freeze(a.divide(result, a.array(np.broadcast_to(self.rows[:, None], shape))))


@dataclass(frozen=True)
class TerminalPort:
    id: str
    outward_normal: int
    area: float

    def __post_init__(self) -> None:
        if not self.id or self.outward_normal not in {-1, 1} or not np.isfinite(self.area) or self.area <= 0:
            raise ContractError("invalid_port")

    def charge(self, displacement: ArrayLike | PhysicalArray, *,
               arithmetic: AssemblyArithmetic | None = None, unit: Unit = COULOMB / AREA) -> Any:
        if unit != COULOMB / AREA:
            raise ContractError("port_unit_mismatch")
        a = arithmetic if arithmetic is not None else FloatArithmetic()
        value = a.array(displacement)
        return a.freeze(a.weighted(value, np.full(value.shape, -self.area * self.outward_normal)))

    def current(self, conduction: ArrayLike | PhysicalArray, displacement_rate: ArrayLike | PhysicalArray, *,
                arithmetic: AssemblyArithmetic | None = None,
                conduction_unit: Unit = COULOMB / AREA / SECOND,
                displacement_rate_unit: Unit = COULOMB / AREA / SECOND) -> Any:
        if conduction_unit != COULOMB / AREA / SECOND or displacement_rate_unit != conduction_unit:
            raise ContractError("port_unit_mismatch")
        a = arithmetic if arithmetic is not None else FloatArithmetic()
        jc, ddot = a.array(conduction), a.array(displacement_rate)
        if jc.shape != ddot.shape:
            raise ContractError("port_shape_mismatch")
        return a.freeze(a.weighted(a.add(jc, ddot), np.full(jc.shape, -self.area * self.outward_normal)))


@dataclass(frozen=True)
class AcceptedStep(ImmutableArrays):
    left: Point
    right: Point
    increment: StateIncrement
    storage_delta: Vector
    displacement_delta: Vector
    derivative: Vector
    input_rate: Vector
    method: str
    event_side: str
    acceptance_metrics: tuple[tuple[str, float], ...]
    physical_checks_passed: bool
    arithmetic: AssemblyArithmetic | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.increment.validate(self.left, self.right)
        if self.right.time <= self.left.time:
            raise ContractError("nonpositive_step")
        if self.event_side not in {"continuous", "left", "right"} or not self.method:
            raise ContractError("invalid_step_history")
        if not self.physical_checks_passed or not self.acceptance_metrics:
            raise ContractError("physical_step_rejected")
        metrics = tuple((name, float(value)) for name, value in self.acceptance_metrics)
        if any(not isinstance(name, str) or not name for name, _ in metrics):
            raise ContractError("invalid_acceptance_metric")
        if any(not np.isfinite(value) for _, value in metrics):
            raise ContractError("nonfinite_acceptance_metric")
        object.__setattr__(self, "acceptance_metrics", metrics)
        a = self.arithmetic if self.arithmetic is not None else FloatArithmetic()
        for name in ("storage_delta", "displacement_delta"):
            object.__setattr__(self, name, a.freeze(a.array(getattr(self, name))))
        for name in ("derivative", "input_rate"):
            object.__setattr__(self, name, frozen_array(getattr(self, name)))
        if self.derivative.shape != self.right.y.shape or self.input_rate.shape != self.right.inputs.shape:
            raise ContractError("step_derivative_shape_mismatch")
