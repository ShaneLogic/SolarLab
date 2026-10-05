"""Isolated P02 numerical contracts; no production entry point imports this module.

The independent references live in tests/numerical_contracts.  This module owns
interfaces and candidate arithmetic, never reference answers or acceptance gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import math
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

    def linear_form(self, form: PhysicalLinearForm, *, arithmetic=None,
                    point: Point | None = None, sources=()) -> LinearFormResult:
        return _linear_provider(arithmetic).linear_form(form, self, point=point, sources=sources)


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

    def linear_form(self, left: Point, right: Point, form: PhysicalLinearForm, *,
                    arithmetic=None, sources=()) -> LinearFormResult:
        return _linear_provider(arithmetic).linear_form(
            form, self, left=left, right=right, sources=sources)

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


def _linear_scalar(value: Any) -> float:
    """A coefficient is an exact supplied binary64 value, never an expression."""
    raw = np.asarray(value)
    if (raw.shape != () or raw.dtype.kind not in {"i", "u", "f"}
            or raw.dtype.kind == "f" and raw.dtype.itemsize > 8):
        raise ContractError("linear_factor_type")
    result = float(raw)
    if not math.isfinite(result):
        raise ContractError("linear_nonfinite_factor")
    if raw.dtype.kind in {"i", "u"} and int(raw) != int(result):
        raise ContractError("linear_inexact_integer")
    return result


def _linear_array(value: ArrayLike) -> Vector:
    raw = np.asarray(value)
    if raw.dtype.kind not in {"i", "u", "f"} or raw.dtype.kind == "f" and raw.dtype.itemsize > 8:
        raise ContractError("linear_word_dtype")
    result = frozen_array(raw)
    if raw.dtype.kind in {"i", "u"} and any(int(a) != int(b) for a, b in zip(raw.flat, result.flat)):
        raise ContractError("linear_inexact_integer")
    return result


def _linear_unit(unit: Unit) -> Unit:
    if not isinstance(unit, Unit):
        raise ContractError("linear_invalid_unit")
    return Unit(tuple(unit.powers))


@dataclass(frozen=True)
class LinearFactor:
    """One source-bound physical coefficient; products stay factored."""

    id: str
    value: float
    unit: Unit
    source_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id or not isinstance(self.source_identity, str) or not self.source_identity:
            raise ContractError("linear_factor_identity")
        object.__setattr__(self, "value", _linear_scalar(self.value))
        object.__setattr__(self, "unit", _linear_unit(self.unit))

    @property
    def identity(self) -> str:
        return sha256(repr((self.id, self.value.hex(), self.unit, self.source_identity)).encode()).hexdigest()


@dataclass(frozen=True)
class LinearSourceSpec:
    """A resolved source value from one named evaluation, in its actual units."""

    id: str
    support: str
    shape: tuple[int, ...]
    unit: Unit
    source_identity: str

    def __post_init__(self) -> None:
        if (not isinstance(self.id, str) or not self.id or not isinstance(self.support, str) or not self.support
                or not isinstance(self.source_identity, str) or not self.source_identity):
            raise ContractError("linear_source_identity")
        shape = tuple(self.shape)
        if not shape or any(type(n) is not int or n < 0 for n in shape):
            raise ContractError("linear_source_shape")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "unit", _linear_unit(self.unit))


@dataclass(frozen=True)
class LinearTerm:
    """One stored COO contribution, including declared numeric zeros.

    ``field=None`` is exact unity for state-only fixed backgrounds. A divisor
    requests the separately bounded quotient operation, not a rounded inverse.
    """

    row: int
    field: str | None
    index: int
    factors: tuple[LinearFactor, ...] = ()
    sign: int = 1
    divisor: LinearFactor | None = None

    def __post_init__(self) -> None:
        if isinstance(self.row, (bool, np.bool_)) or isinstance(self.index, (bool, np.bool_)):
            raise ContractError("invalid_index")
        indices = integer_indices([self.row, self.index])
        if indices.shape != (2,) or np.any(indices < 0):
            raise ContractError("linear_index_outside_support")
        if self.field is not None and (not isinstance(self.field, str) or not self.field):
            raise ContractError("linear_unknown_field")
        factors = tuple(self.factors)
        if len(factors) > 4 or not all(isinstance(factor, LinearFactor) for factor in factors):
            raise ContractError("linear_factor_arity_or_type")
        if type(self.sign) is not int or self.sign not in {-1, 1}:
            raise ContractError("linear_invalid_sign")
        if self.divisor is not None and (not isinstance(self.divisor, LinearFactor) or self.divisor.value == 0):
            raise ContractError("linear_invalid_divisor")
        object.__setattr__(self, "row", int(indices[0]))
        object.__setattr__(self, "index", int(indices[1]))
        object.__setattr__(self, "factors", factors)

    def _key(self) -> tuple:
        return (self.row, self.field or "", self.index, self.sign,
                tuple(factor.identity for factor in self.factors),
                self.divisor.identity if self.divisor is not None else "")


@dataclass(frozen=True)
class PhysicalLinearForm:
    """Small sparse physical action; output rows never modify Point layouts.

    ``declared_nnz`` counts stored contributions, including duplicate positions
    and zeros. Each term has at most four numerator factors and one explicit
    divisor. Total work is bounded by the actual declaration and its caller's
    resource budget; this metadata imposes no device/grid-size ceiling.
    """

    layout: Layout
    row_ids: tuple[str, ...]
    row_units: tuple[Unit, ...]
    terms: tuple[LinearTerm, ...]
    source_identity: str
    kind: str
    declared_nnz: int
    sources: tuple[LinearSourceSpec, ...] = ()
    identity: str = field(init=False)

    def __post_init__(self) -> None:
        if (not isinstance(self.layout, Layout) or self.kind not in {"state", "increment", "rate"}
                or not isinstance(self.source_identity, str) or not self.source_identity):
            raise ContractError("linear_form_identity_or_kind")
        rows, units, terms, sources = tuple(self.row_ids), tuple(self.row_units), tuple(self.terms), tuple(self.sources)
        if (len(rows) != len(units) or len(set(rows)) != len(rows)
                or any(not isinstance(row, str) or not row for row in rows)):
            raise ContractError("linear_row_metadata")
        if (type(self.declared_nnz) is not int or self.declared_nnz != len(terms)
                or not all(isinstance(term, LinearTerm) for term in terms)):
            raise ContractError("linear_declared_nnz")
        units = tuple(_linear_unit(unit) for unit in units)
        specs = {spec.id: spec for spec in self.layout.variables}
        supports = {support.id: support for support in self.layout.supports}
        if (not all(isinstance(source, LinearSourceSpec) for source in sources)
                or len({source.id for source in sources}) != len(sources)
                or {source.id for source in sources} & specs.keys()):
            raise ContractError("linear_source_ownership")
        for source in sources:
            if source.support not in supports or source.shape != supports[source.support].shape:
                raise ContractError("linear_source_support")
            if source.source_identity != self.source_identity:
                raise ContractError("linear_source_mismatch")
        external = {source.id: source for source in sources}
        for term in terms:
            if term.row >= len(rows):
                raise ContractError("linear_row_outside_support")
            if term.field is None:
                if self.kind != "state" or term.index != 0:
                    raise ContractError("linear_constant_requires_state")
                unit = ONE
            else:
                if term.field not in specs and term.field not in external:
                    raise ContractError("linear_unknown_field")
                spec = specs[term.field] if term.field in specs else external[term.field]
                if term.index >= int(np.prod(spec.shape)):
                    raise ContractError("linear_index_outside_support")
                unit = spec.unit / SECOND if self.kind == "rate" and term.field in specs else spec.unit
            for factor in term.factors + ((term.divisor,) if term.divisor is not None else ()):
                if factor.source_identity != self.source_identity:
                    raise ContractError("linear_factor_source_mismatch")
            for factor in term.factors:
                unit = unit * factor.unit
            if term.divisor is not None:
                unit = unit / term.divisor.unit
            if unit != units[term.row]:
                raise ContractError("linear_unit_mismatch")
        terms = tuple(sorted(terms, key=LinearTerm._key))
        sources = tuple(sorted(sources, key=lambda value: value.id))
        for name, value in (("row_ids", rows), ("row_units", units), ("terms", terms), ("sources", sources)):
            object.__setattr__(self, name, value)
        payload = ("physical-linear-form-v1", self.layout.identity, rows, units,
                   tuple(term._key() for term in terms), sources, self.source_identity, self.kind, self.declared_nnz)
        object.__setattr__(self, "identity", sha256(repr(payload).encode()).hexdigest())


def _source_evaluation_identity(point_identity: str, entries) -> str:
    digest = sha256(b"physical-linear-source-evaluation-v1:" + point_identity.encode())
    for spec, value in sorted(entries, key=lambda item: item[0].id):
        digest.update(repr(spec).encode() + value.identity_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class BoundLinearSource:
    spec: LinearSourceSpec
    value: PhysicalArray
    point_identity: str
    evaluation_identity: str

    def __post_init__(self) -> None:
        if (not isinstance(self.spec, LinearSourceSpec) or not self.point_identity or not self.evaluation_identity
                or not callable(getattr(self.value, "is_finite", None))
                or self.value.shape != self.spec.shape or not self.value.is_finite()
                or getattr(self.value, "authority", None) is not None):
            raise ContractError("linear_source_binding")
        object.__setattr__(self, "value", self.value.immutable_copy())

    @classmethod
    def bind(cls, point: Point, entries: Iterable[tuple[LinearSourceSpec, PhysicalArray]]) -> tuple[BoundLinearSource, ...]:
        """Freeze the actual evaluated packet; an arbitrary label is not proof.

        The caller first validates its physical evaluation against ``point``.
        This factory binds the returned values, not the accuracy of that model.
        """
        entries = tuple((spec, value.immutable_copy()) for spec, value in entries)
        if (not isinstance(point, Point) or not all(isinstance(spec, LinearSourceSpec) for spec, _ in entries)
                or len({spec.id for spec, _ in entries}) != len(entries)):
            raise ContractError("linear_source_packet_mismatch")
        identity = _source_evaluation_identity(point.identity, entries)
        return tuple(cls(spec, value, point.identity, identity) for spec, value in entries)

    @property
    def identity(self) -> str:
        return sha256(repr((self.spec, self.point_identity, self.evaluation_identity)).encode()
                      + self.value.identity_bytes()).hexdigest()


@dataclass(frozen=True, init=False)
class RateView(ImmutableArrays):
    """Physical rate words bound to a Point and their declared raw origin.

    ``values`` may be a four-word input primitive; it is never converted to a
    DoubleArray. The mapping factory owns mapping correctness. This record
    owns actual mapped/raw bytes and rejects a different Point, input rate or
    source. Rate fields do not inherit nonnegative state-domain constraints.
    """

    point: Point
    values: Any
    input_rate: Vector
    source_identity: str
    mapping_identity: str
    origin: str
    raw_coordinates: Vector
    raw_rate: Vector
    identity: str
    _words: tuple[Vector, ...]

    def __init__(self, point: Point, values: Any, input_rate: ArrayLike, *, source_identity: str,
                 mapping_identity: str, origin: str, raw_coordinates: ArrayLike, raw_rate: ArrayLike):
        if (not isinstance(point, Point) or not isinstance(source_identity, str) or not source_identity
                or not isinstance(mapping_identity, str) or not mapping_identity
                or origin not in {"physical-rate", "mapped-coordinate-rate"}):
            raise ContractError("linear_rate_identity")
        if not hasattr(values, "immutable_copy"):
            values = FloatArray(_linear_array(values))
        values = values.immutable_copy()
        words = (values.values,) if type(values) is FloatArray else getattr(values, "words", ())
        words = tuple(_linear_array(word) for word in words)
        if (not 1 <= len(words) <= 4 or values.shape != (point.state.layout.size,)
                or any(word.shape != values.shape for word in words)):
            raise ContractError("linear_rate_word_shape_or_count")
        input_rate, raw_coordinates, raw_rate = map(_linear_array, (input_rate, raw_coordinates, raw_rate))
        if (point.y.shape != values.shape or input_rate.shape != point.inputs.shape
                or raw_coordinates.shape != point.y.shape or raw_rate.shape != point.y.shape):
            raise ContractError("rate_shape_mismatch")
        for name, value in (("point", point), ("values", values), ("input_rate", input_rate),
                            ("source_identity", source_identity), ("mapping_identity", mapping_identity),
                            ("origin", origin), ("raw_coordinates", raw_coordinates), ("raw_rate", raw_rate), ("_words", words)):
            object.__setattr__(self, name, value)
        digest = sha256(repr(("physical-rate-view-v1", point.identity, point.state.layout.identity,
                              source_identity, mapping_identity, origin)).encode())
        digest.update(values.identity_bytes() + b"".join(word.tobytes() for word in words))
        digest.update(input_rate.tobytes() + raw_coordinates.tobytes() + raw_rate.tobytes())
        object.__setattr__(self, "identity", digest.hexdigest())

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    @property
    def words(self) -> tuple[Vector, ...]:
        return tuple(word.view() for word in self._words)

    def field(self, variable_id: str) -> Any:
        specs = {spec.id: spec for spec in self.point.state.layout.variables}
        if variable_id not in specs:
            raise ContractError("unknown_variable")
        selection = self.point.state.layout.offsets[variable_id]
        indices = np.arange(selection.start, selection.stop).reshape(specs[variable_id].shape)
        take = getattr(self.values, "take_flat", self.values.take if hasattr(self.values, "take") else None)
        if take is None:
            raise ContractError("linear_rate_field_capability")
        return take(indices)

    def validate(self, point: Point, input_rate: ArrayLike, source_identity: str | None = None) -> None:
        if (point.identity != self.point.identity or point.state.layout.identity != self.point.state.layout.identity
                or source_identity is not None and source_identity != self.source_identity):
            raise ContractError("linear_rate_point_or_source_mismatch")
        input_rate = _linear_array(input_rate)
        if input_rate.shape != self.input_rate.shape or input_rate.tobytes() != self.input_rate.tobytes():
            raise ContractError("linear_rate_input_mismatch")

    def linear_form(self, form: PhysicalLinearForm, *, arithmetic=None, point: Point,
                    sources=()) -> LinearFormResult:
        return _linear_provider(arithmetic).linear_form(form, self, point=point, sources=sources)

    def __array__(self, dtype=None, copy=None):
        raise TypeError("implicit full-word rate projection is forbidden")


@dataclass(frozen=True)
class LinearFormResult(ImmutableArrays):
    value: Any
    units: tuple[Unit, ...]
    form_identity: str
    operand_identity: str
    source_identities: tuple[str, ...]
    absolute_error_bound: FloatArray
    arithmetic_policy: str

    def __post_init__(self) -> None:
        units = tuple(_linear_unit(unit) for unit in self.units)
        value = self.value.immutable_copy() if hasattr(self.value, "immutable_copy") else frozen_array(self.value)
        bound = self.absolute_error_bound.immutable_copy()
        if (value.shape != (len(units),) or bound.shape != value.shape or np.any(bound.values < 0)
                or not self.form_identity or not self.operand_identity or not self.arithmetic_policy):
            raise ContractError("linear_result_metadata")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "source_identities", tuple(self.source_identities))
        object.__setattr__(self, "absolute_error_bound", bound)


def _linear_provider(arithmetic):
    provider = arithmetic if arithmetic is not None else FloatArithmetic()
    if not callable(getattr(provider, "linear_form", None)):
        raise ContractError("physical_linear_action_unavailable")
    return provider


def _word_integer(words) -> int:
    """Exact bounded dyadic readback in units of the minimum binary64 word.

    Only verifies finite-word transforms and computes their error bounds. It
    does not replace the floating point physical action with rational physics.
    A binary64 input needs at most 2098 bits. Summing K submitted finite words
    adds at most ceil(log2(K)) bits. The four-factor bound gives at most 128
    product words per contribution (eight input words times sixteen products),
    and each quotient remainder adds at most eight words. No grid-size limit
    or unbounded expression/history nesting is used to justify that bound.
    """
    total = 0
    for word in words:
        numerator, denominator = float(word).as_integer_ratio()
        total += numerator << (1074 - (denominator.bit_length() - 1))
    return total


def _linear_sum(words) -> float:
    try:
        value = math.fsum(words)
    except (OverflowError, ValueError) as error:
        raise ContractError("linear_sum_range") from error
    if not math.isfinite(value):
        raise ContractError("linear_sum_range")
    return value


def _finite_two_product(a: float, b: float) -> tuple[float, float]:
    """Scaled Dekker product with an exact range/underflow verification.

    Split normalized mantissas, where every intermediate is normal and finite;
    rescale the two product words only afterwards. Exact dyadic verification
    rejects a rescaling that lost a nonzero tail, including gradual underflow.
    """
    if not math.isfinite(a) or not math.isfinite(b):
        raise ContractError("linear_product_nonfinite")
    if a == 0 or b == 0:
        return 0.0, 0.0
    if b in {-1.0, 1.0}:
        return a * b, 0.0
    ma, ea = math.frexp(a)
    mb, eb = math.frexp(b)
    ca, cb = 134217729.0 * ma, 134217729.0 * mb
    ah, bh = ca - (ca - ma), cb - (cb - mb)
    al, bl = ma - ah, mb - bh
    high = ma * mb
    low = al * bl - (((high - ah * bh) - al * bh) - ah * bl)
    try:
        high, low = math.ldexp(high, ea + eb), math.ldexp(low, ea + eb)
    except OverflowError as error:
        raise ContractError("linear_product_overflow") from error
    if not math.isfinite(high) or not math.isfinite(low):
        raise ContractError("linear_product_overflow")
    if _word_integer((a,)) * _word_integer((b,)) != _word_integer((high, low)) << 1074:
        raise ContractError("linear_product_underflow_or_inexact")
    return high, low


def _upper_ratio(numerator: int, denominator: int) -> float:
    """Outward binary64 bound, including a nonzero subnormal remainder."""
    if numerator == 0:
        return 0.0
    try:
        value = numerator / denominator
    except OverflowError as error:
        raise ContractError("linear_error_bound_range") from error
    if not math.isfinite(value):
        raise ContractError("linear_error_bound_range")
    a, b = value.as_integer_ratio()
    if a * denominator < numerator * b:
        value = math.nextafter(value, math.inf)
    if not math.isfinite(value):
        raise ContractError("linear_error_bound_range")
    return value


def _linear_quotient(words: list[float], divisor: float) -> tuple[list[float], float]:
    """Four quotient words and a bound on the exact unrepresented remainder.

    This operation is explicitly approximate. Each remainder update is exact
    in finite words, and its residual divided by the original divisor bounds
    the error before final row rounding. No reciprocal is used as a factor.
    """
    if not math.isfinite(divisor) or divisor == 0:
        raise ContractError("linear_invalid_divisor")
    remainder, quotient = list(words), []
    for _ in range(4):
        lead = _linear_sum(remainder)
        if lead == 0:
            if _word_integer(remainder):
                raise ContractError("linear_quotient_underflow")
            break
        word = lead / divisor
        if not math.isfinite(word):
            raise ContractError("linear_quotient_overflow")
        if word == 0:
            raise ContractError("linear_quotient_underflow")
        high, low = _finite_two_product(word, divisor)
        quotient.append(word)
        remainder.extend((-high, -low))
    dn, dd = divisor.as_integer_ratio()
    bound = _upper_ratio(abs(_word_integer(remainder)) * dd, (1 << 1074) * abs(dn))
    return quotient, bound


def _linear_reduce(form: PhysicalLinearForm, components: Mapping[str, tuple[Vector, ...]],
                   output_words: int) -> tuple[Vector, Vector, Vector]:
    """The single state/increment/rate reduction, before the requested output."""
    groups: list[dict[str, tuple[float | None, list[float]]]] = [{} for _ in form.row_ids]
    for term in form.terms:
        divisor_key = term.divisor.identity if term.divisor is not None else ""
        divisor = term.divisor.value if term.divisor is not None else None
        group = groups[term.row].setdefault(divisor_key, (divisor, []))[1]
        # A zero coefficient has empty numeric words but remains in the form's
        # declared structural support, unit validation and source identity.
        if any(factor.value == 0 for factor in term.factors):
            continue
        words = (1.0,) if term.field is None else tuple(float(word.flat[term.index]) for word in components[term.field])
        for word in words:
            product = [term.sign * word] if word else []
            for factor in term.factors:
                product = [part for value in product for part in _finite_two_product(value, factor.value) if part]
            group.extend(product)
    high, low, bounds = (np.zeros(len(form.row_ids)) for _ in range(3))
    for row, entries in enumerate(groups):
        words, errors = [], []
        for divisor, numerator in entries.values():
            if divisor is None:
                words.extend(numerator)
            else:
                quotient, error = _linear_quotient(numerator, divisor)
                words.extend(quotient)
                errors.append(error)
        high[row] = _linear_sum(words)
        if output_words == 2:
            low[row] = _linear_sum([*words, -high[row]])
            # Normalize the requested two-word result without changing its
            # represented sum; the final bound uses the normalized pair.
            h = _linear_sum((high[row], low[row]))
            l = _linear_sum((high[row], low[row], -h))
            high[row], low[row] = h, l
        remainder = abs(_word_integer([*words, -high[row], -low[row]]))
        errors.append(_upper_ratio(remainder, 1 << 1074))
        bounds[row] = _upper_ratio(_word_integer(errors), 1 << 1074)
    return high, low, bounds


def _linear_context(form, operand, *, point=None, left=None, right=None, sources=()):
    if not isinstance(form, PhysicalLinearForm):
        raise ContractError("physical_linear_form_required")
    if isinstance(operand, StateView):
        layout, expected_kind, identity = operand.layout, "state", operand.authority.identity
        if left is not None or right is not None:
            raise ContractError("linear_operand_binding")
        if point is not None and (point.state.authority.identity != operand.authority.identity
                                  or point.state.layout.identity != layout.identity):
            raise ContractError("linear_state_point_mismatch")
    elif isinstance(operand, StateIncrement):
        if not isinstance(left, Point) or not isinstance(right, Point) or point is not None:
            raise ContractError("linear_increment_points_required")
        operand.validate(left, right)
        layout, expected_kind, point = right.state.layout, "increment", right
        identity = sha256((left.identity + ":" + right.identity).encode()).hexdigest()
    elif isinstance(operand, RateView):
        if not isinstance(point, Point) or left is not None or right is not None:
            raise ContractError("linear_rate_point_required")
        operand.validate(point, operand.input_rate, form.source_identity)
        layout, expected_kind, identity = point.state.layout, "rate", operand.identity
    else:
        raise ContractError("unsupported_linear_operand")
    # Recheck metadata rather than trusting a cached identity on an externally
    # supplied layout whose nested objects may have been changed by a caller.
    actual_layout = sha256(repr((layout.supports, layout.variables, layout.equations)).encode()).hexdigest()
    if form.kind != expected_kind or layout.identity != form.layout.identity or actual_layout != layout.identity:
        raise ContractError("linear_layout_or_kind_mismatch")
    sources = tuple(sources)
    if (not all(isinstance(source, BoundLinearSource) for source in sources)
            or tuple(sorted((source.spec for source in sources), key=lambda spec: spec.id)) != form.sources):
        raise ContractError("linear_source_packet_mismatch")
    if sources:
        if point is None or any(source.point_identity != point.identity for source in sources):
            raise ContractError("linear_source_point_mismatch")
        expected = _source_evaluation_identity(point.identity, ((source.spec, source.value) for source in sources))
        if any(source.evaluation_identity != expected for source in sources):
            raise ContractError("linear_source_evaluation_mismatch")
    identity = sha256(repr((identity, point.identity if point is not None else None)).encode()).hexdigest()
    return identity, sources


def _apply_linear_form(form, operand, *, point=None, left=None, right=None, sources=(),
                       array_words, state_words, increment_words, output_words, finish):
    identity, sources = _linear_context(form, operand, point=point, left=left, right=right, sources=sources)
    external = {source.spec.id: source for source in sources}
    specs = {spec.id: spec for spec in form.layout.variables}
    components = {}
    for name in {term.field for term in form.terms if term.field is not None}:
        if name in external:
            parts, shape = array_words(external[name].value), external[name].spec.shape
        else:
            shape = specs[name].shape
            if isinstance(operand, StateView):
                parts = state_words(operand, name)
            elif isinstance(operand, StateIncrement):
                parts = increment_words(left.state, right.state, name)
            else:
                selection = form.layout.offsets[name]
                parts = tuple(word[selection].reshape(shape) for word in operand.words)
        if (not 1 <= len(parts) <= 8 or any(part.shape != shape or part.dtype != np.dtype(np.float64)
                                         or not np.isfinite(part).all() for part in parts)):
            raise ContractError("linear_source_word_shape_or_count")
        components[name] = parts
    high, low, bound = _linear_reduce(form, components, output_words)
    try:
        value = finish(high, low)
    except ArithmeticError as error:
        raise ContractError("linear_output_provider_range") from error
    return LinearFormResult(value, form.row_units, form.identity, identity,
                            tuple(source.identity for source in sources), FloatArray(bound),
                            f"finite-eft-factor4/quotient4/final-word{output_words}-v1")


def _float_linear_words(value) -> tuple[Vector, ...]:
    if type(value) is not FloatArray:
        raise ContractError("linear_precision_provider_required")
    return (value.values,)


def _resolved_linear_words(state, name) -> tuple[Vector, ...]:
    if type(state.authority) is not ResolvedAuthority:
        raise ContractError("linear_authority_provider_required")
    return _float_linear_words(state.authority.field(name))


def _resolved_increment_words(left, right, name) -> tuple[Vector, ...]:
    return _resolved_linear_words(right, name) + tuple(-word for word in _resolved_linear_words(left, name))


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
    def linear_form(self, form: PhysicalLinearForm, operand: Any, *,
                    point=None, left=None, right=None, sources=()) -> LinearFormResult: ...
    def validate_rate(self, value: RateView) -> None: ...


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

    def validate_rate(self, value: RateView) -> None:
        if len(value.words) != 1 or type(value.point.state.authority) is not ResolvedAuthority:
            raise ContractError("linear_precision_provider_required")

    def linear_form(self, form: PhysicalLinearForm, operand: Any, *,
                    point=None, left=None, right=None, sources=()) -> LinearFormResult:
        if isinstance(operand, RateView):
            self.validate_rate(operand)
        return _apply_linear_form(
            form, operand, point=point, left=left, right=right, sources=sources,
            array_words=_float_linear_words, state_words=_resolved_linear_words,
            increment_words=_resolved_increment_words, output_words=1,
            finish=lambda high, low: self.freeze(high))


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
        if isinstance(ydot, RateView):
            raise ContractError("full_word_rate_requires_explicit_problem_callback")
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
    accepted_rate_mapping_identity: str | None = None

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
        if self.accepted_rate_mapping_identity is not None and (
                not isinstance(self.accepted_rate_mapping_identity, str) or not self.accepted_rate_mapping_identity):
            raise ContractError("problem_rate_mapping_identity")

    def _point(self, point: Point) -> None:
        if (point.state.layout.identity != self.layout.identity or point.y.shape != (self.layout.size,)
                or point.inputs.shape != (self.input_count,)):
            raise ContractError("problem_point_mismatch")
        if self.point_validator is not None:
            self.point_validator(point)

    def _rates(self, point: Point, ydot: Any, adot: Vector) -> tuple[Any, Vector]:
        self._point(point)
        if isinstance(ydot, RateView):
            if (self.accepted_rate_mapping_identity is None
                    or ydot.mapping_identity != self.accepted_rate_mapping_identity):
                raise ContractError("problem_full_word_rate_not_admitted")
            ydot.validate(point, adot, self.source_identity)
            provider = _linear_provider(self.arithmetic)
            if not callable(getattr(provider, "validate_rate", None)):
                raise ContractError("problem_full_word_rate_unavailable")
            provider.validate_rate(ydot)
            return ydot, frozen_array(adot)
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
