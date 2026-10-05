"""Isolated P02 numerical contracts; no production entry point imports this module.

The independent references live in tests/numerical_contracts.  This module owns
interfaces and candidate arithmetic, never reference answers or acceptance gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

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


@dataclass(frozen=True)
class Unit:
    """SI exponents (length, time, charge, particle number); no implicit scale."""

    powers: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if len(self.powers) != 4 or any(type(x) is not int for x in self.powers):
            raise ContractError("invalid_unit")

    def __mul__(self, other: Unit) -> Unit:
        return Unit(tuple(a + b for a, b in zip(self.powers, other.powers)))

    def __truediv__(self, other: Unit) -> Unit:
        return Unit(tuple(a - b for a, b in zip(self.powers, other.powers)))


ONE = Unit((0, 0, 0, 0))
SECOND = Unit((0, 1, 0, 0))
VOLUME = Unit((3, 0, 0, 0))
AREA = Unit((2, 0, 0, 0))
PARTICLE = Unit((0, 0, 0, 1))
COULOMB = Unit((0, 0, 1, 0))


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
    def identity_bytes(self) -> bytes: ...


@dataclass(frozen=True)
class FloatArray:
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

    def identity_bytes(self) -> bytes:
        return b"float64:" + repr(self.shape).encode() + self.values.tobytes()


@dataclass(frozen=True, init=False)
class StateView:
    layout: Layout
    fields: Mapping[str, PhysicalArray]

    def __init__(self, layout: Layout, fields: Iterable[tuple[str, PhysicalArray]]):
        owned: dict[str, PhysicalArray] = {}
        specs = {x.id: x for x in layout.variables}
        for key, value in fields:
            if key in owned:
                raise ContractError("duplicate_state_write")
            if key not in specs:
                raise ContractError("unknown_variable")
            if value.shape != specs[key].shape:
                raise ContractError("state_shape_mismatch")
            owned[key] = value.immutable_copy()
        if set(owned) != set(specs):
            raise ContractError("incomplete_state")
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "fields", MappingProxyType(owned))

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


@dataclass(frozen=True)
class Point:
    time: float
    y: Vector
    inputs: Vector
    state: StateView
    coordinate_reference: str
    identity: str = field(init=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.time) or not self.coordinate_reference:
            raise ContractError("invalid_point")
        object.__setattr__(self, "y", frozen_array(self.y))
        object.__setattr__(self, "inputs", frozen_array(self.inputs))
        if self.y.ndim != 1 or self.inputs.ndim != 1:
            raise ContractError("invalid_coordinate_shape")
        digest = sha256(repr((self.time, self.coordinate_reference, self.state.layout.identity)).encode())
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

    def validate(self, left: Point, right: Point) -> None:
        if (left.identity, right.identity) != (self.left_identity, self.right_identity):
            raise ContractError("increment_endpoint_mismatch")
        if left.state.layout.identity != right.state.layout.identity:
            raise ContractError("increment_layout_mismatch")
        if set(self.fields) != set(left.state.fields):
            raise ContractError("incomplete_increment")
        if any(self.fields[key].shape != left.state.field(key).shape for key in self.fields):
            raise ContractError("increment_shape_mismatch")


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
class Geometry:
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
    values: Vector
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


class TermSink:
    """Typed unweighted contributions; geometry is applied here exactly once."""

    def __init__(self, layout: Layout, equation_id: str, geometry: Geometry):
        equations = {x.id: x for x in layout.equations}
        if equation_id not in equations:
            raise ContractError("unknown_equation")
        self.equation = equations[equation_id]
        self.supports = {x.id: x for x in layout.supports}
        self.geometry = geometry
        for support_id, location, shape in (
            (geometry.cell_support, "cell", geometry.volumes.shape),
            (geometry.face_support, "face", geometry.face_measures.shape),
        ):
            support = self.supports.get(support_id)
            if support is None or support.location != location or support.shape != shape:
                raise ContractError("geometry_support_mismatch")
        self._seen: set[str] = set()
        self._value = np.zeros(self.equation.shape)

    def add(self, term: Contribution) -> None:
        if not term.id or term.id in self._seen:
            raise ContractError("duplicate_contribution")
        if term.support not in self.supports:
            raise ContractError("unknown_support")
        support = self.supports[term.support]
        target = self.supports[self.equation.support]
        values = frozen_array(term.values)
        if values.shape != support.shape:
            raise ContractError("term_shape_mismatch")
        geometry = self.geometry
        if isinstance(term, CellSource):
            if support != target or support.location != "cell" or values.shape != geometry.volumes.shape:
                raise ContractError("wrong_source_support")
            if support.id != geometry.cell_support:
                raise ContractError("geometry_support_mismatch")
            expected = term.unit * geometry.volume_unit
            contribution = values * geometry.volumes
        elif isinstance(term, FaceFlux):
            if (support.location != "face" or target.location != "cell"
                    or values.shape != geometry.face_measures.shape
                    or target.shape != geometry.volumes.shape):
                raise ContractError("wrong_flux_support")
            if support.id != geometry.face_support or target.id != geometry.cell_support:
                raise ContractError("geometry_support_mismatch")
            expected = term.unit * geometry.face_unit
            contribution = np.zeros_like(self._value)
            weighted = values * geometry.face_measures
            np.add.at(contribution, geometry.pairs[:, 0], -weighted)
            np.add.at(contribution, geometry.pairs[:, 1], weighted)
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
            contribution = values * geometry.face_measures
        else:
            raise ContractError("untyped_contribution")
        if expected != self.equation.unit:
            raise ContractError("unit_mismatch")
        self._value += contribution
        self._seen.add(term.id)

    def value(self) -> Vector:
        return frozen_array(self._value)


@dataclass(frozen=True)
class StoragePartials:
    y: Vector
    inputs: Vector
    time: Vector


class StorageLaw(Protocol):
    def value(self, point: Point) -> Vector: ...
    def delta(self, left: Point, right: Point, increment: StateIncrement) -> Vector: ...
    def first(self, point: Point) -> StoragePartials: ...
    def rate_jvp(self, point: Point, ydot: Vector, adot: Vector,
                 dy: Vector, da: Vector, dt: float) -> Vector: ...


@dataclass(frozen=True)
class LinearStorage:
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

    def ida_matrix(self, cj: float) -> Vector:
        return self.y + cj * self.ydot


@dataclass(frozen=True)
class ImplicitSystem:
    storage: StorageLaw
    terms: Callable[[Point], BalanceTerms]

    def evaluate(self, point: Point) -> tuple[StoragePartials, BalanceTerms]:
        first, terms = self.storage.first(point), self.terms(point)
        n, m = point.y.size, point.inputs.size
        if terms.rate.ndim != 1 or terms.algebraic.ndim != 1:
            raise ContractError("invalid_equation_shape")
        k, l = terms.rate.size, terms.algebraic.size
        arrays = (
            (first.y, (k, n)), (first.inputs, (k, m)), (first.time, (k,)),
            (terms.Ry, (k, n)), (terms.Ra, (k, m)), (terms.Rt, (k,)),
            (terms.gy, (l, n)), (terms.ga, (l, m)), (terms.gt, (l,)),
        )
        if k + l != n or any(array.shape != shape for array, shape in arrays):
            raise ContractError("equation_derivative_shape_mismatch")
        if any(not np.isfinite(array).all() for array, _ in arrays):
            raise ContractError("nonfinite_derivative")
        if not np.isfinite(terms.rate).all() or not np.isfinite(terms.algebraic).all():
            raise ContractError("nonfinite_residual")
        return first, terms

    @staticmethod
    def _check_rates(point: Point, ydot: Vector, adot: Vector) -> None:
        if ydot.shape != point.y.shape or adot.shape != point.inputs.shape:
            raise ContractError("rate_shape_mismatch")
        if not np.isfinite(ydot).all() or not np.isfinite(adot).all():
            raise ContractError("nonfinite_rate")

    @staticmethod
    def _storage_vector(value: ArrayLike, count: int) -> Vector:
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (count,):
            raise ContractError("storage_callback_shape_mismatch")
        if not np.isfinite(array).all():
            raise ContractError("nonfinite_storage_callback")
        return array

    def residual(self, point: Point, ydot: Vector, adot: Vector) -> Vector:
        self._check_rates(point, ydot, adot)
        first, terms = self.evaluate(point)
        rate = first.y @ ydot + first.inputs @ adot + first.time
        return np.concatenate((rate - terms.rate, terms.algebraic))

    def linearize(self, point: Point, ydot: Vector, adot: Vector) -> Linearization:
        self._check_rates(point, ydot, adot)
        first, terms = self.evaluate(point)
        n, m = point.y.size, point.inputs.size
        zero_y, zero_a = np.zeros(n), np.zeros(m)
        def contraction(dy: Vector, da: Vector, dt: float) -> Vector:
            return self._storage_vector(
                self.storage.rate_jvp(point, ydot, adot, dy, da, dt), terms.rate.size,
            )
        rate_y = (np.column_stack([
            contraction(direction, zero_a, 0.0)
            for direction in np.eye(n)
        ]) if n else np.empty((terms.rate.size, 0)))
        rate_a = (np.column_stack([
            contraction(zero_y, direction, 0.0)
            for direction in np.eye(m)
        ]) if m else np.empty((terms.rate.size, 0)))
        rate_t = contraction(zero_y, zero_a, 1.0)
        return Linearization(
            np.vstack((rate_y - terms.Ry, terms.gy)),
            np.vstack((first.y, np.zeros_like(terms.gy))),
            np.vstack((rate_a - terms.Ra, terms.ga)),
            np.vstack((first.inputs, np.zeros_like(terms.ga))),
            np.concatenate((rate_t - terms.Rt, terms.gt)),
        )

    def conservative_residual(self, left: Point, right: Point,
                              increment: StateIncrement) -> Vector:
        h = right.time - left.time
        if h <= 0:
            raise ContractError("nonpositive_step")
        _, terms = self.evaluate(right)
        delta = self._storage_vector(self.storage.delta(left, right, increment), terms.rate.size)
        return np.concatenate((delta - h * terms.rate,
                               terms.algebraic))

    def conservative_jacobian(self, left: Point, right: Point) -> Vector:
        h = right.time - left.time
        if h <= 0:
            raise ContractError("nonpositive_step")
        first, terms = self.evaluate(right)
        return np.vstack((first.y - h * terms.Ry, terms.gy))


@dataclass(frozen=True)
class TerminalPort:
    id: str
    outward_normal: int
    area: float

    def __post_init__(self) -> None:
        if not self.id or self.outward_normal not in {-1, 1} or not np.isfinite(self.area) or self.area <= 0:
            raise ContractError("invalid_port")

    def charge(self, displacement: ArrayLike) -> Vector:
        return -self.area * self.outward_normal * frozen_array(displacement)

    def current(self, conduction: ArrayLike, displacement_rate: ArrayLike) -> Vector:
        jc, ddot = frozen_array(conduction), frozen_array(displacement_rate)
        if jc.shape != ddot.shape:
            raise ContractError("port_shape_mismatch")
        return -self.area * self.outward_normal * (jc + ddot)


@dataclass(frozen=True)
class AcceptedStep:
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
        for name in ("storage_delta", "displacement_delta", "derivative", "input_rate"):
            object.__setattr__(self, name, frozen_array(getattr(self, name)))
        if self.derivative.shape != self.right.y.shape or self.input_rate.shape != self.right.inputs.shape:
            raise ContractError("step_derivative_shape_mismatch")
