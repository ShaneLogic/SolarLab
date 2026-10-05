"""P02 storage/initialization probes through the public numerical contracts.

This is isolated candidate arithmetic, not a production integrator.  The only
time adapters are a bounded, fixed-step conservative BE experiment and the
installed IDA binding.  Independent references and gates belong to the tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from typing import Callable, Protocol

import numpy as np
from numpy.typing import ArrayLike

from scripts.benchmarks.contract_prototype import (
    AREA, ONE, PARTICLE, SECOND, VOLUME, BalanceTerms, ContractError,
    EquationSpec, FaceFlux, FloatArray, Geometry, ImplicitSystem, Layout,
    LinearCoordinates, LinearStorage, Point, StateIncrement, StateView,
    StoragePartials, Support, TermSink, VariableSpec, Vector, frozen_array,
)


class Coordinates(Protocol):
    def point(self, y: ArrayLike, time: float = 0.0,
              inputs: ArrayLike = ()) -> Point: ...

    def advance(self, left: Point, dy: ArrayLike, time: float,
                inputs: ArrayLike = ()) -> tuple[Point, StateIncrement]: ...


def physical_field(point: Point, name: str) -> Vector:
    value = point.state.field(name)
    if not isinstance(value, FloatArray):
        raise ContractError("explicit_precision_adapter_required")
    return value.values


def scalar_layout(variable: str = "storage") -> Layout:
    return Layout(
        (Support("scalar", "global", (1,)),),
        (VariableSpec(variable, "storage-probe", "scalar", (1,), PARTICLE),),
        (EquationSpec("balance", "storage-probe", "scalar", (1,),
                      PARTICLE / SECOND, derivative_support=(variable,)),),
    )


def no_constraints(rate: ArrayLike, Ry: ArrayLike, input_count: int = 0) -> BalanceTerms:
    rate, Ry = frozen_array(rate), frozen_array(Ry)
    n = rate.size
    return BalanceTerms(rate, np.empty(0), Ry, np.zeros((n, input_count)),
                        np.zeros(n), np.empty((0, n)),
                        np.empty((0, input_count)), np.empty(0))


@dataclass(frozen=True)
class DiffusionProbe:
    coordinates: LinearCoordinates
    system: ImplicitSystem
    geometry: Geometry


def diffusion_probe(volumes: ArrayLike, faces: ArrayLike,
                    conductance: ArrayLike) -> DiffusionProbe:
    """Unweighted face rate assembled once; Q_i=V_i*n_i."""
    volumes, conductance = frozen_array(volumes), frozen_array(conductance)
    geometry = Geometry(volumes, np.asarray(faces), np.ones(conductance.size))
    if conductance.shape != geometry.face_measures.shape or np.any(conductance < 0):
        raise ContractError("invalid_conductance")
    layout = Layout(
        (Support("cells", "cell", volumes.shape),
         Support("faces", "face", conductance.shape)),
        (VariableSpec("density", "diffusion", "cells", volumes.shape,
                      PARTICLE / VOLUME, lower=0.0),),
        (EquationSpec("balance", "diffusion", "cells", volumes.shape,
                      PARTICLE / SECOND, derivative_support=("density",)),),
    )
    coordinates = LinearCoordinates(layout)
    laplacian = np.zeros((volumes.size, volumes.size))
    for (i, j), k in zip(geometry.pairs, conductance, strict=True):
        laplacian[i, i] -= k
        laplacian[i, j] += k
        laplacian[j, i] += k
        laplacian[j, j] -= k

    def terms(point: Point) -> BalanceTerms:
        difference = point.state.face_difference("density", geometry.pairs)
        if not isinstance(difference, FloatArray):
            raise ContractError("explicit_precision_adapter_required")
        sink = TermSink(layout, "balance", geometry)
        sink.add(FaceFlux("transport", "faces", -conductance * difference.values,
                          PARTICLE / AREA / SECOND))
        return no_constraints(sink.value(), laplacian, point.inputs.size)

    return DiffusionProbe(coordinates, ImplicitSystem(LinearStorage(np.diag(volumes)), terms),
                          geometry)


def trap_exchange_probe(Nt: float, capture: float, emission: float
                        ) -> tuple[LinearCoordinates, ImplicitSystem]:
    if not np.isfinite([Nt, capture, emission]).all() or Nt <= 0 or min(capture, emission) < 0:
        raise ContractError("invalid_reaction_parameters")
    layout = Layout(
        (Support("scalar", "global", (1,)),),
        (VariableSpec("carrier", "exchange", "scalar", (1,), PARTICLE, lower=0),
         VariableSpec("occupancy", "exchange", "scalar", (1,), ONE, lower=0, upper=1)),
        tuple(EquationSpec(name, "exchange", "scalar", (1,), PARTICLE / SECOND,
                           derivative_support=("carrier", "occupancy"))
              for name in ("carrier_balance", "trap_balance")),
    )

    def terms(point: Point) -> BalanceTerms:
        n = physical_field(point, "carrier")[0]
        f = physical_field(point, "occupancy")[0]
        r = capture * n * (1.0 - f) - emission * Nt * f
        dn, df = capture * (1.0 - f), -capture * n - emission * Nt
        return no_constraints([-r, r], [[-dn, -df], [dn, df]])

    return LinearCoordinates(layout), ImplicitSystem(LinearStorage(np.diag([1.0, Nt])), terms)


def _positive_exp(value: float) -> float:
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        try:
            result = float(np.exp(value))
        except FloatingPointError as exc:
            raise ContractError("unrepresentable_positive_population") from exc
    if result == 0.0 or not np.isfinite(result):
        raise ContractError("unrepresentable_positive_population")
    return result


@dataclass(frozen=True)
class ExponentialStorage:
    """Scalar nonlinear Q with explicit state, input and time dependence.

    The increment evaluates the finite exponential identity from coordinate
    differences.  Float64 Point fields remain rounded values; this is not a
    compensated physical-state representation or a sub-ULP R1 qualification.
    """

    by: float = 2.0
    ba: float = 3.0
    bt: float = 5.0

    @property
    def reference(self) -> str:
        return f"exponential-storage:{self.by!r}:{self.ba!r}:{self.bt!r}"

    def point(self, y: ArrayLike, time: float = 0.0, inputs: ArrayLike = (0.0,)) -> Point:
        y, a = frozen_array(y), frozen_array(inputs)
        if y.shape != (1,) or a.shape != (1,):
            raise ContractError("coordinate_shape_mismatch")
        Q = _positive_exp(fsum((self.by * y[0], self.ba * a[0], self.bt * time)))
        return Point(time, y, a, StateView(scalar_layout(), [("storage", FloatArray([Q]))]),
                     self.reference)

    def value(self, point: Point) -> Vector:
        if point.coordinate_reference != self.reference:
            raise ContractError("storage_coordinate_reference_mismatch")
        return physical_field(point, "storage")

    def advance(self, left: Point, dy: ArrayLike, time: float,
                inputs: ArrayLike = (0.0,)) -> tuple[Point, StateIncrement]:
        right = self.point(left.y + frozen_array(dy), time, inputs)
        dz = fsum((self.by * (right.y[0] - left.y[0]),
                   self.ba * (right.inputs[0] - left.inputs[0]),
                   self.bt * (right.time - left.time)))
        delta = self.value(left) * np.expm1(dz)
        return right, StateIncrement(left.identity, right.identity, {"storage": FloatArray(delta)})

    def delta(self, left: Point, right: Point, increment: StateIncrement) -> Vector:
        increment.validate(left, right)
        self.value(left)
        self.value(right)
        value = increment.field("storage")
        if not isinstance(value, FloatArray):
            raise ContractError("explicit_precision_adapter_required")
        return value.values

    def first(self, point: Point) -> StoragePartials:
        Q = self.value(point)
        return StoragePartials(self.by * Q[:, None], self.ba * Q[:, None], self.bt * Q)

    def rate_jvp(self, point: Point, ydot: Vector, adot: Vector,
                 dy: Vector, da: Vector, dt: float) -> Vector:
        rate = fsum((self.by * ydot[0], self.ba * adot[0], self.bt))
        direction = fsum((self.by * dy[0], self.ba * da[0], self.bt * dt))
        return self.value(point) * rate * direction

    def system(self) -> ImplicitSystem:
        return ImplicitSystem(self, lambda point: no_constraints([0.0], [[0.0]], 1))


@dataclass(frozen=True)
class CarrierPotentialStorage:
    """n=exp(q+phi+alpha*a+beta*t), g=phi+n-a-gamma*t."""

    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0

    @property
    def reference(self) -> str:
        return f"carrier-potential:{self.alpha!r}:{self.beta!r}:{self.gamma!r}"

    def point(self, y: ArrayLike, time: float = 0.0, inputs: ArrayLike = (0.0,)) -> Point:
        y, a = frozen_array(y), frozen_array(inputs)
        if y.shape != (2,) or a.shape != (1,):
            raise ContractError("coordinate_shape_mismatch")
        n = _positive_exp(fsum((y[0], y[1], self.alpha * a[0], self.beta * time)))
        layout = Layout(
            (Support("scalar", "global", (1,)),),
            (VariableSpec("carrier", "ic-probe", "scalar", (1,), PARTICLE),
             VariableSpec("potential", "ic-probe", "scalar", (1,), ONE, role="constraint")),
            (EquationSpec("balance", "ic-probe", "scalar", (1,), PARTICLE / SECOND,
                          derivative_support=("carrier", "potential")),
             EquationSpec("constraint", "ic-probe", "scalar", (1,), ONE, role="constraint",
                          derivative_support=("carrier", "potential"))),
        )
        return Point(time, y, a, StateView(layout, [("carrier", FloatArray([n])),
                                                   ("potential", FloatArray([y[1]]))]), self.reference)

    def value(self, point: Point) -> Vector:
        if point.coordinate_reference != self.reference:
            raise ContractError("storage_coordinate_reference_mismatch")
        return physical_field(point, "carrier")

    def advance(self, left: Point, dy: ArrayLike, time: float,
                inputs: ArrayLike = (0.0,)) -> tuple[Point, StateIncrement]:
        right = self.point(left.y + frozen_array(dy), time, inputs)
        dz = fsum((*list(right.y - left.y), self.alpha * (right.inputs[0] - left.inputs[0]),
                   self.beta * (right.time - left.time)))
        inc = StateIncrement(left.identity, right.identity, {
            "carrier": FloatArray(self.value(left) * np.expm1(dz)),
            "potential": FloatArray([right.y[1] - left.y[1]]),
        })
        return right, inc

    def delta(self, left: Point, right: Point, increment: StateIncrement) -> Vector:
        increment.validate(left, right)
        self.value(left)
        self.value(right)
        value = increment.field("carrier")
        if not isinstance(value, FloatArray):
            raise ContractError("explicit_precision_adapter_required")
        return value.values

    def first(self, point: Point) -> StoragePartials:
        n = self.value(point)
        return StoragePartials(np.column_stack((n, n)), self.alpha * n[:, None], self.beta * n)

    def rate_jvp(self, point: Point, ydot: Vector, adot: Vector,
                 dy: Vector, da: Vector, dt: float) -> Vector:
        zdot = fsum((*list(ydot), self.alpha * adot[0], self.beta))
        dz = fsum((*list(dy), self.alpha * da[0], self.beta * dt))
        return self.value(point) * zdot * dz

    def terms(self, point: Point) -> BalanceTerms:
        n = self.value(point)[0]
        return BalanceTerms(np.zeros(1), np.array([point.y[1] + n - point.inputs[0]
                                                   - self.gamma * point.time]),
                            np.zeros((1, 2)), np.zeros((1, 1)), np.zeros(1),
                            np.array([[n, n + 1.0]]), np.array([[self.alpha * n - 1.0]]),
                            np.array([self.beta * n - self.gamma]))

    def system(self) -> ImplicitSystem:
        return ImplicitSystem(self, self.terms)


@dataclass(frozen=True)
class SharedCapacityStorage:
    """One shared vacancy, explicit inactive species, C=a+capacity_rate*t.

    Inactive populations retain exact zero fields and no logarithmic coordinate.
    A solver must compile the matching reduced equation set before integrating
    that topology.  This prototype measures its storage and active rank only.
    """

    active: tuple[bool, ...] = (True, True)
    capacity_rate: float = 0.0

    @property
    def reference(self) -> str:
        return f"shared-capacity:{self.active!r}:{self.capacity_rate!r}"

    def capacity(self, point: Point) -> float:
        if point.coordinate_reference != self.reference or point.inputs.shape != (1,):
            raise ContractError("storage_coordinate_reference_mismatch")
        C = point.inputs[0] + self.capacity_rate * point.time
        if not np.isfinite(C) or C <= 0:
            raise ContractError("unsupported_capacity")
        return float(C)

    def _theta(self, q: Vector) -> Vector:
        if q.shape != (sum(self.active),):
            raise ContractError("active_coordinate_shape_mismatch")
        theta = np.zeros(len(self.active))
        if q.size:
            shift = max(0.0, float(np.max(q)))
            weights = np.exp(q - shift)
            vacancy = float(np.exp(-shift))
            if vacancy == 0.0 or np.any(weights == 0.0):
                raise ContractError("unsupported_singular_capacity_boundary")
            theta[np.asarray(self.active)] = weights / fsum((*weights, vacancy))
        return theta

    def point(self, y: ArrayLike, time: float = 0.0, inputs: ArrayLike = (1.0,)) -> Point:
        y, inputs = frozen_array(y), frozen_array(inputs)
        if inputs.shape != (1,):
            raise ContractError("input_shape_mismatch")
        C = inputs[0] + self.capacity_rate * time
        if not np.isfinite(C) or C <= 0:
            raise ContractError("unsupported_capacity")
        theta = self._theta(y)
        layout = Layout(
            (Support("sites", "global", (len(self.active),)),),
            (VariableSpec("populations", "shared-sites", "sites", (len(self.active),),
                          PARTICLE, lower=0),),
            (EquationSpec("balance", "shared-sites", "sites", (len(self.active),),
                          PARTICLE / SECOND, derivative_support=("populations",)),),
        )
        return Point(time, y, inputs, StateView(layout, [("populations", FloatArray(C * theta))]),
                     self.reference)

    def from_occupancies(self, theta: ArrayLike, time: float = 0.0,
                         inputs: ArrayLike = (1.0,)) -> Point:
        theta = frozen_array(theta)
        active = np.asarray(self.active)
        if theta.shape != active.shape or np.any(theta < 0) or fsum(theta) > 1.0:
            raise ContractError("joint_capacity_infeasible")
        if np.any(theta[~active] != 0.0) or np.any(theta[active] == 0.0):
            raise ContractError("inactive_identity_mismatch")
        vacancy = 1.0 - fsum(theta)
        if vacancy <= 0.0:
            raise ContractError("unsupported_singular_capacity_boundary")
        return self.point(np.log(theta[active] / vacancy), time, inputs)

    def value(self, point: Point) -> Vector:
        self.capacity(point)
        return physical_field(point, "populations")

    def theta(self, point: Point) -> Vector:
        return self.value(point) / self.capacity(point)

    def theta_jacobian(self, point: Point) -> Vector:
        theta = self.theta(point)
        full = np.diag(theta) - np.outer(theta, theta)
        return full[:, np.asarray(self.active)]

    def advance(self, left: Point, dy: ArrayLike, time: float,
                inputs: ArrayLike = (1.0,)) -> tuple[Point, StateIncrement]:
        right = self.point(left.y + frozen_array(dy), time, inputs)
        dq = right.y - left.y
        theta_left, theta_right = self.theta(left), self.theta(right)
        active = np.asarray(self.active)
        dtheta = np.zeros(len(self.active))
        if dq.size and np.max(np.abs(dq)) <= 0.5:
            em1 = np.expm1(dq)
            denominator_change = fsum(theta_left[active] * em1)
            dtheta[active] = theta_left[active] * (em1 - denominator_change) / (1 + denominator_change)
        elif dq.size:
            dtheta = theta_right - theta_left
        dC = fsum((right.inputs[0] - left.inputs[0],
                   self.capacity_rate * (right.time - left.time)))
        dQ = self.capacity(left) * dtheta + dC * theta_right
        return right, StateIncrement(left.identity, right.identity, {"populations": FloatArray(dQ)})

    def delta(self, left: Point, right: Point, increment: StateIncrement) -> Vector:
        increment.validate(left, right)
        self.value(left)
        self.value(right)
        value = increment.field("populations")
        if not isinstance(value, FloatArray):
            raise ContractError("explicit_precision_adapter_required")
        return value.values

    def first(self, point: Point) -> StoragePartials:
        theta = self.theta(point)
        return StoragePartials(self.capacity(point) * self.theta_jacobian(point),
                               theta[:, None], self.capacity_rate * theta)

    def rate_jvp(self, point: Point, ydot: Vector, adot: Vector,
                 dy: Vector, da: Vector, dt: float) -> Vector:
        theta, jac = self.theta(point), self.theta_jacobian(point)
        dtheta = jac @ dy
        djac = (np.diag(dtheta) - np.outer(dtheta, theta)
                - np.outer(theta, dtheta))[:, np.asarray(self.active)]
        dC = da[0] + self.capacity_rate * dt
        Cdot = adot[0] + self.capacity_rate
        return dC * (jac @ ydot) + self.capacity(point) * (djac @ ydot) + Cdot * dtheta


@dataclass(frozen=True)
class RankEvidence:
    shape: tuple[int, int]
    rank: int
    threshold: float
    singular_values: Vector


def rank_evidence(matrix: ArrayLike) -> RankEvidence:
    matrix = frozen_array(matrix)
    if matrix.ndim != 2:
        raise ContractError("invalid_rank_matrix")
    singular = np.linalg.svd(matrix, compute_uv=False)
    threshold = 64 * np.finfo(float).eps * max(matrix.shape, default=0) * (
        float(singular[0]) if singular.size else 0.0)
    return RankEvidence(matrix.shape, int(np.count_nonzero(singular > threshold)),
                        threshold, frozen_array(singular))


def rank_aware_solve(matrix: ArrayLike, rhs: ArrayLike, *, gauge_nullspace: bool = False
                     ) -> tuple[Vector, RankEvidence]:
    matrix, rhs = frozen_array(matrix), frozen_array(rhs)
    evidence = rank_evidence(matrix)
    if rhs.shape != (matrix.shape[0],):
        raise ContractError("linear_rhs_shape_mismatch")
    if evidence.rank < min(matrix.shape) or matrix.shape[0] != matrix.shape[1]:
        augmented = rank_evidence(np.column_stack((matrix, rhs)))
        if augmented.rank > evidence.rank:
            raise ContractError("incompatible_fixed_physical_quantities")
        raise ContractError("missing_gauge" if gauge_nullspace else "dependent_constraints")
    return frozen_array(np.linalg.solve(matrix, rhs)), evidence


@dataclass(frozen=True)
class InitialStateResult:
    point: Point
    derivative: Vector
    input_rate: Vector
    history: tuple[Point, ...]
    storage_error: Vector
    residual: Vector
    tangent: Vector
    rank: RankEvidence
    iterations: int


def consistent_initial_state(system: ImplicitSystem, coordinates: Coordinates, seed: Point,
                             preserved_storage: ArrayLike, input_rate: ArrayLike, *,
                             history: tuple[Point, ...] = (), tolerance: float = 1e-14,
                             max_iterations: int = 16, gauge_nullspace: bool = False
                             ) -> InitialStateResult:
    """Solve [Q-Qfixed,g], then [Qy;gy]ydot with Qa/Qt and ga/gt.

    Physical storage rows and coordinate columns have independent roles.  The
    caller declares which storage/port quantities are to be fixed and retains
    immutable pre-event history.  No state is clipped or normalized afterward.
    """
    target, adot = frozen_array(preserved_storage), frozen_array(input_rate)
    if adot.shape != seed.inputs.shape:
        raise ContractError("input_rate_shape_mismatch")
    _, seed_terms = system.evaluate(seed)
    if target.shape != (seed_terms.rate.size,):
        raise ContractError("preserved_storage_shape_mismatch")
    point = seed
    for iteration in range(max_iterations + 1):
        first, terms = system.evaluate(point)
        error = np.concatenate((system.storage.value(point) - target, terms.algebraic))
        matrix = np.vstack((first.y, terms.gy))
        correction, rank = rank_aware_solve(matrix, -error, gauge_nullspace=gauge_nullspace)
        if np.max(np.abs(error), initial=0.0) <= tolerance:
            break
        if iteration == max_iterations:
            raise ContractError("initial_state_iteration_budget_exhausted")
        # IC is not a trajectory step; its original history stays unchanged.
        point = coordinates.point(point.y + correction, point.time, point.inputs)
    rhs = np.concatenate((terms.rate - first.inputs @ adot - first.time,
                          -terms.ga @ adot - terms.gt))
    derivative, rank = rank_aware_solve(matrix, rhs, gauge_nullspace=gauge_nullspace)
    residual = system.residual(point, derivative, adot)
    tangent = terms.gy @ derivative + terms.ga @ adot + terms.gt
    return InitialStateResult(point, derivative, adot, history,
                              frozen_array(system.storage.value(point) - target),
                              frozen_array(residual), frozen_array(tangent), rank, iteration)


@dataclass(frozen=True)
class ConservativeStep:
    left: Point
    right: Point
    increment: StateIncrement
    residual: Vector
    newton_iterations: int
    line_search_evaluations: int


def conservative_be_step(system: ImplicitSystem, coordinates: Coordinates, left: Point,
                         h: float, *, inputs: ArrayLike | None = None, tolerance: float = 2e-15,
                         feasible: Callable[[Point], bool] = lambda point: True,
                         max_iterations: int = 12, max_line_search: int = 16
                         ) -> ConservativeStep:
    """One declared BE step, with finite delta_Q; never adapt h or method."""
    if not np.isfinite(h) or h <= 0:
        raise ContractError("nonpositive_step")
    if inputs is None:
        inputs = left.inputs
    right, increment = coordinates.advance(left, np.zeros_like(left.y), left.time + h, inputs)
    searches = 0
    for iteration in range(max_iterations + 1):
        if not feasible(right):
            raise ContractError("infeasible_be_candidate")
        residual = system.conservative_residual(left, right, increment)
        norm = float(np.max(np.abs(residual), initial=0))
        if norm <= tolerance:
            return ConservativeStep(left, right, increment, frozen_array(residual), iteration, searches)
        if iteration == max_iterations:
            raise ContractError("be_iteration_budget_exhausted")
        step, _ = rank_aware_solve(system.conservative_jacobian(left, right), -residual)
        for slot in range(max_line_search):
            searches += 1
            trial, trial_inc = coordinates.advance(
                left, right.y - left.y + 2.0 ** (-slot) * step, left.time + h, inputs)
            if feasible(trial):
                trial_r = system.conservative_residual(left, trial, trial_inc)
                if np.max(np.abs(trial_r), initial=0) < norm:
                    right, increment = trial, trial_inc
                    break
        else:
            raise ContractError("be_line_search_budget_exhausted")
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class NativeObservation:
    point: Point
    derivative: Vector
    residual: Vector
    status: int


@dataclass(frozen=True)
class IDAProbeResult:
    observations: tuple[NativeObservation, ...]
    jacobian_calls: int
    residual_calls: int
    completed: bool
    failure_reason: str | None
    options: dict


def ida_diffusion_probe(probe: DiffusionProbe, initial: ArrayLike, end_time: float, *,
                        max_step: float, max_order: int, rtol: float, atol: float,
                        accepted_step_budget: int = 100000) -> IDAProbeResult:
    """Use actual scikit-SUNDAE sparse/onestep capabilities, no private shim.

    Returned records are native solver observations awaiting independent physical
    and trajectory gates, not automatically qualified AcceptedStep objects.
    Failed/invalid native payloads are never published as an accepted endpoint.
    """
    import sksundae as sun
    from scipy.sparse import csc_matrix

    system, coordinates = probe.system, probe.coordinates
    start = coordinates.point(initial)
    first, terms = system.evaluate(start)
    yp, _ = rank_aware_solve(first.y, terms.rate)
    initial_residual = system.residual(start, yp, np.empty(0))
    if np.max(np.abs(initial_residual), initial=0) > 1e-14:
        raise ContractError("inconsistent_ida_initial_state")
    observations = [NativeObservation(start, yp, frozen_array(initial_residual), 0)]
    sparsity = csc_matrix((first.y != 0) | (terms.Ry != 0))
    rows, columns = sparsity.indices, np.repeat(np.arange(start.y.size), np.diff(sparsity.indptr))
    counts = {"jac": 0, "res": 0}

    def residual(t, y, ydot, output):
        counts["res"] += 1
        output[:] = system.residual(coordinates.point(y, t), ydot, np.empty(0))

    def jacobian(t, y, ydot, residual_value, cj, values):
        counts["jac"] += 1
        matrix = system.linearize(coordinates.point(y, t), ydot, np.empty(0)).ida_matrix(cj)
        values[:] = matrix[rows, columns]

    options = {"linsolver": "sparse", "nthreads": 1, "max_step": max_step,
               "max_order": max_order, "rtol": rtol, "atol": [atol] * start.y.size}
    solver = sun.ida.IDA(residual, jacfn=jacobian, sparsity=sparsity, **options)
    result = solver.init_step(start.time, start.y.copy(), yp.copy())
    failure = None
    if not result.success:
        failure = "ida_initialization_failed"
    else:
        for _ in range(accepted_step_budget):
            # The actual wrapper clears tstop after each call.
            result = solver.step(end_time, method="onestep", tstop=end_time)
            if not result.success:
                failure = "ida_native_step_failed"
                break
            t = float(result.t)
            if (not np.isfinite(t) or t <= observations[-1].point.time or t > end_time
                    or not np.isfinite(result.y).all() or not np.isfinite(result.yp).all()):
                failure = "invalid_native_payload"
                break
            point = coordinates.point(result.y, t)
            if np.any(point.y < 0):
                failure = "native_physical_population_negative"
                break
            derivative = frozen_array(result.yp)
            observed_residual = system.residual(point, derivative, np.empty(0))
            observations.append(NativeObservation(point, derivative,
                                                  frozen_array(observed_residual), int(result.status)))
            if t == end_time:
                break
        else:
            failure = "accepted_step_budget_exhausted"
    return IDAProbeResult(tuple(observations), counts["jac"], counts["res"],
                          failure is None and observations[-1].point.time == end_time,
                          failure, options)
