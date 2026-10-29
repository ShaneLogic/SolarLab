"""Independent, fixed-grid Radau IIA(3) in physical control-volume inventories.

Only static assembled F/J/S/S_u are shared with other time routes. Each
stage and output solves [S_D(u)-z, F_A(t,u)]=0; no coordinate interpolation,
BDF history, storage secants, time-step selection or accounting is reused.
The cubic inventory polynomial is retained verbatim, including the defect
of an inexact stage solve. A successful numerical trajectory is unqualified.

``linear_tube_bound`` is a small exact-rational certificate for a DECLARED
constant affine reduced ODE, not an enclosure of nonlinear device physics.
It bounds the exact polynomial encoded by binary64 coefficients, its jumps,
initial uncertainty and output-evaluation roundoff. Nonlinear E/I physics,
Poisson reconstruction and terminal-observable enclosures remain required
before these trajectories can supply C2 admission references.

Integration conditions: optional, unqualified reference infrastructure for
restricted declared E/I inputs and analytic unit fixtures only. Independence
is static-only: no BDF stepping, history, interpolation or accounting is used.
Retained failed attempts stay failures and qualification flags stay false.
Any future nonzero initial uncertainty must be expressed in the selected
weighted log norm; a numerical trajectory alone does not qualify a reference.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
import math
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.equations.assembly.assembler import AssembledSystem
from solarlab.equations.assembly.classical_1d import AssembledDC1DProblem
from solarlab.equations.assembly.ionic_1d import AssembledIonic1DProblem
from solarlab.equations.assembly.layout import SystemLayout
from solarlab.equations.assembly.terms import OperatingInputs
from solarlab.physics.ions import DIFFUSION_ONLY_STERIC_V1

__all__ = [
    "InventoryDAE", "InventoryPoint", "LinearTubeBound", "RadauControls",
    "RadauCounters", "RadauFailure", "RadauStep", "RadauTrajectory",
    "assembled_inventory_dae", "integrate_radau", "linear_tube_bound",
]

Vector: TypeAlias = NDArray[np.float64]
Evaluate: TypeAlias = Callable[[float, Vector], AssembledSystem]
Drive: TypeAlias = Callable[[float], OperatingInputs]


def _frozen(value: ArrayLike) -> Vector:
    result = np.array(value, dtype=float, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError("radau_nonfinite_array")
    result.setflags(write=False)
    return result


_ROOT6 = math.sqrt(6.0)
_C = _frozen([(4 - _ROOT6) / 10, (4 + _ROOT6) / 10, 1.0])
_A = _frozen([
    [(88 - 7 * _ROOT6) / 360, (296 - 169 * _ROOT6) / 1800, (-2 + 3 * _ROOT6) / 225],
    [(296 + 169 * _ROOT6) / 1800, (88 + 7 * _ROOT6) / 360, (-2 - 3 * _ROOT6) / 225],
    [(16 - _ROOT6) / 36, (16 + _ROOT6) / 36, 1 / 9],
])
# Columns are the three Lagrange basis polynomials in ascending powers.
_BASIS = _frozen(np.linalg.solve(np.vander(_C, 3, increasing=True), np.eye(3)))


@dataclass(frozen=True)
class RadauControls:
    """Numerical solve controls only; none is a physical admission gate."""

    scaled_newton_tolerance: float = 1.0e-12
    max_newton_iterations: int = 16
    max_backtracks: int = 12
    max_steps: int = 4096

    def __post_init__(self) -> None:
        if not math.isfinite(self.scaled_newton_tolerance) or self.scaled_newton_tolerance <= 0:
            raise ValueError("radau_invalid_newton_tolerance")
        for count in (self.max_newton_iterations, self.max_backtracks, self.max_steps):
            if type(count) is not int or count <= 0:
                raise ValueError("radau_invalid_iteration_cap")


@dataclass
class RadauCounters:
    assembly_evaluations: int = 0
    reconstructions: int = 0
    reconstruction_updates: int = 0
    stage_updates: int = 0
    linear_solves: int = 0
    output_reconstructions: int = 0


class RadauFailure(RuntimeError):
    """Retained failed trial plus all completed steps; no automatic retry."""

    def __init__(self, code: str, time_s: float, trial: ArrayLike):
        super().__init__(code)
        self.code = code
        self.time_s = time_s
        self.trial = _frozen(trial)
        self.completed_steps: tuple[RadauStep, ...] = ()
        self.failed_step: int | None = None


def _solve(matrix: Vector, rhs: Vector, counters: RadauCounters) -> Vector:
    counters.linear_solves += 1
    result: Vector = np.linalg.solve(matrix, rhs)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("radau_nonfinite_linear_solution")
    return result


@dataclass(frozen=True)
class InventoryPoint:
    time_s: float
    inventory: Vector
    u: Vector
    rhs: Vector
    algebraic_residual: Vector
    inventory_residual: Vector
    scaled_reconstruction_residual: float
    rhs_jacobian: Vector | None = None


@dataclass(frozen=True)
class InventoryDAE:
    """Index-1 map using an explicit partition and dimensional row scales.

    ``differential_rows`` identifies physical storage rows, not coordinates
    whose derivative happens to be nonzero at a single state. All remaining
    rows, including interior Poisson rows, are algebraic. The reduced map
    exists where B=[S_u[D,:]; J[A,:]] is nonsingular. A floating Newton solve
    neither proves nonsingularity on a tube nor certifies its inverse.
    """

    layout: SystemLayout
    evaluate: Evaluate
    differential_rows: tuple[int, ...]
    inventory_scale: Vector
    algebraic_scale: Vector
    inventory_capacity: Vector | None = None
    algebraic_rows: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        rows = self.differential_rows
        if (not rows or any(type(row) is not int for row in rows)
                or tuple(sorted(set(rows))) != rows or rows[0] < 0 or rows[-1] >= self.layout.size):
            raise ValueError("radau_invalid_differential_partition")
        algebraic = tuple(row for row in range(self.layout.size) if row not in rows)
        for value, size in ((self.inventory_scale, len(rows)), (self.algebraic_scale, len(algebraic))):
            array = np.asarray(value)
            if array.shape != (size,) or not np.all(np.isfinite(array)) or np.any(array <= 0):
                raise ValueError("radau_requires_explicit_positive_row_scales")
        object.__setattr__(self, "algebraic_rows", algebraic)
        object.__setattr__(self, "inventory_scale", _frozen(self.inventory_scale))
        object.__setattr__(self, "algebraic_scale", _frozen(self.algebraic_scale))
        if self.inventory_capacity is not None:
            capacity = np.array(self.inventory_capacity, dtype=float, copy=True)
            if capacity.shape != (len(rows),) or np.any(np.isnan(capacity)) or np.any(capacity <= 0):
                raise ValueError("radau_invalid_inventory_capacity")
            capacity.setflags(write=False)
            object.__setattr__(self, "inventory_capacity", capacity)

    def assembled(self, time_s: float, u: Vector, counters: RadauCounters) -> AssembledSystem:
        if not math.isfinite(time_s) or u.shape != (self.layout.size,):
            raise ValueError("radau_invalid_time_or_coordinates")
        counters.assembly_evaluations += 1
        value = self.evaluate(time_s, _frozen(u))
        if not value.admissible:
            raise FloatingPointError("radau_inadmissible_static_state")
        if value.layout.identity != self.layout.identity:
            raise ValueError("radau_layout_changed")
        if value.S is None or value.S_u is None or value.J is None:
            raise ValueError("radau_requires_F_J_S_Su")
        size = self.layout.size
        if (value.F.shape != (size,) or value.S.shape != (size,)
                or value.J.shape != (size, size) or value.S_u.shape != (size, size)):
            raise ValueError("radau_invalid_static_shape")
        if not all(np.all(np.isfinite(x)) for x in (value.F, value.S, value.J.data, value.S_u.data)):
            raise FloatingPointError("radau_nonfinite_static_evaluation")
        if (np.any(value.S[list(self.algebraic_rows)] != 0)
                or np.any(value.S_u[list(self.algebraic_rows), :].data != 0)):
            raise ValueError("radau_algebraic_rows_have_storage")
        return value

    def reconstruct(self, time_s: float, inventory: ArrayLike, guess: ArrayLike,
                    controls: RadauControls, counters: RadauCounters, *,
                    derivatives: bool = False) -> InventoryPoint:
        z, u = _frozen(inventory), np.array(guess, dtype=float, copy=True)
        d, a = list(self.differential_rows), list(self.algebraic_rows)
        if z.shape != (len(d),) or u.shape != (self.layout.size,):
            raise ValueError("radau_invalid_reconstruction_shape")
        if np.any(z <= 0) or (self.inventory_capacity is not None and np.any(z >= self.inventory_capacity)):
            raise FloatingPointError("radau_inventory_outside_physical_domain")
        counters.reconstructions += 1
        scale = np.r_[self.inventory_scale, self.algebraic_scale]

        def evaluate(trial: Vector) -> tuple[Vector, Vector, AssembledSystem]:
            value = self.assembled(time_s, trial, counters)
            assert value.S is not None and value.S_u is not None and value.J is not None
            residual = np.r_[value.S[d] - z, value.F[a]]
            matrix = np.vstack((value.S_u[d, :].toarray(), value.J[a, :].toarray()))
            return residual, matrix, value

        for _ in range(controls.max_newton_iterations + 1):
            residual, matrix, value = evaluate(u)
            norm = float(np.max(np.abs(residual / scale)))
            if norm <= controls.scaled_newton_tolerance:
                jacobian = None
                if derivatives:
                    rhs = np.zeros((self.layout.size, len(d)))
                    rhs[:len(d), :] = np.eye(len(d))
                    try:
                        lift = _solve(matrix / scale[:, None], rhs / scale[:, None], counters)
                    except np.linalg.LinAlgError as error:
                        raise RadauFailure("radau_singular_reconstruction", time_s, u) from error
                    assert value.J is not None
                    jacobian = _frozen(value.J[d, :] @ lift)
                return InventoryPoint(time_s, z, _frozen(u), _frozen(value.F[d]),
                                      _frozen(value.F[a]), _frozen(residual[:len(d)]), norm, jacobian)
            if _ == controls.max_newton_iterations:
                break
            counters.reconstruction_updates += 1
            try:
                delta = _solve(matrix / scale[:, None], -residual / scale, counters)
            except np.linalg.LinAlgError as error:
                raise RadauFailure("radau_singular_reconstruction", time_s, u) from error
            for backtrack in range(controls.max_backtracks):
                trial = u + (0.5 ** backtrack) * delta
                try:
                    trial_residual = evaluate(trial)[0]
                except (FloatingPointError, OverflowError):
                    continue
                if np.max(np.abs(trial_residual / scale)) < norm:
                    u = trial
                    break
            else:
                raise RadauFailure("radau_reconstruction_line_search", time_s, u)
        raise RadauFailure("radau_reconstruction_iteration_cap", time_s, u)


def assembled_inventory_dae(
    problem: AssembledDC1DProblem | AssembledIonic1DProblem, drive: Drive, *,
    inventory_scale: ArrayLike, algebraic_scale: ArrayLike,
) -> InventoryDAE:
    """Structural E/I adapter; exact admission deck/protocol binding is separate.

    Fixed carrier reservoirs and optional confined, mobile, positive finite-
    site ions only. Original ion rows remain differential, including both
    blocking endpoints; no steady inventory-replacement row is imported.
    """
    if not isinstance(problem, (AssembledDC1DProblem, AssembledIonic1DProblem)):
        raise TypeError("radau_requires_public_assembled_E_or_I_problem")
    if isinstance(problem, AssembledIonic1DProblem):
        ion = problem.device
        electronic = ion.electronic_component
        if (ion.ion_species_names != ("positive",)
                or ion.ion_site_contract != "independent_finite_sites"
                or ion.steric_model != DIFFUSION_ONLY_STERIC_V1
                or np.any(ion.positive_ion_diffusivity_faces_m2_s <= 0)
                or ion.ion_node_indices[0] == 0 or ion.ion_node_indices[-1] == ion.node_count - 1):
            raise ValueError("radau_ionic_scope_not_supported")
    else:
        electronic = problem.device
    if any(channel.kind != "fixed_reservoir" for channel in electronic.channels):
        raise ValueError("radau_requires_four_fixed_reservoirs")
    if electronic.field_transport is not None or electronic.additional_recombination is not None:
        raise ValueError("radau_additional_transport_or_reaction_not_supported")
    layout = problem.binding.layout
    mask = layout.constraint_mask()
    mask[layout.rows("potential")] = True
    rows = tuple(int(row) for row in np.flatnonzero(~mask))

    def evaluate(time_s: float, u: Vector) -> AssembledSystem:
        inputs = drive(time_s)
        return problem.at(inputs.voltage_V, inputs.generation_scale).assemble(u, storage=True)

    capacity = np.full(layout.size, np.inf)
    if isinstance(problem, AssembledIonic1DProblem):
        block = layout.blocks[3]
        weights = problem.binding.assembler.mesh.support(block.node_set).geometry.volumes
        capacity[layout.rows(block.id)] = weights * problem.device.positive_ion_site_limit_m3
    return InventoryDAE(layout, evaluate, rows, _frozen(inventory_scale), _frozen(algebraic_scale),
                        capacity[list(rows)])


@dataclass(frozen=True)
class RadauStep:
    left_s: float
    right_s: float
    coefficients: Vector  # [power 0..3, inventory component], theta in [0,1]
    stages: tuple[InventoryPoint, ...]
    stage_residual: Vector
    endpoint: InventoryPoint

    def inventory(self, time_s: float) -> Vector:
        if not math.isfinite(time_s) or not self.left_s <= time_s <= self.right_s:
            raise ValueError("radau_output_outside_step")
        theta = (time_s - self.left_s) / (self.right_s - self.left_s)
        p = self.coefficients
        return _frozen(p[0] + theta * (p[1] + theta * (p[2] + theta * p[3])))

    def inventory_rate(self, time_s: float) -> Vector:
        self.inventory(time_s)  # Same finite, closed-domain check.
        h = self.right_s - self.left_s
        theta = (time_s - self.left_s) / h
        p = self.coefficients
        return _frozen((p[1] + theta * (2 * p[2] + theta * 3 * p[3])) / h)


@dataclass(frozen=True)
class RadauTrajectory:
    model: InventoryDAE
    initial: InventoryPoint
    steps: tuple[RadauStep, ...]
    controls: RadauControls
    counters: RadauCounters
    qualified: bool = field(default=False, init=False)

    def sample(self, time_s: float) -> InventoryPoint:
        """Reconstruct at every output; never interpolate algebraic coordinates."""
        self.counters.output_reconstructions += 1
        if time_s == self.initial.time_s:
            return self.model.reconstruct(time_s, self.initial.inventory, self.initial.u,
                                          self.controls, self.counters)
        for step in self.steps:
            if step.left_s <= time_s <= step.right_s:
                return self.model.reconstruct(time_s, step.inventory(time_s), step.endpoint.u,
                                              self.controls, self.counters)
        raise ValueError("radau_output_outside_trajectory")


def _step(model: InventoryDAE, start: InventoryPoint, right_s: float,
          controls: RadauControls, counters: RadauCounters) -> RadauStep:
    h = right_s - start.time_s
    times = start.time_s + h * _C
    times[-1] = right_s
    z = np.tile(start.inventory, (3, 1))
    if times[0] <= start.time_s or np.any(np.diff(times) <= 0):
        raise RadauFailure("radau_stage_clock_resolution", start.time_s, z)
    size = start.inventory.size
    guesses = [start.u] * 3

    def stages(trial: Vector, *, derivatives: bool) -> tuple[Vector, tuple[InventoryPoint, ...]]:
        points = tuple(model.reconstruct(float(t), row, guess, controls, counters, derivatives=derivatives)
                       for t, row, guess in zip(times, trial, guesses))
        rhs = np.array([point.rhs for point in points])
        return trial - start.inventory - h * (_A @ rhs), points

    for iteration in range(controls.max_newton_iterations + 1):
        residual, points = stages(z, derivatives=True)
        norm = float(np.max(np.abs(residual / model.inventory_scale)))
        if norm <= controls.scaled_newton_tolerance:
            rates = np.array([point.rhs for point in points])
            coefficients = _frozen(np.vstack((start.inventory,
                                    h * (_BASIS @ rates) / np.arange(1, 4)[:, None])))
            end_z = coefficients[0] + (coefficients[1] + (coefficients[2] + coefficients[3]))
            endpoint = model.reconstruct(right_s, end_z, points[-1].u, controls, counters)
            return RadauStep(start.time_s, right_s, coefficients, points, _frozen(residual), endpoint)
        if iteration == controls.max_newton_iterations:
            break
        counters.stage_updates += 1
        guesses = [point.u for point in points]
        matrix = np.zeros((3 * size, 3 * size))
        for i in range(3):
            for j, point in enumerate(points):
                assert point.rhs_jacobian is not None
                matrix[i * size:(i + 1) * size, j * size:(j + 1) * size] = -h * _A[i, j] * point.rhs_jacobian
            matrix[i * size:(i + 1) * size, i * size:(i + 1) * size] += np.eye(size)
        scale = np.tile(model.inventory_scale, 3)
        try:
            delta = _solve(matrix / scale[:, None], -residual.ravel() / scale, counters).reshape(z.shape)
        except np.linalg.LinAlgError as error:
            raise RadauFailure("radau_singular_stage_system", start.time_s, z) from error
        for backtrack in range(controls.max_backtracks):
            trial = z + (0.5 ** backtrack) * delta
            try:
                trial_residual = stages(trial, derivatives=False)[0]
            except (RadauFailure, FloatingPointError, OverflowError):
                continue
            if np.max(np.abs(trial_residual / model.inventory_scale)) < norm:
                z = trial
                break
        else:
            raise RadauFailure("radau_stage_line_search", start.time_s, z)
    raise RadauFailure("radau_stage_iteration_cap", start.time_s, z)


def integrate_radau(model: InventoryDAE, initial_u: ArrayLike, times_s: ArrayLike, *,
                    controls: RadauControls = RadauControls(),
                    initial_inventory: ArrayLike | None = None) -> RadauTrajectory:
    """One continuous-drive segment on an explicit grid; no adaptive retries.

    The caller must split at drive knots and jumps. A jump needs independent
    left/right reconstruction at identical inventories (``initial_inventory``)
    and separate impulse accounting; this driver does not provide that packet
    or admit arbitrary checkpoint continuation. ``initial_u`` is only a
    reconstruction guess when an explicit physical inventory is supplied.
    """
    times = _frozen(times_s)
    if (times.ndim != 1 or times.size < 2 or not np.all(np.isfinite(np.diff(times)))
            or np.any(np.diff(times) <= 0)):
        raise ValueError("radau_requires_increasing_grid")
    if times.size - 1 > controls.max_steps:
        raise ValueError("radau_step_cap")
    counters = RadauCounters()
    raw = model.assembled(float(times[0]), _frozen(initial_u), counters)
    assert raw.S is not None
    z0 = raw.S[list(model.differential_rows)] if initial_inventory is None else _frozen(initial_inventory)
    initial = model.reconstruct(float(times[0]), z0, initial_u, controls, counters)
    start = initial
    completed: list[RadauStep] = []
    for index, right in enumerate(times[1:]):
        try:
            step = _step(model, start, float(right), controls, counters)
        except RadauFailure as failure:
            failure.completed_steps = tuple(completed)
            failure.failed_step = index
            raise
        except (FloatingPointError, np.linalg.LinAlgError) as error:
            wrapped = RadauFailure("radau_static_evaluation_failed", start.time_s, start.u)
            wrapped.completed_steps = tuple(completed)
            wrapped.failed_step = index
            raise wrapped from error
        completed.append(step)
        start = step.endpoint
    return RadauTrajectory(model, initial, tuple(completed), controls, counters)


def _q(value: float) -> Fraction:
    if not math.isfinite(value):
        raise ValueError("radau_bound_nonfinite_input")
    return Fraction.from_float(float(value))


def _upper(value: Fraction) -> float:
    rounded = float(value)
    if not math.isfinite(rounded):
        raise ValueError("radau_bound_overflow")
    if _q(rounded) < value:
        rounded = float(np.nextafter(rounded, np.inf))
    return rounded


def _exp_enclosure(value: Fraction) -> tuple[Fraction, Fraction]:
    """Taylor sum and geometric tail, all rational; bounded tiny-case utility."""
    if abs(value) > 16:
        raise ValueError("radau_bound_exponential_argument_cap")
    if value < 0:
        lower, upper = _exp_enclosure(-value)
        return 1 / upper, 1 / lower
    term = total = Fraction(1)
    for k in range(1, 257):
        term *= value / k
        total += term
        ratio = value / (k + 2)
        if ratio < 1:
            tail = term * value / (k + 1) / (1 - ratio)
            if tail <= Fraction(1, 10**45):
                return total, total + tail
    raise ValueError("radau_bound_exponential_iteration_cap")


@dataclass(frozen=True)
class LinearTubeBound:
    """Exact affine-inventory proof only; not a device/reference qualification."""

    scaled_error_upper: float
    log_norm_upper: float
    defect_upper: tuple[float, ...]
    interval_error_upper: tuple[float, ...]
    output_evaluation_error_upper: tuple[float, ...]
    tube_validated: bool
    within_budget: bool
    status: str
    full_state_reconstruction_bound: None = None
    reference_qualified: bool = field(default=False, init=False)


def linear_tube_bound(
    trajectory: RadauTrajectory, matrix: ArrayLike, exact_initial: ArrayLike, *,
    component_scale: ArrayLike, tube_radius: float, reference_budget: float,
    initial_radius: ArrayLike, forcing: ArrayLike | None = None,
    output_times_s: Sequence[float] = (),
) -> LinearTubeBound:
    """Certify z'=Az+b with constant exact binary64 A,b and initial box.

    On each entire interval Bernstein coefficients enclose the defect of the
    stored cubic. The scaled infinity logarithmic norm is exact rational;
    exponential propagation uses a rigorous Taylor-tail enclosure. Center
    jumps and actual output Horner/clock roundoff are added, never reset.
    The affine ODE is globally defined, so a radius check gives a validated
    trajectory tube without presuming the truth of a nonlinear local box.
    A loose radius or budget returns an unusable certificate. This function
    does not assert that an arbitrary assembled device has affine physics.
    """
    n = trajectory.initial.inventory.size
    A = np.asarray(matrix, dtype=float)
    initial = np.asarray(exact_initial, dtype=float)
    scale = np.asarray(component_scale, dtype=float)
    radius = np.asarray(initial_radius, dtype=float)
    b = np.zeros(n) if forcing is None else np.asarray(forcing, dtype=float)
    if (A.shape != (n, n) or any(x.shape != (n,) for x in (initial, scale, radius, b))
            or not all(np.all(np.isfinite(x)) for x in (A, initial, scale, radius, b))
            or np.any(scale <= 0) or np.any(radius < 0)
            or not math.isfinite(tube_radius) or tube_radius <= 0
            or not math.isfinite(reference_budget) or reference_budget <= 0):
        raise ValueError("radau_bound_invalid_affine_data")
    aq: list[list[Fraction]] = [[_q(v) for v in row] for row in A]
    sq, bq = [_q(v) for v in scale], [_q(v) for v in b]
    mu: Fraction = max(aq[i][i] + sum((abs(aq[i][j]) * sq[j] / sq[i]
                                      for j in range(n) if j != i), Fraction(0)) for i in range(n))
    initial_q = [_q(v) for v in initial]
    error = max(_q(radius[i]) / sq[i] for i in range(n))
    last_center = initial_q
    interval_errors: list[Fraction] = []
    defects: list[Fraction] = []
    previous_time = _q(trajectory.initial.time_s)
    for step in trajectory.steps:
        if _q(step.left_s) != previous_time:
            raise ValueError("radau_bound_discontinuous_clock")
        p = [[_q(v) for v in row] for row in step.coefficients]
        h = _q(step.right_s) - _q(step.left_s)
        error += max(abs(p[0][i] - last_center[i]) / sq[i] for i in range(n))
        # Power coefficients of p'(theta)/h - A p(theta) - b.
        defect = [[(Fraction(k + 1) * p[k + 1][i] / h if k < 3 else Fraction(0))
                   - sum((aq[i][j] * p[k][j] for j in range(n)), Fraction(0))
                   - (bq[i] if k == 0 else Fraction(0)) for i in range(n)] for k in range(4)]
        bernstein = [[sum((Fraction(math.comb(j, k), math.comb(3, k)) * defect[k][i]
                          for k in range(j + 1)), Fraction(0)) for i in range(n)] for j in range(4)]
        delta = max(abs(row[i]) / sq[i] for row in bernstein for i in range(n))
        lower, upper = _exp_enclosure(mu * h)
        integral = h if mu == 0 else (upper - 1) / mu if mu > 0 else (1 - lower) / (-mu)
        after = upper * error + integral * delta
        interval_errors.append(max(error, after))
        defects.append(delta)
        error = after
        last_center = [sum((p[k][i] for k in range(4)), Fraction(0)) for i in range(n)]
        previous_time = _q(step.right_s)
    if not interval_errors:
        raise ValueError("radau_bound_empty_trajectory")
    evaluations: list[Fraction] = []
    for time_s in output_times_s:
        for step in trajectory.steps:
            if step.left_s <= time_s <= step.right_s:
                theta = (_q(time_s) - _q(step.left_s)) / (_q(step.right_s) - _q(step.left_s))
                exact = [sum((_q(step.coefficients[k, i]) * theta**k for k in range(4)), Fraction(0))
                         for i in range(n)]
                floating = step.inventory(time_s)
                evaluations.append(max(abs(exact[i] - _q(floating[i])) / sq[i] for i in range(n)))
                break
        else:
            raise ValueError("radau_bound_output_outside_trajectory")
    tube_error = max(interval_errors)
    total_error = tube_error + max(evaluations, default=Fraction(0))
    tube_valid = tube_error <= _q(tube_radius)
    budget_valid = tube_valid and total_error <= _q(reference_budget)
    status = "bounded_affine_inventories" if budget_valid else "reference_budget_exceeded" if tube_valid else "tube_radius_exceeded"
    return LinearTubeBound(_upper(total_error), _upper(mu), tuple(map(_upper, defects)),
                           tuple(map(_upper, interval_errors)), tuple(map(_upper, evaluations)),
                           tube_valid, budget_valid, status)
