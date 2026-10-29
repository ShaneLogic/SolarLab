"""Bracketed direct operating points of the actual new-core DC equations.

An input bracket comes from accepted SteadyJVResult solutions. Every refined
point is another solve of the same DC equations and physical controls. The
reported voltage estimate belongs to an accepted solved state; interpolation
of current is never used to manufacture an operating point.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from solarlab.equations.dc_1d import CompiledDevice1D, DC1DProblem
from solarlab.experiments.jv.steady import SteadyJVResult
from solarlab.solvers.nonlinear.newton import NewtonResult
from solarlab.solvers.nonlinear.problem import state_digest
from solarlab.solvers.nonlinear.production import (
    production_convergence_norm, production_residual_scaling,
)
from solarlab.solvers.nonlinear.steady_1d import (
    SteadyOptions,
    SteadyState1D,
    SteadyStateError,
    physical_gates,
    solve_steady_state,
)

__all__ = [
    "DirectVocError",
    "DirectVocResult",
    "VocBracketError",
    "VocFailure",
    "solve_direct_voc",
]


class VocBracketError(ValueError):
    """The supplied solutions do not identify one actual DC current root."""


@dataclass(frozen=True)
class VocFailure:
    """A failed DC step, with its accepted seed and original diagnostics."""

    voltage_V: float
    seed_voltage_V: float
    seed_digest: str
    subdivisions: int
    exception_type: str
    message: str
    newton_result: NewtonResult | None


class DirectVocError(RuntimeError):
    """A bounded refinement failed; accepted states and failures are retained."""

    def __init__(
        self,
        message: str,
        lower: SteadyState1D,
        upper: SteadyState1D,
        solutions: tuple[SteadyState1D, ...],
        failed_attempts: tuple[VocFailure, ...],
        solve_calls: int,
    ) -> None:
        self.lower = lower
        self.upper = upper
        self.solutions = solutions
        self.failed_attempts = failed_attempts
        self.solve_calls = solve_calls
        super().__init__(message)

    @property
    def interval_V(self) -> tuple[float, float]:
        return self.lower.voltage_V, self.upper.voltage_V


@dataclass(frozen=True)
class DirectVocResult:
    """An accepted solved estimate inside a sign-bracketed voltage interval."""

    solution: SteadyState1D
    lower: SteadyState1D
    upper: SteadyState1D
    initial_endpoints: tuple[SteadyState1D, SteadyState1D]
    solutions: tuple[SteadyState1D, ...]
    failed_attempts: tuple[VocFailure, ...]
    solve_calls: int
    voltage_tolerance_V: float
    voltage_polarity: float
    compiled_identity: str
    generation_scale: float
    solver_controls: dict[str, object]

    @property
    def voltage_V(self) -> float:
        return self.solution.voltage_V

    @property
    def interval_V(self) -> tuple[float, float]:
        return self.lower.voltage_V, self.upper.voltage_V

    @property
    def interval_width_V(self) -> float:
        return self.upper.voltage_V - self.lower.voltage_V

    @property
    def output_current_A_m2(self) -> float:
        return _output_current(self.solution, self.voltage_polarity)


def _output_current(point: SteadyState1D, polarity: float) -> float:
    return float(-polarity * point.current_density_A_m2[0])


def _unique_root_indices(current: np.ndarray) -> tuple[int, int]:
    """Select one exact zero or adjacent strict sign change, without products."""
    zeros = np.flatnonzero(current == 0.0)
    signs = np.sign(current)
    changes = np.flatnonzero(signs[:-1] * signs[1:] < 0.0)
    if zeros.size == 1 and changes.size == 0:
        node = int(zeros[0])
        return node, node
    if zeros.size > 1 or zeros.size + changes.size > 1:
        raise VocBracketError("multiple or ambiguous DC current roots in the bracket")
    if changes.size == 1 and zeros.size == 0:
        node = int(changes[0])
        return node, node + 1
    raise VocBracketError("no actual solved sign-changing DC current bracket")


def _verify_point(
    compiled: CompiledDevice1D,
    point: SteadyState1D,
    generation_scale: float,
    controls: SteadyOptions,
) -> None:
    """Validate saved acceptance against the same actual physical equations."""
    if not isinstance(point, SteadyState1D) or not point.converged:
        raise VocBracketError("every input point must be an accepted steady solve")
    if (
        point.state.device_sha256 != compiled.identity
        or point.state.voltage_V != point.voltage_V
        or point.generation_scale != generation_scale
        or not np.array_equal(point.newton.state, point.state.u)
    ):
        raise VocBracketError("steady point identity, voltage, or state is inconsistent")
    required = {"poisson_balance", "continuity_current_bound", "all_face_current_spread"}
    if not required <= {gate.name for gate in point.physical_gates} or not all(
        gate.passed for gate in point.physical_gates
    ):
        raise VocBracketError("steady point lacks passed production physical gates")
    state = compiled.state(point.state.u, voltage_V=point.voltage_V)
    for name in ("potential_V", "electron_density_m3", "hole_density_m3"):
        if not np.array_equal(getattr(state, name), getattr(point.state, name)):
            raise VocBracketError("saved fields do not match the actual state coordinates")
    problem = DC1DProblem(compiled, point.voltage_V, generation_scale)
    current = problem.current_density(point.state.u)
    contact = problem.contact_current_density(point.state.u)
    if not np.array_equal(current, point.current_density_A_m2) or not np.array_equal(
        contact, point.contact_current_density_A_m2
    ):
        raise VocBracketError("saved current is not the current of the solved DC state")
    scaling = production_residual_scaling(
        compiled,
        poisson_tolerance=controls.poisson_tolerance,
        continuity_tolerance_A_m2=controls.continuity_tolerance_A_m2,
        contact_log_tolerance=controls.contact_log_tolerance,
    )
    residual = scaling.scaled(problem.residual(point.state.u))
    norm = production_convergence_norm(compiled, convergence_policy=controls.convergence_policy)
    if not np.all(np.isfinite(residual)) or (
        float(np.max(np.abs(residual))) if norm is None else norm(residual)
    ) > 1.0:
        raise VocBracketError("saved point fails the production residual controls")
    if np.ptp(np.r_[current, contact]) > controls.current_spread_tolerance_A_m2:
        raise VocBracketError("saved point fails the all-face current gate")
    if any(not gate(point.state.u).passed for gate in physical_gates(
        problem, controls, equilibrium=point.voltage_V == 0.0 and generation_scale == 0.0,
    )):
        raise VocBracketError("saved point fails a recomputed production physical gate")


def solve_direct_voc(
    compiled: CompiledDevice1D,
    bracket: SteadyJVResult,
    *,
    voltage_tolerance_V: float = 1.0e-6,
    max_solve_calls: int = 64,
    max_continuation_subdivisions: int = 16,
    options: SteadyOptions | None = None,
) -> DirectVocResult:
    """Refine one actual DC current bracket to a positive voltage-width control.

    A failed step may use bounded finer continuation from the nearest accepted
    same-device state. All failed attempts remain diagnostic evidence; their
    states are never seeds. The final estimate is whichever accepted endpoint
    has smaller absolute output current, with its enclosing interval reported.
    Voltage width is a numerical stopping control, not scientific qualification.
    """
    if not math.isfinite(voltage_tolerance_V) or voltage_tolerance_V <= 0.0:
        raise ValueError("voltage_tolerance_V must be finite and positive")
    for name, value, minimum in (
        ("max_solve_calls", max_solve_calls, 1),
        ("max_continuation_subdivisions", max_continuation_subdivisions, 2),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not isinstance(bracket, SteadyJVResult):
        raise VocBracketError("an actual SteadyJVResult bracket is required")
    if (
        bracket.compiled_identity != compiled.identity
        or bracket.voltage_polarity != compiled.voltage_polarity
        or not np.array_equal(bracket.nodes_m, compiled.nodes_m)
    ):
        raise VocBracketError("the bracket must belong to the same compiled device")
    controls = SteadyOptions(**bracket.solver_controls)
    if options is not None and options != controls:
        raise VocBracketError("direct refinement must retain the bracket's solver controls")
    voltage = np.asarray(bracket.voltage_V, dtype=float)
    output = np.asarray(bracket.output_current_A_m2, dtype=float)
    current = np.asarray(bracket.current_density_A_m2, dtype=float)
    count = voltage.size
    if (
        voltage.shape != (count,) or count < 2
        or output.shape != (count,)
        or current.shape != (count, compiled.node_count - 1)
        or len(bracket.solutions) != count
        or not np.all(np.isfinite(voltage))
        or not np.all(np.isfinite(output))
        or np.unique(voltage).size != count
    ):
        raise VocBracketError("bracket arrays and accepted solutions are inconsistent")
    accepted: dict[float, SteadyState1D] = {}
    for index, point in enumerate(bracket.solutions):
        _verify_point(compiled, point, bracket.generation_scale, controls)
        if point.voltage_V != float(voltage[index]) or not np.array_equal(
            current[index], point.current_density_A_m2
        ) or output[index] != _output_current(point, compiled.voltage_polarity):
            raise VocBracketError("bracket currents/voltages are not actual solved outputs")
        accepted[point.voltage_V] = point
    ordered = sorted(accepted.values(), key=lambda point: point.voltage_V)
    left, right = _unique_root_indices(
        np.array([_output_current(point, compiled.voltage_polarity) for point in ordered])
    )
    lower, upper = ordered[left], ordered[right]
    initial = lower, upper
    failures: list[VocFailure] = []
    calls = 0

    def saved() -> tuple[SteadyState1D, ...]:
        return tuple(sorted(accepted.values(), key=lambda point: point.voltage_V))

    def fail(message: str) -> DirectVocError:
        return DirectVocError(message, lower, upper, saved(), tuple(failures), calls)

    def attempt(target: float, subdivisions: int) -> SteadyState1D | None:
        nonlocal calls
        if target in accepted:
            return accepted[target]
        if calls >= max_solve_calls:
            raise fail("direct V_oc exhausted the DC solve-call budget")
        seed = min(
            accepted.values(),
            key=lambda point: (abs(point.voltage_V - target), point.voltage_V),
        )
        calls += 1
        try:
            point = solve_steady_state(
                compiled, initial_state=seed.state, voltage_V=target,
                generation_scale=bracket.generation_scale, options=controls,
            )
        except (SteadyStateError, ArithmeticError) as error:
            failures.append(VocFailure(
                target, seed.voltage_V, state_digest(seed.state.u), subdivisions,
                type(error).__name__, str(error),
                error.result if isinstance(error, SteadyStateError) else None,
            ))
            return None
        except Exception as error:
            failures.append(VocFailure(
                target, seed.voltage_V, state_digest(seed.state.u), subdivisions,
                type(error).__name__, str(error), None,
            ))
            raise fail("DC step raised a non-recoverable error") from error
        try:
            _verify_point(compiled, point, bracket.generation_scale, controls)
            if point.voltage_V != target:
                raise VocBracketError("DC solve returned a different requested voltage")
        except Exception as error:
            failures.append(VocFailure(
                target, seed.voltage_V, state_digest(seed.state.u), subdivisions,
                type(error).__name__, str(error),
                getattr(point, "newton", None),
            ))
            raise fail("DC step did not return an accepted physical operating point") from error
        accepted[target] = point
        return point

    def reach(target: float) -> SteadyState1D:
        point = attempt(target, 0)
        if point is not None:
            return point
        subdivisions = 2
        while subdivisions <= max_continuation_subdivisions:
            start = min(
                accepted.values(),
                key=lambda point: (abs(point.voltage_V - target), point.voltage_V),
            )
            success = True
            for value in np.linspace(start.voltage_V, target, subdivisions + 1)[1:]:
                point = attempt(float(value), subdivisions)
                if point is None:
                    success = False
                    break
                if _output_current(point, compiled.voltage_polarity) == 0.0:
                    return point
            if success:
                assert point is not None
                return point
            subdivisions *= 2
        raise fail("bounded finer same-equation continuation could not reach the midpoint")

    while upper.voltage_V - lower.voltage_V > voltage_tolerance_V:
        midpoint = lower.voltage_V + 0.5 * (upper.voltage_V - lower.voltage_V)
        if midpoint == lower.voltage_V or midpoint == upper.voltage_V:
            raise fail("binary64 cannot further subdivide the accepted voltage bracket")
        reach(midpoint)
        candidates = [
            point for point in saved()
            if lower.voltage_V <= point.voltage_V <= upper.voltage_V
        ]
        try:
            left, right = _unique_root_indices(np.array([
                _output_current(point, compiled.voltage_polarity) for point in candidates
            ]))
        except VocBracketError as error:
            raise fail("new solved points make the DC root bracket ambiguous") from error
        lower, upper = candidates[left], candidates[right]
    estimate = min(
        (lower, upper),
        key=lambda point: (abs(_output_current(point, compiled.voltage_polarity)), point.voltage_V),
    )
    return DirectVocResult(
        estimate, lower, upper, initial, saved(), tuple(failures), calls,
        voltage_tolerance_V, compiled.voltage_polarity, compiled.identity,
        bracket.generation_scale, asdict(controls),
    )
