"""Driven voltage/generation transients from an accepted same-device DC state.

The caller specifies every time and endpoint voltage/generation value. There are no
automatic substeps/retries, steady settling stages, or scientific certificates.
Newton acceptance concerns the physical backward-Euler residual, not constant
DC conduction current. The first failed attempt and accepted prefix are kept.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import math
import time
from typing import TYPE_CHECKING, Literal, cast, overload

import numpy as np

from solarlab.contracts import frozen_array
from solarlab.equations.assembly.classical_1d import AssembledDC1DProblem, classical_dc_1d
from solarlab.equations.dc_1d import CompiledDevice1D, DC1DProblem, State1D
from solarlab.equations.implementation import equation_implementation, selected_implementation, use_assembled
from solarlab.experiments.jv.operating_points import _verify_point
from solarlab.solvers.nonlinear.newton import NewtonResult, NewtonSettings, solve_newton
from solarlab.solvers.nonlinear.problem import CallableProblem, Matrix, SolveBudget
from solarlab.solvers.nonlinear.production import (
    HISTORICAL_MAX_ROW_V1, POISSON_CONTACT_BALANCE_V1, production_convergence_norm,
    production_coordinates, production_residual_scaling,
)
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, SteadyState1D
from solarlab.solvers.time import BackwardEuler1DStep, step_accounting
from solarlab.solvers.time.assembled_storage import AssembledStorage1D

if TYPE_CHECKING:
    from solarlab.solvers.time.bdf import BDFOptions, BDFResult
    from solarlab.solvers.time.bdf_1d import BDFDrive

__all__ = ["TransientProtocol", "TransientOptions", "TransientAttempt", "TransientResult1D", "solve_transient"]


@dataclass(frozen=True)
class TransientProtocol:
    """Explicit voltage/generation samples at each BE right endpoint.

    There is no inferred event interpolation or steady solve at intermediate
    biases. The samples drive a time history, not a steady JV scan.
    """

    times_s: np.ndarray
    voltage_V: np.ndarray
    generation_scale: np.ndarray

    def __post_init__(self) -> None:
        for name in ("times_s", "voltage_V", "generation_scale"):
            if np.iscomplexobj(getattr(self, name)):
                raise ValueError(f"{name} must be real; complex values are unsupported")
            array = frozen_array(getattr(self, name))
            object.__setattr__(self, name, array)
        times = self.times_s
        with np.errstate(over="ignore"):
            differences = np.diff(times) if times.ndim == 1 else np.asarray([])
        if (times.ndim != 1 or times.size < 2 or not np.all(np.isfinite(differences))
                or np.any(differences <= 0.0)):
            raise ValueError("times_s must be an explicit strictly increasing finite vector")
        if any(a.shape != times.shape for a in (self.voltage_V, self.generation_scale)):
            raise ValueError("voltage and generation must have one value per requested time")
        if np.any(self.generation_scale < 0.0):
            raise ValueError("generation_scale must be nonnegative")


@dataclass(frozen=True)
class TransientOptions:
    """Required numerical tolerances, step limits, and finite work budgets."""

    poisson_tolerance: float
    continuity_tolerance_A_m2: float
    contact_log_tolerance: float
    potential_step_V: float
    quasi_fermi_step: float
    newton_budget: SolveBudget
    max_steps: int
    wall_s: float
    predictor: str
    convergence_policy: str

    def __post_init__(self) -> None:
        for name in ("poisson_tolerance", "continuity_tolerance_A_m2", "contact_log_tolerance",
                     "potential_step_V", "quasi_fermi_step", "wall_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if not isinstance(self.newton_budget, SolveBudget):
            raise ValueError("newton_budget must be explicit SolveBudget controls")
        if self.predictor not in ("previous", "generation_inventory"):
            raise ValueError("predictor must be previous or generation_inventory")
        if self.convergence_policy not in (HISTORICAL_MAX_ROW_V1, POISSON_CONTACT_BALANCE_V1):
            raise ValueError("unknown convergence policy")


@dataclass(frozen=True)
class TransientAttempt:
    start_s: float
    end_s: float
    voltage_V: float
    generation_scale: float
    accepted: bool
    termination: str
    initial_coordinates: np.ndarray
    candidate_coordinates: np.ndarray
    newton: NewtonResult | None
    accounting: dict[str, object] | None
    wall_s: float
    error: str | None = None

    def __post_init__(self) -> None:
        for name in ("initial_coordinates", "candidate_coordinates"):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class TransientResult1D:
    protocol: TransientProtocol
    accepted_times_s: np.ndarray
    states: tuple[State1D, ...]
    attempts: tuple[TransientAttempt, ...]
    initial: SteadyState1D
    compiled_identity: str
    solver_controls: dict[str, object]
    initial_controls: dict[str, object]
    wall_s: float
    physical_certificate_status: str = "not_evaluated"

    def __post_init__(self) -> None:
        object.__setattr__(self, "accepted_times_s", frozen_array(self.accepted_times_s))

    @property
    def completed(self) -> bool:
        return len(self.states) == self.protocol.times_s.size and all(a.accepted for a in self.attempts)

    @property
    def failed_attempts(self) -> tuple[TransientAttempt, ...]:
        return tuple(a for a in self.attempts if not a.accepted)


class _WallBudgetExceeded(RuntimeError):
    pass


@overload
def solve_transient(compiled: CompiledDevice1D, initial: SteadyState1D, protocol: TransientProtocol, *,
    options: TransientOptions, initial_options: SteadyOptions, method: Literal["backward_euler"] = "backward_euler",
    bdf_options: BDFOptions | None = None, bdf_drive: BDFDrive | None = None) -> TransientResult1D: ...


@overload
def solve_transient(compiled: CompiledDevice1D, initial: SteadyState1D, protocol: TransientProtocol, *,
    options: TransientOptions, initial_options: SteadyOptions, method: Literal["bdf"],
    bdf_options: BDFOptions, bdf_drive: BDFDrive | None = None) -> BDFResult: ...


def solve_transient(
    compiled: CompiledDevice1D,
    initial: SteadyState1D,
    protocol: TransientProtocol,
    *,
    options: TransientOptions,
    initial_options: SteadyOptions,
    method: str = "backward_euler",
    bdf_options: BDFOptions | None = None,
    bdf_drive: BDFDrive | None = None,
) -> TransientResult1D | BDFResult:
    """Run exactly the requested BE steps; stop and retain the first failure.

The wall budget is monotonic and checked around each residual/Jacobian and
accounting call; individual SciPy operations are not preempted. No wall clock
or evaluation budget is reset after failure. Initial DC gates are rechecked
only for admission of the initial state, never used to gate transient steps.
For a changing voltage, the recorded Newton seed is rebased and harmonically
lifted to the new electrode potentials while preserving old carrier densities.
The full physical secant already includes reference-voltage changes, so no
additional voltage-rate storage term is added.
"""
    if method == "bdf":
        from solarlab.solvers.time.bdf_1d import solve_electronic_bdf
        if bdf_options is None:
            raise ValueError("BDF selection requires explicit BDFOptions")
        with equation_implementation(selected_implementation(operator="BDFStep:DC1DProblem")):
            return solve_electronic_bdf(compiled, initial, protocol, options=options,
                initial_options=initial_options, bdf_options=bdf_options, drive=bdf_drive)
    if method != "backward_euler":
        raise ValueError("unknown time integrator")
    started = time.monotonic()
    if not isinstance(compiled, CompiledDevice1D) or not isinstance(initial, SteadyState1D):
        raise ValueError("an actual compiled device and accepted same-device DC initial state are required")
    if not isinstance(protocol, TransientProtocol) or not isinstance(options, TransientOptions):
        raise ValueError("an explicit TransientProtocol and TransientOptions are required")
    if not isinstance(initial_options, SteadyOptions):
        raise ValueError("initial_options must contain the DC admission controls")
    if not {"ion_free", "classical_1d_dc", "maxwell_boltzmann"} <= set(compiled.capabilities):
        raise ValueError("transient capability is limited to ion-free classical 1D")
    if protocol.times_s.size - 1 > options.max_steps:
        raise ValueError("requested timepoints exceed max_steps")
    if initial.voltage_V != protocol.voltage_V[0] or initial.generation_scale != protocol.generation_scale[0]:
        raise ValueError("initial voltage/generation must equal the first protocol sample")
    if not np.array_equal(compiled.geometry.cell_measures_m[1:-1], compiled.poisson.interior_widths_m):
        raise ValueError("carrier and Poisson control volumes must agree")
    if any(channel.kind not in {"fixed_reservoir", "robin", "blocking"} for channel in compiled.channels):
        raise ValueError("unsupported contact channel for the transient slice")
    _verify_point(compiled, initial, initial.generation_scale, initial_options)
    settings = NewtonSettings(
        production_residual_scaling(compiled, poisson_tolerance=options.poisson_tolerance,
                                    continuity_tolerance_A_m2=options.continuity_tolerance_A_m2,
                                    contact_log_tolerance=options.contact_log_tolerance),
        1.0, production_coordinates(compiled, potential_step_V=options.potential_step_V,
                                    quasi_fermi_step=options.quasi_fermi_step),
        options.newton_budget, (), False,
    )
    deadline = started + options.wall_s
    # One bound assembled system serves every step (the BE bridge's storage adapter).
    binding = classical_dc_1d(compiled) if use_assembled() else None
    states, accepted_times, attempts = [initial.state], [float(protocol.times_s[0])], []
    for index in range(1, protocol.times_s.size):
        start, end = float(protocol.times_s[index - 1]), float(protocol.times_s[index])
        voltage, generation = float(protocol.voltage_V[index]), float(protocol.generation_scale[index])
        # The storage adapter has the step's endpoint surface (as in the BE bridge).
        endpoint = (DC1DProblem(compiled, voltage, generation) if binding is None
                    else cast(DC1DProblem, AssembledStorage1D(AssembledDC1DProblem(compiled, voltage, generation,
                                                                                   binding))))
        step = BackwardEuler1DStep(endpoint, states[-1], end - start)
        candidate = step.initial_coordinates(voltage_lift=True)
        initial_guess = candidate.copy()
        attempt_start = time.monotonic()
        newton, accounting, error = None, None, None

        def checkpoint() -> None:
            if time.monotonic() >= deadline:
                raise _WallBudgetExceeded("transient wall budget exhausted; no retry")

        def evaluate(function: Callable[[np.ndarray], Matrix], u: np.ndarray) -> Matrix:
            nonlocal candidate
            candidate = np.array(u, copy=True)
            checkpoint()
            value = function(u)
            checkpoint()
            return value

        try:
            checkpoint()
            candidate = step.initial_coordinates(options.predictor, voltage_lift=True)
            initial_guess = candidate.copy()
            newton = solve_newton(
                CallableProblem(lambda u: evaluate(step.residual, u), lambda u: evaluate(step.jacobian, u)),
                candidate, settings,
                convergence_norm=production_convergence_norm(compiled, convergence_policy=options.convergence_policy),
            )
            candidate = newton.state
            checkpoint()
            if np.all(np.isfinite(candidate)):
                _, n, p = compiled.fields(candidate, voltage_V=voltage)
                if np.all(np.isfinite(n)) and np.all(np.isfinite(p)) and np.all(n > 0) and np.all(p > 0):
                    accounting = step_accounting(step, candidate)
            checkpoint()
            accepted, termination = newton.accepted, newton.termination.value
            if accepted and (accounting is None or any(
                not np.all(np.isfinite(value)) for value in accounting.values()
                if isinstance(value, (np.ndarray, float))
            )):
                accepted, termination = False, "nonfinite_accounting"
        except Exception as exc:
            accepted = False
            termination = "wall_budget_exhausted" if isinstance(exc, _WallBudgetExceeded) else "evaluation_error"
            error = f"{type(exc).__name__}: {str(exc)[:400]}"
        attempts.append(TransientAttempt(start, end, voltage, generation, accepted, termination,
                                         initial_guess, candidate, newton, accounting,
                                         time.monotonic() - attempt_start, error))
        if not accepted:
            break
        states.append(compiled.state(candidate, voltage_V=voltage))
        accepted_times.append(end)
    recorded_controls = asdict(options)
    if np.any(protocol.voltage_V != protocol.voltage_V[0]):
        recorded_controls["voltage_seed_policy"] = "harmonic_potential_preserve_densities_when_voltage_changes"
    return TransientResult1D(protocol, np.asarray(accepted_times), tuple(states), tuple(attempts), initial,
                             compiled.identity, recorded_controls, asdict(initial_options), time.monotonic() - started)
