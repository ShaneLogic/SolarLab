"""Dark equilibrium and illuminated classical 1D DC using unified Newton.

Poisson-Boltzmann prepares a common-Fermi seed. Coupled Poisson/continuity
certify the same production equations at every illuminated and biased point.
Bounded light/bias continuation records each Newton result. Failed residuals
or physical gates raise SteadyStateError; no legacy/transient fallback exists.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.sparse import diags

from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.equations.dc_1d import CompiledDevice1D, DC1DProblem, State1D
from solarlab.physics.contacts import ContactEndpoint, assess_contact_thermodynamics
from solarlab.solvers.nonlinear.newton import (
    NewtonResult, NewtonSettings, factorize, gate_from, solve_newton,
)
from solarlab.solvers.nonlinear.problem import (
    CallableProblem, GateOutcome, PhysicalGate, ResidualScaling, SolveBudget, StepControl,
)
from solarlab.solvers.nonlinear.production import (
    poisson_row_weights, production_coordinates, production_residual_scaling,
)

__all__ = [
    "SteadyOptions", "SteadyState1D", "SteadyStateError",
    "solve_equilibrium", "solve_steady_state",
]

Q = ELEMENTARY_CHARGE_C


@dataclass(frozen=True)
class SteadyOptions:
    """Physical limits and bounded numerical work for this DC slice."""

    poisson_tolerance: float = 1.0e-11
    continuity_tolerance_A_m2: float = 1.0e-8
    current_spread_tolerance_A_m2: float = 1.0e-8
    equilibrium_current_tolerance_A_m2: float = 1.0e-10
    contact_log_tolerance: float = 1.0e-11
    potential_step_V: float = 0.1
    quasi_fermi_step: float = 2.0
    max_iterations: int = 100
    max_residual_evaluations: int = 10000
    max_jacobian_evaluations: int = 2000
    max_backtracks: int = 40
    max_bias_step_V: float = 0.1
    max_continuation_steps: int = 100

    def __post_init__(self) -> None:
        for name in (
            "poisson_tolerance", "continuity_tolerance_A_m2",
            "current_spread_tolerance_A_m2", "equilibrium_current_tolerance_A_m2",
            "contact_log_tolerance", "potential_step_V", "quasi_fermi_step",
            "max_bias_step_V",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        self.budget()
        if type(self.max_continuation_steps) is not int or self.max_continuation_steps < 1:
            raise ValueError("max_continuation_steps must be a positive integer")

    def budget(self) -> SolveBudget:
        return SolveBudget(
            self.max_iterations, self.max_residual_evaluations,
            self.max_jacobian_evaluations, self.max_backtracks,
        )


@dataclass(frozen=True)
class SteadyState1D:
    """A physically accepted operating point and its Newton evidence."""

    state: State1D
    current_density_A_m2: np.ndarray
    contact_current_density_A_m2: np.ndarray
    voltage_V: float
    generation_scale: float
    newton: NewtonResult
    preparation: tuple[NewtonResult, ...] = ()
    density_basin_initializations: int = 0

    def __post_init__(self) -> None:
        for name, shape in (
            ("current_density_A_m2", (self.state.potential_V.size - 1,)),
            ("contact_current_density_A_m2", (2,)),
        ):
            array = np.array(getattr(self, name), dtype=float, copy=True)
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite with shape {shape}")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if not self.newton.accepted:
            raise ValueError("an unaccepted Newton result cannot be a steady point")
        if type(self.density_basin_initializations) is not int or self.density_basin_initializations < 0:
            raise ValueError("density_basin_initializations must be a nonnegative integer")

    @property
    def converged(self) -> bool:
        return self.newton.accepted

    @property
    def physical_gates(self) -> tuple[GateOutcome, ...]:
        return self.newton.gates


class SteadyStateError(RuntimeError):
    """A bounded solve/gate failed; the attached result is diagnostic only."""

    def __init__(self, stage: str, result: NewtonResult):
        self.stage = stage
        self.result = result
        super().__init__(
            f"{stage}: {result.termination.value}: {result.message}; "
            f"scaled residual={result.scaled_residual:.6g}"
        )


def _continuity_current_bound(problem: DC1DProblem, u: np.ndarray) -> float:
    count = problem.device.node_count
    residual = problem.residual(u)
    total = 0.0
    for carrier, offset in (("n", count), ("p", 2 * count)):
        rows = np.abs(residual[offset:offset + count]).copy()
        for side, node in (("left", 0), ("right", count - 1)):
            if problem.device.channel(carrier, side).kind == "fixed_reservoir":
                rows[node] = 0.0
        total += float(np.sum(rows))
    return Q * total


def _physical_gates(
    problem: DC1DProblem, options: SteadyOptions, *, equilibrium: bool
) -> tuple[PhysicalGate, ...]:
    device = problem.device
    count = device.node_count

    def all_currents(u: np.ndarray) -> np.ndarray:
        contact = problem.contact_current_density(u)
        return np.r_[contact[:1], problem.current_density(u), contact[1:]]

    gates: list[PhysicalGate] = [
        gate_from(
            "poisson_balance",
            lambda u: float(np.max(np.abs(problem.residual(u)[:count] * poisson_row_weights(device)))),
            options.poisson_tolerance, "maximum normalized Gauss/contact residual",
        ),
        gate_from(
            "continuity_current_bound", lambda u: _continuity_current_bound(problem, u),
            options.continuity_tolerance_A_m2,
            "q * sum of absolute carrier balances, including finite-rate contact cells",
        ),
        gate_from(
            "all_face_current_spread", lambda u: float(np.ptp(all_currents(u))),
            options.current_spread_tolerance_A_m2,
            "maximum minus minimum over every internal/external contact face",
        ),
    ]
    if equilibrium:
        def thermodynamics(u: np.ndarray) -> GateOutcome:
            state = device.state(u, voltage_V=problem.voltage_V)
            bands = device.parameters.band_structure

            def endpoint(node: int) -> ContactEndpoint:
                return ContactEndpoint(
                    float(state.electron_density_m3[node]), float(state.hole_density_m3[node]),
                    float(device.parameters.conduction_dos_m3[node]),
                    float(device.parameters.valence_dos_m3[node]),
                    float(bands.physical_electron_affinity_eV[node]),
                    float(bands.physical_band_gap_eV[node]),
                )

            assessment = assess_contact_thermodynamics(
                built_in_potential_mode="semiconductor_work_function",
                left=endpoint(0), right=endpoint(-1),
                V_bi_bc=float(state.potential_V[-1] - state.potential_V[0]),
                thermal_voltage_V=device.thermal_voltage_V, has_band_reference=True,
            )
            span = assessment.fermi_level_span_eV
            return GateOutcome(
                "contact_thermodynamics", assessment.certified,
                math.inf if span is None else span, assessment.tolerance_eV, assessment.status,
            )

        gates.extend((
            gate_from(
                "zero_equilibrium_carrier_current",
                lambda u: float(max(
                    np.max(np.abs(problem.fluxes(u).electron_current())),
                    np.max(np.abs(problem.fluxes(u).hole_current())),
                    np.max(np.abs(problem.contact_current_density(u))),
                )),
                options.equilibrium_current_tolerance_A_m2,
                "each carrier on every face and both physical contacts",
            ),
            thermodynamics,
        ))
    return tuple(gates)


def _solve_point(
    device: CompiledDevice1D, u: np.ndarray, voltage_V: float,
    generation_scale: float, options: SteadyOptions, *, equilibrium: bool = False,
) -> tuple[DC1DProblem, NewtonResult]:
    problem = DC1DProblem(device, voltage_V, generation_scale)
    settings = NewtonSettings(
        production_residual_scaling(
            device, poisson_tolerance=options.poisson_tolerance,
            continuity_tolerance_A_m2=options.continuity_tolerance_A_m2,
            contact_log_tolerance=options.contact_log_tolerance,
        ),
        1.0,
        production_coordinates(
            device, potential_step_V=options.potential_step_V,
            quasi_fermi_step=options.quasi_fermi_step,
        ),
        options.budget(), _physical_gates(problem, options, equilibrium=equilibrium), False,
    )
    result = solve_newton(problem, np.asarray(u, dtype=float), settings)
    if not result.accepted:
        raise SteadyStateError(f"DC V={voltage_V:g}, generation_scale={generation_scale:g}", result)
    return problem, result


def _point_result(
    problem: DC1DProblem, result: NewtonResult, preparation: tuple[NewtonResult, ...] = (),
    density_basin_initializations: int = 0,
) -> SteadyState1D:
    return SteadyState1D(
        problem.device.state(result.state, voltage_V=problem.voltage_V), problem.current_density(result.state),
        problem.contact_current_density(result.state), problem.voltage_V,
        problem.generation_scale, result, preparation, density_basin_initializations,
    )


def _density_basin_seed(problem: DC1DProblem, u: np.ndarray) -> np.ndarray:
    """One physical density linearization at frozen phi, used only as a seed.

    The full production Jacobian supplies SG, contact, and recombination
    derivatives. Solving its carrier block in density coordinates avoids an
    enormous logarithmic step out of an almost empty carrier population.
    The seed receives no acceptance certificate; coupled Newton still judges it.
    """
    count = problem.device.node_count
    _, n, p = problem.device.fields(u, voltage_V=problem.voltage_V)
    residual = problem.residual(u)[count:]
    density_inverse = np.r_[1.0 / n, -1.0 / p]
    matrix = problem.jacobian(u)[count:, count:] @ diags(density_inverse)
    increment = factorize(matrix).solve(-residual)
    trial_n, trial_p = n + increment[:count], p + increment[count:]
    if not all(np.all(np.isfinite(value)) and np.all(value > 0.0) for value in (trial_n, trial_p)):
        raise ValueError("cold density preparation did not produce finite positive carriers")
    seed = np.array(u, copy=True)
    for offset, density, trial, delta, sign in (
        (count, n, trial_n, increment[:count], 1.0),
        (2 * count, p, trial_p, increment[count:], -1.0),
    ):
        with np.errstate(over="ignore", divide="ignore"):
            relative = delta / density
            change = np.log1p(relative)
        large = ~np.isfinite(change)
        change[large] = np.log(trial[large]) - np.log(density[large])
        seed[offset:offset + count] += sign * change
    return seed


def solve_equilibrium(
    device: CompiledDevice1D, *, options: SteadyOptions | None = None
) -> SteadyState1D:
    """Dark equilibrium of physical bands, contact reservoirs, and Poisson."""
    controls = SteadyOptions() if options is None else options
    count = device.node_count

    def densities(phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, n, p = device.fields(np.r_[phi, np.zeros(2 * count)])
        return n, p

    def residual(phi: np.ndarray) -> np.ndarray:
        n, p = densities(phi)
        if not np.all(np.isfinite(n)) or not np.all(np.isfinite(p)):
            return np.full(count, np.nan)
        rho = device.charge_ledger(n, p).space_charge_C_m3()
        return device.poisson.residual(phi, rho, *device.contact_potentials(0.0))

    def jacobian(phi: np.ndarray) -> object:
        n, p = densities(phi)
        slope = np.zeros(count)
        slope[1:-1] = -Q * device.poisson.interior_widths_m * (n[1:-1] + p[1:-1]) / device.thermal_voltage_V
        return device.poisson.jacobian() + diags(slope)

    settings = NewtonSettings(
        ResidualScaling(poisson_row_weights(device) / controls.poisson_tolerance),
        1.0, StepControl(np.full(count, controls.potential_step_V)), controls.budget(), (), False,
    )
    prepared = solve_newton(
        CallableProblem(residual, jacobian), device.reference_potential_V, settings,
    )
    if not prepared.accepted:
        raise SteadyStateError("Poisson-Boltzmann equilibrium preparation", prepared)
    u = np.r_[prepared.state, np.zeros(2 * count)]
    problem, result = _solve_point(device, u, 0.0, 0.0, controls, equilibrium=True)
    return _point_result(problem, result, (prepared,))


def solve_steady_state(
    device: CompiledDevice1D,
    *,
    initial_state: State1D | None = None,
    voltage_V: float = 0.0,
    generation_scale: float = 1.0,
    options: SteadyOptions | None = None,
) -> SteadyState1D:
    """Solve a real operating point; an accepted state may seed the next point."""
    controls = SteadyOptions() if options is None else options
    target = DC1DProblem(device, voltage_V, generation_scale)
    if initial_state is not None:
        if not isinstance(initial_state, State1D) or initial_state.device_sha256 != device.sha256:
            raise ValueError("initial_state must belong to the same compiled device")
        problem, result = _solve_point(
            device,
            device.rebase_coordinates(initial_state.u, initial_state.voltage_V, target.voltage_V),
            target.voltage_V, target.generation_scale, controls,
            equilibrium=voltage_V == 0.0 and generation_scale == 0.0,
        )
        return _point_result(problem, result)
    equilibrium = solve_equilibrium(device, options=controls)
    if generation_scale == 0.0 and voltage_V == 0.0:
        return equilibrium
    u = equilibrium.state.u
    preparation = [*equilibrium.preparation, equilibrium.newton]
    light_steps = [fraction * generation_scale for fraction in (
        1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0,
    )] if generation_scale > 0.0 else []
    bias_count = int(math.ceil(abs(voltage_V) / controls.max_bias_step_V))
    if len(light_steps) + bias_count > controls.max_continuation_steps:
        raise ValueError("requested initialization exceeds the continuation-step budget")
    points = [(0.0, value) for value in light_steps]
    points.extend(
        (float(value), generation_scale)
        for value in np.linspace(0.0, voltage_V, bias_count + 1)[1:]
    )
    previous_voltage = 0.0
    density_initializations = 0
    for index, (voltage, light) in enumerate(points):
        u = device.rebase_coordinates(u, previous_voltage, voltage)
        if index == 0 and light > 0.0:
            u = _density_basin_seed(DC1DProblem(device, voltage, light), u)
            density_initializations += 1
        problem, result = _solve_point(device, u, voltage, light, controls)
        u = result.state
        previous_voltage = voltage
        if index != len(points) - 1:
            preparation.append(result)
    return _point_result(problem, result, tuple(preparation), density_initializations)
