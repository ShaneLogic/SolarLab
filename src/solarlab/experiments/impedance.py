"""Ion-free classical 1D fixed/finite/selective small signal about accepted DC.

Phasors use exp(+i*omega*t). A 1 V normalization describes a derivative,
not a finite one-volt experiment. Coordinates use the DC state's fixed
physical Fermi reference; moving electrode-reference terms are retained
explicitly when converting the reported coordinate perturbation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csc_matrix, eye

from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.equations.dc_1d import (
    CompiledDevice1D, DC1DProblem, _quasi_fermi_potential_tangents,
)
from solarlab.experiments.jv.operating_points import _verify_point
from solarlab.physics.contacts.boundary_flux import boundary_particle_flux
from solarlab.solvers.linear import FrequencyOptions, FrequencySolution, solve_frequency_systems
from solarlab.solvers.nonlinear.problem import state_digest
from solarlab.solvers.nonlinear.production import production_convergence_norm, production_residual_scaling
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, SteadyState1D

__all__ = ["SmallSignalResult1D", "voltage_reference_direction", "solve_impedance"]

Q = ELEMENTARY_CHARGE_C


def voltage_reference_direction(device: CompiledDevice1D) -> np.ndarray:
    """du_moving/dV at fixed physical fields, from the affine DC references."""
    count = device.node_count
    result = np.zeros(3 * count)
    zero = device.carrier_reference_energies(0.0)
    one = device.carrier_reference_energies(1.0)
    for offset, before, after in zip((count, 2 * count), zero, one):
        result[offset:offset + count] = -(after - before) / device.thermal_voltage_V
    return result


def _sources(*, field_transport: bool = False) -> dict[str, str]:
    names: tuple[str, ...] = (
        __name__, "solarlab.equations.dc_1d",
        "solarlab.physics.transport.electrochemical_flux",
        "solarlab.discretization.stable_bernoulli",
        "solarlab.physics.recombination.bulk", "solarlab.physics.recombination.reaction",
        "solarlab.physics.contacts.boundary_flux", "solarlab.physics.contacts.exchange",
        "solarlab.physics.electrostatics.poisson",
        "solarlab.solvers.linear.frequency_domain",
        "solarlab.solvers.nonlinear.production", "solarlab.solvers.nonlinear.steady_1d",
        "solarlab.experiments.jv.operating_points",
    )
    if field_transport:
        names += ("solarlab.equations.field_transport_1d",
                  "solarlab.physics.transport.field_mobility",
                  "solarlab.physics.transport.field_mobility_tangents")
    result = {}
    for name in names:
        path = sys.modules[name].__file__
        if path is None:
            raise RuntimeError(f"small-signal source cannot be pinned: {name}")
        result[name] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return result


@dataclass(frozen=True)
class SmallSignalResult1D:
    """Complex field/current phasors for the requested linear normalization.

    All face currents are oriented along +x. Terminal columns are left/right;
    the passive input port uses polarity times the positive-electrode current.
    Y is in S/m^2, and its inverse is an impedance-area product in ohm*m^2.
    """

    frequency_Hz: np.ndarray
    voltage_phasor_V: complex
    nodes_m: np.ndarray
    fixed_coordinate_perturbation: np.ndarray
    moving_coordinate_perturbation: np.ndarray
    reference_direction_per_V: np.ndarray
    potential_perturbation_V: np.ndarray
    electric_field_perturbation_V_m: np.ndarray
    electron_density_perturbation_m3: np.ndarray
    hole_density_perturbation_m3: np.ndarray
    conduction_current_density_A_m2: np.ndarray
    displacement_current_density_A_m2: np.ndarray
    total_current_density_A_m2: np.ndarray
    terminal_conduction_current_density_A_m2: np.ndarray
    terminal_displacement_current_density_A_m2: np.ndarray
    terminal_current_density_A_m2: np.ndarray
    input_current_density_A_m2: np.ndarray
    admittance_density_S_m2: np.ndarray
    maxwell_spread_A_m2_V: np.ndarray
    residual_balance_bound_A_m2_V: np.ndarray
    linear_solutions: tuple[FrequencySolution, ...]
    compiled_identity: str
    working_state_identity: str
    working_state_digest: str
    operating_voltage_V: float
    generation_scale: float
    voltage_polarity: float
    source_identity: str
    source_sha256: dict[str, str]
    revalidation_solver_controls: dict[str, object]
    linear_solver_controls: dict[str, object]

    def __post_init__(self) -> None:
        real = {"frequency_Hz", "nodes_m", "reference_direction_per_V", "maxwell_spread_A_m2_V", "residual_balance_bound_A_m2_V"}
        for name, value in vars(self).items():
            if isinstance(value, np.ndarray):
                array = np.array(value, dtype=float if name in real else complex, copy=True)
                if not np.all(np.isfinite(array)):
                    raise ValueError(f"nonfinite small-signal output: {name}")
                array.setflags(write=False)
                object.__setattr__(self, name, array)

    @property
    def impedance_area_ohm_m2(self) -> np.ndarray:
        if np.any(self.admittance_density_S_m2 == 0.0):
            raise ValueError("zero admittance has no finite impedance-area product")
        result = 1.0 / self.admittance_density_S_m2
        if not np.all(np.isfinite(result)):
            raise ValueError("impedance-area product is outside the finite numerical envelope")
        return result


def _particle_perturbations(
    problem: DC1DProblem, u: np.ndarray, x: np.ndarray, voltage: complex,
) -> tuple[np.ndarray, np.ndarray]:
    """Use the same stable quasi-Fermi tangent blocks as the production Fy."""
    device = problem.device
    count = device.node_count
    fluxes = problem.fluxes(u)
    transport = None if device.field_transport is None else device.field_transport.evaluate(u[:count])
    _, n, p = device.fields(u, voltage_V=problem.voltage_V)
    references = tuple(
        after - before for before, after in zip(
            device.carrier_reference_energies(0.0), device.carrier_reference_energies(1.0),
        )
    )
    result = []
    for offset, density, flux, sign, reference in (
        (count, n, fluxes.electron, 1.0, references[0]),
        (2 * count, p, fluxes.hole, -1.0, references[1]),
    ):
        drive = sign * np.diff(u[offset:offset + count])
        left_phi, _right_phi = _quasi_fermi_potential_tangents(flux, drive, sign, device.thermal_voltage_V)
        if transport is not None:
            diffusivity = (transport.electron_diffusivity_m2_s if offset == count
                           else transport.hole_diffusivity_m2_s)
            d_diffusivity = (transport.electron_d_diffusivity_d_field if offset == count
                             else transport.hole_d_diffusivity_d_field)
            # The CT chain is antisymmetric in phi_L/phi_R. SG is linear in D;
            # its carrier sign is already in Gamma, never applied a second time.
            left_phi = left_phi + (flux.flux / diffusivity) * d_diffusivity / np.diff(device.nodes_m)
        left_z = sign * flux.left_density_derivative * density[:-1]
        # A uniform Fermi or potential shift scales the particle flux. Group
        # that exact homogeneity mode before multiplying the large SG slopes.
        result.append(
            left_phi * (x[:count - 1] - x[1:count])
            + left_z * (x[offset:offset + count - 1] - x[offset + 1:offset + count])
            + sign * flux.flux * (
                x[offset + 1:offset + count] + (x[1:count] + reference * voltage) / device.thermal_voltage_V
            )
        )
    return result[0], result[1]


def _moving_voltage_partial(
    problem: DC1DProblem, u: np.ndarray, coordinate_direction: np.ndarray | None = None,
) -> np.ndarray:
    """Actual F_V+J*direction, combining density/contact terms before cancellation."""
    device = problem.device
    count = device.node_count
    _, n, p = device.fields(u, voltage_V=problem.voltage_V)
    references = tuple(
        after - before for before, after in zip(
            device.carrier_reference_energies(0.0), device.carrier_reference_energies(1.0),
        )
    )
    direction = np.zeros(3 * count) if coordinate_direction is None else coordinate_direction
    phi = direction[:count]
    dn = n * ((phi + references[0]) / device.thermal_voltage_V + direction[count:2 * count])
    dp = -p * ((phi + references[1]) / device.thermal_voltage_V + direction[2 * count:])
    rate = problem.recombination_rate(n, p)
    assert rate.d_rate_dn is not None and rate.d_rate_dp is not None
    source = -(rate.d_rate_dn * dn + rate.d_rate_dp * dp)
    fn, fp = _particle_perturbations(problem, u, direction, 1.0)
    partial = np.zeros(3 * count)
    partial[:count] = device.poisson.jacobian() @ phi + device.poisson.charge_jacobian() * Q * (dp - dn)
    partial[count - 1] += device.voltage_polarity
    for carrier, offset, density_derivative, face, sign, reference in (
        ("n", count, dn, fn, 1.0, references[0]),
        ("p", 2 * count, dp, fp, -1.0, references[1]),
    ):
        rows = -np.diff(np.r_[0.0, face, 0.0]) + device.geometry.cell_measures_m * source
        for side, node in (("left", 0), ("right", count - 1)):
            channel = device.channel(carrier, side)
            if channel.kind == "fixed_reservoir":
                rows[node] = sign * (
                    (phi[node] + reference) / device.thermal_voltage_V + direction[offset + node]
                )
            else:
                slope = boundary_particle_flux(channel, channel.reservoir_density_m3).density_derivative
                rows[node] += (1.0 if side == "left" else -1.0) * slope * density_derivative[node]
        partial[offset:offset + count] = rows
    return partial


def _lifted_storage_drive(problem: DC1DProblem, u: np.ndarray, lift: np.ndarray) -> np.ndarray:
    """K*(c-lift), evaluated as the merged physical density drive."""
    device = problem.device
    count = device.node_count
    _, n, p = device.fields(u, voltage_V=problem.voltage_V)
    references = tuple(
        after - before for before, after in zip(
            device.carrier_reference_energies(0.0), device.carrier_reference_energies(1.0),
        )
    )
    result = np.zeros(3 * count)
    for carrier, offset, density, sign, reference in (
        ("n", count, n, 1.0, references[0]), ("p", 2 * count, p, -1.0, references[1]),
    ):
        result[offset:offset + count] = device.geometry.cell_measures_m * sign * density * (
            (lift[:count] + reference) / device.thermal_voltage_V + lift[offset:offset + count]
        )
        for side, node in (("left", 0), ("right", count - 1)):
            if device.channel(carrier, side).kind == "fixed_reservoir":
                result[offset + node] = 0.0
    return result


def _uniform_carrier_partial(problem: DC1DProblem, u: np.ndarray, shifted_carrier: str) -> np.ndarray:
    """Production derivative for a uniform dimensionless carrier offset.

    A unit electron offset gives dn=n; a unit hole offset gives dp=-p.
    The corresponding particle flux scales by +Gamma_n or -Gamma_p.
    """
    device = problem.device
    count = device.node_count
    _, n, p = device.fields(u, voltage_V=problem.voltage_V)
    dn = n if shifted_carrier == "n" else np.zeros(count)
    dp = -p if shifted_carrier == "p" else np.zeros(count)
    rate = problem.recombination_rate(n, p)
    assert rate.d_rate_dn is not None and rate.d_rate_dp is not None
    source = -(rate.d_rate_dn * dn + rate.d_rate_dp * dp)
    fluxes = problem.fluxes(u)
    result = np.zeros(3 * count)
    result[:count] = device.poisson.charge_jacobian() * Q * (dp - dn)
    for carrier, offset, density, flux, sign in (
        ("n", count, dn, fluxes.electron, 1.0), ("p", 2 * count, dp, fluxes.hole, -1.0),
    ):
        face = sign * flux.flux if carrier == shifted_carrier else np.zeros(count - 1)
        rows = -np.diff(np.r_[0.0, face, 0.0]) + device.geometry.cell_measures_m * source
        for side, node in (("left", 0), ("right", count - 1)):
            channel = device.channel(carrier, side)
            if channel.kind == "fixed_reservoir":
                rows[node] = sign if carrier == shifted_carrier else 0.0
            else:
                slope = boundary_particle_flux(channel, channel.reservoir_density_m3).density_derivative
                rows[node] += (1.0 if side == "left" else -1.0) * slope * density[node]
        result[offset:offset + count] = rows
    return result


def _reference_anchor_system(
    problem: DC1DProblem, u: np.ndarray, J: csc_matrix, K: csc_matrix,
) -> tuple[csc_matrix, csc_matrix, csc_matrix, tuple[int, int]]:
    """Sparse carrier-offset basis with analytically grouped uniform columns.

    q_i=eta_i+eta_anchor off the anchor, q_anchor=eta_anchor. Thus the
    anchor column is the uniform carrier derivative. Evaluating it from
    the production kernels retains the small physical mode that can be
    lost when large opposite entries of the nodal Jacobian are summed.
    """
    device = problem.device
    count = device.node_count
    transform = eye(3 * count, format="lil")
    anchors = []
    for carrier, offset, preferred in (
        ("n", count, "right" if device.voltage_polarity > 0.0 else "left"),
        ("p", 2 * count, "left" if device.voltage_polarity > 0.0 else "right"),
    ):
        fixed = [side for side in ("left", "right") if device.channel(carrier, side).kind == "fixed_reservoir"]
        side = fixed[0] if fixed else preferred
        anchor = offset + (0 if side == "left" else count - 1)
        anchors.append(anchor)
        transform[offset:offset + count, anchor] = np.ones(count)
    P = transform.tocsc()
    matrix = (J @ P).tolil()
    for carrier, anchor in zip(("n", "p"), anchors):
        uniform = _uniform_carrier_partial(problem, u, carrier)
        matrix[:, anchor] = uniform[:, None]
    return matrix.tocsc(), (K @ P).tocsc(), P, (anchors[0], anchors[1])


def solve_impedance(
    compiled: CompiledDevice1D,
    operating_point: SteadyState1D,
    frequencies_Hz: ArrayLike,
    *,
    voltage_phasor_V: complex = 1.0,
    steady_options: SteadyOptions | None = None,
    linear_options: FrequencyOptions | None = None,
    current_absolute_tolerance_A_m2_V: float = 1e-8,
    current_relative_tolerance: float = 1e-10,
) -> SmallSignalResult1D:
    """Actual fixed/Robin/selective linearization within checked numerical limits.

    Input fields/gates are revalidated against supplied current controls and
    code. A bare SteadyState1D does not bind complete historical solver/source
    provenance; the checks here do not grant that stronger guarantee.

    With c=du_moving/dV at fixed physical fields, the moving-reference
    storage drive is -K*c*Vdot and F_V=polarity*e_phi_right-J*c. For the fixed-reference
    perturbation x=du_moving-c*v this cancels, leaving
    (J+i*omega*K)x=-polarity*e_phi_right*v. Here K is exactly F_udot.
    """
    if not isinstance(compiled, CompiledDevice1D):
        raise ValueError("an actual CompiledDevice1D is required")
    if not isinstance(operating_point, SteadyState1D):
        raise ValueError("an accepted same-device SteadyState1D is required")
    if not {"ion_free", "classical_1d_dc", "maxwell_boltzmann"} <= set(compiled.capabilities):
        raise ValueError("small-signal capability is limited to ion-free classical 1D")
    voltage = complex(voltage_phasor_V)
    if not np.isfinite(voltage) or voltage == 0.0:
        raise ValueError("voltage_phasor_V must be finite and nonzero")
    if not all(math.isfinite(value) and value > 0.0 for value in (
        current_absolute_tolerance_A_m2_V, current_relative_tolerance,
    )):
        raise ValueError("numerical current-closure controls must be finite and positive")
    controls = SteadyOptions() if steady_options is None else steady_options
    linear = FrequencyOptions() if linear_options is None else linear_options
    _verify_point(compiled, operating_point, operating_point.generation_scale, controls)
    limits = {
        "poisson_balance": controls.poisson_tolerance,
        "continuity_current_bound": controls.continuity_tolerance_A_m2,
        "all_face_current_spread": controls.current_spread_tolerance_A_m2,
        "zero_equilibrium_carrier_current": controls.equilibrium_current_tolerance_A_m2,
    }
    if any(gate.name in limits and gate.limit != limits[gate.name] for gate in operating_point.physical_gates):
        raise ValueError("supplied DC controls disagree with recorded physical gate limits")
    widths = compiled.geometry.cell_measures_m
    if not np.array_equal(widths[1:-1], compiled.poisson.interior_widths_m):
        raise ValueError("small signal requires matching carrier and Poisson interior control widths")
    for channel in compiled.channels:
        if channel.kind not in {"fixed_reservoir", "robin", "blocking"}:
            raise ValueError(f"unsupported small-signal contact kind: {channel.kind}")
    if np.any(np.asarray(frequencies_Hz) == 0.0):
        for carrier in ("n", "p"):
            if all(compiled.channel(carrier, side).kind == "blocking" for side in ("left", "right")):
                raise ValueError(f"zero-frequency {carrier} blocked at both ends is not validated by this small-signal slice")
    source_pins = _sources(field_transport=compiled.field_transport is not None)
    problem = DC1DProblem(compiled, operating_point.voltage_V, operating_point.generation_scale)
    u = operating_point.state.u
    J, K = problem.jacobian(u), problem.storage_jacobian(u)
    scaled = production_residual_scaling(
        compiled, poisson_tolerance=controls.poisson_tolerance,
        continuity_tolerance_A_m2=controls.continuity_tolerance_A_m2,
        contact_log_tolerance=controls.contact_log_tolerance,
    ).scaled(problem.residual(u))
    norm = production_convergence_norm(compiled, convergence_policy=controls.convergence_policy)
    recorded_norm = float(np.max(abs(scaled))) if norm is None else norm(scaled)
    if recorded_norm != operating_point.newton.scaled_residual:
        raise ValueError("supplied DC convergence norm does not match the recorded metric")
    count = compiled.node_count
    c = voltage_reference_direction(compiled)
    lift = np.zeros(3 * count)
    lift[count - 1] = -compiled.voltage_polarity
    fixed_coordinates: dict[int, complex] = {0: 0.0j, count - 1: 0.0j}
    for carrier, offset in (("n", count), ("p", 2 * count)):
        for side, node in (("left", 0), ("right", count - 1)):
            if compiled.channel(carrier, side).kind == "fixed_reservoir":
                lift[offset + node] = (
                    -lift[node] / compiled.thermal_voltage_V + c[offset + node]
                )
                fixed_coordinates[offset + node] = 0.0j
    rhs = -_moving_voltage_partial(problem, u, lift) * voltage
    rhs_rate = _lifted_storage_drive(problem, u, lift) * voltage
    transform = eye(3 * count, format="csc")
    solve_J, solve_K = J, K
    anchors: tuple[int, int] | None = None
    if any(channel.kind != "fixed_reservoir" for channel in compiled.channels):
        solve_J, solve_K, transform, anchors = _reference_anchor_system(problem, u, J, K)
    reduced_solutions = solve_frequency_systems(
        solve_J, solve_K, rhs, frequencies_Hz, options=linear, fixed_coordinates=fixed_coordinates, rhs_rate=rhs_rate,
    )
    original_rhs = -_moving_voltage_partial(problem, u) * voltage
    original_rate = (K @ c) * voltage
    full_solutions = []
    for item in reduced_solutions:
        omega = 2.0 * math.pi * item.frequency_Hz
        matrix = J + 1j * omega * K
        y = transform @ item.solution + lift * voltage
        b = original_rhs + 1j * omega * original_rate
        residual = matrix @ y - b
        denominator = abs(matrix) @ abs(y) + abs(b)
        error = float(np.max(np.divide(
            abs(residual), denominator, out=np.zeros_like(denominator), where=denominator != 0.0,
        )))
        if not math.isfinite(error) or error > linear.relative_tolerance:
            raise ArithmeticError("reconstructed small-signal state fails the full original system")
        full_solutions.append(FrequencySolution(item.frequency_Hz, y, residual, error, item.refinement_steps))
    solved = tuple(full_solutions)
    moving_x = np.stack([item.solution for item in solved])
    fixed_x = moving_x - c * voltage
    phi = fixed_x[:, :count]
    n, p = operating_point.state.electron_density_m3, operating_point.state.hole_density_m3
    references = tuple(
        after - before for before, after in zip(
            compiled.carrier_reference_energies(0.0), compiled.carrier_reference_energies(1.0),
        )
    )
    dn = n * ((phi + references[0] * voltage) / compiled.thermal_voltage_V + moving_x[:, count:2 * count])
    dp = -p * ((phi + references[1] * voltage) / compiled.thermal_voltage_V + moving_x[:, 2 * count:])
    charge = Q * (dp - dn)
    field = -np.diff(phi, axis=1) / np.diff(compiled.nodes_m)
    displacement = -np.diff(phi, axis=1) * compiled.poisson.face_capacitance_F_m2
    terminal_D = np.stack((
        displacement[:, 0] - widths[0] * charge[:, 0],
        displacement[:, -1] + widths[-1] * charge[:, -1],
    ), axis=1)
    rate = problem.recombination_rate(n, p)
    assert rate.d_rate_dn is not None and rate.d_rate_dp is not None
    pair_perturbation = -widths * (rate.d_rate_dn * dn + rate.d_rate_dp * dp)
    conduction_rows: list[np.ndarray] = []
    terminal_rows: list[list[complex]] = []
    bounds: list[float] = []
    for index, item in enumerate(solved):
        omega = 2.0 * math.pi * item.frequency_Hz
        fn, fp = _particle_perturbations(problem, u, moving_x[index], voltage)
        conduction_rows.append(-Q * fn + Q * fp)
        terminals = []
        for side, node, face, orientation in (("left", 0, 0, -1.0), ("right", count - 1, -1, 1.0)):
            current = 0.0j
            for carrier, density_perturbation, flux, charge_sign in (("n", dn[index], fn, -Q), ("p", dp[index], fp, Q)):
                channel = compiled.channel(carrier, side)
                if channel.kind == "fixed_reservoir":
                    exchange = flux[face] + orientation * (
                        pair_perturbation[index, node] - 1j * omega * widths[node] * density_perturbation[node]
                    )
                else:
                    slope = boundary_particle_flux(channel, channel.reservoir_density_m3).density_derivative
                    exchange = slope * density_perturbation[node]
                current += charge_sign * exchange
            terminals.append(current)
        terminal_rows.append(terminals)
        residual = abs(item.residual[count:]).copy()
        for carrier, offset in (("n", 0), ("p", count)):
            for side, node in (("left", 0), ("right", count - 1)):
                if compiled.channel(carrier, side).kind == "fixed_reservoir":
                    residual[offset + node] = 0.0
        bounds.append((Q * float(np.sum(residual)) + omega * float(np.sum(abs(item.residual[1:count - 1])))) / abs(voltage))
    conduction = np.asarray(conduction_rows)
    terminal_conduction = np.asarray(terminal_rows)
    omegas = 2.0 * math.pi * np.asarray([item.frequency_Hz for item in solved])
    displacement_current = 1j * omegas[:, None] * displacement
    terminal_displacement = 1j * omegas[:, None] * terminal_D
    total = conduction + displacement_current
    terminal_total = terminal_conduction + terminal_displacement
    all_faces = np.c_[terminal_total[:, :1], total, terminal_total[:, 1:]] / voltage
    spread = np.max(abs(all_faces - all_faces[:, :1]), axis=1)
    scale = np.max(abs(all_faces), axis=1)
    if np.any(spread > current_absolute_tolerance_A_m2_V + current_relative_tolerance * scale):
        raise ArithmeticError(f"small-signal Maxwell-current closure failed: {float(np.max(spread)):g} A/m^2/V")
    positive_electrode = 0 if compiled.voltage_polarity > 0.0 else 1
    input_current = compiled.voltage_polarity * terminal_total[:, positive_electrode]
    if _sources(field_transport=compiled.field_transport is not None) != source_pins:
        raise RuntimeError("small-signal source changed during the solve")
    digest = state_digest(u)
    identity = hashlib.sha256(json.dumps((
        compiled.identity, float(operating_point.voltage_V).hex(), float(operating_point.generation_scale).hex(), digest,
    )).encode()).hexdigest()
    source_identity = hashlib.sha256(json.dumps(source_pins, sort_keys=True).encode()).hexdigest()
    return SmallSignalResult1D(
        np.asarray([item.frequency_Hz for item in solved]), voltage, compiled.nodes_m,
        fixed_x, moving_x, c, phi, field, dn, dp, conduction, displacement_current,
        total, terminal_conduction, terminal_displacement, terminal_total, input_current,
        input_current / voltage, spread, np.asarray(bounds), solved, compiled.identity,
        identity, digest, operating_point.voltage_V, operating_point.generation_scale,
        compiled.voltage_polarity, source_identity, source_pins, asdict(controls), {
            **asdict(linear), "fixed_coordinate_indices": sorted(fixed_coordinates),
            "carrier_reference_anchor_coordinates": anchors,
            "coordinate_form": "boundary-voltage lifted moving reference; full original residual verified",
            "current_absolute_tolerance_A_m2_V": current_absolute_tolerance_A_m2_V,
            "current_relative_tolerance": current_relative_tolerance,
        },
    )
