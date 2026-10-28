"""Test-local DC device problems for the nonlinear layer (not a production binding).

The P04-03 DC scaffold (``tests/unit/physics/transport/p0403_dc_scaffold.py``,
read only) supplies the residual and analytic Jacobian of a p-HTL / absorber /
n-ETL device in the test's own coordinates ``(phi, ln n, ln p)``. This module
adds what the solver layer needs from a caller -- row weights, step limits,
staged preparation and physical gates -- without choosing anything inside the
solver. Gates measure every face (maximum or spread), never a median.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from solarlab.physics.contacts import ContactEndpoint, assess_contact_thermodynamics
from solarlab.physics.transport.electrochemical_flux import carrier_particle_fluxes
from solarlab.solvers.nonlinear import legacy_normalized_poisson_residual
from solarlab.solvers.nonlinear.constraints import DcPortProblem
from solarlab.solvers.nonlinear.newton import NewtonSettings, gate_from
from solarlab.solvers.nonlinear.problem import (
    CallableProblem, GateOutcome, ResidualScaling, SolveBudget, StepControl,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "physics" / "transport"))
import p0403_dc_scaffold as scaffold  # noqa: E402

Q = scaffold.Q
POTENTIAL_STEP_V = 0.1
LOG_DENSITY_STEP = 2.0


def polarity(device):
    return -1.0 if device.V_bi_bc < 0.0 else 1.0


# ---- stage A: Poisson-Boltzmann equilibrium on phi --------------------------------------

def boltzmann_problem(device):
    def charge_and_slope(phi):
        n, p = scaffold.equilibrium_densities(device, phi)
        return Q * (p - n + device.N_D - device.N_A), -Q * (n + p) / device.V_T

    def residual(phi):
        rho, _ = charge_and_slope(phi)
        return device.poisson.residual(phi, rho, 0.0, device.V_bi_bc)

    def jacobian(phi):
        _, slope = charge_and_slope(phi)
        matrix = device.poisson.jacobian().toarray()
        inner = np.arange(1, device.size - 1)
        matrix[inner, inner] += device.poisson.interior_widths_m * slope[1:-1]
        return matrix
    return CallableProblem(residual, jacobian)


def boltzmann_weights(device):
    """The legacy normalized-Poisson scale (interior) and 1/V_ref (contact rows)."""
    size = device.size
    reference = max(abs(device.V_bi_bc), device.V_T)
    probe = np.zeros(size - 2)
    weights = np.empty(size)
    for i in range(size - 2):      # 1 / scale_i, recovered row by row from the M criterion
        probe[:] = 0.0
        probe[i] = 1.0
        weights[i + 1] = legacy_normalized_poisson_residual(
            probe, face_capacitance_F_m2=device.poisson.face_capacitance_F_m2,
            interior_widths_m=device.poisson.interior_widths_m,
            acceptor_density_m3=device.N_A, donor_density_m3=device.N_D,
            V_bi_bc=device.V_bi_bc, thermal_voltage_V=device.V_T)
    weights[0] = weights[-1] = 1.0 / reference
    return weights


def boltzmann_settings(device, *, iterations=80):
    size = device.size
    return NewtonSettings(ResidualScaling(boltzmann_weights(device)), 1e-12,
                          StepControl(np.full(size, POTENTIAL_STEP_V)),
                          SolveBudget(iterations, 20_000, 2_000, 30), (), False)


def linear_potential(device, V=0.0):
    x = device.x
    return (device.V_bi_bc - polarity(device) * V) * (x - x[0]) / (x[-1] - x[0])


def boltzmann_state(device, phi):
    n, p = scaffold.equilibrium_densities(device, phi)
    return np.r_[phi, np.log(n), np.log(p)]


# ---- stage B: the coupled DC system ---------------------------------------------------------

def dc_problem(device, V=0.0, generation_scale=0.0):
    size = device.size

    def evaluate(u):
        return scaffold.residual_and_jacobian(device, scaffold.DcState.from_vector(u, size),
                                              V_app=V, generation_scale=generation_scale)
    return CallableProblem(lambda u: evaluate(u)[0], lambda u: evaluate(u)[1])


def flux_scale(device):
    """``k_max c_ref`` [m^-2 s^-1]: the rounding scale of every carrier row."""
    k = max(np.max(device.D_n_face), np.max(device.D_p_face)) / np.min(np.diff(device.x))
    return k * max(device.n_eq + device.p_eq)


def current_scale(device):
    return Q * flux_scale(device)


def dc_weights(device):
    size = device.size
    weights = np.ones(3 * size)
    weights[:size] = boltzmann_weights(device)
    weights[size:] = 1.0 / flux_scale(device)
    for carrier, offset in (("n", size), ("p", 2 * size)):
        for side, node in (("left", 0), ("right", size - 1)):
            if device.channels[carrier + "_" + side].kind == "fixed_reservoir":
                weights[offset + node] = 1.0          # ln c - ln c_eq is dimensionless
    return weights


def dc_settings(device, *, gates=(), iterations=60, tolerance=1e-10, backtracks=30):
    size = device.size
    step = np.r_[np.full(size, POTENTIAL_STEP_V), np.full(2 * size, LOG_DENSITY_STEP)]
    return NewtonSettings(ResidualScaling(dc_weights(device)), tolerance, StepControl(step),
                          SolveBudget(iterations, 20_000, 2_000, backtracks), tuple(gates),
                          False)


def state_of(device, u):
    return scaffold.DcState.from_vector(np.asarray(u, dtype=float), device.size)


def face_currents(device, u):
    return scaffold.terminal_current_density(device, state_of(device, u))


# ---- physical gates (every face) --------------------------------------------------------------

def zero_current_gate(device, relative=1e-12):
    """Dark equilibrium: max over faces |J_n + J_p| within rounding of the current scale."""
    return gate_from("zero_equilibrium_current",
                     lambda u: float(np.max(np.abs(face_currents(device, u))))
                     / current_scale(device), relative, "every face, no median")


def uniform_current_gate(device, relative=1e-10):
    """DC conservation: the spread of J over all faces, relative to the current scale."""
    return gate_from("face_current_spread",
                     lambda u: float(np.ptp(face_currents(device, u))) / current_scale(device),
                     relative, "max - min over every face")


def generation_limit_gate(device, slack=1e-9):
    """|J| on every face cannot exceed q * integral(G): collected pairs <= generated pairs."""
    return gate_from("generation_limit",
                     lambda u: float(np.max(np.abs(face_currents(device, u))))
                     / device.generation_limit_A_m2, 1.0 + slack, "every face")


def contact_certificate_gate(device):
    def endpoint(node, n, p):
        return ContactEndpoint(n, p, float(device.Nc[node]), float(device.Nv[node]),
                               float(device.bands.physical_electron_affinity_eV[node]),
                               float(device.bands.physical_band_gap_eV[node]))

    def gate(u):
        state = state_of(device, u)
        assessment = assess_contact_thermodynamics(
            built_in_potential_mode="semiconductor_work_function",
            left=endpoint(0, float(state.n[0]), float(state.p[0])),
            right=endpoint(-1, float(state.n[-1]), float(state.p[-1])),
            V_bi_bc=float(state.phi[-1] - state.phi[0]), thermal_voltage_V=device.V_T,
            has_band_reference=True)
        span = assessment.fermi_level_span_eV
        return GateOutcome("contact_thermodynamics",
                           assessment.certified, float("inf") if span is None else span,
                           assessment.tolerance_eV, assessment.status)
    return gate


# ---- direct V_oc: the DC terminal current at the left contact face ------------------------------

def direct_voc_problem(device, generation_scale=1.0):
    size = device.size

    def fluxes(u):
        state = state_of(device, u)
        return state, carrier_particle_fluxes(device.x, state.phi, state.n, state.p,
                                              device.D_n_face, device.D_p_face, device.V_T,
                                              device.bands)

    def current(u, V):
        _, f = fluxes(u)
        return float(-Q * f.electron.flux[0] + Q * f.hole.flux[0])

    def gradient(u, V):
        state, f = fluxes(u)
        grad = np.zeros(3 * size)
        for sign, flux, offset, density in ((-Q, f.electron, size, state.n),
                                            (Q, f.hole, 2 * size, state.p)):
            grad[0] += sign * flux.left_potential_derivative[0]
            grad[1] += sign * flux.right_potential_derivative[0]
            grad[offset] += sign * flux.left_density_derivative[0] * density[0]
            grad[offset + 1] += sign * flux.right_density_derivative[0] * density[1]
        return grad

    def dresidual_dV(u, V):
        column = np.zeros(3 * size)
        column[size - 1] = polarity(device)        # phi[-1] - (V_bi_bc - polarity * V)
        return column

    return DcPortProblem(lambda V: dc_problem(device, V, generation_scale), dresidual_dV,
                         current, gradient, lambda u, V: 0.0,
                         current_scale=device.generation_limit_A_m2)
