"""Continuous open-circuit evolution for the production 1D density equations.

The terminal voltage is an unknown alongside the densities. Differentiating
the same discrete Poisson equation gives the instantaneous displacement
current, so all terms in the Maxwell current have the same time argument.
Independent quadrature of conduction current checks the accepted trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from perovskite_sim.constants import Q
from perovskite_sim.experiments.jv_sweep import (
    _state_fields,
    compute_current_components,
)
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.physics.poisson import solve_poisson_prefactored
from perovskite_sim.physics.recombination import _observe_srh_denominators
from perovskite_sim.solver.mol import MaterialArrays, StateVec, assemble_rhs
from perovskite_sim.solver.numerical_diagnostics import (
    NumericalDiagnosticsMonitor,
    NumericalDiagnosticsPolicy,
    NumericalDiagnosticsReport,
    StateLayout,
)


class OpenCircuitError(RuntimeError):
    """No accepted open-circuit trajectory exists for the requested interval."""


@dataclass(frozen=True)
class OpenCircuitTrace:
    t: np.ndarray
    y: np.ndarray
    V: np.ndarray
    J: np.ndarray
    max_face_current_A_m2: np.ndarray
    interval_current_residual_A_m2: np.ndarray
    charge_voltage_error_V: np.ndarray
    quadrature_error_C_m2: np.ndarray
    valid: np.ndarray
    numerical_diagnostics: NumericalDiagnosticsReport
    nfev: int


class OpenCircuitSystem:
    """Density MoL plus the voltage rate required by zero external current."""

    def __init__(self, x: np.ndarray, stack: DeviceStack, mat: MaterialArrays):
        self.x = np.asarray(x, dtype=float)
        self.stack = stack
        self.mat = mat
        if mat.N_iface_state:
            raise OpenCircuitError(
                "open-circuit evolution requires a closed Poisson charge model; "
                "the interface-plane density-state charge is not implemented"
            )
        phi_voltage = solve_poisson_prefactored(
            mat.poisson_factor, np.zeros_like(self.x),
            phi_left=0.0, phi_right=-float(mat.junction_polarity),
        )
        self.displacement_voltage_derivative = self._displacement(phi_voltage)
        self.capacitance_A = float(-self.displacement_voltage_derivative[0])
        if not np.isfinite(self.capacitance_A) or self.capacitance_A <= 0.0:
            raise OpenCircuitError("the geometric capacitance must be positive")
        self.layout = StateLayout(
            n_nodes=self.x.size,
            has_dual_ions=mat.has_dual_ions,
            positive_ion_active=tuple(np.asarray(mat.P_ion0) > 0.0),
            negative_ion_active=(
                tuple(np.asarray(mat.P_ion0_neg) > 0.0)
                if mat.has_dual_ions else ()
            ),
        )

    def _displacement(self, potential: np.ndarray) -> np.ndarray:
        # Solar current convention: -polarity * epsilon * E.
        return self.mat.junction_polarity * self.mat.poisson_factor.C * np.diff(potential)

    def canonical_state(self, y: np.ndarray, voltage: float) -> np.ndarray:
        self.layout.split(y)
        n, p, _, state = _state_fields(
            self.x, y, self.stack, voltage, self.mat,
            phi_frozen=np.zeros_like(self.x),
        )
        return StateVec.pack(n, p, state.P, state.P_neg)

    def conduction_current(self, y: np.ndarray, voltage: float) -> np.ndarray:
        return compute_current_components(
            self.x, y, self.stack, voltage, mat=self.mat,
        ).J_total

    def displacement_change(
        self, y_before: np.ndarray, voltage_before: float,
        y_after: np.ndarray, voltage_after: float,
    ) -> np.ndarray:
        before = self.canonical_state(y_before, voltage_before)
        after = self.canonical_state(y_after, voltage_after)
        delta = StateVec.unpack(after - before, self.x.size)
        rho_change = Q * (delta.p - delta.n + delta.P)
        if delta.P_neg is not None:
            rho_change -= Q * delta.P_neg
        phi_change = solve_poisson_prefactored(
            self.mat.poisson_factor, rho_change, phi_left=0.0,
            phi_right=-self.mat.junction_polarity * (voltage_after - voltage_before),
        )
        return self._displacement(phi_change)

    def rates(
        self, t: float, y: np.ndarray, voltage: float,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        dy = assemble_rhs(t, y, self.x, self.stack, self.mat, V_app=voltage)
        state_rate = StateVec.unpack(dy, self.x.size)
        rho_rate = Q * (state_rate.p - state_rate.n + state_rate.P)
        if state_rate.P_neg is not None:
            rho_rate -= Q * state_rate.P_neg
        phi_rate_at_fixed_voltage = solve_poisson_prefactored(
            self.mat.poisson_factor, rho_rate, phi_left=0.0, phi_right=0.0,
        )
        current_at_fixed_voltage = (
            self.conduction_current(y, voltage)
            + self._displacement(phi_rate_at_fixed_voltage)
        )
        voltage_rate = float(current_at_fixed_voltage[0] / self.capacitance_A)
        maxwell_current = (
            current_at_fixed_voltage
            + self.displacement_voltage_derivative * voltage_rate
        )
        if (not np.isfinite(voltage_rate) or not np.all(np.isfinite(dy))
                or not np.all(np.isfinite(maxwell_current))):
            raise OpenCircuitError(f"non-finite open-circuit derivative at t={t:.9g} s")
        return dy, voltage_rate, maxwell_current

    def validate_state(self, y: np.ndarray, voltage: float, t: float) -> None:
        blocks = self.layout.split(y)
        if not np.isfinite(voltage) or not np.all(np.isfinite(y)):
            raise OpenCircuitError(f"non-finite accepted state at t={t:.9g} s")
        if any(np.any(values < 0.0) for values in blocks.values()):
            raise OpenCircuitError(f"negative accepted density at t={t:.9g} s")
        mat = self.mat
        for name, capacity in (("P", mat.P_lim_node), ("P_neg", mat.P_lim_neg_node)):
            if name in blocks and capacity is not None:
                if np.any(blocks[name] > np.asarray(capacity)):
                    raise OpenCircuitError(f"{name} exceeds site capacity at t={t:.9g} s")
        if mat.has_dual_ions and mat.ion_steric_shared_site and mat.ion_steric_diffusion_only:
            if np.any(blocks["P"] + blocks["P_neg"] > np.asarray(mat.P_lim_node)):
                raise OpenCircuitError(f"combined ions exceed site capacity at t={t:.9g} s")


def integrate_open_circuit(
    system: OpenCircuitSystem,
    y0: np.ndarray,
    voltage0: float,
    times: np.ndarray,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    voltage_atol: float = 1e-9,
    max_step: float = np.inf,
    max_current_error_A_m2: float = 0.05,
    max_voltage_error_V: float = 1e-6,
    max_nfev: int = 100000,
) -> OpenCircuitTrace:
    """Integrate one continuous illumination phase and certify charge balance.

    ``times`` includes both endpoints. Output sampling does not change the
    integration steps. The extra unknown is the change in terminal dielectric
    charge divided by geometric capacitance, in volts. Its derivative is the
    conduction current divided by capacitance; the voltage follows from the
    same Poisson equation and the evolving space charge.
    The charge defect includes quadrature uncertainty and is converted to a
    voltage using the geometric capacitance, independently of a decay fit.
    """
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size < 2 or not np.all(np.isfinite(times)):
        raise ValueError("times must contain at least two finite endpoints")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly increasing")
    for name, value in (
        ("rtol", rtol), ("atol", atol), ("voltage_atol", voltage_atol),
        ("max_current_error_A_m2", max_current_error_A_m2),
        ("max_voltage_error_V", max_voltage_error_V),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if np.isnan(max_step) or max_step <= 0.0:
        raise ValueError("max_step must be positive")
    if isinstance(max_nfev, bool) or not isinstance(max_nfev, int) or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer")
    y0 = system.canonical_state(np.asarray(y0, dtype=float), voltage0)
    system.validate_state(y0, voltage0, times[0])
    monitor = NumericalDiagnosticsMonitor(
        system.layout, NumericalDiagnosticsPolicy(mode="research_strict"),
    )
    calls = 0

    def voltage_at(state):
        space_charge_change = system.displacement_change(
            y0, voltage0, state[:-1], voltage0,
        )[0]
        return float(voltage0 + state[-1] + space_charge_change / system.capacitance_A)

    def rhs(t, state):
        nonlocal calls
        calls += 1
        if calls > max_nfev:
            raise OpenCircuitError(f"open-circuit integration exceeded max_nfev={max_nfev}")
        physical_state = state[:-1]
        voltage = voltage_at(state)
        monitor.observe_trial_state(physical_state)
        with _observe_srh_denominators(monitor.observe_srh_denominator):
            dy = assemble_rhs(t, physical_state, system.x, system.stack, system.mat, V_app=voltage)
        monitor.observe_rhs(dy)
        charge_rate = system.conduction_current(physical_state, voltage)[0] / system.capacitance_A
        if not np.isfinite(charge_rate):
            raise OpenCircuitError(f"non-finite electrode charge derivative at t={t:.9g} s")
        return np.append(dy, charge_rate)

    def jacobian(t, state):
        # The charge coordinate starts at zero. A default finite difference
        # based on its tiny tolerance can disappear on adding the DC voltage.
        floor = np.append(np.full(y0.size, atol), system.mat.V_T_device)
        steps = np.sqrt(np.finfo(float).eps) * np.maximum(np.abs(state), floor)
        base = rhs(t, state)
        jac = np.empty((state.size, state.size))
        for j, step in enumerate(steps):
            trial = state.copy()
            trial[j] += step
            jac[:, j] = (rhs(t, trial) - base) / (trial[j] - state[j])
        return jac

    try:
        sol = solve_ivp(
            rhs, (times[0], times[-1]), np.append(y0, 0.0), method="Radau",
            rtol=rtol, atol=np.append(np.full(y0.size, atol), voltage_atol),
            max_step=max_step, dense_output=True, jac=jacobian,
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        raise OpenCircuitError(f"open-circuit integration failed: {exc}") from exc
    if not sol.success or sol.sol is None or sol.t[-1] != times[-1]:
        raise OpenCircuitError(f"open-circuit integration failed: {sol.message}")
    for t, state in zip(sol.t, sol.y.T):
        system.validate_state(state[:-1], voltage_at(state), float(t))
    sampled = sol.sol(times)
    y = sampled[:-1]
    voltage = np.array([voltage_at(state) for state in sampled.T])
    current = np.empty(times.size)
    face_max = np.empty(times.size)
    for k, t in enumerate(times):
        system.validate_state(y[:, k], voltage[k], float(t))
        _, _, faces = system.rates(float(t), y[:, k], voltage[k])
        current[k] = faces[0]
        face_max[k] = np.max(np.abs(faces))
    if np.any(face_max > max_current_error_A_m2):
        raise OpenCircuitError(
            f"instantaneous Maxwell current exceeds {max_current_error_A_m2:.6g} A/m2: "
            f"{np.max(face_max):.6g} A/m2"
        )

    # Integrate each accepted Radau interpolant independently, split also at
    # requested samples. Comparing orders detects underresolved quadrature.
    edges = np.unique(np.concatenate((sol.t, times)))
    rules = [np.polynomial.legendre.leggauss(order) for order in (3, 5)]
    conduction_integral = np.zeros((times.size - 1, system.x.size - 1))
    quadrature_error = np.zeros_like(conduction_integral)
    for left, right in zip(edges[:-1], edges[1:]):
        midpoint, half_width = (left + right) * 0.5, (right - left) * 0.5
        integrals = []
        for nodes, weights in rules:
            nodes_t = midpoint + half_width * nodes
            states = sol.sol(nodes_t)
            values = []
            for t, state in zip(nodes_t, states.T):
                v = voltage_at(state)
                physical_state = state[:-1]
                system.validate_state(physical_state, v, float(t))
                conduction = system.conduction_current(physical_state, v)
                if not np.all(np.isfinite(conduction)):
                    raise OpenCircuitError(f"non-finite conduction current at t={t:.9g} s")
                values.append(conduction)
            integrals.append(half_width * (weights @ np.asarray(values)))
        interval = min(int(np.searchsorted(times, midpoint, side="right")) - 1, times.size - 2)
        conduction_integral[interval] += integrals[1]
        quadrature_error[interval] += np.abs(integrals[1] - integrals[0])
    interval_residual = np.empty(times.size)
    interval_residual[0] = face_max[0]
    voltage_error = np.zeros(times.size)
    quad_error = np.zeros(times.size)
    for k in range(1, times.size):
        delta_d = system.displacement_change(
            y[:, k - 1], voltage[k - 1], y[:, k], voltage[k],
        )
        defect = np.abs(conduction_integral[k - 1] + delta_d) + quadrature_error[k - 1]
        bound = float(np.max(defect))
        interval_residual[k] = bound / (times[k] - times[k - 1])
        voltage_error[k] = voltage_error[k - 1] + bound / system.capacitance_A
        quad_error[k] = float(np.max(quadrature_error[k - 1]))
    if np.max(interval_residual) > max_current_error_A_m2:
        raise OpenCircuitError(
            f"integrated Maxwell current exceeds {max_current_error_A_m2:.6g} A/m2: "
            f"{np.max(interval_residual):.6g} A/m2"
        )
    if voltage_error[-1] > max_voltage_error_V:
        raise OpenCircuitError(
            f"integrated charge error corresponds to {voltage_error[-1]:.6g} V; "
            f"limit is {max_voltage_error_V:.6g} V"
        )
    report = monitor.finalize(y[:, -1], solver_success=True)
    return OpenCircuitTrace(
        t=times.copy(), y=y, V=voltage, J=current,
        max_face_current_A_m2=face_max,
        interval_current_residual_A_m2=interval_residual,
        charge_voltage_error_V=voltage_error,
        quadrature_error_C_m2=quad_error,
        valid=np.ones(times.size, dtype=bool),
        numerical_diagnostics=report, nfev=calls,
    )
