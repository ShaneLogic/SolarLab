"""Ideal-step preparation and regular currents for the explicit R1-1 lane.

An ideal voltage step has a finite electrode charge impulse.  The state on
its right has unchanged dynamic populations but a new algebraic solution;
its regular current includes the right derivative of the displacement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from perovskite_sim.constants import EPS_0, Q
from perovskite_sim.experiments.interface_defect_ion_transient import (
    InterfaceDefectIonTransientError,
)
from perovskite_sim.physics.physical_control_volume import physical_contact_displacement
from perovskite_sim.physics.poisson import solve_poisson_prefactored
from perovskite_sim.physics.recombination import total_recombination
from perovskite_sim.solver.mol import poisson_right_boundary


_GAUSS_CHARGE_SCALE = Q * 1e15
_GAUSS_LIMIT = 1e-10


@dataclass(frozen=True)
class CurrentAtState:
    """Regular right-limit currents; arrays in metrics use legacy polarity."""

    metrics: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]
    evidence: dict[str, Any]


@dataclass(frozen=True)
class InitialStepResult:
    system: Any
    zero_plus: Any
    initial_current_metrics: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float
    ]
    event: dict[str, Any]


def _fail(message: str, evidence: dict[str, Any] | None = None) -> None:
    error = InterfaceDefectIonTransientError(message)
    error.result = evidence
    raise error


def _json_array(value: Any) -> Any:
    return None if value is None else np.asarray(value).tolist()


def _electrostatics(system, state) -> dict[str, Any]:
    mat, widths = system.material, system.widths
    rho, _ = system.system._bulk_space_charge_and_tangent(
        state.n, state.p, positive_ion_density_m3=state.positive,
        negative_ion_density_m3=state.negative,
    )
    face_d = -mat.poisson_factor.C * np.diff(state.phi)
    contact_d = physical_contact_displacement(face_d, rho, widths)
    full_charge = float(np.dot(rho, widths) + np.sum(state.sheet_charge))
    gauss = float(contact_d[1] - contact_d[0] - full_charge)
    # This direct residual is independent of the incremental voltage lift.
    poisson = np.diff(mat.poisson_factor.C * np.diff(state.phi))
    poisson += rho[1:-1] * mat.poisson_factor.h_cell
    for k, (left, right) in enumerate(zip(system.left_nodes, system.right_nodes)):
        wl, wr = system._sheet_weights(k)
        poisson[left - 1] += wl * state.sheet_charge[k]
        poisson[right - 1] += wr * state.sheet_charge[k]
    return {
        "rho_C_m3": _json_array(rho),
        "physical_displacement_C_m2": _json_array(contact_d),
        "electrode_charge_C_m2": _json_array(contact_d * [1., -1.]),
        "full_charge_C_m2": full_charge,
        "gauss_error_C_m2": gauss,
        "gauss_normalized": abs(gauss) / _GAUSS_CHARGE_SCALE,
        "direct_poisson_residual_C_m2": _json_array(poisson),
    }


def _state_record(system, state, voltage: float, label: str) -> dict[str, Any]:
    record = {
        "label": label, "time_s": 0., "voltage_V": float(voltage),
        "electron_density_m3": _json_array(state.n),
        "hole_density_m3": _json_array(state.p),
        "positive_ion_density_m3": _json_array(state.positive),
        "negative_ion_density_m3": _json_array(state.negative),
        "interface_occupancy": _json_array(state.occupancy),
        "interface_reference_occupancy": _json_array(system.equilibrium_occupancy),
        "electrostatic_potential_V": _json_array(state.phi),
        "interface_trace_potential_V": _json_array([v.trace_potential for v in state.local]),
        "interface_trace_density_m3": _json_array([v.state_m3 for v in state.local]),
        "interface_sheet_charge_C_m2": _json_array(state.sheet_charge),
        "local_residual": _json_array(state.local_residual),
        "positive_ion_component_inventory_m2": [
            float(np.dot(state.positive[np.asarray(nodes)], system.widths[np.asarray(nodes)]))
            for nodes in system.ion_layout.positive_components
        ],
    }
    record.update(_electrostatics(system, state))
    return record


def _check_algebraic(system, state, policy) -> dict[str, float]:
    evidence = _electrostatics(system, state)
    carrier, gauss = system.local_normalized_residuals(state)
    poisson = np.asarray(evidence["direct_poisson_residual_C_m2"])
    factor = system.material.poisson_factor
    poisson_normalized = float(np.max(
        np.abs(poisson) / ((factor.C[:-1] + factor.C[1:]) * system.thermal_voltage)
    ))
    trace_error = max(
        (abs(float(v.electrostatic_residual[0])) for v in state.local), default=0.,
    )
    result = {
        "direct_poisson_normalized": poisson_normalized,
        "local_carrier_normalized": float(carrier),
        "local_gauss_normalized": float(gauss),
        "trace_potential_error_V": trace_error,
        "full_gauss_normalized": evidence["gauss_normalized"],
    }
    limits = {
        "direct_poisson_normalized": policy.maximum_dc_poisson_residual,
        "local_carrier_normalized": policy.maximum_local_carrier_normalized_residual,
        "local_gauss_normalized": policy.maximum_local_gauss_normalized_residual,
        "trace_potential_error_V": (
            policy.interface_potential_atol_V
            + policy.interface_algebraic_relative_tolerance * system.thermal_voltage
        ),
        "full_gauss_normalized": _GAUSS_LIMIT,
    }
    if any(not np.isfinite(value) or value > limits[key] for key, value in result.items()):
        _fail("R1-1 initial algebraic state failed its original gates", {
            "values": result, "limits": limits,
        })
    return result


def _fixed_population_electrostatic_correction(system, voltage: float):
    """Solve the inherited algebraic defect in small potential increments.

    A certified DC field still has a finite Poisson/Gauss residual. Leaving
    that defect to the first time step converts it into a spurious current
    proportional to 1/dt. Here it is actually solved at fixed populations,
    retaining the previous residual and an independent absolute-field check.
    """
    coordinate = np.zeros(system.dimension)
    before = system.evaluate(coordinate, voltage)
    factor = system.material.poisson_factor
    source = np.zeros(system.node_count)
    source[1:-1] = before.poisson_residual / factor.h_cell
    delta_phi = solve_poisson_prefactored(factor, source, 0., 0.)
    reduced = delta_phi[1:-1] / system.thermal_voltage
    coordinate[system.potential_slice] = reduced
    coordinate[system.electron_slice] = -reduced
    coordinate[system.hole_slice] = reduced
    delta_trace = np.empty((system.interface_count, 2))
    for k, (left, right) in enumerate(zip(system.left_nodes, system.right_nodes)):
        mat = system.material
        cl = EPS_0 * mat.eps_r[left] / mat.iface_qss_left_distances_m[k]
        cr = EPS_0 * mat.eps_r[right] / mat.iface_qss_right_distances_m[k]
        jump_residual, gauss_residual = before.local_residual[6*k:6*k+2]
        trace_left = (
            cl * delta_phi[left] + cr * delta_phi[right]
            - gauss_residual + cr * jump_residual
        ) / (cl + cr)
        delta_trace[k] = [trace_left, trace_left - jump_residual]
        block = system._local_block_slice(k)
        coordinate[block.start:block.start+2] = delta_trace[k] / system.thermal_voltage
    after = system.evaluate(coordinate, voltage)
    # These are computed residual sums, never a zero assignment. Save both
    # representations because rounding an absolute potential can hide the
    # correction even while its incremental representation is resolved.
    evidence = {
        "potential_correction_V": _json_array(delta_phi),
        "trace_potential_correction_V": _json_array(delta_trace),
        "poisson_residual_before_C_m2": _json_array(before.poisson_residual),
        "poisson_residual_after_C_m2": _json_array(after.poisson_residual),
        "linear_poisson_residual_C_m2": _json_array(
            before.poisson_residual + np.diff(factor.C * np.diff(delta_phi))
        ),
        "local_electrostatic_residual_before": _json_array(
            np.asarray(before.local_residual).reshape(-1, 6)[:, :2]
        ),
        "local_electrostatic_residual_after": _json_array(
            np.asarray(after.local_residual).reshape(-1, 6)[:, :2]
        ),
        "direct_poisson_residual_after_C_m2": _json_array(after.direct_poisson_residual),
        "maximum_potential_correction_V": float(np.max(np.abs(delta_phi), initial=0.)),
        "representation": "fixed-population potential increments with inherited residual retained",
    }
    return coordinate, evidence


def _solve_local_carriers(system, voltage: float, policy, coordinate=None):
    """Keep dynamic and electrostatic coordinates fixed; solve local carriers."""
    coordinate = np.zeros(system.dimension) if coordinate is None else np.asarray(coordinate).copy()
    rows = np.asarray([
        6 * k + j for k in range(system.interface_count) for j in range(2, 6)
    ], dtype=int)
    columns = np.asarray([
        system._local_block_slice(k).start + j
        for k in range(system.interface_count) for j in range(2, 6)
    ], dtype=int)
    scale = system.local_algebraic_scale(policy)[rows]
    maximum = policy.maximum_scaled_nonlinear_residual
    for iteration in range(policy.maximum_newton_iterations + 1):
        state = system.evaluate(coordinate, voltage)
        residual = state.local_residual[rows] / scale
        norm = float(np.max(np.abs(residual), initial=0.))
        if norm <= maximum:
            return state, iteration, norm
        if iteration == policy.maximum_newton_iterations:
            break
        jacobian = state.local_jacobian[rows][:, columns].toarray() / scale[:, None]
        try:
            delta = np.linalg.solve(jacobian, -residual)
        except np.linalg.LinAlgError as exc:
            _fail(f"R1-1 fixed-population local solve is singular: {exc}")
        if not np.all(np.isfinite(delta)):
            _fail("R1-1 fixed-population local solve produced a nonfinite step")
        damping = min(1., 2. / max(float(np.max(np.abs(delta))), 2.))
        for _ in range(policy.maximum_line_search_steps):
            candidate = coordinate.copy()
            candidate[columns] += damping * delta
            try:
                trial = system.evaluate(candidate, voltage)
                trial_norm = float(np.max(np.abs(trial.local_residual[rows] / scale), initial=0.))
            except (InterfaceDefectIonTransientError, RuntimeError, ValueError, FloatingPointError):
                trial_norm = np.inf
            if np.isfinite(trial_norm) and (trial_norm <= maximum or trial_norm < norm):
                coordinate = candidate
                break
            damping *= .5
        else:
            _fail("R1-1 fixed-population local line search stalled", {
                "iterations": iteration, "scaled_residual": norm,
            })
    _fail("R1-1 fixed-population local iteration limit", {
        "iterations": policy.maximum_newton_iterations, "scaled_residual": norm,
    })


def effective_regular_limits(policy, *, require_relative_closure=True):
    """Return the actual original regular-current conservation gate limits."""
    limits = {"charge_balance_normalized": policy.maximum_charge_balance_relative_error,
              "differentiated_poisson_normalized": _GAUSS_LIMIT}
    if require_relative_closure:
        limits.update({"face_current_spread_relative": policy.maximum_all_face_current_spread_relative,
            "interface_current_spread_relative": policy.maximum_two_sided_interface_total_current_relative_error,
            "contact_internal_current_spread_relative": policy.maximum_all_face_current_spread_relative})
    else:
        limits.update({"electron_continuity_A_m2": policy.maximum_dc_continuity_bound_A_m2,
            "hole_continuity_A_m2": policy.maximum_dc_continuity_bound_A_m2,
            "electron_normalized_residual": policy.maximum_dc_normalized_residual,
            "hole_normalized_residual": policy.maximum_dc_normalized_residual,
            "trap_charge_rate_A_m2": policy.maximum_dc_continuity_bound_A_m2,
            "absolute_current_spread_A_m2": policy.maximum_dc_face_current_spread_A_m2,
            "ionic_face_current_A_m2": policy.maximum_dc_ionic_face_current_A_m2})
    return limits


def regular_current_at_state(system, state, *, policy, require_relative_closure=True) -> CurrentAtState:
    """Evaluate dD/dt from the differentiated Poisson equation at fixed voltage.

    This is the regular right limit at a step, or the instantaneous regular
    current at any positive-time state. It contains no voltage impulse.

    An exact zero-excitation equilibrium can request the original absolute
    DC checks instead. Its measured relative errors remain visible and are
    explicitly not certified as finite-signal current closure.
    """
    mat, widths = system.material, system.widths
    rho_dot = system._increment_charge_density(state.rate)
    start = 2 * system.interior_count
    sigma_dot = -Q * state.rate[start:start + system.interface_count]
    effective_rho_dot = rho_dot.copy()
    for k, (left, right) in enumerate(zip(system.left_nodes, system.right_nodes)):
        wl, wr = system._sheet_weights(k)
        effective_rho_dot[left] += wl * sigma_dot[k] / widths[left]
        effective_rho_dot[right] += wr * sigma_dot[k] / widths[right]
    phi_dot = solve_poisson_prefactored(mat.poisson_factor, effective_rho_dot, 0., 0.)
    face_d_dot = -mat.poisson_factor.C * np.diff(phi_dot)
    contact_d_dot = physical_contact_displacement(face_d_dot, rho_dot, widths)
    interface_d_dot = np.empty((system.interface_count, 2))
    trace_phi_dot = np.empty((system.interface_count, 2))
    for k, (left, right) in enumerate(zip(system.left_nodes, system.right_nodes)):
        cl = EPS_0 * mat.eps_r[left] / mat.iface_qss_left_distances_m[k]
        cr = EPS_0 * mat.eps_r[right] / mat.iface_qss_right_distances_m[k]
        trace = (cl * phi_dot[left] + cr * phi_dot[right] + sigma_dot[k]) / (cl + cr)
        trace_phi_dot[k] = trace
        interface_d_dot[k] = [-cl * (trace - phi_dot[left]), cr * (trace - phi_dot[right])]
        face_d_dot[system.interface_faces[k]] = interface_d_dot[k, 0]
    interface_conduction, _, _ = system.interface_current_sides(state)
    displacement = system.polarity * face_d_dot
    interface_displacement = system.polarity * interface_d_dot
    total = state.conduction + displacement
    interface_total = interface_conduction + interface_displacement
    recombination = total_recombination(
        state.n, state.p, mat.ni_sq, mat.tau_n, mat.tau_p,
        mat.n1, mat.p1, mat.B_rad, mat.C_n, mat.C_p,
    )
    contact_n = state.current_n[[0, -1]] + Q * recombination[[0, -1]] * widths[[0, -1]] * [-1., 1.]
    contact_p = state.current_p[[0, -1]] + Q * recombination[[0, -1]] * widths[[0, -1]] * [1., -1.]
    contact_conduction = contact_n + contact_p
    contact_total = contact_conduction + contact_d_dot
    all_total = np.r_[total / system.polarity, interface_total.ravel() / system.polarity, contact_total]
    scale = max(float(np.max(np.abs(all_total), initial=0.)), 1e-20)
    face_error = float(np.ptp(total)) / max(float(np.max(np.abs(total), initial=0.)), 1e-20)
    interface_error = float(np.max(
        np.abs(interface_total[:, 0] - interface_total[:, 1])
        / np.maximum(np.max(np.abs(interface_total), axis=1), 1e-20), initial=0.,
    ))
    contact_error = float(np.ptp(all_total)) / scale
    charge_rate = float(np.dot(rho_dot, widths) + np.sum(sigma_dot))
    charge_error = charge_rate - float(contact_conduction[0] - contact_conduction[1])
    charge_scale = max(abs(charge_rate), abs(float(contact_conduction[0] - contact_conduction[1])),
                       float(np.max(np.abs(state.conduction), initial=0.)), 1.)
    poisson_dot = np.diff(mat.poisson_factor.C * np.diff(phi_dot)) + effective_rho_dot[1:-1] * mat.poisson_factor.h_cell
    poisson_scale = max(float(np.max(np.abs(effective_rho_dot[1:-1] * widths[1:-1]), initial=0.)),
                        float(np.max(np.abs(mat.poisson_factor.C * np.diff(phi_dot)), initial=0.)), 1.)
    evidence = {
        "kind": "regular_current_at_fixed_voltage",
        "current_direction": "+x; report current is junction_polarity times +x",
        "potential_derivative_V_s": _json_array(phi_dot),
        "trace_potential_derivative_V_s": _json_array(trace_phi_dot),
        "charge_density_derivative_C_m3_s": _json_array(rho_dot),
        "sheet_charge_derivative_C_m2_s": _json_array(sigma_dot),
        "contact_electron_current_A_m2": _json_array(contact_n),
        "contact_hole_current_A_m2": _json_array(contact_p),
        "contact_ion_current_A_m2": [0., 0.],
        "contact_conduction_A_m2": _json_array(contact_conduction),
        "contact_displacement_A_m2": _json_array(contact_d_dot),
        "contact_maxwell_A_m2": _json_array(contact_total),
        "report_contact_current_A_m2": _json_array(system.polarity * contact_total),
        "internal_maxwell_A_m2": _json_array(total / system.polarity),
        "interface_maxwell_A_m2": _json_array(interface_total / system.polarity),
        "charge_rate_A_m2": charge_rate,
        "charge_balance_error_A_m2": charge_error,
        "charge_balance_normalized": abs(charge_error) / charge_scale,
        "contact_internal_current_spread_relative": contact_error,
        "face_current_spread_relative": face_error,
        "interface_current_spread_relative": interface_error,
        "differentiated_poisson_residual_A_m2": _json_array(poisson_dot),
        "differentiated_poisson_normalized": float(np.max(np.abs(poisson_dot), initial=0.)) / poisson_scale,
        "relative_current_certified": bool(require_relative_closure),
        "relative_current_status": (
            "certified" if require_relative_closure
            else "relative_current_not_certified_zero_excitation"
        ),
    }
    limits = effective_regular_limits(policy, require_relative_closure=require_relative_closure)
    evidence["limits"] = limits
    gate_values = evidence.copy()
    if not require_relative_closure:
        count = system.interior_count
        equilibrium = {
            "electron_continuity_A_m2": float(np.sum(np.abs(Q * state.rate[:count] * widths[1:-1]))),
            "hole_continuity_A_m2": float(np.sum(np.abs(Q * state.rate[count:2*count] * widths[1:-1]))),
            "electron_normalized_residual": float(np.max(np.abs(Q * state.rate[:count] * widths[1:-1]))) / system.system.current_scale,
            "hole_normalized_residual": float(np.max(np.abs(Q * state.rate[count:2*count] * widths[1:-1]))) / system.system.current_scale,
            "trap_charge_rate_A_m2": float(np.max(np.abs(sigma_dot), initial=0.)),
            "absolute_current_spread_A_m2": float(np.ptp(all_total)),
            "ionic_face_current_A_m2": float(np.max(np.abs(state.positive_current), initial=0.)),
        }
        evidence["zero_excitation_absolute_checks"] = equilibrium
        gate_values.update(equilibrium)
        evidence["algebraic_certificate"] = _check_algebraic(system, state, policy)
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_standard import (
        verify_execution_policy, verify_used_execution_limits,
    )
    verify_execution_policy(policy)
    verify_used_execution_limits("published_regular_relative" if require_relative_closure
                                 else "published_regular_zero_excitation", limits)
    gates = [(gate_values[key], limit) for key, limit in limits.items()]
    arrays = (phi_dot, total, interface_total, contact_total, state.rate)
    if any(not np.all(np.isfinite(v)) for v in arrays) or any(not np.isfinite(v) or v > limit for v, limit in gates):
        _fail("R1-1 regular current failed its original conservation gates", evidence)
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_independent_regular import independent_regular_current
    independent = independent_regular_current(system, state, policy=policy,
        require_relative_closure=require_relative_closure, reported=evidence)
    evidence["independent_physics"] = independent
    if not independent["passed"]:
        _fail("R1-1 regular current failed independent reconstruction: " + ", ".join(independent["reasons"]), evidence)
    return CurrentAtState(
        (displacement, total, interface_conduction, interface_displacement, face_error, interface_error),
        evidence,
    )


def build_initial_step(system, zero_minus_state, voltage_after: float, *, policy) -> InitialStepResult:
    """Apply the ideal voltage step without evolving any dynamic population."""
    voltage_after = float(voltage_after)
    if not np.isfinite(voltage_after):
        raise ValueError("R1-1 step voltage must be finite")
    if system.material.physical_cell_faces_m is None:
        raise ValueError("R1-1 initial step requires physical control volumes")
    before = zero_minus_state
    voltage_before = float((poisson_right_boundary(system.material, 0.) - before.phi[-1]) / system.polarity)
    working, local_before = system.rebase(before)
    working.set_voltage_lift(voltage_after, local_before)
    coordinate, correction = _fixed_population_electrostatic_correction(working, voltage_after)
    zero_plus, iterations, norm = _solve_local_carriers(working, voltage_after, policy, coordinate)
    for field in ("n", "p", "positive", "negative", "occupancy", "sheet_charge"):
        old, new = getattr(before, field), getattr(zero_plus, field)
        if (old is None) != (new is None) or (old is not None and not np.array_equal(old, new)):
            _fail(f"R1-1 ideal step changed dynamic population or charge: {field}")
    minus_record = _state_record(system, before, voltage_before, "0-")
    plus_record = _state_record(working, zero_plus, voltage_after, "0+")
    algebraic = _check_algebraic(working, zero_plus, policy)
    delta_d = np.asarray(plus_record["physical_displacement_C_m2"]) - minus_record["physical_displacement_C_m2"]
    capacitance = float(1. / np.sum(1. / system.material.poisson_factor.C))
    delta_voltage = voltage_after - voltage_before
    expected_impulse = capacitance * delta_voltage
    observed_impulse = system.polarity * delta_d
    error = float(np.max(np.abs(observed_impulse - expected_impulse)))
    charge_error = float(plus_record["full_charge_C_m2"] - minus_record["full_charge_C_m2"])
    if error > _GAUSS_CHARGE_SCALE * _GAUSS_LIMIT or abs(charge_error) > _GAUSS_CHARGE_SCALE * _GAUSS_LIMIT:
        _fail("R1-1 ideal-step charge failed the dielectric/Gauss oracle", {
            "impulse_error_C_m2": error, "full_charge_jump_C_m2": charge_error,
        })
    # Rebase once more and remove the old voltage lift. Otherwise the zero
    # coordinate would apply the voltage jump for a second time on integration.
    integration_system, _ = working.rebase(zero_plus)
    integration_system.set_voltage_lift(voltage_after, zero_plus)
    integration_state = integration_system.evaluate(np.zeros(system.dimension), voltage_after)
    for field in ("n", "p", "positive", "negative", "occupancy", "phi", "sheet_charge"):
        old, new = getattr(zero_plus, field), getattr(integration_state, field)
        if (old is None) != (new is None) or (old is not None and not np.array_equal(old, new)):
            _fail(f"R1-1 rebasing changed the prepared right-limit state: {field}")
    current = regular_current_at_state(
        integration_system, integration_state, policy=policy,
        require_relative_closure=voltage_after != voltage_before,
    )
    event = {
        "schema": "R1IdealStepV1", "zero_minus": minus_record, "zero_plus": plus_record,
        "voltage_jump_V": delta_voltage,
        "capacitance_infinity_F_m2": capacitance,
        "contact_displacement_jump_C_m2": _json_array(delta_d),
        "electrode_charge_jump_C_m2": _json_array(delta_d * [1., -1.]),
        "impulse_charge_C_m2": expected_impulse,
        "observed_impulse_charge_C_m2": _json_array(observed_impulse),
        "impulse_error_C_m2": error, "full_charge_jump_C_m2": charge_error,
        "local_iterations": iterations, "local_scaled_residual": norm,
        "fixed_population_algebraic_correction": correction,
        "algebraic_certificate": algebraic, "regular_current": current.evidence,
        "certified": True,
    }
    return InitialStepResult(integration_system, integration_state, current.metrics, event)
