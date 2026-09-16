"""Controlled R1 DC and direct small-signal response on physical volumes.

This opt-in research module uses the *transient* A-D operator, including its
fixed reference, pinned populations and physical contact faces. It does not
certify a time/frequency band. See OneDimensionalMechanismR1ResponseV1.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from perovskite_sim.constants import EPS_0, Q
from perovskite_sim.experiments.defect_ion_combined_impedance import (
    _component_inventories,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_dynamics import (
    R1DynamicsControls,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import r1_policy
from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import (
    restore_common_state, snapshot,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_step import _electrostatics
from perovskite_sim.physics.physical_control_volume import physical_contact_displacement
from perovskite_sim.physics.recombination import total_recombination


class R1ResponseError(RuntimeError):
    """A bounded solve failed; the partial result remains inspectable."""

    def __init__(self, message, result):
        self.result = result
        super().__init__(message)


@dataclass(frozen=True)
class R1DCResponse:
    system: Any
    state: Any
    evidence: dict


def _finite_scalar(value, name):
    raw = np.asarray(value)
    if raw.shape or raw.dtype.kind not in "iuf" or not np.isfinite(raw):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(raw)


def _frequencies(value):
    raw = np.asarray(value)
    if raw.ndim != 1 or not raw.size or raw.dtype.kind not in "iuf":
        raise ValueError("frequency_Hz must be a nonempty real vector")
    result = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(result)) or result[0] < 0 or np.any(np.diff(result) <= 0):
        raise ValueError("frequencies must be finite, nonnegative and increasing")
    return result


def _solve(matrix, rhs):
    """Row equilibrate without altering the equations or physical tolerances."""
    matrix = sparse.csr_matrix(matrix)
    norm = np.asarray(abs(matrix).max(axis=1).toarray()).ravel()
    if np.any(norm == 0) or not np.all(np.isfinite(norm)):
        raise ValueError("singular or nonfinite response equation")
    left = sparse.diags(1 / norm)
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        result = spsolve((left @ matrix).tocsc(), np.asarray(rhs) / norm)
    if not np.all(np.isfinite(result)):
        raise ValueError("nonfinite linear solution")
    return result


def _backward_error(matrix, solution, rhs):
    residual = np.asarray(matrix @ solution - rhs)
    scale = np.asarray(abs(matrix) @ np.abs(solution)) + np.abs(rhs)
    return float(np.max(np.abs(residual) / np.maximum(scale, np.finfo(float).tiny)))


def descriptor_frequency_response(frequency_Hz, *, storage, rate, forcing,
                                  storage_voltage, conduction, conduction_voltage,
                                  displacement, displacement_voltage):
    """Solve (s M-A)x=b-s mV; return Cx+cV+s(Dx+dV).

    The algebraic rows of M may be zero. All observations use the same state;
    D is stored displacement charge, not a second conduction channel. This
    small public primitive also admits independent R/C/single-pole oracles.
    """
    frequency = _frequencies(frequency_Hz)
    mass, operator = sparse.csr_matrix(storage), sparse.csr_matrix(rate)
    n = mass.shape[0]
    if mass.shape != (n, n) or operator.shape != (n, n):
        raise ValueError("storage and rate must be equally sized square matrices")
    b, mv = np.asarray(forcing), np.asarray(storage_voltage)
    c, d = np.atleast_2d(conduction), np.atleast_2d(displacement)
    cv, dv = np.atleast_1d(conduction_voltage), np.atleast_1d(displacement_voltage)
    if b.shape != (n,) or mv.shape != (n,) or c.shape != d.shape or c.shape[1] != n:
        raise ValueError("descriptor input or observation shapes disagree")
    if cv.shape != (c.shape[0],) or dv.shape != cv.shape:
        raise ValueError("voltage observation shapes disagree")
    if not all(np.all(np.isfinite(x)) for x in (mass.data, operator.data, b, mv, c, d, cv, dv)):
        raise ValueError("descriptor coefficients must be finite")
    states, current, charge, residual = [], [], [], []
    for f in frequency:
        s = 2j * np.pi * f
        matrix, rhs = s * mass - operator, b - s * mv
        x = _solve(matrix, rhs)
        states.append(x)
        current.append(c @ x + cv)
        charge.append(d @ x + dv)
        residual.append(_backward_error(matrix, x, rhs))
    current, charge = np.asarray(current), np.asarray(charge)
    return {
        "frequency_Hz": frequency, "state_per_V": np.asarray(states),
        "conduction_S_m2": current, "displacement_F_m2": charge,
        "admittance_S_m2": current + 2j * np.pi * frequency[:, None] * charge,
        "linear_backward_error": np.asarray(residual),
    }


def physical_observations(system, state):
    """Separate n/p/ion currents and D at contacts, bulk, both interface sides.

    All signs are the declared report sign j=polarity*J_x. The ion contacts
    are blocking. Reservoir carrier densities are pinned, hence no omitted
    endpoint carrier storage is introduced by their recombination correction.
    """
    mat, w = system.material, system.widths
    rho, _ = system.system._bulk_space_charge_and_tangent(
        state.n, state.p, positive_ion_density_m3=state.positive,
        negative_ion_density_m3=state.negative,
    )
    face_d = -mat.poisson_factor.C * np.diff(state.phi)
    contact_d = physical_contact_displacement(face_d, rho, w)
    rec = total_recombination(state.n, state.p, mat.ni_sq, mat.tau_n, mat.tau_p,
                              mat.n1, mat.p1, mat.B_rad, mat.C_n, mat.C_p)
    contact_n = state.current_n[[0, -1]] + Q * rec[[0, -1]] * w[[0, -1]] * [-1, 1]
    contact_p = state.current_p[[0, -1]] + Q * rec[[0, -1]] * w[[0, -1]] * [1, -1]
    jn, jp = state.current_n.copy(), state.current_p.copy()
    ji = Q * state.positive_flux.copy()
    if state.negative_flux is not None:
        ji -= Q * state.negative_flux
    labels = [f"internal_face_{i}" for i in range(len(jn))]
    for k in reversed(range(system.interface_count)):
        face = system.interface_faces[k]
        left, right = system.left_nodes[k], system.right_nodes[k]
        local = state.local[k]
        flux = local.tangent.balance.bulk_flux_m2_s
        cl = EPS_0 * mat.eps_r[left] / mat.iface_qss_left_distances_m[k]
        cr = EPS_0 * mat.eps_r[right] / mat.iface_qss_right_distances_m[k]
        jn[face], jp[face] = -Q * flux[0], Q * flux[1]
        face_d[face] = -cl * (local.trace_potential[0] - state.phi[left])
        jn = np.insert(jn, face + 1, Q * flux[2])
        jp = np.insert(jp, face + 1, -Q * flux[3])
        ji = np.insert(ji, face + 1, ji[face])
        face_d = np.insert(face_d, face + 1, cr * (local.trace_potential[1] - state.phi[right]))
        labels[face:face + 1] = [f"interface_{k}_left", f"interface_{k}_right"]
    values = system.polarity * np.array([
        np.r_[contact_n[0], jn, contact_n[-1]],
        np.r_[contact_p[0], jp, contact_p[-1]],
        np.r_[0., ji, 0.],
        np.r_[contact_d[0], face_d, contact_d[-1]],
    ])
    return values, ["left_contact", *labels, "right_contact"]


def _inventory_rows(system, state):
    rows = []
    lookup = {int(node): k for k, node in enumerate(system.positive_nodes)}
    for nodes, target in zip(system.ion_layout.positive_components, system.positive_targets):
        nodes = np.asarray(nodes, dtype=int)
        row = np.zeros(system.dimension)
        columns = system.positive_slice.start + np.array([lookup[int(n)] for n in nodes])
        row[columns] = system.widths[nodes] * state.positive[nodes] / target
        equation = system.positive_slice.start + lookup[int(nodes[-1])]
        rows.append((equation, row, float(np.dot(system.widths[nodes], state.positive[nodes]) / target - 1)))
    return rows


def _dc_equations(system, state):
    residual = np.r_[state.rate, state.poisson_residual, state.local_residual]
    matrix = sparse.vstack((state.rate_jacobian, state.poisson_jacobian,
                            state.local_jacobian), format="lil")
    frozen = []
    if not system.controls.nu_t:
        frozen.extend(range(system.trap_slice.start, system.trap_slice.stop))
    if not system.controls.nu_I:
        frozen.extend(range(system.positive_slice.start, system.positive_slice.stop))
    for row in frozen:
        residual[row] = state.coordinate[row]
        matrix[row] = 0.
        matrix[row, row] = 1.
    if system.controls.nu_I:
        for row, derivative, value in _inventory_rows(system, state):
            residual[row], matrix[row] = value, derivative
    return residual, matrix.tocsr()


def _dc_metrics(system, state, initial, policy):
    n = system.interior_count
    w = system.widths[1:-1]
    values, _ = physical_observations(system, state)
    current = np.sum(values[:3], axis=0)
    carrier, local_gauss = system.local_normalized_residuals(state)
    inventory = _component_inventories(state.positive, system.ion_layout.positive_components, system.widths)
    electrostatic = _electrostatics(system, state)
    metrics = {
        "electron_continuity_A_m2": float(np.sum(np.abs(Q * state.rate[:n] * w))),
        "hole_continuity_A_m2": float(np.sum(np.abs(Q * state.rate[n:2*n] * w))),
        "electron_normalized_residual": float(np.max(np.abs(Q * state.rate[:n] * w))) / system.system.current_scale,
        "hole_normalized_residual": float(np.max(np.abs(Q * state.rate[n:2*n] * w))) / system.system.current_scale,
        "trap_charge_rate_A_m2": float(np.max(np.abs(Q * state.rate[system.trap_slice]))),
        "ionic_face_current_A_m2": float(np.max(np.abs(Q * state.positive_flux))),
        "inventory_relative_error": float(np.max(np.abs(inventory / system.positive_targets - 1))),
        "all_physical_face_current_spread_A_m2": float(np.ptp(current)),
        "poisson_normalized": float(np.max(np.abs(state.poisson_residual))) / (Q * 1e15),
        "full_gauss_normalized": electrostatic["gauss_normalized"],
        "local_carrier_residual": carrier, "local_gauss_residual": local_gauss,
        "frozen_ion_relative_change": 0. if system.controls.nu_I else float(np.max(np.abs(state.positive-initial.positive) / np.maximum(initial.positive, 1.))),
        "frozen_trap_absolute_change": 0. if system.controls.nu_t else float(np.max(np.abs(state.occupancy-initial.occupancy))),
    }
    limits = {
        "electron_continuity_A_m2": policy.maximum_dc_continuity_bound_A_m2,
        "hole_continuity_A_m2": policy.maximum_dc_continuity_bound_A_m2,
        "electron_normalized_residual": policy.maximum_dc_normalized_residual,
        "hole_normalized_residual": policy.maximum_dc_normalized_residual,
        "trap_charge_rate_A_m2": policy.maximum_dc_continuity_bound_A_m2,
        "ionic_face_current_A_m2": policy.maximum_dc_ionic_face_current_A_m2,
        "inventory_relative_error": 1e-10,
        "all_physical_face_current_spread_A_m2": policy.maximum_dc_face_current_spread_A_m2,
        "poisson_normalized": 1e-10, "full_gauss_normalized": 1e-10,
        "local_carrier_residual": policy.maximum_local_carrier_normalized_residual,
        "local_gauss_residual": policy.maximum_local_gauss_normalized_residual,
        "frozen_ion_relative_change": 0., "frozen_trap_absolute_change": 0.,
    }
    reasons = [k for k, value in metrics.items() if not np.isfinite(value) or value > limits[k]]
    return {"metrics": metrics, "limits": limits, "certified": not reasons, "reasons": reasons}


def solve_controlled_dc(stack, intervals, binding, prepared, *, control="D",
                        voltage_V=0., policy=None):
    """Bounded direct DC on the controlled transient equations and inventory.

    Freeze means retain the common prepared population, including its charge.
    This function never calls a full-D biased solver for A-C. A converged
    nonlinear solve is distinct from numerical accuracy of a tiny DC current.
    """
    voltage = _finite_scalar(voltage_V, "voltage_V")
    policy = policy or r1_policy()
    controls = R1DynamicsControls.from_label(control)
    system, initial = restore_common_state(prepared, stack, intervals, binding,
                                           controls=controls, policy=policy)
    system, initial = system.rebase(initial)
    coordinate = system.initial_coordinate()
    history, reason = [], "newton_iteration_budget_exhausted"
    # A row norm is a coordinate conditioning device, not an acceptance gate.
    seed = system.evaluate(coordinate, voltage)
    _, seed_jacobian = _dc_equations(system, seed)
    row_scale = np.asarray(abs(seed_jacobian).sum(axis=1)).ravel()
    row_scale = np.maximum(row_scale, np.finfo(float).tiny)
    for iteration in range(policy.maximum_newton_iterations + 1):
        state = system.evaluate(coordinate, voltage)
        residual, jacobian = _dc_equations(system, state)
        scaled = residual / row_scale
        norm = float(np.max(np.abs(scaled)))
        checks = _dc_metrics(system, state, initial, policy)
        history.append({"iteration": iteration, "scaled_residual": norm,
                        "physical_reasons": checks["reasons"]})
        # Converge the equation further than its physical gate where the
        # fixed coordinate's floating-point resolution permits this.
        if checks["certified"] and norm <= 1e-12:
            reason = "converged"
            break
        if iteration == policy.maximum_newton_iterations:
            break
        delta = _solve(jacobian, -residual)
        alpha, accepted = min(1., 1. / max(float(np.max(np.abs(delta))), 1.)), False
        for _ in range(policy.maximum_line_search_steps):
            trial = coordinate + alpha * delta
            try:
                trial_state = system.evaluate(trial, voltage)
                trial_residual, _ = _dc_equations(system, trial_state)
                if np.max(np.abs(trial_residual / row_scale)) < norm:
                    coordinate, accepted = trial, True
                    break
            except (ValueError, RuntimeError, FloatingPointError):
                pass
            alpha *= .5
        if not accepted:
            reason = "line_search_stalled"
            break
    observations, labels = physical_observations(system, state)
    current = np.sum(observations[:3], axis=0)
    evidence = {
        "schema": "R1ControlledDCResponseV1", "control": controls.label,
        "voltage_V": voltage, "intervals": int(intervals),
        "prepared_sha256": prepared.sha256 if hasattr(prepared, "sha256") else prepared["sha256"],
        "reference_sha256": binding["sha256"], "state": snapshot(system, state),
        "coordinate": state.coordinate.copy(), "checks": checks, "history": history,
        "stop_reason": reason, "physical_face_labels": labels,
        "electron_current_A_m2": observations[0], "hole_current_A_m2": observations[1],
        "ion_current_A_m2": observations[2], "displacement_C_m2": observations[3],
        "current_A_m2": current, "terminal_current_A_m2": float(current[0]),
        "current_sign_convention": "reported j = junction_polarity * J_x",
        "certified": reason == "converged" and checks["certified"],
        "scope": "controlled_discrete_dc_equations_only",
    }
    result = R1DCResponse(system, state, evidence)
    if not evidence["certified"]:
        raise R1ResponseError("controlled DC did not converge within fixed limits", evidence)
    return result


def dc_conductance_study(stack, intervals, binding, prepared, *, control="D", policy=None):
    """Independent +/-0.1,0.05,0.025 mV steady states, without tail fitting."""
    widths = np.array([1e-4, 5e-5, 2.5e-5])
    pairs, conductance, error_indicators = [], [], []
    baseline = solve_controlled_dc(stack, intervals, binding, prepared, control=control, policy=policy)
    for h in widths:
        minus = solve_controlled_dc(stack, intervals, binding, prepared, control=control,
                                    voltage_V=-h, policy=policy)
        plus = solve_controlled_dc(stack, intervals, binding, prepared, control=control,
                                   voltage_V=h, policy=policy)
        pairs.append({"half_width_V": h, "minus": minus.evidence, "plus": plus.evidence})
        conductance.append((plus.evidence["terminal_current_A_m2"] - minus.evidence["terminal_current_A_m2"]) / (2*h))
        error_indicators.append((plus.evidence["checks"]["metrics"]["electron_continuity_A_m2"]
                                + plus.evidence["checks"]["metrics"]["hole_continuity_A_m2"]
                                + minus.evidence["checks"]["metrics"]["electron_continuity_A_m2"]
                                + minus.evidence["checks"]["metrics"]["hole_continuity_A_m2"]) / (2*h))
    values = np.asarray(conductance)
    limit = 1e-8 + .01 * max(abs(values[-1]), abs(values[-2]))
    return {
        "schema": "R1DCConductanceStudyV1", "control": control,
        "baseline": baseline.evidence, "pairs": pairs, "half_width_V": widths,
        "conductance_S_m2": values,
        "finest_pair_absolute_difference_S_m2": abs(values[-1]-values[-2]),
        "finest_pair_limit_S_m2": limit,
        "finest_pair_agrees": bool(abs(values[-1]-values[-2]) <= limit),
        "continuity_error_indicator_S_m2": np.asarray(error_indicators),
        "absolute_error_bound_S_m2": None,
        "scope": "dc_step_refinement_only_not_an_absolute_error_bound",
    }


def _linear_coefficients(system, state, voltage, h):
    """Direct M/A/H; central voltage derivative and physical observations."""
    n, d = system.dimension, len(state.storage)
    mass = sparse.vstack((state.storage_jacobian, sparse.csr_matrix((n-d, n))), format="csr")
    operator = sparse.vstack((state.rate_jacobian, -state.poisson_jacobian,
                              -state.local_jacobian), format="csr")
    plus, minus = system.evaluate(state.coordinate, voltage+h), system.evaluate(state.coordinate, voltage-h)
    b = np.r_[(plus.rate-minus.rate)/(2*h),
              -(plus.poisson_residual-minus.poisson_residual)/(2*h),
              -(plus.local_residual-minus.local_residual)/(2*h)]
    mv = np.r_[(plus.storage-minus.storage)/(2*h), np.zeros(n-d)]
    observation, labels = physical_observations(system, state)
    op = physical_observations(system, plus)[0]
    om = physical_observations(system, minus)[0]
    observation_v = (op-om)/(2*h)
    observation_x = np.empty((*observation.shape, n))
    differences, unresolved_elementwise, physical_steps = [], [], []
    equation_scales = np.maximum(np.asarray(abs(operator).max(axis=1).toarray()).ravel(), 1.)
    captures_x = np.empty((system.interface_count, 4, n))
    captures_v = (np.asarray([s.tangent.balance.capture_flux_m2_s for s in plus.local])
                  - np.asarray([s.tangent.balance.capture_flux_m2_s for s in minus.local]))/(2*h)
    for k in range(n):
        delta = np.zeros(n)
        delta[k] = h
        up, down = system.evaluate(state.coordinate+delta, voltage), system.evaluate(state.coordinate-delta, voltage)
        physical_steps.append({
            "coordinate_index": k,
            **{name: max(float(np.max(np.abs(getattr(up, name)-getattr(state, name)))),
                         float(np.max(np.abs(getattr(down, name)-getattr(state, name)))))
               for name in ("n", "p", "positive", "occupancy", "phi")},
        })
        observation_x[..., k] = (physical_observations(system, up)[0]-physical_observations(system, down)[0])/(2*h)
        captures_x[..., k] = (np.asarray([s.tangent.balance.capture_flux_m2_s for s in up.local])
                              - np.asarray([s.tangent.balance.capture_flux_m2_s for s in down.local]))/(2*h)
        fd = np.r_[(up.rate-down.rate)/(2*h), -(up.poisson_residual-down.poisson_residual)/(2*h),
                   -(up.local_residual-down.local_residual)/(2*h)]
        direct = operator[:, k].toarray().ravel()
        # Keep both a dimensionless equation-scaled column norm and the
        # cancellation-sensitive elementwise diagnostic. Small entries of
        # a column are not independently resolved by subtracting large rates.
        scale = np.maximum(np.maximum(np.abs(fd), np.abs(direct)), 1.)
        unresolved_elementwise.append(float(np.max(np.abs(fd-direct)/scale)))
        differences.append(float(np.linalg.norm((fd-direct)/equation_scales)
                                 / max(np.linalg.norm(direct/equation_scales), 1e-12)))
    return (mass, operator, b, mv, observation_x, observation_v, labels, differences,
            unresolved_elementwise, captures_x, captures_v, physical_steps)


def _constrained_descriptor(system, state, mass, operator, b, mv):
    """Exact frozen populations and one inventory equation per component."""
    mass, operator, b, mv = mass.tolil(copy=True), operator.tolil(copy=True), b.copy(), mv.copy()
    frozen = []
    if not system.controls.nu_t:
        frozen.extend(range(system.trap_slice.start, system.trap_slice.stop))
    if not system.controls.nu_I:
        frozen.extend(range(system.positive_slice.start, system.positive_slice.stop))
    for row in frozen:
        mass[row], operator[row], b[row], mv[row] = 0., 0., 0., 0.
        operator[row, row] = -1.
    if system.controls.nu_I:
        for row, derivative, _ in _inventory_rows(system, state):
            mass[row], operator[row], b[row], mv[row] = 0., -derivative, 0., 0.
    return mass.tocsr(), operator.tocsr(), b, mv


def small_signal_response(dc: R1DCResponse, frequency_Hz, *, derivative_steps=(1e-5, 5e-6, 2.5e-6)):
    """Direct sparse response plus three independent derivative-step records.

    Direct state Jacobians come from the same A-D transient operator. The
    finite differences differentiate voltage forcing and physical currents;
    their numerical steps are not experimental perturbation amplitudes.
    """
    if not isinstance(dc, R1DCResponse) or not dc.evidence.get("certified"):
        raise ValueError("small signal requires a certified controlled DC result")
    frequency = _frequencies(frequency_Hz)
    steps = tuple(_finite_scalar(h, "derivative step") for h in derivative_steps)
    if steps != (1e-5, 5e-6, 2.5e-6):
        raise ValueError("the declared three derivative steps must remain 1e-5, 5e-6, 2.5e-6")
    system, voltage = dc.system, dc.evidence["voltage_V"]
    # A previously computed pass flag cannot replace the present equations.
    # Reevaluate the live coordinate before using its tangents or currents.
    state = system.evaluate(dc.state.coordinate, voltage)
    current_checks = _dc_metrics(system, state, system._step_reference, r1_policy())
    if not current_checks["certified"]:
        raise R1ResponseError("AC operating state no longer satisfies controlled DC", current_checks)
    records = []
    for h in steps:
        (mass, operator, b, mv, ox, ov, labels, differences,
         elementwise, captures_x, captures_v, physical_steps) = _linear_coefficients(system, state, voltage, h)
        constrained = _constrained_descriptor(system, state, mass, operator, b, mv)
        result = descriptor_frequency_response(
            frequency, storage=constrained[0], rate=constrained[1], forcing=constrained[2],
            storage_voltage=constrained[3], conduction=np.sum(ox[:3], axis=0),
            conduction_voltage=np.sum(ov[:3], axis=0), displacement=ox[3], displacement_voltage=ov[3],
        )
        response = result["state_per_V"]
        decomposed = np.stack([response @ ox[k].T + ov[k] for k in range(3)], axis=1)
        full_residual = []
        for f, value in zip(frequency, response):
            s = 2j*np.pi*f
            full_residual.append(_backward_error(s*mass-operator, value, b-s*mv))
        inventory = []
        for _, row, _ in _inventory_rows(system, state):
            inventory.append(response @ row)
        inventory = np.asarray(inventory).T
        storage_response = response @ state.storage_jacobian.T + mv[:len(state.storage)]
        ion_response = storage_response[:, system.positive_slice]
        inventory_absolute = inventory * system.positive_targets
        positive_scale = np.abs(ion_response) @ system.widths[system.positive_nodes]
        legacy_inventory = np.divide(
            np.abs(ion_response @ system.widths[system.positive_nodes]), positive_scale,
            out=np.zeros(len(frequency)), where=positive_scale > np.finfo(float).tiny,
        )
        capture = np.einsum("fn,ikn->fik", response, captures_x) + captures_v
        trap_capture = capture[:, :, 0] + capture[:, :, 2] - capture[:, :, 1] - capture[:, :, 3]
        trap_storage = 2j*np.pi*frequency[:, None]*storage_response[:, system.trap_slice]
        capture_scale = np.maximum(np.maximum(np.abs(trap_capture), np.abs(trap_storage)),
                                   np.max(np.abs(capture), axis=2))
        capture_relative = np.divide(np.abs(trap_capture-trap_storage), capture_scale,
                                     out=np.zeros_like(capture_scale), where=capture_scale > np.finfo(float).tiny)
        admittance = result["admittance_S_m2"]
        spread = np.max(np.abs(admittance-admittance[:, :1]), axis=1) / np.maximum(np.max(np.abs(admittance), axis=1), 1e-20)
        decomposition = np.max(np.abs(result["conduction_S_m2"]-np.sum(decomposed, axis=1)), axis=1) / np.maximum(np.max(np.abs(result["conduction_S_m2"]), axis=1), 1e-20)
        phi_response = np.zeros((len(frequency), system.node_count), dtype=complex)
        phi_response[:, 1:-1] = system.thermal_voltage*response[:, system.potential_slice]
        phi_response[:, -1] = -system.polarity
        # The ideal-step algebraic solve holds every dynamic population fixed.
        d = len(state.storage)
        initial_matrix = sparse.vstack((state.storage_jacobian, state.poisson_jacobian,
                                        state.local_jacobian), format="csr")
        x0 = _solve(initial_matrix, np.r_[-mv[:d], b[d:]])
        impulse = ox[3] @ x0 + ov[3]
        records.append({
            **result, "derivative_step": h,
            "voltage_difference_half_width_V": h,
            "coordinate_difference_half_width": h,
            "potential_coordinate_half_width_V": h*system.thermal_voltage,
            "coordinate_actual_maximum_physical_perturbations": physical_steps,
            "physical_perturbation_units": {"n": "m-3", "p": "m-3", "positive": "m-3", "occupancy": "1", "phi": "V"},
            "state_jacobian_fd_column_relative_error": np.asarray(differences),
            "state_jacobian_fd_elementwise_relative_error": np.asarray(elementwise),
            "state_jacobian_fd_note": "elementwise ratios may be unresolved after cancellation; column norms use direct equation scales",
            "unreplaced_equation_backward_error": np.asarray(full_residual),
            "electron_admittance_S_m2": decomposed[:, 0],
            "hole_admittance_S_m2": decomposed[:, 1], "ion_admittance_S_m2": decomposed[:, 2],
            "inventory_response_relative_per_V": inventory,
            "inventory_response_m2_per_V": inventory_absolute,
            "inventory_response_thermal_normalized": system.thermal_voltage*np.abs(inventory),
            "legacy_inventory_response_relative": legacy_inventory,
            "thermal_voltage_V": system.thermal_voltage,
            "inventory_targets_m2": system.positive_targets.copy(),
            "capture_flux_per_V_m2_s": capture,
            "trap_storage_rate_per_V_m2_s": trap_storage,
            "capture_storage_error_per_V_m2_s": trap_capture-trap_storage,
            "capture_storage_relative_error": np.max(capture_relative, axis=1),
            "current_decomposition_relative_error": decomposition,
            "storage_response_per_V": storage_response,
            "electron_density_response_m3_per_V": storage_response[:, system.electron_slice],
            "hole_density_response_m3_per_V": storage_response[:, system.hole_slice],
            "carrier_response_nodes": np.arange(1, system.node_count-1),
            "positive_density_response_m3_per_V": ion_response,
            "positive_response_nodes": system.positive_nodes.copy(),
            "occupancy_response_per_V": storage_response[:, system.trap_slice]/system.trap_density,
            "potential_response_V_per_V": phi_response,
            "physical_face_spread_relative": spread,
            "initial_algebraic_state_per_V": x0,
            "impulse_capacitance_F_m2": impulse,
            "impulse_linear_backward_error": _backward_error(initial_matrix, x0, np.r_[-mv[:d], b[d:]]),
        })
    latest, previous = records[-1], records[-2]
    fine, coarse = latest["admittance_S_m2"], previous["admittance_S_m2"]
    refinement = np.max(np.abs(fine-coarse), axis=1) / np.maximum(np.max(np.maximum(np.abs(fine), np.abs(coarse)), axis=1), 1e-20)
    adjacent_refinements = []
    for left, right in zip(records[:-1], records[1:]):
        a, z = left["admittance_S_m2"], right["admittance_S_m2"]
        adjacent_refinements.append(np.max(np.abs(a-z), axis=1)
                                    / np.maximum(np.max(np.maximum(np.abs(a), np.abs(z)), axis=1), 1e-20))
    component_difference = np.stack((np.abs(fine.real-coarse.real), np.abs(fine.imag-coarse.imag)), axis=-1)
    component_limits = 1e-8+.01*np.stack((np.maximum(np.abs(fine.real), np.abs(coarse.real)),
                                        np.maximum(np.abs(fine.imag), np.abs(coarse.imag))), axis=-1)
    checks = {
        "linear_backward_error": latest["linear_backward_error"] <= 1e-10,
        "all_equations_backward_error": latest["unreplaced_equation_backward_error"] <= 1e-10,
        "physical_face_spread": latest["physical_face_spread_relative"] <= 5e-4,
        "derivative_refinement": np.max(adjacent_refinements, axis=0) <= 2e-3,
        "derivative_component_refinement": np.all(component_difference <= component_limits, axis=(1, 2)),
        "inventory": np.max(latest["inventory_response_thermal_normalized"], axis=1) <= 1e-10,
        "legacy_inventory": latest["legacy_inventory_response_relative"] <= 1e-8,
        "capture_storage": latest["capture_storage_relative_error"] <= 1e-3,
        "current_decomposition": latest["current_decomposition_relative_error"] <= 1e-7,
        "direct_tangent_difference": np.full(len(frequency), np.max(latest["state_jacobian_fd_column_relative_error"]) <= 3e-4),
    }
    eligible = np.logical_and.reduce(tuple(checks.values()))
    return {
        "schema": "R1ControlledSmallSignalV1", "control": dc.evidence["control"],
        "voltage_V": voltage, "prepared_sha256": dc.evidence["prepared_sha256"],
        "reference_sha256": dc.evidence["reference_sha256"], "frequency_Hz": frequency,
        "physical_face_labels": labels, "derivative_levels": records,
        "admittance_S_m2": fine[:, 0], "dc_state": dc.evidence,
        "derivative_refinement_relative": refinement, "checks": checks,
        "all_adjacent_derivative_refinement_relative": np.asarray(adjacent_refinements),
        "derivative_component_difference_S_m2": component_difference,
        "derivative_component_limits_S_m2": component_limits,
        "numerically_eligible_frequency_points": eligible,
        "frequency_window_complete": False, "double_domain_consistent": False,
        "equilibrium_dissipation_sign_observation": (
            fine[:, 0].real >= -1e-8 if voltage == 0. else None
        ),
        "missing_validation": ["frequency_window_coverage", "spatial_refinement",
                               "finite_amplitude_linearity", "time_window_and_early_interval",
                               "time_reconstruction_error_budget"],
        "scope": "direct_controlled_linear_response_only",
    }


def compare_transient_tail(dc: R1DCResponse, *, initial_state, tail_state,
                           tail_regular_current_A_m2, time_s):
    """Compare a same-grid finite-time state with its controlled biased DC.

    This check does not infer a bound on the omitted infinite-time integral.
    The caller must provide exactly the same preparation/control and bias;
    identities are part of the required state envelope, not inferred from
    numerical proximity. State dictionaries use the R1 snapshot field names.
    """
    if not dc.evidence.get("certified"):
        raise ValueError("tail comparison requires certified same-model DC")
    for key in ("prepared_sha256", "control", "voltage_V"):
        if tail_state.get(key) != dc.evidence[key]:
            raise ValueError(f"tail/DC identity mismatch: {key}")
    t = _finite_scalar(time_s, "time_s")
    if t <= 0:
        raise ValueError("a finite-time tail must have positive time")
    state = dc.state
    expected = {"n_m3": state.n, "p_m3": state.p, "positive_m3": state.positive,
                "occupancy": state.occupancy, "phi_V": state.phi,
                "trace_state_m3": np.asarray([local.state_m3 for local in state.local]),
                "trace_potential_V": np.asarray([local.trace_potential for local in state.local])}
    arrays = {}
    for key, target in expected.items():
        tail, initial = np.asarray(tail_state[key], dtype=float), np.asarray(initial_state[key], dtype=float)
        if tail.shape != target.shape or initial.shape != target.shape or not np.all(np.isfinite(tail)) or not np.all(np.isfinite(initial)):
            raise ValueError(f"tail/DC state shape or finite mismatch: {key}")
        arrays[key] = tail, initial, target
    comparisons = {}
    for key in ("n_m3", "p_m3", "trace_state_m3"):
        tail, _, target = arrays[key]
        if np.any(tail <= 0) or np.any(target <= 0):
            raise ValueError("carrier densities must be positive")
        difference = np.abs(np.log(tail/target))
        comparisons[key] = {"maximum_log_difference": float(np.max(difference)),
                            "limit": .01, "agrees": bool(np.all(difference <= .01))}
    for key, absolute in (("phi_V", 5e-5), ("trace_potential_V", 5e-5), ("occupancy", 1e-7)):
        tail, initial, target = arrays[key]
        difference = np.abs(tail-target)
        limit = absolute+.01*np.maximum(np.abs(tail-initial), np.abs(target-initial))
        comparisons[key] = {"maximum_absolute_difference": float(np.max(difference)),
                            "maximum_limit_ratio": float(np.max(difference/limit)),
                            "agrees": bool(np.all(difference <= limit))}
    tail, initial_ions, target = arrays["positive_m3"]
    scale = dc.system.material.P_ion0
    difference = np.abs(tail-target)
    comparisons["positive_m3"] = {"maximum_scaled_difference": float(np.max(difference/np.maximum(scale, 1.))),
                                  "limit": .01, "agrees": bool(np.all(difference <= .01*scale))}
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_spatial import ConservativeIonProfile

    faces = dc.system.material.physical_cell_faces_m
    components = dc.system.ion_layout.positive_components
    cuts = {int(np.searchsorted(dc.system.grid, boundary))
            for boundary in dc.system.material.iface_qss_interface_positions_m}
    for nodes in components:
        cuts.update(index for index in (nodes[0], nodes[-1]+1) if 0 < index < dc.system.node_count)
    def centroid(density):
        profile = ConservativeIonProfile(faces, density, tuple(sorted(cuts)))
        result = []
        for nodes in components:
            mass, first = profile.moments(faces[nodes[0]], faces[nodes[-1]+1])
            if mass <= 0:
                raise ValueError("tail active component has no ion inventory")
            result.append(first/mass)
        return np.asarray(result)
    z, z_initial, z_dc = centroid(tail), centroid(initial_ions), centroid(target)
    z_limit = 5e-11+.01*np.maximum(np.abs(z-z_initial), np.abs(z_dc-z_initial))
    comparisons["ion_centroid_m"] = {"absolute_difference": np.abs(z-z_dc), "limit": z_limit,
                                    "agrees": bool(np.all(np.abs(z-z_dc) <= z_limit)),
                                    "definition": "exact first moment of conservative ion reconstruction over each active component"}
    current = _finite_scalar(tail_regular_current_A_m2, "tail_regular_current_A_m2")
    jdc = dc.evidence["terminal_current_A_m2"]
    limit = 1e-7+.005*max(abs(current), abs(jdc))
    comparisons["current_A_m2"] = {"absolute_difference": abs(current-jdc), "limit": limit,
                                   "agrees": abs(current-jdc) <= limit}
    return {"schema": "R1TransientTailDCComparisonV1", "time_s": t,
            "prepared_sha256": dc.evidence["prepared_sha256"], "control": dc.evidence["control"],
            "voltage_V": dc.evidence["voltage_V"], "comparisons": comparisons,
            "all_observables_agree": all(item["agrees"] for item in comparisons.values()),
            "infinite_tail_integral_bound_F_m2": None,
            "scope": "same_grid_pointwise_tail_vs_controlled_dc_only"}


def compare_reconstructed_response(reconstruction, ac):
    """Compare actual common frequencies; unknown tails never pass a budget."""
    frequency = np.asarray(ac["frequency_Hz"])
    if not np.array_equal(reconstruction.frequency_Hz, frequency):
        raise ValueError("time reconstruction and AC frequencies must match exactly")
    direct, time = np.asarray(ac["admittance_S_m2"]), reconstruction.admittance_S_m2
    limits = np.array([1e-8+.01*np.maximum(np.abs(direct.real), np.abs(time.real)),
                       1e-8+.01*np.maximum(np.abs(direct.imag), np.abs(time.imag))]).T
    differences = np.array([np.abs(direct.real-time.real), np.abs(direct.imag-time.imag)]).T
    budget = reconstruction.total_error_estimate_S_m2[:, None]
    return {
        "frequency_Hz": frequency, "difference_components_S_m2": differences,
        "limits_components_S_m2": limits, "component_agreement": np.all(differences <= limits, axis=1),
        "error_budget_agreement": np.all(budget <= limits, axis=1),
        "unknown_error_sources": reconstruction.unknown_error_sources,
        "double_domain_consistent": False,
        "scope": "pointwise_comparison_only_other_study_gates_required",
    }


__all__ = ["R1ResponseError", "R1DCResponse", "solve_controlled_dc", "dc_conductance_study",
           "small_signal_response", "physical_observations", "descriptor_frequency_response",
           "compare_reconstructed_response", "compare_transient_tail"]
