"""Initialize an R1 compensated state from the unchanged QF definitions.

This is a new 0-minus algebraic initialization, before an ideal step or any
accepted trajectory.  It does not repair an already accepted state.  Inputs
to the QF law, ion populations, occupancy and endpoint carrier reservoirs
remain fixed.  The potential is solved anew and carrier populations follow
that QF law.  Diagnostics are attached to the system separately from the
returned dictionary of physical ``DD`` arrays.
"""

from __future__ import annotations

from dataclasses import replace
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from perovskite_sim.constants import EPS_0, Q
from perovskite_sim.physics.compensated import DD
from perovskite_sim.physics.two_sided_interface import (
    EquilibriumReferencedSheetCharge,
    InterfaceTracePotentials,
    _material_two_sided_interface_problem,
    fixed_occupancy_carrier_tangent_from_density,
    solve_fixed_occupancy_two_sided_interface,
)
from perovskite_sim.solver.mol import poisson_right_boundary


_MAX_CORRECTIONS = 12
_POTENTIAL_CORRECTION_LIMIT_V = 1e-28
_LOCAL_CARRIER_LIMIT = 1e-9  # Existing fixed-occupancy local solver default.


def _put(value: DD, index, part) -> DD:
    part = part if isinstance(part, DD) else DD(part)
    hi, lo = value.hi.copy(), value.lo.copy()
    hi[index], lo[index] = part.hi, part.lo
    return DD(hi, lo)


def _maximum(value: DD) -> float:
    return float(np.max(np.abs(value.to_float()), initial=0.0))


def _words(value: DD) -> dict:
    return {"hi": value.hi.tolist(), "lo": value.lo.tolist()}


def initialize_fine_reference(system) -> dict[str, DD]:
    """Return a QF-consistent fine reference before ``_fine_reference`` exists.

    ``system.precision_initialization_diagnostics`` records the actual
    residuals, corrections, fixed inputs, local-carrier decisions and result.
    The caller owns installing the returned reference and serializing those
    diagnostics.  Unsupported charge models and nonconvergence fail closed.
    """
    if hasattr(system, "_fine_reference"):
        raise ValueError("fine initialization must precede installation of a fine reference")
    mat = system.material
    if (
        mat.has_dual_ions
        or not mat.ion_steric_diffusion_only
        or system.reference_negative is not None
        or system.negative_nodes.size
        or system.interface_count != 1
        or getattr(mat, "monovalent_bulk_defects", None) is not None
        or getattr(mat, "multivalent_bulk_defects", None) is not None
        or getattr(mat, "frozen_metastable_defects", None) is not None
        or getattr(system, "illuminated", False)
    ):
        raise ValueError("fine initialization supports the frozen dark, single-positive-ion R1 model")
    if float(mat.T_device) != 300.0:
        raise ValueError("fine initialization requires the frozen 300 K material")
    voltage = float(system.system.V_app)
    if voltage != 0.0:
        raise ValueError("fine initialization is the 0-minus reference at zero applied voltage")

    qn, qp = DD(system.dqfn_dc), DD(system.dqfp_dc)
    positive = DD(system.reference_positive)
    occupancy = DD(system.reference_occupancy)
    if np.any(positive < 0) or np.any((occupancy <= 0) | (occupancy >= 1)):
        raise ValueError("fine initial populations or occupancy are invalid")
    system._site_fraction(positive.hi, None, reject=True)
    vt = DD(system.thermal_voltage)
    phi0 = DD(system.system.phi0)
    log_n0, log_p0 = DD(system.system.log_n0), DD(system.system.log_p0)
    phi = _put(DD(system.reference_phi), [0, -1], [0.0, poisson_right_boundary(mat, voltage)])
    fixed_n, fixed_p = DD(system.reference_n[[0, -1]]), DD(system.reference_p[[0, -1]])
    sigma_dynamic = DD(Q) * DD(system.trap_density) * (DD(system.equilibrium_occupancy) - occupancy)

    # Read the actual original healthy local state once, before installing any
    # fine state.  In particular do not make a new exp(log(state)) round trip
    # when retaining its trace-density seed.
    healthy = system.evaluate(system.initial_coordinate(), voltage)
    geometries, weights, capacitances = [], [], []
    for k in range(system.interface_count):
        geometry, _, _ = _material_two_sided_interface_problem(
            mat, system.stack, healthy.n, healthy.p, healthy.phi, k,
            cross_transmission=system.dark_reference.interface_transmission,
        )
        geometries.append(geometry)
        weights.append(system._sheet_weights(k))
        capacitances.append((
            EPS_0 * geometry.eps_r_left / geometry.left_distance_m,
            EPS_0 * geometry.eps_r_right / geometry.right_distance_m,
        ))
    sigma_static = DD([g.fixed_sheet_charge_C_m2 for g in geometries])
    sigma = sigma_static + sigma_dynamic
    jumps = DD([g.potential_jump_right_minus_left_V for g in geometries])
    if np.any(jumps != 0):
        # The frozen device Poisson assembly contains no prescribed-jump
        # source term.  Extending it here would silently add a new model.
        raise ValueError("nonzero prescribed jumps are outside the frozen R1 Poisson assembly")

    factor = mat.poisson_factor
    face_capacitance, cell_width = DD(factor.C), DD(factor.h_cell)
    dopants = DD(mat.N_D) - DD(mat.N_A)
    ion_background = DD(mat.P_ion0)

    def populations(potential):
        potential_change = potential - phi0
        n = (log_n0 + (qn + potential_change) / vt).exp()
        p = (log_p0 + (qp - potential_change) / vt).exp()
        return _put(n, [0, -1], fixed_n), _put(p, [0, -1], fixed_p)

    def poisson(potential, n, p):
        rho = DD(Q) * (p - n + dopants + positive - ion_background)
        face_term = face_capacitance * (potential[1:] - potential[:-1])
        residual = face_term[1:] - face_term[:-1] + rho[1:-1] * cell_width
        for k, (left, right) in enumerate(zip(system.left_nodes, system.right_nodes)):
            wl, wr = weights[k]
            residual = _put(residual, left - 1, residual[left - 1] + DD(wl) * sigma[k])
            residual = _put(residual, right - 1, residual[right - 1] + DD(wr) * sigma[k])
        return residual

    diagnostics = {
        "schema": "R1FineInitializationV1",
        "scope": "new_zero_minus_algebraic_initialization_not_an_accepted_trajectory",
        "status": "running",
        "qf_definition": "log_n0+(dqfn+phi-phi0)/VT; log_p0+(dqfp-phi+phi0)/VT",
        "qf_inputs_unchanged": True,
        "endpoint_carrier_populations_fixed": True,
        "ion_and_occupancy_inputs_fixed": True,
        "poisson_maximum_corrections": _MAX_CORRECTIONS,
        "potential_correction_limit_V": _POTENTIAL_CORRECTION_LIMIT_V,
        "poisson_corrections_V": [],
        "poisson_maximum_residuals_C_m2": [],
        "local_carriers": [],
        "static_sheet_charge_C_m2": _words(sigma_static),
        "dynamic_sheet_charge_C_m2": _words(sigma_dynamic),
        "total_sheet_charge_C_m2": _words(sigma),
        "prescribed_trace_jump_V": _words(jumps),
        "trace_capacitances_F_m2": [list(pair) for pair in capacitances],
        "poisson_sheet_weights": [list(pair) for pair in weights],
    }
    system.precision_initialization_diagnostics = diagnostics
    converged = False
    for iteration in range(1, _MAX_CORRECTIONS + 1):
        n, p = populations(phi)
        residual = poisson(phi, n, p)
        diagnostics["poisson_maximum_residuals_C_m2"].append(_maximum(residual))
        if iteration == 1:
            diagnostics["poisson_initial_residual_C_m2"] = _words(residual)
        jacobian = system._poisson_laplacian[:, 1:-1] - sparse.diags(
            Q * (n.hi[1:-1] + p.hi[1:-1]) * factor.h_cell / system.thermal_voltage
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            correction = np.asarray(spsolve(jacobian.tocsr(), -residual.hi))
        if correction.shape != (system.node_count - 2,) or not np.all(np.isfinite(correction)):
            diagnostics["status"] = "failed_nonfinite_poisson_correction"
            raise ArithmeticError("fine initial Poisson returned an invalid correction")
        maximum = float(np.max(np.abs(correction), initial=0.0))
        diagnostics["poisson_corrections_V"].append(maximum)
        phi = _put(phi, slice(1, -1), phi[1:-1] + DD(correction))
        if maximum < _POTENTIAL_CORRECTION_LIMIT_V:
            converged = True
            break
    if not converged:
        diagnostics["status"] = "failed_poisson_convergence"
        raise RuntimeError("fine initial Poisson did not reach its potential-correction target")
    n, p = populations(phi)
    residual = poisson(phi, n, p)
    diagnostics["poisson_iterations"] = iteration
    diagnostics["poisson_final_residual_C_m2"] = _words(residual)
    diagnostics["poisson_final_maximum_residual_C_m2"] = _maximum(residual)
    diagnostics["maximum_phi_change_from_original_V"] = _maximum(phi - DD(system.reference_phi))
    diagnostics["maximum_n_change_from_original_m3"] = _maximum(n - DD(system.reference_n))
    diagnostics["maximum_p_change_from_original_m3"] = _maximum(p - DD(system.reference_p))

    trace_hi, trace_lo = np.zeros((system.interface_count, 2)), np.zeros((system.interface_count, 2))
    density_hi, density_lo = np.zeros((system.interface_count, 4)), np.zeros((system.interface_count, 4))
    trace_residuals = []
    for k, (left, right) in enumerate(zip(system.left_nodes, system.right_nodes)):
        cl, cr = (DD(v) for v in capacitances[k])
        trace_left = (cl * phi[left] + cr * phi[right] + sigma[k] - cr * jumps[k]) / (cl + cr)
        trace_right = trace_left + jumps[k]
        trace_hi[k] = [trace_left.hi, trace_right.hi]
        trace_lo[k] = [trace_left.lo, trace_right.lo]
        jump_error = trace_right - trace_left - jumps[k]
        gauss_error = cl * (trace_left - phi[left]) + cr * (trace_right - phi[right]) - sigma[k]
        trace_residuals.append({"jump_V": _words(jump_error), "gauss_C_m2": _words(gauss_error)})

        geometry, physics, bulk = _material_two_sided_interface_problem(
            mat, system.stack, n.hi, p.hi, phi.hi, k,
            cross_transmission=system.dark_reference.interface_transmission,
        )
        charged = replace(geometry, fixed_sheet_charge_C_m2=float(sigma[k].to_float()))
        trace = InterfaceTracePotentials(float(trace_left.hi), float(trace_right.hi))
        density = DD(healthy.local[k].state_m3)

        def local_balance(values):
            return fixed_occupancy_carrier_tangent_from_density(
                values.hi, charged, physics, bulk, float(occupancy[k].hi), trace,
                capture_multiplier=system.capture_multiplier, paired_bernoulli=True,
            ).balance

        balance = local_balance(density)
        norm_before = float(np.max(np.abs(balance.residual_m2_s / system.reference_local_scale[k, 2:])))
        evaluations = 0
        if norm_before > _LOCAL_CARRIER_LIMIT:
            solved = solve_fixed_occupancy_two_sided_interface(
                geometry, physics, bulk, float(occupancy[k].hi),
                EquilibriumReferencedSheetCharge(float(system.equilibrium_occupancy[k]),
                                                 float(system.trap_density[k])),
                initial_state_m3=density.hi, residual_tolerance=_LOCAL_CARRIER_LIMIT,
                capture_multiplier=system.capture_multiplier,
            )
            density = DD(solved.qss.state_m3)
            evaluations = int(solved.qss.evaluations)
            balance = local_balance(density)
        norm_after = float(np.max(np.abs(balance.residual_m2_s / system.reference_local_scale[k, 2:])))
        diagnostics["local_carriers"].append({
            "interface": k, "seed": "original_healthy_local_state_without_new_log_exp_round_trip",
            "precision": "existing_binary64_constitutive_balance_on_fine_state_high_words",
            "normalized_residual_before": norm_before, "normalized_residual_after": norm_after,
            "fixed_occupancy_limit": _LOCAL_CARRIER_LIMIT, "new_solver_evaluations": evaluations,
            "re_solved": bool(evaluations),
        })
        if not np.isfinite(norm_after) or norm_after > _LOCAL_CARRIER_LIMIT:
            diagnostics["status"] = "failed_local_carrier_balance"
            raise RuntimeError("fine initial trace carriers do not satisfy the original local balance")
        density_hi[k], density_lo[k] = density.hi, density.lo
    diagnostics["trace_electrostatic_residuals"] = trace_residuals
    diagnostics["status"] = "completed"
    return {
        "phi_V": phi, "dqfn_V": qn, "dqfp_V": qp,
        "n_m3": n, "p_m3": p, "positive_m3": positive,
        "occupancy": occupancy,
        "trace_potential_V": DD(trace_hi, trace_lo),
        "trace_state_m3": DD(density_hi, density_lo),
    }
