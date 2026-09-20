"""Independent fixed-voltage regular current, including the nonzero-step 0+.

Currents, rates and sheet capture are freshly assembled from physical state.
Integrated Gauss' law solves the derivative electrostatics without the solver's
Poisson factors, derivative state, residual or stored currents. Constitutive
laws are shared and declared; conservation is not a solution-accuracy bound.
"""
from __future__ import annotations

from math import fsum

import numpy as np

from perovskite_sim.constants import EPS_0, Q
from perovskite_sim.physics.recombination import total_recombination
from perovskite_sim.experiments import one_dimensional_mechanism_r1_independent_physics as finite

SCHEMA = "R1IndependentRegularCurrentV1"
SCOPE = "fixed_voltage_regular_current_independent_assembly_not_solution_error_bound"
ROW_FIELDS = frozenset({"schema", "scope", "relative_current_applicable", "shared_constitutive_dependencies",
    "arrays", "metrics", "limits", "checks", "assessment", "reasons", "passed", "content_comparison"})
ARRAY_FIELDS = frozenset({"potential_derivative_V_s", "trace_potential_derivative_V_s",
    "charge_density_derivative_C_m3_s", "sheet_charge_derivative_C_m2_s", "electron_rate_m3_s",
    "hole_rate_m3_s", "positive_ion_rate_m3_s", "internal_conduction_A_m2",
    "internal_displacement_A_m2", "internal_maxwell_A_m2", "interface_maxwell_A_m2",
    "contact_conduction_A_m2", "contact_displacement_A_m2", "contact_maxwell_A_m2",
    "physical_widths_m", "face_capacitance_F_m2", "charge_balance_absolute_A_m2"})
SHARED_CONSTITUTIVE_DEPENDENCIES = finite.SHARED_CONSTITUTIVE_DEPENDENCIES + (
    "recombination.total_recombination",)


def effective_regular_limits(policy, *, require_relative_closure=True):
    """Return the limits actually consumed and published by this reconstruction."""
    limits = {key: finite.METRIC_LIMITS[key] for key in (
        "internal_face_current_spread_relative", "contact_internal_current_spread_relative",
        "interface_current_spread_relative", "charge_balance_normalized")}
    if not require_relative_closure:
        limits.update({"electron_continuity_A_m2": policy.maximum_dc_continuity_bound_A_m2,
            "hole_continuity_A_m2": policy.maximum_dc_continuity_bound_A_m2,
            "electron_normalized_residual": policy.maximum_dc_normalized_residual,
            "hole_normalized_residual": policy.maximum_dc_normalized_residual,
            "trap_charge_rate_A_m2": policy.maximum_dc_continuity_bound_A_m2,
            "absolute_current_spread_A_m2": policy.maximum_dc_face_current_spread_A_m2,
            "ionic_face_current_A_m2": policy.maximum_dc_ionic_face_current_A_m2})
    return limits


def _geometry(system):
    mat = system.material
    widths = finite._volumes(system)
    eps = np.asarray(mat.eps_r)
    capacitance = EPS_0 * 2. * eps[:-1] * eps[1:] / (eps[:-1] + eps[1:]) / np.diff(system.grid)
    sides = []
    for k, (face, left, right) in enumerate(zip(system.interface_faces, system.left_nodes, system.right_nodes)):
        cl = EPS_0 * eps[left] / mat.iface_qss_left_distances_m[k]
        cr = EPS_0 * eps[right] / mat.iface_qss_right_distances_m[k]
        capacitance[face] = cl * cr / (cl + cr)
        sides.append((cl, cr))
    return widths, capacitance, sides


def _rates(system, state, widths, electron, hole, ion_flux):
    mat = system.material
    recombination = total_recombination(state.n, state.p, mat.ni_sq, mat.tau_n, mat.tau_p,
        mat.n1, mat.p1, mat.B_rad, mat.C_n, mat.C_p)
    transport_n, transport_p = electron.copy(), hole.copy()
    transport_n[list(system.interface_faces)] = 0.
    transport_p[list(system.interface_faces)] = 0.
    rate_n = -recombination + np.diff(np.r_[0., transport_n, 0.]) / (Q * widths)
    rate_p = -recombination - np.diff(np.r_[0., transport_p, 0.]) / (Q * widths)
    sheet_rate = []
    for k, (left, right) in enumerate(zip(system.left_nodes, system.right_nodes)):
        balance = finite._interface_balance(system, state, k)
        flux = balance.bulk_flux_m2_s
        rate_n[left] -= flux[0] / widths[left]
        rate_p[left] -= flux[1] / widths[left]
        rate_n[right] -= flux[2] / widths[right]
        rate_p[right] -= flux[3] / widths[right]
        capture = balance.capture_flux_m2_s
        sheet_rate.append(-Q * (capture[0] + capture[2] - (capture[1] + capture[3])))
    # Fixed reservoir populations have no storage derivative.
    rate_n[[0, -1]], rate_p[[0, -1]] = 0., 0.
    ion_rate = -np.diff(np.r_[0., ion_flux, 0.]) / widths
    return rate_n, rate_p, ion_rate, np.asarray(sheet_rate)


def _gauss_derivative(widths, capacitance, rho_dot, sigma_dot, sides, system):
    charge = rho_dot * widths
    for k, (left, right) in enumerate(zip(system.left_nodes, system.right_nodes)):
        cl, cr = sides[k]
        charge[left] += cl / (cl + cr) * sigma_dot[k]
        charge[right] += cr / (cl + cr) * sigma_dot[k]
    cumulative = np.r_[0., np.cumsum(charge[1:-1])]
    resistance = 1. / capacitance
    first = -fsum((cumulative * resistance).tolist()) / fsum(resistance.tolist())
    displacement = first + cumulative
    phi_dot = np.r_[0., np.cumsum(-displacement * resistance)]
    phi_dot[-1] = 0.  # exact prescribed fixed-voltage endpoint
    contact = np.array([displacement[0] - rho_dot[0] * widths[0],
                        displacement[-1] + rho_dot[-1] * widths[-1]])
    interface = np.empty((len(sides), 2))
    traces = np.empty_like(interface)
    for k, (left, right) in enumerate(zip(system.left_nodes, system.right_nodes)):
        cl, cr = sides[k]
        trace = (cl * phi_dot[left] + cr * phi_dot[right] + sigma_dot[k]) / (cl + cr)
        traces[k] = trace
        interface[k] = [-cl * (trace - phi_dot[left]), cr * (trace - phi_dot[right])]
        displacement[system.interface_faces[k]] = interface[k, 0]
    return phi_dot, traces, displacement, interface, contact


def independent_regular_current(system, state, *, policy, require_relative_closure=True, reported=None):
    """Check original relative gates or original zero-excitation absolute gates.

    Published-array agreement allows only a declared floating-point assembly
    allowance (machine epsilon times operation count and component scale).
    It is not a physical error floor or a replacement for any physical limit.
    """
    if system.material.has_dual_ions:
        raise ValueError("independent regular R1 assembly requires the positive-ion model")
    widths, capacitance, sides = _geometry(system)
    electron, hole, ion_flux, interface_conduction = finite._currents(system, state)
    rn, rp, ri, sigma_dot = _rates(system, state, widths, electron, hole, ion_flux)
    rho_dot = Q * (rp - rn + ri)
    phi_dot, trace_dot, displacement, interface_d, contact_d = _gauss_derivative(
        widths, capacitance, rho_dot, sigma_dot, sides, system)
    conduction = electron + hole + Q * ion_flux
    internal = conduction + displacement
    interface = interface_conduction + interface_d
    contact_c = (electron + hole)[[0, -1]]
    contact = contact_c + contact_d
    all_total = np.r_[internal, interface.ravel(), contact]
    charge_rate = float(np.dot(rho_dot, widths) + np.sum(sigma_dot))
    boundary = float(contact_c[0] - contact_c[1])
    absolute = abs(charge_rate - boundary)
    scale = max(abs(charge_rate), abs(boundary), float(np.max(np.abs(conduction))),
                finite.CHARGE_SCALE_FLOOR_A_M2)
    metrics = {
        "internal_face_current_spread_relative": finite._spread(internal),
        "contact_internal_current_spread_relative": finite._spread(all_total),
        "interface_current_spread_relative": max((finite._spread(pair) for pair in interface), default=0.),
        "charge_balance_normalized": absolute / scale,
    }
    limits = effective_regular_limits(policy, require_relative_closure=require_relative_closure)
    checks = {key: {"applicable": key == "charge_balance_normalized" or bool(require_relative_closure),
                    "passed": bool(np.isfinite(value) and value <= limits[key])
                    if key == "charge_balance_normalized" or require_relative_closure else None}
              for key, value in metrics.items()}
    if not require_relative_closure:
        metrics.update({"electron_continuity_A_m2": float(np.sum(np.abs(Q * rn[1:-1] * widths[1:-1]))),
            "hole_continuity_A_m2": float(np.sum(np.abs(Q * rp[1:-1] * widths[1:-1]))),
            "electron_normalized_residual": float(np.max(np.abs(Q * rn[1:-1] * widths[1:-1]))) / system.system.current_scale,
            "hole_normalized_residual": float(np.max(np.abs(Q * rp[1:-1] * widths[1:-1]))) / system.system.current_scale,
            "trap_charge_rate_A_m2": float(np.max(np.abs(sigma_dot), initial=0.)),
            "absolute_current_spread_A_m2": float(np.ptp(all_total)),
            "ionic_face_current_A_m2": float(np.max(np.abs(Q * ion_flux), initial=0.))})
        for key in metrics:
            if key not in checks:
                checks[key] = {"applicable": True, "passed": bool(np.isfinite(metrics[key]) and metrics[key] <= limits[key])}
    arrays = {"potential_derivative_V_s": phi_dot, "trace_potential_derivative_V_s": trace_dot,
        "charge_density_derivative_C_m3_s": rho_dot, "sheet_charge_derivative_C_m2_s": sigma_dot,
        "electron_rate_m3_s": rn, "hole_rate_m3_s": rp, "positive_ion_rate_m3_s": ri,
        "internal_conduction_A_m2": conduction, "internal_displacement_A_m2": displacement,
        "internal_maxwell_A_m2": internal, "interface_maxwell_A_m2": interface,
        "contact_conduction_A_m2": contact_c, "contact_displacement_A_m2": contact_d,
        "contact_maxwell_A_m2": contact, "physical_widths_m": widths,
        "face_capacitance_F_m2": capacitance, "charge_balance_absolute_A_m2": absolute}
    checks["physical_geometry_matches"] = {"applicable": True, "passed": bool(np.array_equal(widths, system.widths))}
    current_scale = max(float(np.max(np.abs(conduction), initial=0.)),
                        float(np.max(np.abs(displacement), initial=0.)), np.finfo(float).tiny)
    arithmetic = {"kind": "floating_point_assembly_only_not_physical_error_budget",
                  "relative_allowance": 32. * len(widths) * np.finfo(float).eps,
                  "current_component_scale_A_m2": current_scale}
    if reported is not None:
        for key in ("internal_maxwell_A_m2", "interface_maxwell_A_m2", "contact_conduction_A_m2",
                    "contact_displacement_A_m2", "contact_maxwell_A_m2"):
            actual = np.asarray(reported.get(key, []))
            expected = np.asarray(arrays[key])
            if expected.size == 0 and actual.size == 0 and key in reported:
                actual = actual.reshape(expected.shape)
            match = actual.shape == expected.shape and np.all(np.isfinite(actual)) and np.allclose(
                actual, expected, rtol=arithmetic["relative_allowance"],
                atol=arithmetic["relative_allowance"] * current_scale)
            checks[key + "_reported_content_matches"] = {"applicable": True, "passed": bool(match)}
    checks["finite_values"] = {"applicable": True, "passed": all(np.all(np.isfinite(value)) for value in arrays.values())}
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_standard import verify_used_execution_limits
    verify_used_execution_limits("independent_regular_relative" if require_relative_closure
                                 else "independent_regular_zero_excitation", limits)
    reasons = [key for key, check in checks.items() if check["applicable"] and check["passed"] is not True]
    return finite._json({"schema": SCHEMA, "scope": SCOPE,
        "relative_current_applicable": bool(require_relative_closure),
        "shared_constitutive_dependencies": finite.shared_constitutive_dependencies(system) + (
            "recombination.total_recombination",),
        "arrays": arrays, "metrics": metrics, "limits": limits, "checks": checks,
        "assessment": finite._assessment(checks), "reasons": reasons, "passed": not reasons,
        "content_comparison": arithmetic})
