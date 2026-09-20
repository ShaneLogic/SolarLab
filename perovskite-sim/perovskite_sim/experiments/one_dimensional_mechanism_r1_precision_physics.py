"""Independent R1 current/charge assembly from persistent fine state arrays.

The shared constitutive laws are recomputed through the existing independent
``_currents`` entry point.  All increments, displacement and charge are
assembled here, without solver metric helpers or saved flux/residual arrays.
This checks assembly/content, not independence of the constitutive laws.

Rounding contract: promote each frozen binary64 coefficient to DD, evaluate
the displayed multiplication/division order in DD, and take the normalized
high word only at the final current/charge output.  The experiment polarity
is applied after that rounding.  Totals then use the original binary64
conduction-plus-displacement order, matching the published current contract.
"""

from __future__ import annotations

import numpy as np

from perovskite_sim.constants import EPS_0, Q
from perovskite_sim.physics.compensated import DD
from . import one_dimensional_mechanism_r1_independent_physics as finite


_LEGACY_ROW = finite.independent_physics_row
SHARED_CONSTITUTIVE_DEPENDENCIES = (
    "one_dimensional_mechanism_r1_precision.carrier_currents_pair",
    "one_dimensional_mechanism_r1_precision.ion_flux_pair",
    "two_sided_interface._material_two_sided_interface_problem",
    "two_sided_interface.fixed_occupancy_carrier_tangent_from_density",
    "compensated.DD",
)


def _difference(value):
    return value[1:] - value[:-1]


def _fine_inputs(state):
    values = getattr(state, "fine", None)
    if not isinstance(values, dict):
        raise ValueError("independent fine assembly requires persistent state arrays")
    required = ("n_m3", "p_m3", "positive_m3", "occupancy", "phi_V", "trace_potential_V")
    if any(not isinstance(values.get(key), DD) for key in required):
        raise ValueError("independent fine assembly is missing a DD physical state field")
    return values


def independent_physics_row_pair(system, state, previous, dt, *, reported=None):
    """Reconstruct the original row contract with fine charge/displacement.

    Saved current fields and ``reported`` are read only for the unchanged
    exact content comparisons, after all physical outputs are reconstructed.
    Legacy states retain the original implementation outside the opt-in lane.
    """
    if not hasattr(state, "fine"):
        return _LEGACY_ROW(system, state, previous, dt, reported=reported)
    f = _fine_inputs(state)
    is_finite = previous is not None
    if is_finite != bool(dt > 0.0) or not np.isfinite(dt):
        raise ValueError("independent physics requires a positive dt exactly for finite steps")
    if system.material.has_dual_ions:
        raise ValueError("independent R1 assembly requires the declared positive-ion model")
    widths = finite._volumes(system)
    spacing = np.diff(system.grid)
    electron, hole, ion_flux, interface_conduction = finite._currents(system, state)
    conduction = electron + hole + Q * ion_flux
    internal_displacement = np.zeros_like(conduction)
    contact_displacement = np.zeros(2)
    interface_displacement = np.zeros_like(interface_conduction)
    charge_rate = 0.0

    if is_finite:
        p = _fine_inputs(previous)
        dn = f["n_m3"] - p["n_m3"]
        dp = f["p_m3"] - p["p_m3"]
        dion = f["positive_m3"] - p["positive_m3"]
        occupied = DD(system.trap_density) * (f["occupancy"] - p["occupancy"])
        rho = DD(Q) * (dp - dn + dion)
        dphi = f["phi_V"] - p["phi_V"]
        dtrace = f["trace_potential_V"] - p["trace_potential_V"]
        duration = DD(float(dt))

        # Keep this sequence explicit; replacing eps/dx with another frozen
        # coefficient or rounding dphi before differencing changes last bits.
        bulk = DD(system.eps_face) * (-_difference(dphi) / DD(spacing)) / duration
        internal_displacement = bulk.hi.copy()
        face_charge = -DD(system.material.poisson_factor.C) * _difference(dphi)
        left_contact = (face_charge[0] - rho[0] * DD(widths[0])) / duration
        right_contact = (face_charge[-1] + rho[-1] * DD(widths[-1])) / duration
        contact_displacement = np.array([left_contact.hi, right_contact.hi], dtype=float)
        charge = ((rho * DD(widths)).sum() - DD(Q) * occupied.sum()) / duration
        charge_rate = float(charge.hi)
        for k, (left, right, face) in enumerate(
            zip(system.left_nodes, system.right_nodes, system.interface_faces)
        ):
            mat = system.material
            cl = EPS_0 * mat.eps_r[left] / mat.iface_qss_left_distances_m[k]
            cr = EPS_0 * mat.eps_r[right] / mat.iface_qss_right_distances_m[k]
            left_displacement = -DD(cl) * (dtrace[k, 0] - dphi[left]) / duration
            right_displacement = DD(cr) * (dtrace[k, 1] - dphi[right]) / duration
            interface_displacement[k] = [left_displacement.hi, right_displacement.hi]
            internal_displacement[face] = interface_displacement[k, 0]

    contact_conduction = (electron + hole)[[0, -1]]
    contact = contact_conduction + contact_displacement
    internal = conduction + internal_displacement
    interface = interface_conduction + interface_displacement
    boundary_rate = float(contact_conduction[0] - contact_conduction[1])
    charge_absolute = abs(charge_rate - boundary_rate)
    charge_scale = max(abs(charge_rate), abs(boundary_rate),
                       float(np.max(np.abs(conduction))), finite.CHARGE_SCALE_FLOOR_A_M2)
    initial = DD(system.common_dc_state.positive_ion_density_m3)
    inventories, initial_inventories = [], []
    for component in system.ion_layout.positive_components:
        component = np.asarray(component, dtype=int)
        cell_width = DD(widths[component])
        inventories.append(float((f["positive_m3"][component] * cell_width).sum().hi))
        initial_inventories.append(float((initial[component] * cell_width).sum().hi))
    drift = max((abs(value / reference - 1.0)
                 for value, reference in zip(inventories, initial_inventories)), default=0.0)

    arrays = {
        "electron_current_A_m2": electron, "hole_current_A_m2": hole,
        "positive_ion_flux_m2_s": ion_flux, "internal_conduction_A_m2": conduction,
        "internal_displacement_A_m2": internal_displacement, "internal_total_A_m2": internal,
        "contact_conduction_A_m2": contact_conduction, "contact_displacement_A_m2": contact_displacement,
        "contact_total_A_m2": contact, "interface_conduction_A_m2": interface_conduction,
        "interface_displacement_A_m2": interface_displacement, "interface_total_A_m2": interface,
        "physical_widths_m": widths, "positive_inventory_m2": inventories,
        "initial_positive_inventory_m2": initial_inventories, "charge_rate_A_m2": charge_rate,
        "charge_balance_absolute_A_m2": charge_absolute, "charge_balance_scale_A_m2": charge_scale,
    }
    metrics = {
        "internal_face_current_spread_relative": finite._spread(internal),
        "contact_internal_current_spread_relative": finite._spread(np.r_[internal, contact]),
        "interface_current_spread_relative": max((finite._spread(pair) for pair in interface), default=0.0),
        "charge_balance_normalized": charge_absolute / charge_scale,
        "inventory_relative_drift": drift,
    }
    checks = finite._metric_checks(metrics, is_finite)
    checks["physical_geometry_matches"] = {
        "applicable": True, "passed": bool(np.array_equal(widths, system.widths))}
    checks["finite_positive_populations"] = {
        "applicable": True,
        "passed": bool(
            all(np.all(np.isfinite(f[key].hi)) and np.all(np.isfinite(f[key].lo))
                for key in ("n_m3", "p_m3", "positive_m3", "phi_V", "occupancy"))
            and np.all(f["n_m3"] > 0) and np.all(f["p_m3"] > 0)
            and np.all(f["positive_m3"] > 0)
            and np.all((f["occupancy"] >= 0) & (f["occupancy"] <= 1))) }
    limits = DD(np.broadcast_to(system.material.P_lim_node, f["positive_m3"].shape))
    active = system.positive_nodes
    checks["ion_site_occupancy_physical"] = {
        "applicable": True,
        "passed": bool(np.all(f["positive_m3"][active] / limits[active] < finite.ION_SITE_OCCUPANCY_CEILING))}
    for key, actual in (("electron_current_A_m2", state.current_n),
                        ("hole_current_A_m2", state.current_p),
                        ("positive_ion_flux_m2_s", state.positive_flux)):
        checks[key + "_state_content_matches"] = {
            "applicable": True, "passed": bool(np.array_equal(actual, arrays[key]))}
    if reported is not None:
        keys = list(finite.BASE_REPORTED_FIELDS)
        if is_finite:
            keys += finite.FINITE_REPORTED_FIELDS
        for key in keys:
            expected = arrays[key] if key in finite.BASE_REPORTED_FIELDS else system.polarity * arrays[key]
            checks[key + "_reported_content_matches"] = {
                "applicable": True,
                "passed": key in reported and bool(np.array_equal(reported[key], expected))}
    reasons = [name for name, check in checks.items()
               if check["applicable"] and check["passed"] is not True]
    return finite._json({
        "schema": finite.SCHEMA, "scope": finite.SCOPE, "finite_step": is_finite,
        "shared_constitutive_dependencies": SHARED_CONSTITUTIVE_DEPENDENCIES,
        "arrays": arrays, "metrics": metrics, "limits": finite.METRIC_LIMITS,
        "checks": checks, "reasons": reasons, "passed": not reasons,
        "assessment": finite._assessment(checks),
    })
