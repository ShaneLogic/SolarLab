"""R1 current/charge checks assembled independently of solver diagnostics.

The source state and its exact incremental coordinates are inputs. No solver
metric, stored current, stored residual, or stored storage increment is used
to construct the answer. Shared constitutive primitives are declared below;
this is an independent assembly check, not a full independent DAE solution.
"""
from __future__ import annotations

import numpy as np

from perovskite_sim.constants import EPS_0, Q
from perovskite_sim.discretization.fe_operators import bernoulli
from perovskite_sim.physics.ion_migration import ion_face_flux
from perovskite_sim.physics.two_sided_interface import (
    InterfaceTracePotentials, _material_two_sided_interface_problem,
    fixed_occupancy_carrier_tangent_from_density,
)

SCHEMA = "R1IndependentPhysicsRowV1"
SCOPE = "saved_state_independent_current_charge_assembly_not_full_DAE_oracle"
SHARED_CONSTITUTIVE_DEPENDENCIES = (
    "fe_operators.bernoulli", "ion_migration.ion_face_flux",
    "two_sided_interface._material_two_sided_interface_problem",
    "two_sided_interface.fixed_occupancy_carrier_tangent_from_density",
)
METRIC_LIMITS = {
    "internal_face_current_spread_relative": 2e-6,
    "contact_internal_current_spread_relative": 2e-6,
    "interface_current_spread_relative": 2e-6,
    "charge_balance_normalized": 1e-10,
    "inventory_relative_drift": 1e-10,
}
SPREAD_FLOOR_A_M2 = 1e-20
CHARGE_SCALE_FLOOR_A_M2 = 1.
ION_SITE_OCCUPANCY_CEILING = .999
METRIC_APPLICABILITY = {
    "internal_face_current_spread_relative": "finite_step",
    "contact_internal_current_spread_relative": "finite_step",
    "interface_current_spread_relative": "finite_step",
    "charge_balance_normalized": "finite_step",
    "inventory_relative_drift": "all_saved_rows",
}
METRIC_UNITS = {
    "internal_face_current_spread_relative": "1",
    "contact_internal_current_spread_relative": "1",
    "interface_current_spread_relative": "1",
    "charge_balance_normalized": "1",
    "inventory_relative_drift": "1",
}
FINITE_STEP_METRICS = frozenset(name for name, scope in METRIC_APPLICABILITY.items()
                              if scope == "finite_step")
ASSESSMENT_FIELDS = frozenset({"content_consistent", "conservation_compliant",
    "reference_accuracy_qualified", "reference_accuracy_status"})
ROW_FIELDS = frozenset({"schema", "scope", "finite_step", "shared_constitutive_dependencies",
    "arrays", "metrics", "limits", "checks", "reasons", "passed", "assessment"})
ARRAY_FIELDS = frozenset({"electron_current_A_m2", "hole_current_A_m2", "positive_ion_flux_m2_s",
    "internal_conduction_A_m2", "internal_displacement_A_m2", "internal_total_A_m2",
    "contact_conduction_A_m2", "contact_displacement_A_m2", "contact_total_A_m2",
    "interface_conduction_A_m2", "interface_displacement_A_m2", "interface_total_A_m2",
    "physical_widths_m", "positive_inventory_m2", "initial_positive_inventory_m2",
    "charge_rate_A_m2", "charge_balance_absolute_A_m2", "charge_balance_scale_A_m2"})
BASE_CHECK_FIELDS = frozenset(METRIC_LIMITS) | {
    "physical_geometry_matches", "finite_positive_populations", "ion_site_occupancy_physical",
    "electron_current_A_m2_state_content_matches", "hole_current_A_m2_state_content_matches",
    "positive_ion_flux_m2_s_state_content_matches"}
BASE_REPORTED_FIELDS = ("electron_current_A_m2", "hole_current_A_m2", "positive_ion_flux_m2_s")
FINITE_REPORTED_FIELDS = ("internal_displacement_A_m2", "internal_total_A_m2",
    "interface_conduction_A_m2", "interface_displacement_A_m2", "interface_total_A_m2")


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    return value


def _volumes(system):
    x = np.asarray(system.grid)
    boundaries = np.concatenate(([x[0]], .5*(x[:-1]+x[1:]), [x[-1]]))
    for position in system.material.iface_qss_interface_positions_m:
        boundaries[np.searchsorted(x, position)] = position
    return np.diff(boundaries)


def _difference(forward, backward, drive):
    result = np.empty_like(drive)
    positive = drive >= 0.
    result[positive] = forward[positive]*(-np.expm1(-drive[positive]))
    result[~positive] = backward[~positive]*np.expm1(drive[~positive])
    return result


def _interface_balance(system, state, index):
    geometry, physics, bulk = _material_two_sided_interface_problem(
        system.material, system.stack, state.n, state.p, state.phi, index,
        cross_transmission=system.dark_reference.interface_transmission)
    local = state.local[index]
    return fixed_occupancy_carrier_tangent_from_density(
        local.state_m3, geometry, physics, bulk, state.occupancy[index],
        InterfaceTracePotentials(*local.trace_potential),
        capture_multiplier=system.controls.nu_t,
        paired_bernoulli=getattr(system, "_step_reference", None) is not None).balance


def _currents(system, state):
    from .one_dimensional_mechanism_r1_backend import backend_for
    if backend_for(system).is_pair:
        from .one_dimensional_mechanism_r1_precision import independent_currents_pair
        return independent_currents_pair(system, state)
    mat, vt = system.material, system.thermal_voltage
    spacing = np.diff(system.grid)
    xi_n = np.diff(state.phi+mat.chi)/vt
    xi_p = np.diff(state.phi+mat.chi+mat.Eg)/vt
    drive_n = np.diff(system.qfn_reference)/vt+np.diff(state.dqfn)/vt
    drive_p = -(np.diff(system.qfp_reference)/vt+np.diff(state.dqfp)/vt)
    electron = Q*mat.D_n_face/spacing*_difference(
        bernoulli(xi_n)*state.n[1:], bernoulli(-xi_n)*state.n[:-1], drive_n)
    hole = Q*mat.D_p_face/spacing*_difference(
        bernoulli(xi_p)*state.p[:-1], bernoulli(-xi_p)*state.p[1:], drive_p)
    ion_flux = np.zeros_like(spacing)
    if system.controls.nu_I:
        ion_flux = ion_face_flux(state.phi, state.positive, spacing, mat.D_ion_face,
            vt, mat.P_lim_face, steric_diffusion_only=mat.ion_steric_diffusion_only,
            P_lim_node=mat.P_lim_node)
    ionic = Q*ion_flux
    interface = np.empty((system.interface_count, 2))
    for index, face in enumerate(system.interface_faces):
        # The fixed-occupancy carrier law depends on the supplied trace
        # potential; never consume state.local[index].tangent or its fluxes.
        balance = _interface_balance(system, state, index)
        flux = balance.bulk_flux_m2_s
        electron[face], hole[face] = -Q*flux[0], Q*flux[1]
        interface[index] = Q*np.array([-flux[0]+flux[1], flux[2]-flux[3]])+ionic[face]
    return electron, hole, ion_flux, interface


def _increments(system, state, previous):
    from .one_dimensional_mechanism_r1_backend import backend_for
    if backend_for(system).is_pair:
        from .one_dimensional_mechanism_r1_precision import independent_increments_pair
        return independent_increments_pair(system, state, previous)
    z = state.coordinate-previous.coordinate
    du = z[system.potential_slice]
    electron = np.zeros(system.node_count)
    hole = np.zeros(system.node_count)
    electron[1:-1] = previous.n[1:-1]*np.expm1(z[system.electron_slice]+du)
    hole[1:-1] = previous.p[1:-1]*np.expm1(z[system.hole_slice]-du)
    positive = np.zeros(system.node_count)
    active = system.positive_nodes
    positive[active] = previous.positive[active]*np.expm1(z[system.positive_slice])
    odds_change = np.expm1(z[system.trap_slice])
    f = previous.occupancy
    occupied = system.trap_density*(f*(1.-f)*odds_change/(1.+f*odds_change))
    rho = Q*(hole-electron+positive)
    potential = np.zeros(system.node_count)
    potential[[0,-1]] = state.phi[[0,-1]]-previous.phi[[0,-1]]
    potential[1:-1] = system.thermal_voltage*du
    # R1's harmonic boundary lift is data reconstructed from the actual
    # boundary change, not a solver-provided potential_increment helper.
    resistance = 1/system.material.poisson_factor.C
    lift = potential[-1]*np.r_[0., np.cumsum(resistance)]/np.sum(resistance)
    lift[-1] = potential[-1]
    potential[1:-1] += lift[1:-1]
    traces = np.asarray([system.thermal_voltage*z[system.local_slice.start+6*k:
                          system.local_slice.start+6*k+2] for k in range(system.interface_count)])
    lift_displacement = -potential[-1]/np.sum(resistance)
    for k,left in enumerate(system.left_nodes):
        cl = EPS_0*system.material.eps_r[left]/system.material.iface_qss_left_distances_m[k]
        traces[k] += lift[left]-lift_displacement/cl
    return rho, occupied, potential, traces


def _spread(values):
    values = np.asarray(values)
    if not values.size:
        return 0.
    return float(np.ptp(values))/max(float(np.max(np.abs(values))), SPREAD_FLOOR_A_M2)


def _assessment(checks):
    """Keep content, conservation and unassessed solution accuracy distinct."""
    content = {name: value for name, value in checks.items()
               if "content_matches" in name or name == "physical_geometry_matches"}
    conservation = {name: value for name, value in checks.items() if name not in content}
    def passed(group):
        return all(not check["applicable"] or check["passed"] is True for check in group.values())
    return {"content_consistent": passed(content), "conservation_compliant": passed(conservation),
            "reference_accuracy_qualified": None,
            "reference_accuracy_status": "not_assessed_shared_constitutive_laws"}


def _metric_checks(metrics, finite):
    return {name: {"applicable": finite or METRIC_APPLICABILITY[name] == "all_saved_rows",
                   "passed": bool(np.isfinite(value) and value <= METRIC_LIMITS[name])
                   if finite or METRIC_APPLICABILITY[name] == "all_saved_rows" else None}
            for name, value in metrics.items()}


def independent_physics_row(system, state, previous, dt, *, reported=None, backend=None):
    """Dispatch from the immutable backend attached to this physical system."""
    from .one_dimensional_mechanism_r1_backend import backend_for
    if backend_for(system, backend).is_pair:
        from .one_dimensional_mechanism_r1_precision_physics import independent_physics_row_pair
        return independent_physics_row_pair(system, state, previous, dt, reported=reported)
    return _legacy_independent_physics_row(system, state, previous, dt, reported=reported)


def shared_constitutive_dependencies(system):
    from .one_dimensional_mechanism_r1_backend import backend_for
    if backend_for(system).is_pair:
        from .one_dimensional_mechanism_r1_precision_physics import SHARED_CONSTITUTIVE_DEPENDENCIES as dependencies
        return dependencies
    return SHARED_CONSTITUTIVE_DEPENDENCIES


def _legacy_independent_physics_row(system, state, previous, dt, *, reported=None):
    """Reconstruct a row; acceptance checks cover every supplied time level.

    ``reported`` is optional diagnostic content to check, never an input to
    the physical reconstruction. Currents are expressed along positive x,
    before the experiment's reporting polarity.
    """
    finite = previous is not None
    if finite != bool(dt > 0.) or not np.isfinite(dt):
        raise ValueError("independent physics requires a positive dt exactly for finite steps")
    if system.material.has_dual_ions:
        raise ValueError("independent R1 assembly is restricted to the declared positive-ion model")
    widths = _volumes(system)
    electron, hole, ion_flux, interface_conduction = _currents(system, state)
    conduction = electron+hole+Q*ion_flux
    internal_displacement = np.zeros_like(conduction)
    contact_displacement = np.zeros(2)
    interface_displacement = np.zeros_like(interface_conduction)
    charge_rate = 0.
    if finite:
        rho, occupied, dphi, trace_change = _increments(system, state, previous)
        internal_displacement = system.eps_face*(-np.diff(dphi)/np.diff(system.grid))/dt
        face_charge_change = -system.material.poisson_factor.C*np.diff(dphi)
        contact_displacement = np.array([
            face_charge_change[0]-rho[0]*widths[0],
            face_charge_change[-1]+rho[-1]*widths[-1]])/dt
        charge_rate = float((np.dot(rho,widths)-Q*np.sum(occupied))/dt)
        for k,(left,right,face) in enumerate(zip(system.left_nodes,system.right_nodes,system.interface_faces)):
            mat = system.material
            cl = EPS_0*mat.eps_r[left]/mat.iface_qss_left_distances_m[k]
            cr = EPS_0*mat.eps_r[right]/mat.iface_qss_right_distances_m[k]
            drops = trace_change[k]-dphi[[left,right]]
            interface_displacement[k] = [-cl*drops[0]/dt, cr*drops[1]/dt]
            internal_displacement[face] = interface_displacement[k,0]
    contact_conduction = (electron+hole)[[0,-1]]
    contact = contact_conduction+contact_displacement
    internal = conduction+internal_displacement
    interface = interface_conduction+interface_displacement
    charge_absolute = abs(charge_rate-float(contact_conduction[0]-contact_conduction[1]))
    charge_scale = max(abs(charge_rate),abs(float(contact_conduction[0]-contact_conduction[1])),
                       float(np.max(np.abs(conduction))),CHARGE_SCALE_FLOOR_A_M2)
    initial = system.common_dc_state.positive_ion_density_m3
    inventories, initial_inventories = [], []
    for component in system.ion_layout.positive_components:
        component = np.asarray(component, dtype=int)
        inventories.append(float(np.dot(state.positive[component],widths[component])))
        initial_inventories.append(float(np.dot(initial[component],widths[component])))
    drift = max((abs(value/reference-1.) for value,reference in zip(inventories,initial_inventories)),default=0.)
    arrays = {
        "electron_current_A_m2": electron, "hole_current_A_m2": hole,
        "positive_ion_flux_m2_s": ion_flux, "internal_conduction_A_m2": conduction,
        "internal_displacement_A_m2": internal_displacement,"internal_total_A_m2":internal,
        "contact_conduction_A_m2":contact_conduction,"contact_displacement_A_m2":contact_displacement,
        "contact_total_A_m2":contact,"interface_conduction_A_m2":interface_conduction,
        "interface_displacement_A_m2":interface_displacement,"interface_total_A_m2":interface,
        "physical_widths_m":widths,"positive_inventory_m2":inventories,
        "initial_positive_inventory_m2":initial_inventories,"charge_rate_A_m2":charge_rate,
        "charge_balance_absolute_A_m2":charge_absolute,"charge_balance_scale_A_m2":charge_scale,
    }
    metrics = {"internal_face_current_spread_relative":_spread(internal),
        "contact_internal_current_spread_relative":_spread(np.r_[internal,contact]),
        "interface_current_spread_relative":max((_spread(pair) for pair in interface),default=0.),
        "charge_balance_normalized":charge_absolute/charge_scale,"inventory_relative_drift":drift}
    checks = _metric_checks(metrics, finite)
    checks["physical_geometry_matches"]={"applicable":True,"passed":bool(np.array_equal(widths,system.widths))}
    checks["finite_positive_populations"]={"applicable":True,"passed":bool(
        all(np.all(np.isfinite(value)) for value in (state.n,state.p,state.positive,state.phi,state.occupancy))
        and np.all(state.n>0) and np.all(state.p>0) and np.all(state.positive>0)
        and np.all((state.occupancy>=0)&(state.occupancy<=1)))}
    limits = np.broadcast_to(system.material.P_lim_node,state.positive.shape)
    checks["ion_site_occupancy_physical"]={"applicable":True,"passed":bool(np.all(
        state.positive[system.positive_nodes]/limits[system.positive_nodes] < ION_SITE_OCCUPANCY_CEILING))}
    for key,actual in (("electron_current_A_m2",state.current_n),("hole_current_A_m2",state.current_p),
                       ("positive_ion_flux_m2_s",state.positive_flux)):
        checks[key+"_state_content_matches"]={"applicable":True,"passed":bool(np.array_equal(actual,arrays[key]))}
    if reported is not None:
        keys=list(BASE_REPORTED_FIELDS)
        if finite:
            keys += FINITE_REPORTED_FIELDS
        for key in keys:
            expected = arrays[key] if key in keys[:3] else system.polarity*arrays[key]
            checks[key+"_reported_content_matches"]={"applicable":True,
                "passed":key in reported and bool(np.array_equal(reported[key],expected))}
    reasons=[name for name,check in checks.items() if check["applicable"] and check["passed"] is not True]
    return _json({"schema":SCHEMA,"scope":SCOPE,"finite_step":finite,
        "shared_constitutive_dependencies":SHARED_CONSTITUTIVE_DEPENDENCIES,
        "arrays":arrays,"metrics":metrics,"limits":METRIC_LIMITS,"checks":checks,
        "reasons":reasons,"passed":not reasons,"assessment":_assessment(checks)})
