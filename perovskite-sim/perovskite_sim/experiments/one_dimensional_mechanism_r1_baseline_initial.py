"""One-time binary64 R1 initialization from the original fixed-QF root.

The independent solve is an initialization operation, before any accepted
state. Only its rounded potential is retained. Carrier populations are then
evaluated by the original binary64 global-QF map, with unchanged contacts.
This helper must not be used to reconstruct or repair an accepted state.
"""

from dataclasses import replace
import json

import numpy as np

from perovskite_sim.constants import EPS_0, Q
from perovskite_sim.physics.compensated import DD
from perovskite_sim.physics.two_sided_interface import _material_two_sided_interface_problem
from perovskite_sim.solver.mol import poisson_right_boundary


def _words(value):
    value = DD(np.asarray(value, dtype=float))
    return {"hi": value.hi.tolist(), "lo": value.lo.tolist()}


def canonicalize_baseline_initial_dc(stack, material, qf, dc, dynamic, *,
                                     equilibrium_occupancy, trap_density_m2):
    """Return new n/p/phi arrays; preserve QF, ions, occupancy and DC evidence.

There is one call to the existing bounded independent Poisson solver, with
the same global anchors and fixed inputs used by the elimination reference.
No optimizer, transient step, neighboring-float search or retry runs here.
The existing constructor subsequently builds local states from these actual
binary64 arrays. Import paths retain the saved arrays and never call this.
"""
    from .one_dimensional_mechanism_r1_precision import (
        IndependentPoissonInputs, solve_independent_poisson,
    )

    if (float(qf.V_app) != 0.0 or material.has_dual_ions
            or not material.ion_steric_diffusion_only
            or dc.negative_ion_density_m3 is not None
            or dc.bulk_trap_occupancy is not None):
        raise ValueError("baseline initialization requires the original zero-voltage positive-ion model")
    left = tuple(int(value) for value in material.iface_qss_left_nodes)
    right = tuple(int(value) for value in material.iface_qss_right_nodes)
    if len(left) != 1 or len(right) != 1:
        raise ValueError("baseline initialization requires the original single interface")
    count = np.asarray(qf.phi0).size
    seed_n = np.asarray(dynamic.y[:count], dtype=float)
    seed_p = np.asarray(dynamic.y[count:2 * count], dtype=float)
    geometry, _, _ = _material_two_sided_interface_problem(
        material, stack, seed_n, seed_p, dynamic.phi, 0, cross_transmission=1.0,
    )
    if (geometry.fixed_sheet_charge_C_m2 != 0.0
            or geometry.potential_jump_right_minus_left_V != 0.0):
        raise ValueError("baseline initialization requires zero static sheet charge and trace jump")
    # Match the original physical-volume sheet weights and multiplication
    # association; no direct state sheet or electrostatic residual is imported.
    c_left = (EPS_0 * float(material.eps_r[left[0]])
              / float(material.iface_qss_left_distances_m[0]))
    c_right = (EPS_0 * float(material.eps_r[right[0]])
               / float(material.iface_qss_right_distances_m[0]))
    total = c_left + c_right
    sheet = DD(Q) * DD(trap_density_m2) * (DD(equilibrium_occupancy) - DD(dc.interface_occupancy))
    fixed = {
        "dqfn_V": _words(dc.electron_qf_increment_V),
        "dqfp_V": _words(dc.hole_qf_increment_V),
        "positive_m3": _words(dc.positive_ion_density_m3),
        "occupancy": _words(dc.interface_occupancy),
        "sheet_charge_C_m2": {"hi": sheet.hi.tolist(), "lo": sheet.lo.tolist()},
    }
    preparation = {
        "phi0_V": qf.phi0, "log_n0": qf.log_n0, "log_p0": qf.log_p0,
        "contact_n_m3": seed_n[[0, -1]], "contact_p_m3": seed_p[[0, -1]],
        "thermal_voltage_V": material.V_T_device,
        "poisson_capacitance_F_m2": material.poisson_factor.C,
        "poisson_width_m": material.poisson_factor.h_cell,
        "N_D_m3": material.N_D, "N_A_m3": material.N_A,
        "ion_background_m3": material.P_ion0,
        "interface_nodes": list(zip(left, right)),
        "sheet_weights": [[c_left / total, c_right / total]],
        "boundary_phi_V": [0.0, poisson_right_boundary(material, 0.0)],
    }
    preparation = {key: np.asarray(value).tolist() for key, value in preparation.items()}
    encode = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    inputs = IndependentPoissonInputs(
        encode(fixed), encode(preparation), encode(_words(dynamic.phi)),
        "independent_legacy_fixed_qf_evaluation",
    )
    fields, _ = solve_independent_poisson(inputs)
    # Quantize before using phi in the baseline map, as in the baseline
    # eliminated constitutive reference. Never retain fine n/p or phi lows.
    phi = fields["phi_V"].to_float().copy()
    if phi.shape != (count,) or not np.all(np.isfinite(phi)):
        raise ValueError("baseline initial Poisson returned an invalid potential")
    phi[[0, -1]] = preparation["boundary_phi_V"]
    dphi = phi - qf.phi0
    n = np.exp(qf.log_n0 + (dc.electron_qf_increment_V + dphi) / material.V_T_device)
    p = np.exp(qf.log_p0 + (dc.hole_qf_increment_V - dphi) / material.V_T_device)
    if not (np.all(np.isfinite(n)) and np.all(np.isfinite(p))
            and np.all(n > 0.0) and np.all(p > 0.0)):
        raise ValueError("baseline initial global-QF map returned invalid populations")
    n[[0, -1]], p[[0, -1]] = seed_n[[0, -1]], seed_p[[0, -1]]
    return replace(dc, electron_density_m3=n, hole_density_m3=p, potential_V=phi)
