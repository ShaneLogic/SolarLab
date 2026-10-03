"""Absolute electrostatic rows of the R1 binary64 baseline state.

The inputs are the actual one-word state and original material coefficients.
DD is only an expression accumulator: no primary low words are manufactured,
stored, or supplied to a later constitutive evaluation. The public rows are
rounded to binary64 once, after cancellation of the physical terms.
"""
from dataclasses import dataclass

import numpy as np

from perovskite_sim.constants import EPS_0, Q
from perovskite_sim.physics.compensated import DD
from . import one_dimensional_mechanism_r1_local_carrier as local_carrier
from .interface_defect_transient import _InterfaceTransientSystem


@dataclass(frozen=True, slots=True)
class BaselineElectrostaticRows:
    poisson_residual_C_m2: np.ndarray
    local_electrostatic_residual: np.ndarray
    sheet_charge_C_m2: np.ndarray


def _put(value, index, part):
    high, low = value.hi.copy(), value.lo.copy()
    high[index], low[index] = part.hi, part.lo
    return DD._trusted_parts(high, low)


def _diff(value):
    return value[1:] - value[:-1]


def baseline_physical_electrostatic_rows(system, state):
    """Read only primary arrays; saved residuals cannot define the equation.

    This is the existing positive-ion R1 charge model. Reject unsupported
    charge sources instead of silently omitting one from this assembly.
    Geometry and capacitances use the same material helper and binary64
    coefficients as the inherited analytic Jacobian.
    """
    material = system.material
    if (hasattr(state, "fine") or hasattr(state, "input_lift")
            or material.has_dual_ions or state.negative is not None):
        raise ValueError("baseline electrostatics requires a one-word positive-ion state")
    for owner in (material, system.system.source_mat):
        if any(getattr(owner, name, None) is not None for name in (
                "monovalent_bulk_defects", "multivalent_bulk_defects",
                "frozen_metastable_defects")):
            raise ValueError("baseline electrostatics does not support additional bulk charge")

    phi = DD(state.phi)
    sheet = -DD(Q) * DD(system.trap_density) * (
        DD(state.occupancy) - DD(system.equilibrium_occupancy))
    rho = DD(Q) * (DD(state.p) - DD(state.n) + DD(material.N_D)
                   - DD(material.N_A) + DD(state.positive) - DD(material.P_ion0))
    factor = material.poisson_factor
    poisson = (_diff(DD(factor.C) * _diff(phi))
               + rho[1:-1] * DD(factor.h_cell))
    local = np.empty(2 * system.interface_count)
    for index, (left, right, item) in enumerate(
            zip(system.left_nodes, system.right_nodes, state.local)):
        weight_left, weight_right = system._sheet_weights(index)
        poisson = _put(poisson, left - 1,
                       poisson[left - 1] + DD(weight_left) * sheet[index])
        poisson = _put(poisson, right - 1,
                       poisson[right - 1] + DD(weight_right) * sheet[index])
        geometry, _, _ = local_carrier._material_two_sided_interface_problem(
            material, system.stack, state.n, state.p, state.phi, index,
            cross_transmission=system.dark_reference.interface_transmission)
        capacitance_left = EPS_0 * geometry.eps_r_left / geometry.left_distance_m
        capacitance_right = EPS_0 * geometry.eps_r_right / geometry.right_distance_m
        trace = DD(item.trace_potential)
        local[2 * index] = (trace[1] - trace[0]
            - DD(geometry.potential_jump_right_minus_left_V)).to_float().item()
        local[2 * index + 1] = (
            DD(capacitance_left) * (trace[0] - phi[left])
            + DD(capacitance_right) * (trace[1] - phi[right])
            - DD(geometry.fixed_sheet_charge_C_m2) - sheet[index]).to_float().item()
    return BaselineElectrostaticRows(poisson.to_float(), local, sheet.to_float())


def baseline_electrostatic_diagnostics(system, state, previous=None):
    """Reconstruct the old incremental rows as evidence, without mutating state.

    These historical rows intentionally retain the former lift-free evaluation
    and previous saved residual. They are not returned as live Newton rows.
    A caller saving failure/replay evidence can request them for the actual
    state pair without a mutable per-system last-evaluation cache.
    """
    physical = baseline_physical_electrostatic_rows(system, state)
    previous = system._step_reference if previous is None else previous
    result = {
        "schema": "R1BaselineAbsoluteElectrostaticsV1",
        "input_representation": "float64-baseline",
        "evaluation": "absolute_rows_from_actual_binary64_primary_fields",
        "poisson_residual_C_m2": physical.poisson_residual_C_m2,
        "local_electrostatic_residual": physical.local_electrostatic_residual,
        "sheet_charge_C_m2": physical.sheet_charge_C_m2,
        "historical_incremental_is_live_equation": False,
    }
    if previous is None:
        return result
    storage = system.storage_increment(state, previous)
    rho_increment = system._increment_charge_density(storage)
    occupied = storage[2 * system.interior_count:
                       2 * system.interior_count + system.interface_count]
    # Exactly the historical PhysicalInterfaceIonSystem lift-free increment.
    dphi = _InterfaceTransientSystem.potential_increment(system, state, previous)
    if hasattr(system, "_lift"):
        dphi[-1] -= system._lift[-1]
    factor = system.material.poisson_factor
    poisson = (previous.poisson_residual + np.diff(factor.C * np.diff(dphi))
               + rho_increment[1:-1] * factor.h_cell)
    local = np.empty(2 * system.interface_count)
    for index, (left, right) in enumerate(zip(system.left_nodes, system.right_nodes)):
        weight_left, weight_right = system._sheet_weights(index)
        poisson[left - 1] -= weight_left * Q * occupied[index]
        poisson[right - 1] -= weight_right * Q * occupied[index]
        trace = system.thermal_voltage * state.coordinate[system._local_block_slice(index)][:2]
        cl = EPS_0 * system.material.eps_r[left] / system.material.iface_qss_left_distances_m[index]
        cr = EPS_0 * system.material.eps_r[right] / system.material.iface_qss_right_distances_m[index]
        local[2 * index:2 * index + 2] = previous.local_residual[6 * index:6 * index + 2] + np.array([
            trace[1] - trace[0], cl * (trace[0] - dphi[left])
            + cr * (trace[1] - dphi[right]) + Q * occupied[index]])
    result.update({
        "historical_previous_poisson_residual_C_m2": previous.poisson_residual.copy(),
        "historical_previous_local_electrostatic_residual": np.concatenate([
            previous.local_residual[6 * k:6 * k + 2] for k in range(system.interface_count)]),
        "historical_incremental_poisson_residual_C_m2": poisson,
        "historical_incremental_local_electrostatic_residual": local,
        "absolute_minus_historical_poisson_C_m2": physical.poisson_residual_C_m2 - poisson,
        "absolute_minus_historical_local": physical.local_electrostatic_residual - local,
    })
    return result
