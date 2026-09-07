"""Analytic density-state tangent for the supported continuous J-V slice.

Differentiate the existing eliminated-Poisson MOL equations without changing
state coordinates, constitutive laws, or the time integrator.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.discretization.fe_operators import sg_fluxes_n_jacobian, sg_fluxes_p_jacobian
from perovskite_sim.experiments.jv_sweep import _state_fields
from perovskite_sim.physics.generation import dual_cell_widths
from perovskite_sim.physics.ion_migration import ion_face_flux_jacobian
from perovskite_sim.physics.poisson import solve_poisson_prefactored
from perovskite_sim.physics.recombination import total_recombination_derivatives


class WaveformJacobianCapabilityError(ValueError):
    """An active closure is outside this analytic tangent's verified scope."""


def build_waveform_density_jacobian(x, stack, mat, voltage) -> Callable:
    unsupported = {
        "dual ions": mat.has_dual_ions,
        "interface states": bool(mat.N_iface_state),
        "interface transport/recombination": bool(mat.interface_faces) or bool(np.any(stack.interfaces)),
        "field-dependent mobility": mat.has_field_mobility,
        "radiative reabsorption": mat.has_radiative_reabsorption,
        "recombination de-spike": bool(mat.het_recomb_despike),
        "charged interfaces": stack.interface_charge_closure != "off",
        "carrier statistics": mat.carrier_statistics != "maxwell_boltzmann",
        "recombination closure": mat.degenerate_recombination_model != "maxwell_boltzmann",
        "dopant ionization": mat.dopant_ionization_model != "fully_ionized",
        "explicit bulk defects": any(getattr(mat, key, None) is not None for key in (
            "neutral_bulk_defects", "monovalent_bulk_defects", "multivalent_bulk_defects", "frozen_metastable_defects")),
        "mobile ions at outer contacts": bool(mat.D_ion_face[0] or mat.D_ion_face[-1]),
        "exclusive interface transport": bool(mat.iface_qss_exclusive_transport),
    }
    active = [name for name, enabled in unsupported.items() if enabled]
    if active:
        raise WaveformJacobianCapabilityError("Unsupported analytic waveform Jacobian: " + ", ".join(active))
    widths = dual_cell_widths(x)
    mobile = mat.D_ion_node > 0
    if np.any(mobile):
        maximum_density = float(mat.P_ion0 @ widths)/float(np.min(widths))
        if maximum_density >= 0.99*float(np.min(mat.P_lim_node[mobile])):
            raise WaveformJacobianCapabilityError("Ionic inventory may reach the steric clipping threshold")
        mixed_empty = (mat.P_ion0[:-1] == 0) != (mat.P_ion0[1:] == 0)
        if np.any(mixed_empty & (mat.D_ion_face > 0)):
            raise WaveformJacobianCapabilityError("Initial ionic profile touches a one-sided occupancy kink")
    count = len(x)
    spacing = np.diff(x)
    cell_widths = np.r_[spacing[0], 0.5*(spacing[:-1]+spacing[1:]), spacing[-1]]
    contact_speeds = ((mat.S_n_L, mat.S_n_R, mat.S_p_L, mat.S_p_R)
                      if mat.has_selective_contacts else (None, None, None, None))
    eye = np.eye(count)
    # Poisson is linear in charge. Unit-charge solves give its discrete
    # sensitivity without subtracting almost identical potential profiles.
    sensitivity = np.column_stack([solve_poisson_prefactored(
        mat.poisson_factor, Q*eye[:, index], 0.0, 0.0) for index in range(count)])
    phi_jacobian = np.concatenate((-sensitivity, sensitivity, sensitivity), axis=1)
    boundary_indices = (0, count-1, count, 2*count-1)
    pinned = [index for index, speed in zip(boundary_indices, contact_speeds) if speed is None]
    phi_jacobian[:, pinned] = 0.0
    faces = np.arange(count-1)
    interior = np.arange(1, count-1)

    def jacobian(time, state):
        bias = voltage(time) if callable(voltage) else voltage
        n, p, phi, state_fields = _state_fields(x, state, stack, bias, mat)
        electron = sg_fluxes_n_jacobian(phi+mat.chi, n, spacing, mat.D_n_face, mat.V_T_device)
        hole = sg_fluxes_p_jacobian(phi+mat.chi+mat.Eg, p, spacing, mat.D_p_face, mat.V_T_device)
        result = np.zeros((3*count, 3*count))
        for local, offset, sign, left_speed, right_speed in (
            (electron, 0, 1, *contact_speeds[:2]), (hole, count, -1, *contact_speeds[2:]),
        ):
            current_jacobian = (local.potential_left_derivative[:, None]*phi_jacobian[:-1]
                                + local.potential_right_derivative[:, None]*phi_jacobian[1:])
            current_jacobian[faces, offset+faces] += local.density_left_derivative
            current_jacobian[faces, offset+faces+1] += local.density_right_derivative
            current_jacobian[:, pinned] = 0.0
            padded = np.zeros((count+1, 3*count))
            padded[1:-1] = current_jacobian
            if left_speed is not None:
                padded[0, offset] = sign*Q*left_speed
            if right_speed is not None:
                padded[-1, offset+count-1] = -sign*Q*right_speed
            # Match the production boundary control volumes, including its
            # full outer intervals when a carrier has a finite-rate contact.
            result[offset:offset+count] = sign*np.diff(padded, axis=0)/(Q*cell_widths[:, None])
        reaction = total_recombination_derivatives(n, p, mat.ni_sq, mat.tau_n, mat.tau_p,
            mat.n1, mat.p1, mat.B_rad, mat.C_n, mat.C_p)
        for offset in (0, count):
            nodes = np.arange(count)
            result[offset+nodes, nodes] -= reaction.electron_density_derivative
            result[offset+nodes, count+nodes] -= reaction.hole_density_derivative
        ion = ion_face_flux_jacobian(phi, state_fields.P, spacing, mat.D_ion_face,
            mat.V_T_device, mat.P_lim_face, steric_diffusion_only=mat.ion_steric_diffusion_only,
            P_lim_node=mat.P_lim_node)
        # At two empty endpoints crowding enters at second order, so bare
        # SG is the unique first derivative despite the occupancy kink.
        empty_face = (state_fields.P[:-1] == 0) & (state_fields.P[1:] == 0)
        if not np.all(ion.differentiable_faces | empty_face):
            raise WaveformJacobianCapabilityError("Ion occupancy reached a non-differentiable clipping face")
        ion_faces = (ion.potential_left_derivative[:, None]*phi_jacobian[:-1]
                     + ion.potential_right_derivative[:, None]*phi_jacobian[1:])
        ion_faces[faces, 2*count+faces] += ion.density_left_derivative
        ion_faces[faces, 2*count+faces+1] += ion.density_right_derivative
        result[2*count+interior] = -np.diff(ion_faces, axis=0)/mat.dx_cell[interior, None]
        result[pinned] = 0.0
        result[:, pinned] = 0.0
        return result

    return jacobian
