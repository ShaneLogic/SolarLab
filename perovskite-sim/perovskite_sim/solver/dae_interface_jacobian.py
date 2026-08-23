"""Analytic tangent for the clamp-inactive algebraic-interface DAE slice."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.special import expit
from scipy.sparse import coo_matrix, csr_matrix

from perovskite_sim.constants import Q
from perovskite_sim.discretization.fe_operators import ScharfetterGummelFaceJacobian
from perovskite_sim.models.device import electrical_interfaces
from perovskite_sim.physics.interface_plane import (
    FERMI_RICHARDSON,
    compute_interface_srh_occupancy_on_state,
)
from perovskite_sim.physics.recombination import (
    bulk_srh_denominator,
    total_recombination_derivatives,
)
from perovskite_sim.solver.dae_interface_states import AlgebraicInterfaceStateDAE
from perovskite_sim.solver.dae_jacobian import build_carrier_face_jacobians


_PLANE_EXPONENT_LIMIT = 30.0
_LOG_ACTIVITY_LIMIT = 700.0
_DENSITY_FLOOR_M3 = 1.0e-300


class AlgebraicInterfaceJacobianCapabilityError(ValueError):
    """The operating point is outside the declared smooth interface slice."""


def _readonly(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise AlgebraicInterfaceJacobianCapabilityError(
            f"{name} must be finite with shape {shape}"
        )
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class AlgebraicInterfaceLocalLinearization:
    """Local fluxes and exact coordinate tangents for one interface.

    Bulk-coordinate columns are ``(log n_R, log p_R, log n_L, log p_L)``;
    potential columns are ``(phi_R, phi_L)``; interface columns follow the
    physical state order ``(n_R, p_R, n_L, p_L)``.
    """

    interface_node: int
    bulk_flux_m2_s: np.ndarray
    cross_flux_m2_s: np.ndarray
    srh_flux_m2_s: np.ndarray
    state_flux_m2_s: np.ndarray
    bulk_bulk_log_jacobian_m2_s: np.ndarray
    bulk_potential_jacobian_m2_s_V: np.ndarray
    bulk_interface_coordinate_jacobian_m2_s: np.ndarray
    cross_interface_coordinate_jacobian_m2_s: np.ndarray
    srh_interface_coordinate_jacobian_m2_s: np.ndarray
    state_bulk_log_jacobian_m2_s: np.ndarray
    state_potential_jacobian_m2_s_V: np.ndarray
    state_interface_coordinate_jacobian_m2_s: np.ndarray
    projected_bulk_state_m3: np.ndarray
    srh_occupancy: float
    srh_denominator_m2_s: float
    minimum_projection_occupation_margin: float
    minimum_cross_occupation_margin: float
    minimum_srh_occupancy_margin: float
    minimum_interface_density_margin_m3: float
    minimum_interface_dos_margin_m3: float


@dataclass(frozen=True, slots=True)
class AlgebraicInterfaceStructuredJacobian:
    """Sparse DAE tangent and its local smooth-capability evidence."""

    matrix: csr_matrix
    nonzero_count: int
    minimum_bulk_srh_denominator_s_m3: float
    local_interface: AlgebraicInterfaceLocalLinearization
    electron_current_faces_A_m2: np.ndarray
    hole_current_faces_A_m2: np.ndarray


def _fermi_projection(
    density_m3: float,
    density_of_states_m3: float,
    exponent: float,
    *,
    label: str,
) -> tuple[float, float, float]:
    """Return projected density, derivative wrt log density, and margin."""
    if (
        not math.isfinite(density_m3)
        or density_m3 <= _DENSITY_FLOOR_M3
        or not math.isfinite(density_of_states_m3)
        or density_of_states_m3 <= 0.0
    ):
        raise AlgebraicInterfaceJacobianCapabilityError(
            f"{label} activates the positive-density clamp"
        )
    log_activity = (
        math.log(density_m3) - math.log(density_of_states_m3) + exponent
    )
    if (
        not math.isfinite(log_activity)
        or abs(log_activity) >= _LOG_ACTIVITY_LIMIT
    ):
        raise AlgebraicInterfaceJacobianCapabilityError(
            f"{label} activates the Fermi log-activity clamp"
        )
    occupation = float(expit(log_activity))
    if not 0.0 < occupation < 1.0:
        raise AlgebraicInterfaceJacobianCapabilityError(
            f"{label} has saturated Fermi occupation"
        )
    projected = density_of_states_m3 * occupation
    slope = projected * (1.0 - occupation)
    return projected, slope, min(occupation, 1.0 - occupation)


def _barrier_occupation(
    state_m3: float,
    density_of_states_m3: float,
    barrier_normalized: float,
    *,
    label: str,
) -> tuple[float, float, float]:
    """Return occupation, derivative wrt physical state, and clamp margin."""
    if (
        not math.isfinite(state_m3)
        or state_m3 <= _DENSITY_FLOOR_M3
        or not math.isfinite(density_of_states_m3)
        or density_of_states_m3 <= 0.0
    ):
        raise AlgebraicInterfaceJacobianCapabilityError(
            f"{label} activates the cross-flux density clamp"
        )
    log_activity = (
        math.log(state_m3)
        - math.log(density_of_states_m3)
        - barrier_normalized
    )
    if (
        not math.isfinite(log_activity)
        or abs(log_activity) >= _LOG_ACTIVITY_LIMIT
    ):
        raise AlgebraicInterfaceJacobianCapabilityError(
            f"{label} activates the cross-flux log-activity clamp"
        )
    occupation = float(expit(log_activity))
    if not 0.0 < occupation < 1.0:
        raise AlgebraicInterfaceJacobianCapabilityError(
            f"{label} has saturated cross-flux occupation"
        )
    derivative = occupation * (1.0 - occupation) / state_m3
    return occupation, derivative, min(occupation, 1.0 - occupation)


def _interface_srh_linearization(
    model: AlgebraicInterfaceStateDAE,
    state_m3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    material = model.material
    interfaces = electrical_interfaces(model.stack)
    if len(interfaces) != 1:
        raise AlgebraicInterfaceJacobianCapabilityError(
            "the analytic interface tangent requires exactly one capture pair"
        )
    v_n, v_p = (float(value) for value in interfaces[0])
    if material.interface_calibration_factor:
        calibration = float(material.interface_calibration_factor[0])
        v_n *= calibration
        v_p *= calibration
    if not all(math.isfinite(value) and value > 0.0 for value in (v_n, v_p)):
        raise AlgebraicInterfaceJacobianCapabilityError(
            "interface capture velocities must be finite and positive"
        )
    if np.any(state_m3 <= 0.0):
        raise AlgebraicInterfaceJacobianCapabilityError(
            "interface SRH positive-part clamp must be inactive"
        )
    if material.interface_n1_L and material.interface_n1_R:
        n1_left = float(material.interface_n1_L[0])
        n1_right = float(material.interface_n1_R[0])
        p1_left = float(material.interface_p1_L[0])
        p1_right = float(material.interface_p1_R[0])
    else:
        n1_left = n1_right = float(material.interface_n1[0])
        p1_left = p1_right = float(material.interface_p1[0])
    trap_levels = np.array(
        [n1_right, p1_right, n1_left, p1_left], dtype=float
    )
    if not np.all(np.isfinite(trap_levels)) or np.any(trap_levels < 0.0):
        raise AlgebraicInterfaceJacobianCapabilityError(
            "interface trap-level densities must be finite and non-negative"
        )

    n_right, p_right, n_left, p_left = state_m3
    numerator = v_n * (n_left + n_right) + v_p * (p1_left + p1_right)
    denominator = v_n * (
        n_left + n_right + n1_left + n1_right
    ) + v_p * (p_left + p_right + p1_left + p1_right)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise AlgebraicInterfaceJacobianCapabilityError(
            "interface SRH denominator must be finite and positive"
        )
    occupancy = numerator / denominator
    if not 0.0 < occupancy < 1.0:
        raise AlgebraicInterfaceJacobianCapabilityError(
            "interface SRH occupancy clamp must be inactive"
        )
    empty = 1.0 - occupancy
    flux = np.array(
        [
            -v_n * (n_right * empty - n1_right * occupancy),
            -v_p * (p_right * occupancy - p1_right * empty),
            -v_n * (n_left * empty - n1_left * occupancy),
            -v_p * (p_left * occupancy - p1_left * empty),
        ],
        dtype=float,
    )

    d_occupancy = np.array(
        [
            v_n * (denominator - numerator) / denominator**2,
            -v_p * numerator / denominator**2,
            v_n * (denominator - numerator) / denominator**2,
            -v_p * numerator / denominator**2,
        ],
        dtype=float,
    )
    physical_jacobian = np.empty((4, 4), dtype=float)
    electron_rows = ((0, n_right, n1_right), (2, n_left, n1_left))
    hole_rows = ((1, p_right, p1_right), (3, p_left, p1_left))
    for row, density, trap_density in electron_rows:
        for column in range(4):
            delta = 1.0 if column == row else 0.0
            physical_jacobian[row, column] = -v_n * (
                delta * empty
                - (density + trap_density) * d_occupancy[column]
            )
    for row, density, trap_density in hole_rows:
        for column in range(4):
            delta = 1.0 if column == row else 0.0
            physical_jacobian[row, column] = -v_p * (
                delta * occupancy
                + (density + trap_density) * d_occupancy[column]
            )
    return flux, physical_jacobian, occupancy, denominator


def linearize_algebraic_interface_response(
    model: AlgebraicInterfaceStateDAE,
    coordinate: np.ndarray,
) -> AlgebraicInterfaceLocalLinearization:
    """Evaluate the exact local tangent while every production clamp is inert."""
    material = model.material
    layout = model.layout
    if layout.interface_count != 1 or len(material.interface_nodes) != 1:
        raise AlgebraicInterfaceJacobianCapabilityError(
            "the first analytic interface tangent supports exactly one interface"
        )
    if (
        material.iface_qss_transport_model != FERMI_RICHARDSON
        or not material.iface_state_physical_offsets
        or material.iface_state_partition
    ):
        raise AlgebraicInterfaceJacobianCapabilityError(
            "the tangent requires unpartitioned physical Fermi-Richardson transport"
        )
    n, p, state, phi = model.physical_fields(coordinate)
    right = int(material.interface_nodes[0])
    left = right - 1
    thermal_voltage = float(material.V_T_device)
    velocity = float(material.iface_state_v_th)
    transmission = float(material.iface_qss_cross_transmission)
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (thermal_voltage, velocity, transmission)
    ):
        raise AlgebraicInterfaceJacobianCapabilityError(
            "interface thermal controls must be finite and positive"
        )

    exponent_left = (float(phi[right]) - float(phi[left])) / thermal_voltage
    if (
        not math.isfinite(exponent_left)
        or abs(exponent_left) >= _PLANE_EXPONENT_LIMIT
    ):
        raise AlgebraicInterfaceJacobianCapabilityError(
            "left plane projection exponent clamp must be inactive"
        )
    bulk_density = np.array(
        [n[right], p[right], n[left], p[left]], dtype=float
    )
    density_of_states = np.array(
        [
            material.N_C_physical[right],
            material.N_V_physical[right],
            material.N_C_physical[left],
            material.N_V_physical[left],
        ],
        dtype=float,
    )
    projection_exponents = np.array(
        [0.0, -0.0, exponent_left, -exponent_left], dtype=float
    )
    projected = np.empty(4, dtype=float)
    projection_slope = np.empty(4, dtype=float)
    projection_margins = np.empty(4, dtype=float)
    for index in range(4):
        projected[index], projection_slope[index], projection_margins[index] = (
            _fermi_projection(
                float(bulk_density[index]),
                float(density_of_states[index]),
                float(projection_exponents[index]),
                label=f"interface projection component {index}",
            )
        )

    state_coordinate_chain = model.interface_coordinate_jacobian_m3(coordinate)
    if np.any(state_coordinate_chain <= 0.0):
        raise AlgebraicInterfaceJacobianCapabilityError(
            "interface logit map must remain strictly inside both DOS bounds"
        )
    bulk_log_jacobian = np.diag(velocity * projection_slope)
    bulk_potential_jacobian = np.zeros((4, 2), dtype=float)
    bulk_potential_jacobian[2, 0] = (
        velocity * projection_slope[2] / thermal_voltage
    )
    bulk_potential_jacobian[2, 1] = -bulk_potential_jacobian[2, 0]
    bulk_potential_jacobian[3, 0] = (
        -velocity * projection_slope[3] / thermal_voltage
    )
    bulk_potential_jacobian[3, 1] = -bulk_potential_jacobian[3, 0]
    bulk_interface_jacobian = np.diag(-velocity * state_coordinate_chain)

    d_chi = float(material.interface_chi_step[0])
    d_eg = float(material.interface_Eg_step[0])
    electron_delta = -d_chi
    hole_delta = d_chi + d_eg
    barriers = np.array(
        [
            max(-electron_delta, 0.0),
            max(-hole_delta, 0.0),
            max(electron_delta, 0.0),
            max(hole_delta, 0.0),
        ],
        dtype=float,
    ) / thermal_voltage
    cross_occupations = np.empty(4, dtype=float)
    cross_physical_slopes = np.empty(4, dtype=float)
    cross_margins = np.empty(4, dtype=float)
    for index in range(4):
        (
            cross_occupations[index],
            cross_physical_slopes[index],
            cross_margins[index],
        ) = _barrier_occupation(
            float(state[index]),
            float(density_of_states[index]),
            float(barriers[index]),
            label=f"cross flux component {index}",
        )
    temperature = float(material.T_device)
    electron_supply = (
        transmission
        * min(float(material.A_star_n[left]), float(material.A_star_n[right]))
        * temperature**2
        / Q
    )
    hole_supply = (
        transmission
        * min(float(material.A_star_p[left]), float(material.A_star_p[right]))
        * temperature**2
        / Q
    )
    if not all(
        math.isfinite(value) and value >= 0.0
        for value in (electron_supply, hole_supply)
    ):
        raise AlgebraicInterfaceJacobianCapabilityError(
            "Richardson supplies must be finite and non-negative"
        )
    cross_physical_jacobian = np.zeros((4, 4), dtype=float)
    cross_physical_jacobian[0, 0] = -electron_supply * cross_physical_slopes[0]
    cross_physical_jacobian[0, 2] = electron_supply * cross_physical_slopes[2]
    cross_physical_jacobian[2] = -cross_physical_jacobian[0]
    cross_physical_jacobian[1, 1] = -hole_supply * cross_physical_slopes[1]
    cross_physical_jacobian[1, 3] = hole_supply * cross_physical_slopes[3]
    cross_physical_jacobian[3] = -cross_physical_jacobian[1]
    cross_interface_jacobian = (
        cross_physical_jacobian * state_coordinate_chain[np.newaxis, :]
    )

    srh_flux, srh_physical_jacobian, srh_occupancy, srh_denominator = (
        _interface_srh_linearization(model, state)
    )
    srh_interface_jacobian = (
        srh_physical_jacobian * state_coordinate_chain[np.newaxis, :]
    )
    response = model.interface_response(n, p, state, phi)
    production_srh = compute_interface_srh_occupancy_on_state(
        state,
        model.stack,
        material,
    )
    if not np.allclose(srh_flux, production_srh, rtol=5.0e-14, atol=1.0e-3):
        raise AlgebraicInterfaceJacobianCapabilityError(
            "analytic SRH branch does not match the production interface response"
        )

    state_interface_jacobian = (
        bulk_interface_jacobian
        + cross_interface_jacobian
        + srh_interface_jacobian
    )
    shape4 = (4,)
    shape44 = (4, 4)
    return AlgebraicInterfaceLocalLinearization(
        interface_node=right,
        bulk_flux_m2_s=_readonly(response.bulk_flux_m2_s, shape4, "bulk flux"),
        cross_flux_m2_s=_readonly(response.cross_flux_m2_s, shape4, "cross flux"),
        srh_flux_m2_s=_readonly(production_srh, shape4, "SRH flux"),
        state_flux_m2_s=_readonly(response.state_flux_m2_s, shape4, "state flux"),
        bulk_bulk_log_jacobian_m2_s=_readonly(
            bulk_log_jacobian, shape44, "bulk/log-density tangent"
        ),
        bulk_potential_jacobian_m2_s_V=_readonly(
            bulk_potential_jacobian, (4, 2), "bulk/potential tangent"
        ),
        bulk_interface_coordinate_jacobian_m2_s=_readonly(
            bulk_interface_jacobian, shape44, "bulk/interface tangent"
        ),
        cross_interface_coordinate_jacobian_m2_s=_readonly(
            cross_interface_jacobian, shape44, "cross/interface tangent"
        ),
        srh_interface_coordinate_jacobian_m2_s=_readonly(
            srh_interface_jacobian, shape44, "SRH/interface tangent"
        ),
        state_bulk_log_jacobian_m2_s=_readonly(
            bulk_log_jacobian, shape44, "state/log-density tangent"
        ),
        state_potential_jacobian_m2_s_V=_readonly(
            bulk_potential_jacobian, (4, 2), "state/potential tangent"
        ),
        state_interface_coordinate_jacobian_m2_s=_readonly(
            state_interface_jacobian, shape44, "state/interface tangent"
        ),
        projected_bulk_state_m3=_readonly(projected, shape4, "projected bulk state"),
        srh_occupancy=float(srh_occupancy),
        srh_denominator_m2_s=float(srh_denominator),
        minimum_projection_occupation_margin=float(np.min(projection_margins)),
        minimum_cross_occupation_margin=float(np.min(cross_margins)),
        minimum_srh_occupancy_margin=float(
            min(srh_occupancy, 1.0 - srh_occupancy)
        ),
        minimum_interface_density_margin_m3=float(np.min(state)),
        minimum_interface_dos_margin_m3=float(
            np.min(layout.interface_capacity_m3 - state)
        ),
    )


def _assemble_algebraic_interface_structured_jacobian(
    model: AlgebraicInterfaceStateDAE,
    coordinate: np.ndarray,
    *,
    derivative: np.ndarray | None,
    backward_euler_dt_s: float | None,
) -> AlgebraicInterfaceStructuredJacobian:
    layout = model.layout
    count = layout.node_count
    if (derivative is None) == (backward_euler_dt_s is None):
        raise ValueError(
            "provide exactly one of derivative or backward_euler_dt_s"
        )
    if derivative is not None:
        rate = np.asarray(derivative, dtype=float)
        if rate.shape != (layout.size,) or not np.all(np.isfinite(rate)):
            raise ValueError(
                "algebraic-interface DAE derivative must be finite and layout-sized"
            )
        dt_s = None
    else:
        dt_s = float(backward_euler_dt_s)
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("backward_euler_dt_s must be finite and positive")
        rate = None

    n_raw, p_raw, _state, phi = model.physical_fields(coordinate)
    material = model.material
    if material.has_field_mobility:
        raise AlgebraicInterfaceJacobianCapabilityError(
            "field mobility is outside the algebraic-interface DAE slice"
        )
    if material.has_radiative_reabsorption:
        raise AlgebraicInterfaceJacobianCapabilityError(
            "self-consistent radiative reabsorption has no local tangent"
        )
    if material.het_recomb_despike > 0.0 and material.het_recomb_nodes:
        raise AlgebraicInterfaceJacobianCapabilityError(
            "heterojunction recombination de-spike has no smooth tangent"
        )

    local = linearize_algebraic_interface_response(model, coordinate)
    interface_face = local.interface_node - 1
    exclusive_faces = tuple(
        int(value)
        for value in material.carrier_params.get("exclusive_interface_faces", ())
    )
    if exclusive_faces != (interface_face,):
        raise AlgebraicInterfaceJacobianCapabilityError(
            "exactly the physical interface face must be excluded from bulk SG"
        )

    # assemble_rhs overwrites the four ohmic carrier reservoirs before it
    # evaluates transport and recombination.
    n = np.array(n_raw, copy=True)
    p = np.array(p_raw, copy=True)
    n[[0, -1]] = (material.n_L, material.n_R)
    p[[0, -1]] = (material.p_L, material.p_R)
    electron_face, hole_face = build_carrier_face_jacobians(model, n, p, phi)
    electron_current = np.array(electron_face.flux, dtype=float, copy=True)
    hole_current = np.array(hole_face.flux, dtype=float, copy=True)
    electron_current[interface_face] = 0.0
    hole_current[interface_face] = 0.0

    denominator = bulk_srh_denominator(
        n,
        p,
        material.tau_n,
        material.tau_p,
        material.n1,
        material.p1,
    )
    if not np.all(np.isfinite(denominator)) or np.any(denominator <= 0.0):
        raise AlgebraicInterfaceJacobianCapabilityError(
            "bulk SRH denominator must be finite and positive"
        )
    reaction = total_recombination_derivatives(
        n,
        p,
        material.ni_sq,
        material.tau_n,
        material.tau_p,
        material.n1,
        material.p1,
        material.B_rad,
        material.C_n,
        material.C_p,
    )
    reaction_arrays = (
        reaction.rate,
        reaction.electron_density_derivative,
        reaction.hole_density_derivative,
    )
    if any(
        np.asarray(value).shape != (count,) or not np.all(np.isfinite(value))
        for value in reaction_arrays
    ):
        raise AlgebraicInterfaceJacobianCapabilityError(
            "bulk recombination tangent is not finite and node matched"
        )

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []

    def add(row: int, column: int, value: float) -> None:
        number = float(value)
        if not math.isfinite(number):
            raise AlgebraicInterfaceJacobianCapabilityError(
                "structured interface DAE assembly produced a non-finite entry"
            )
        if number != 0.0:
            rows.append(row)
            columns.append(column)
            values.append(number)

    for index in (0, count - 1, count, 2 * count - 1):
        add(index, index, 1.0)

    for node in range(1, count - 1):
        electron_row = node
        hole_row = count + node
        electron_scale = layout.electron_rate_scale_m3_s[node]
        hole_scale = layout.hole_rate_scale_m3_s[node]
        electron_storage = n[node] / dt_s if dt_s is not None else n[node] * rate[node]
        hole_storage = (
            p[node] / dt_s
            if dt_s is not None
            else p[node] * rate[count + node]
        )
        add(electron_row, node, electron_storage / electron_scale)
        add(hole_row, count + node, hole_storage / hole_scale)
        d_recombination_dlogn = (
            reaction.electron_density_derivative[node] * n[node]
        )
        d_recombination_dlogp = (
            reaction.hole_density_derivative[node] * p[node]
        )
        for row, scale in (
            (electron_row, electron_scale),
            (hole_row, hole_scale),
        ):
            add(row, node, d_recombination_dlogn / scale)
            add(row, count + node, d_recombination_dlogp / scale)

    potential_offset = layout.potential_slice.start

    def distribute_face(
        face: int,
        face_tangent: ScharfetterGummelFaceJacobian,
        density: np.ndarray,
        density_offset: int,
        *,
        electron: bool,
    ) -> None:
        face_columns: list[tuple[int, float]] = []
        if 0 < face < count - 1:
            face_columns.append(
                (
                    density_offset + face,
                    face_tangent.density_left_derivative[face] * density[face],
                )
            )
        right_node = face + 1
        if 0 < right_node < count - 1:
            face_columns.append(
                (
                    density_offset + right_node,
                    face_tangent.density_right_derivative[face]
                    * density[right_node],
                )
            )
        face_columns.extend(
            (
                (
                    potential_offset + face,
                    face_tangent.potential_left_derivative[face],
                ),
                (
                    potential_offset + right_node,
                    face_tangent.potential_right_derivative[face],
                ),
            )
        )
        if 0 < face < count - 1:
            row = face if electron else count + face
            scale = (
                layout.electron_rate_scale_m3_s[face]
                if electron
                else layout.hole_rate_scale_m3_s[face]
            )
            coefficient = (-1.0 if electron else 1.0) / (
                Q * material.dx_cell[face] * scale
            )
            for column, tangent in face_columns:
                add(row, column, coefficient * tangent)
        if 0 < right_node < count - 1:
            row = right_node if electron else count + right_node
            scale = (
                layout.electron_rate_scale_m3_s[right_node]
                if electron
                else layout.hole_rate_scale_m3_s[right_node]
            )
            coefficient = (1.0 if electron else -1.0) / (
                Q * material.dx_cell[right_node] * scale
            )
            for column, tangent in face_columns:
                add(row, column, coefficient * tangent)

    for face in range(count - 1):
        if face == interface_face:
            continue
        distribute_face(face, electron_face, n, 0, electron=True)
        distribute_face(face, hole_face, p, count, electron=False)

    bulk_columns = (
        local.interface_node,
        count + local.interface_node,
        local.interface_node - 1,
        count + local.interface_node - 1,
    )
    potential_columns = (
        potential_offset + local.interface_node,
        potential_offset + local.interface_node - 1,
    )
    interface_columns = tuple(
        range(layout.interface_slice.start, layout.interface_slice.stop)
    )
    carrier_rows = bulk_columns
    for component, row in enumerate(carrier_rows):
        node = row if row < count else row - count
        scale = (
            layout.electron_rate_scale_m3_s[node]
            if component in (0, 2)
            else layout.hole_rate_scale_m3_s[node]
        )
        coefficient = 1.0 / (material.dx_cell[node] * scale)
        for column_index, column in enumerate(bulk_columns):
            add(
                row,
                column,
                coefficient
                * local.bulk_bulk_log_jacobian_m2_s[component, column_index],
            )
        for column_index, column in enumerate(potential_columns):
            add(
                row,
                column,
                coefficient
                * local.bulk_potential_jacobian_m2_s_V[component, column_index],
            )
        for column_index, column in enumerate(interface_columns):
            add(
                row,
                column,
                coefficient
                * local.bulk_interface_coordinate_jacobian_m2_s[
                    component, column_index
                ],
            )

    for component, row in enumerate(interface_columns):
        scale = layout.interface_flux_scale_m2_s[component]
        for column_index, column in enumerate(bulk_columns):
            add(
                row,
                column,
                local.state_bulk_log_jacobian_m2_s[component, column_index]
                / scale,
            )
        for column_index, column in enumerate(potential_columns):
            add(
                row,
                column,
                local.state_potential_jacobian_m2_s_V[component, column_index]
                / scale,
            )
        for column_index, column in enumerate(interface_columns):
            add(
                row,
                column,
                local.state_interface_coordinate_jacobian_m2_s[
                    component, column_index
                ]
                / scale,
            )

    add(potential_offset, potential_offset, 1.0 / layout.potential_scale_V)
    add(layout.size - 1, layout.size - 1, 1.0 / layout.potential_scale_V)
    capacitance = material.poisson_factor.C
    widths = material.poisson_factor.h_cell
    for local_index, node in enumerate(range(1, count - 1)):
        row = potential_offset + node
        scale = layout.poisson_scale_C_m2[local_index]
        add(row, node, -Q * n[node] * widths[local_index] / scale)
        add(row, count + node, Q * p[node] * widths[local_index] / scale)
        add(row, potential_offset + node - 1, capacitance[node - 1] / scale)
        add(
            row,
            potential_offset + node,
            -(capacitance[node - 1] + capacitance[node]) / scale,
        )
        add(row, potential_offset + node + 1, capacitance[node] / scale)

    matrix = coo_matrix(
        (values, (rows, columns)),
        shape=(layout.size, layout.size),
        dtype=float,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if not np.all(np.isfinite(matrix.data)):
        raise AlgebraicInterfaceJacobianCapabilityError(
            "structured interface DAE matrix contains non-finite entries"
        )
    return AlgebraicInterfaceStructuredJacobian(
        matrix=matrix,
        nonzero_count=int(matrix.nnz),
        minimum_bulk_srh_denominator_s_m3=float(np.min(denominator)),
        local_interface=local,
        electron_current_faces_A_m2=_readonly(
            electron_current,
            (count - 1,),
            "electron current faces",
        ),
        hole_current_faces_A_m2=_readonly(
            hole_current,
            (count - 1,),
            "hole current faces",
        ),
    )


def build_algebraic_interface_structured_state_jacobian(
    model: AlgebraicInterfaceStateDAE,
    coordinate: np.ndarray,
    derivative: np.ndarray,
) -> AlgebraicInterfaceStructuredJacobian:
    """Assemble exact smooth ``dF/dq`` with interface states held algebraic."""
    return _assemble_algebraic_interface_structured_jacobian(
        model,
        coordinate,
        derivative=derivative,
        backward_euler_dt_s=None,
    )


def build_algebraic_interface_structured_backward_euler_jacobian(
    model: AlgebraicInterfaceStateDAE,
    coordinate: np.ndarray,
    dt_s: float,
) -> AlgebraicInterfaceStructuredJacobian:
    """Assemble the complete physical-density backward-Euler tangent."""
    return _assemble_algebraic_interface_structured_jacobian(
        model,
        coordinate,
        derivative=None,
        backward_euler_dt_s=dt_s,
    )


__all__ = [
    "AlgebraicInterfaceJacobianCapabilityError",
    "AlgebraicInterfaceLocalLinearization",
    "AlgebraicInterfaceStructuredJacobian",
    "build_algebraic_interface_structured_backward_euler_jacobian",
    "build_algebraic_interface_structured_state_jacobian",
    "linearize_algebraic_interface_response",
]
