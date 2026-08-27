"""Structured analytic tangents for the dual-mobile-ion research DAE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from perovskite_sim.constants import Q
from perovskite_sim.discretization.fe_operators import ScharfetterGummelFaceJacobian
from perovskite_sim.physics.ion_migration import (
    IonFaceFluxJacobian,
    ion_face_flux_jacobian,
)
from perovskite_sim.physics.recombination import (
    bulk_recombination_denominators,
    total_recombination_derivatives,
)
from perovskite_sim.solver.dae_dual_ions import DualIonDAE
from perovskite_sim.solver.dae_jacobian import (
    DAEStructuredJacobianCapabilityError,
    build_carrier_face_jacobians,
)


@dataclass(frozen=True, slots=True)
class DualIonDAEStructuredJacobian:
    """Sparse tangent and constitutive evidence for the dual-ion slice."""

    matrix: csr_matrix
    nonzero_count: int
    storage_mode: str
    field_mobility_active: bool
    ion_steric_diffusion_only: bool
    shared_site: bool
    minimum_bulk_srh_denominator_s_m3: float
    electron_current_faces_A_m2: np.ndarray
    hole_current_faces_A_m2: np.ndarray
    positive_ion_particle_flux_faces_m2_s: np.ndarray
    negative_ion_particle_flux_faces_m2_s: np.ndarray


def _assemble_dual_ion_structured_jacobian(
    model: DualIonDAE,
    coordinate: np.ndarray,
    *,
    derivative: np.ndarray | None,
    backward_euler_dt_s: float | None,
) -> DualIonDAEStructuredJacobian:
    layout = model.layout
    count = layout.node_count
    if (derivative is None) == (backward_euler_dt_s is None):
        raise ValueError("select exactly one dual-ion storage linearization")
    rate = None
    if derivative is not None:
        rate = np.asarray(derivative, dtype=float)
        if rate.shape != (layout.size,) or not np.all(np.isfinite(rate)):
            raise ValueError("dual-ion DAE derivative must be finite and layout-sized")
        storage_mode = "dae_state"
    else:
        assert backward_euler_dt_s is not None
        if not np.isfinite(backward_euler_dt_s) or backward_euler_dt_s <= 0.0:
            raise ValueError("backward_euler_dt_s must be finite and positive")
        storage_mode = "physical_density_backward_euler"

    n_raw, p_raw, positive_ion, negative_ion, phi = model.physical_fields(
        coordinate
    )
    material = model.material
    if material.has_radiative_reabsorption:
        raise DAEStructuredJacobianCapabilityError(
            "self-consistent radiative reabsorption has no nonlocal DAE tangent"
        )
    if material.het_recomb_despike > 0.0 and material.het_recomb_nodes:
        raise DAEStructuredJacobianCapabilityError(
            "heterojunction recombination de-spike has no DAE tangent"
        )
    if material.interface_faces:
        raise DAEStructuredJacobianCapabilityError(
            "thermionic interface caps are outside the dual-ion DAE slice"
        )
    negative_arrays = (
        material.D_ion_neg_face,
        material.P_lim_neg_face,
        material.P_lim_neg_node,
    )
    if any(value is None for value in negative_arrays):
        raise DAEStructuredJacobianCapabilityError(
            "negative-ion material arrays are incomplete"
        )

    n = np.array(n_raw, copy=True)
    p = np.array(p_raw, copy=True)
    n[[0, -1]] = (material.n_L, material.n_R)
    p[[0, -1]] = (material.p_L, material.p_R)
    electron_face, hole_face = build_carrier_face_jacobians(model, n, p, phi)
    positive_face = ion_face_flux_jacobian(
        phi,
        positive_ion,
        np.diff(model.grid_m),
        material.D_ion_face,
        material.V_T_device,
        material.P_lim_face,
        steric_diffusion_only=material.ion_steric_diffusion_only,
        P_lim_node=material.P_lim_node,
        P_other_node=negative_ion if layout.shared_site else None,
        drift_sign=1.0,
    )
    negative_face = ion_face_flux_jacobian(
        phi,
        negative_ion,
        np.diff(model.grid_m),
        material.D_ion_neg_face,
        material.V_T_device,
        material.P_lim_neg_face,
        steric_diffusion_only=material.ion_steric_diffusion_only,
        P_lim_node=material.P_lim_neg_node,
        P_other_node=positive_ion if layout.shared_site else None,
        drift_sign=-1.0,
    )
    for label, face_tangent in (
        ("positive", positive_face),
        ("negative", negative_face),
    ):
        if not np.all(face_tangent.differentiable_faces):
            faces = tuple(
                np.flatnonzero(~face_tangent.differentiable_faces).tolist()
            )
            raise DAEStructuredJacobianCapabilityError(
                f"{label}-ion steric law is non-differentiable on faces {faces}"
            )

    denominator = bulk_recombination_denominators(
        n,
        p,
        material.tau_n,
        material.tau_p,
        material.n1,
        material.p1,
        neutral_bulk_defects=material.neutral_bulk_defects,
    )
    if not np.all(np.isfinite(denominator)) or np.any(denominator <= 0.0):
        raise DAEStructuredJacobianCapabilityError(
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
        neutral_bulk_defects=material.neutral_bulk_defects,
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
        raise DAEStructuredJacobianCapabilityError(
            "bulk recombination tangent is not finite and node matched"
        )

    ion_mass = model.ion_coordinate_jacobian_m3(coordinate)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []

    def add(row: int, column: int, value: float) -> None:
        number = float(value)
        if not np.isfinite(number):
            raise DAEStructuredJacobianCapabilityError(
                "dual-ion structured DAE assembly produced a non-finite entry"
            )
        if number != 0.0:
            rows.append(row)
            columns.append(column)
            values.append(number)

    for index in (0, count - 1, count, 2 * count - 1):
        add(index, index, 1.0)

    assert (rate is None) != (backward_euler_dt_s is None)
    for node in range(1, count - 1):
        electron_row = node
        hole_row = count + node
        electron_scale = layout.electron_rate_scale_m3_s[node]
        hole_scale = layout.hole_rate_scale_m3_s[node]
        if rate is not None:
            electron_storage = n[node] * rate[node]
            hole_storage = p[node] * rate[count + node]
        else:
            assert backward_euler_dt_s is not None
            electron_storage = n[node] / backward_euler_dt_s
            hole_storage = p[node] / backward_euler_dt_s
        add(electron_row, node, electron_storage / electron_scale)
        add(hole_row, count + node, hole_storage / hole_scale)
        dR_dlogn = reaction.electron_density_derivative[node] * n[node]
        dR_dlogp = reaction.hole_density_derivative[node] * p[node]
        for row, scale in (
            (electron_row, electron_scale),
            (hole_row, hole_scale),
        ):
            add(row, node, dR_dlogn / scale)
            add(row, count + node, dR_dlogp / scale)

    positive_offset = 2 * count
    negative_offset = 3 * count
    ion_offsets = (positive_offset, negative_offset)
    ion_scales = (
        layout.positive_ion_rate_scale_m3_s,
        layout.negative_ion_rate_scale_m3_s,
    )
    if rate is not None:
        ion_coordinate_rate = np.stack(
            (
                rate[layout.positive_ion_slice],
                rate[layout.negative_ion_slice],
            ),
            axis=1,
        )
        storage_tangent = np.einsum(
            "nsjk,nj->nsk",
            model.ion_coordinate_hessian_m3(coordinate),
            ion_coordinate_rate,
        )
    else:
        assert backward_euler_dt_s is not None
        storage_tangent = ion_mass / backward_euler_dt_s
    for node in range(count):
        for species, row_offset in enumerate(ion_offsets):
            for coordinate_species, column_offset in enumerate(ion_offsets):
                add(
                    row_offset + node,
                    column_offset + node,
                    storage_tangent[node, species, coordinate_species]
                    / ion_scales[species][node],
                )

    potential_offset = 4 * count

    def carrier_face_columns(
        face: int,
        local: ScharfetterGummelFaceJacobian,
        density: np.ndarray,
        density_offset: int,
    ) -> list[tuple[int, float]]:
        result: list[tuple[int, float]] = []
        if 0 < face < count - 1:
            result.append(
                (
                    density_offset + face,
                    local.density_left_derivative[face] * density[face],
                )
            )
        right_node = face + 1
        if 0 < right_node < count - 1:
            result.append(
                (
                    density_offset + right_node,
                    local.density_right_derivative[face] * density[right_node],
                )
            )
        result.extend(
            (
                (potential_offset + face, local.potential_left_derivative[face]),
                (
                    potential_offset + right_node,
                    local.potential_right_derivative[face],
                ),
            )
        )
        return result

    def distribute_carrier_face(
        face: int,
        local: ScharfetterGummelFaceJacobian,
        density: np.ndarray,
        density_offset: int,
        *,
        electron: bool,
    ) -> None:
        face_columns = carrier_face_columns(face, local, density, density_offset)
        right_node = face + 1
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
        distribute_carrier_face(face, electron_face, n, 0, electron=True)
        distribute_carrier_face(face, hole_face, p, count, electron=False)

    def ion_face_columns(
        face: int,
        local: IonFaceFluxJacobian,
        *,
        own_species: int,
    ) -> tuple[tuple[int, float], ...]:
        partner_species = 1 - own_species
        right_node = face + 1
        result: list[tuple[int, float]] = []
        for node, own_tangent, partner_tangent in (
            (
                face,
                local.density_left_derivative[face],
                local.partner_left_derivative[face],
            ),
            (
                right_node,
                local.density_right_derivative[face],
                local.partner_right_derivative[face],
            ),
        ):
            for coordinate_species, column_offset in enumerate(ion_offsets):
                tangent = (
                    own_tangent
                    * ion_mass[node, own_species, coordinate_species]
                    + partner_tangent
                    * ion_mass[node, partner_species, coordinate_species]
                )
                result.append((column_offset + node, tangent))
        result.extend(
            (
                (potential_offset + face, local.potential_left_derivative[face]),
                (
                    potential_offset + right_node,
                    local.potential_right_derivative[face],
                ),
            )
        )
        return tuple(result)

    def distribute_ion_face(
        face: int,
        local: IonFaceFluxJacobian,
        *,
        species: int,
    ) -> None:
        right_node = face + 1
        face_columns = ion_face_columns(face, local, own_species=species)
        scale = ion_scales[species]
        row_offset = ion_offsets[species]
        left_coefficient = 1.0 / (material.dx_cell[face] * scale[face])
        right_coefficient = -1.0 / (
            material.dx_cell[right_node] * scale[right_node]
        )
        for column, tangent in face_columns:
            add(row_offset + face, column, left_coefficient * tangent)
            add(row_offset + right_node, column, right_coefficient * tangent)

    for face in range(count - 1):
        distribute_ion_face(face, positive_face, species=0)
        distribute_ion_face(face, negative_face, species=1)

    add(potential_offset, potential_offset, 1.0 / layout.potential_scale_V)
    add(layout.size - 1, layout.size - 1, 1.0 / layout.potential_scale_V)
    capacitance = material.poisson_factor.C
    widths = material.poisson_factor.h_cell
    ion_charge_derivative = ion_mass[:, 0, :] - ion_mass[:, 1, :]
    for local, node in enumerate(range(1, count - 1)):
        row = potential_offset + node
        scale = layout.poisson_scale_C_m2[local]
        add(row, node, -Q * n[node] * widths[local] / scale)
        add(row, count + node, Q * p[node] * widths[local] / scale)
        add(
            row,
            positive_offset + node,
            Q * ion_charge_derivative[node, 0] * widths[local] / scale,
        )
        add(
            row,
            negative_offset + node,
            Q * ion_charge_derivative[node, 1] * widths[local] / scale,
        )
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
        raise DAEStructuredJacobianCapabilityError(
            "dual-ion structured DAE matrix contains non-finite entries"
        )
    return DualIonDAEStructuredJacobian(
        matrix=matrix,
        nonzero_count=int(matrix.nnz),
        storage_mode=storage_mode,
        field_mobility_active=bool(material.has_field_mobility),
        ion_steric_diffusion_only=bool(material.ion_steric_diffusion_only),
        shared_site=bool(layout.shared_site),
        minimum_bulk_srh_denominator_s_m3=(
            float(np.min(denominator)) if denominator.size else float("inf")
        ),
        electron_current_faces_A_m2=np.asarray(electron_face.flux, dtype=float),
        hole_current_faces_A_m2=np.asarray(hole_face.flux, dtype=float),
        positive_ion_particle_flux_faces_m2_s=np.asarray(
            positive_face.flux,
            dtype=float,
        ),
        negative_ion_particle_flux_faces_m2_s=np.asarray(
            negative_face.flux,
            dtype=float,
        ),
    )


def build_dual_ion_structured_state_jacobian(
    model: DualIonDAE,
    coordinate: np.ndarray,
    derivative: np.ndarray,
) -> DualIonDAEStructuredJacobian:
    """Assemble exact smooth ``dF/dq`` for the general dual-ion DAE."""
    return _assemble_dual_ion_structured_jacobian(
        model,
        coordinate,
        derivative=derivative,
        backward_euler_dt_s=None,
    )


def build_dual_ion_structured_backward_euler_jacobian(
    model: DualIonDAE,
    coordinate: np.ndarray,
    dt_s: float,
) -> DualIonDAEStructuredJacobian:
    """Assemble the complete physical-density backward-Euler tangent."""
    return _assemble_dual_ion_structured_jacobian(
        model,
        coordinate,
        derivative=None,
        backward_euler_dt_s=dt_s,
    )


__all__ = [
    "DualIonDAEStructuredJacobian",
    "build_dual_ion_structured_backward_euler_jacobian",
    "build_dual_ion_structured_state_jacobian",
]
