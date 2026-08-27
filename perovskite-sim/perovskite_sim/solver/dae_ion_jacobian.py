"""Structured analytic state Jacobian for the single-positive-ion DAE."""

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
from perovskite_sim.solver.dae_ions import SinglePositiveIonDAE
from perovskite_sim.solver.dae_jacobian import (
    DAEStructuredJacobianCapabilityError,
    build_carrier_face_jacobians,
)


@dataclass(frozen=True, slots=True)
class SingleIonDAEStructuredStateJacobian:
    """Sparse ``dF/dq`` and constitutive evidence for the single-ion slice."""

    matrix: csr_matrix
    nonzero_count: int
    field_mobility_active: bool
    ion_steric_diffusion_only: bool
    minimum_bulk_srh_denominator_s_m3: float
    electron_current_faces_A_m2: np.ndarray
    hole_current_faces_A_m2: np.ndarray
    positive_ion_particle_flux_faces_m2_s: np.ndarray


def build_single_ion_structured_state_jacobian(
    model: SinglePositiveIonDAE,
    coordinate: np.ndarray,
    derivative: np.ndarray,
) -> SingleIonDAEStructuredStateJacobian:
    """Assemble the exact smooth ``dF/dq`` for the single-ion topology."""
    layout = model.layout
    count = layout.node_count
    rate = np.asarray(derivative, dtype=float)
    if rate.shape != (layout.size,) or not np.all(np.isfinite(rate)):
        raise ValueError("single-ion DAE derivative must be finite and layout-sized")
    n_raw, p_raw, positive_ion, phi = model.physical_fields(coordinate)
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
            "thermionic interface caps are outside the single-ion DAE slice"
        )

    n = np.array(n_raw, copy=True)
    p = np.array(p_raw, copy=True)
    n[[0, -1]] = (material.n_L, material.n_R)
    p[[0, -1]] = (material.p_L, material.p_R)
    electron_face, hole_face = build_carrier_face_jacobians(model, n, p, phi)
    ion_face = ion_face_flux_jacobian(
        phi,
        positive_ion,
        np.diff(model.grid_m),
        material.D_ion_face,
        material.V_T_device,
        material.P_lim_face,
        steric_diffusion_only=material.ion_steric_diffusion_only,
        P_lim_node=material.P_lim_node,
        drift_sign=1.0,
    )
    if not np.all(ion_face.differentiable_faces):
        faces = tuple(np.flatnonzero(~ion_face.differentiable_faces).tolist())
        raise DAEStructuredJacobianCapabilityError(
            "positive-ion steric law is non-differentiable on faces "
            f"{faces}"
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
        np.asarray(value).shape != (count,)
        or not np.all(np.isfinite(value))
        for value in reaction_arrays
    ):
        raise DAEStructuredJacobianCapabilityError(
            "bulk recombination tangent is not finite and node matched"
        )

    ion_coordinate_derivative = model.positive_ion_coordinate_derivative_m3(
        coordinate
    )
    theta = positive_ion / layout.positive_ion_site_limit_m3
    ion_coordinate_second_derivative = ion_coordinate_derivative * (
        1.0 - 2.0 * theta
    )
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []

    def add(row: int, column: int, value: float) -> None:
        number = float(value)
        if not np.isfinite(number):
            raise DAEStructuredJacobianCapabilityError(
                "single-ion structured DAE assembly produced a non-finite entry"
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
        add(electron_row, node, n[node] * rate[node] / electron_scale)
        add(
            hole_row,
            count + node,
            p[node] * rate[count + node] / hole_scale,
        )
        dR_dlogn = reaction.electron_density_derivative[node] * n[node]
        dR_dlogp = reaction.hole_density_derivative[node] * p[node]
        for row, scale in (
            (electron_row, electron_scale),
            (hole_row, hole_scale),
        ):
            add(row, node, dR_dlogn / scale)
            add(row, count + node, dR_dlogp / scale)

    ion_offset = 2 * count
    for node in range(count):
        add(
            ion_offset + node,
            ion_offset + node,
            ion_coordinate_second_derivative[node]
            * rate[ion_offset + node]
            / layout.positive_ion_rate_scale_m3_s[node],
        )

    potential_offset = 3 * count

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
                (
                    potential_offset + face,
                    local.potential_left_derivative[face],
                ),
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
        columns = carrier_face_columns(face, local, density, density_offset)
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
            for column, tangent in columns:
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
            for column, tangent in columns:
                add(row, column, coefficient * tangent)

    for face in range(count - 1):
        distribute_carrier_face(face, electron_face, n, 0, electron=True)
        distribute_carrier_face(face, hole_face, p, count, electron=False)

    def ion_face_columns(
        face: int,
        local: IonFaceFluxJacobian,
    ) -> tuple[tuple[int, float], ...]:
        right_node = face + 1
        return (
            (
                ion_offset + face,
                local.density_left_derivative[face]
                * ion_coordinate_derivative[face],
            ),
            (
                ion_offset + right_node,
                local.density_right_derivative[face]
                * ion_coordinate_derivative[right_node],
            ),
            (
                potential_offset + face,
                local.potential_left_derivative[face],
            ),
            (
                potential_offset + right_node,
                local.potential_right_derivative[face],
            ),
        )

    for face in range(count - 1):
        right_node = face + 1
        face_columns = ion_face_columns(face, ion_face)
        left_coefficient = 1.0 / (
            material.dx_cell[face]
            * layout.positive_ion_rate_scale_m3_s[face]
        )
        right_coefficient = -1.0 / (
            material.dx_cell[right_node]
            * layout.positive_ion_rate_scale_m3_s[right_node]
        )
        for column, tangent in face_columns:
            add(ion_offset + face, column, left_coefficient * tangent)
            add(ion_offset + right_node, column, right_coefficient * tangent)

    add(
        potential_offset,
        potential_offset,
        1.0 / layout.potential_scale_V,
    )
    add(layout.size - 1, layout.size - 1, 1.0 / layout.potential_scale_V)
    capacitance = material.poisson_factor.C
    widths = material.poisson_factor.h_cell
    for local, node in enumerate(range(1, count - 1)):
        row = potential_offset + node
        scale = layout.poisson_scale_C_m2[local]
        add(row, node, -Q * n[node] * widths[local] / scale)
        add(row, count + node, Q * p[node] * widths[local] / scale)
        add(
            row,
            ion_offset + node,
            Q * ion_coordinate_derivative[node] * widths[local] / scale,
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
            "single-ion structured DAE matrix contains non-finite entries"
        )
    return SingleIonDAEStructuredStateJacobian(
        matrix=matrix,
        nonzero_count=int(matrix.nnz),
        field_mobility_active=bool(material.has_field_mobility),
        ion_steric_diffusion_only=bool(material.ion_steric_diffusion_only),
        minimum_bulk_srh_denominator_s_m3=(
            float(np.min(denominator)) if denominator.size else float("inf")
        ),
        electron_current_faces_A_m2=np.asarray(electron_face.flux, dtype=float),
        hole_current_faces_A_m2=np.asarray(hole_face.flux, dtype=float),
        positive_ion_particle_flux_faces_m2_s=np.asarray(
            ion_face.flux,
            dtype=float,
        ),
    )


__all__ = [
    "SingleIonDAEStructuredStateJacobian",
    "build_single_ion_structured_state_jacobian",
]
