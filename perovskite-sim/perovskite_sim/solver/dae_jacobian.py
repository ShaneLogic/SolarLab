"""Structured analytic Jacobian for the first research DAE slice."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from perovskite_sim.constants import Q
from perovskite_sim.discretization.fe_operators import (
    ScharfetterGummelFaceJacobian,
    sg_fluxes_n_jacobian,
    sg_fluxes_p_jacobian,
)
from perovskite_sim.physics.field_mobility import linearize_field_mobility
from perovskite_sim.physics.recombination import (
    bulk_recombination_denominators,
    total_recombination_derivatives,
)
from perovskite_sim.solver.dae import NoIonNoInterfaceDAE


class DAEStructuredJacobianCapabilityError(ValueError):
    """An active closure has no smooth tangent in the first DAE slice."""


@dataclass(frozen=True, slots=True)
class DAEStructuredStateJacobian:
    """Sparse ``dF/dq`` plus its constitutive capability evidence."""

    matrix: csr_matrix
    nonzero_count: int
    field_mobility_active: bool
    minimum_bulk_srh_denominator_s_m3: float
    electron_current_faces_A_m2: np.ndarray
    hole_current_faces_A_m2: np.ndarray


def _with_field_mobility_tangent(
    local: ScharfetterGummelFaceJacobian,
    mobility: np.ndarray,
    field_derivative: np.ndarray,
    spacing: np.ndarray,
) -> ScharfetterGummelFaceJacobian:
    if (
        mobility.shape != spacing.shape
        or field_derivative.shape != spacing.shape
        or not np.all(np.isfinite(mobility))
        or not np.all(np.isfinite(field_derivative))
        or np.any(mobility < 0.0)
        or np.any((mobility == 0.0) & (field_derivative != 0.0))
    ):
        raise DAEStructuredJacobianCapabilityError(
            "field-mobility tangent is not finite and face matched"
        )
    potential_left = np.divide(
        local.flux * field_derivative,
        mobility * spacing,
        out=np.zeros_like(mobility),
        where=mobility > 0.0,
    )
    return ScharfetterGummelFaceJacobian(
        flux=local.flux,
        density_left_derivative=local.density_left_derivative,
        density_right_derivative=local.density_right_derivative,
        potential_left_derivative=(
            local.potential_left_derivative + potential_left
        ),
        potential_right_derivative=(
            local.potential_right_derivative - potential_left
        ),
    )


def build_carrier_face_jacobians(
    model: NoIonNoInterfaceDAE,
    n: np.ndarray,
    p: np.ndarray,
    phi: np.ndarray,
) -> tuple[ScharfetterGummelFaceJacobian, ScharfetterGummelFaceJacobian]:
    material = model.material
    spacing = np.diff(model.grid_m)
    if not material.has_field_mobility:
        return (
            sg_fluxes_n_jacobian(
                phi + material.chi,
                n,
                spacing,
                material.D_n_face,
                material.V_T_device,
            ),
            sg_fluxes_p_jacobian(
                phi + material.chi + material.Eg,
                p,
                spacing,
                material.D_p_face,
                material.V_T_device,
            ),
        )

    parameter_arrays = (
        material.v_sat_n_face,
        material.v_sat_p_face,
        material.ct_beta_n_face,
        material.ct_beta_p_face,
        material.pf_gamma_n_face,
        material.pf_gamma_p_face,
    )
    if any(value is None for value in parameter_arrays):
        raise DAEStructuredJacobianCapabilityError(
            "field-mobility material arrays are incomplete"
        )
    electric_field = -np.diff(phi) / spacing
    electron = linearize_field_mobility(
        material.D_n_face / material.V_T_device,
        electric_field,
        material.v_sat_n_face,
        material.ct_beta_n_face,
        material.pf_gamma_n_face,
    )
    hole = linearize_field_mobility(
        material.D_p_face / material.V_T_device,
        electric_field,
        material.v_sat_p_face,
        material.ct_beta_p_face,
        material.pf_gamma_p_face,
    )
    if not np.all(electron.differentiable) or not np.all(hole.differentiable):
        faces = tuple(
            np.flatnonzero(~(electron.differentiable & hole.differentiable)).tolist()
        )
        raise DAEStructuredJacobianCapabilityError(
            "field-mobility operating point is non-differentiable on faces "
            f"{faces}"
        )
    electron_local = sg_fluxes_n_jacobian(
        phi + material.chi,
        n,
        spacing,
        electron.mobility_m2_V_s * material.V_T_device,
        material.V_T_device,
    )
    hole_local = sg_fluxes_p_jacobian(
        phi + material.chi + material.Eg,
        p,
        spacing,
        hole.mobility_m2_V_s * material.V_T_device,
        material.V_T_device,
    )
    return (
        _with_field_mobility_tangent(
            electron_local,
            electron.mobility_m2_V_s,
            electron.field_derivative_m3_V2_s,
            spacing,
        ),
        _with_field_mobility_tangent(
            hole_local,
            hole.mobility_m2_V_s,
            hole.field_derivative_m3_V2_s,
            spacing,
        ),
    )


def build_structured_state_jacobian(
    model: NoIonNoInterfaceDAE,
    coordinate: np.ndarray,
    derivative: np.ndarray,
) -> DAEStructuredStateJacobian:
    """Assemble the exact smooth ``dF/dq`` for the first DAE topology."""
    layout = model.layout
    count = layout.node_count
    rate = np.asarray(derivative, dtype=float)
    if rate.shape != (layout.size,) or not np.all(np.isfinite(rate)):
        raise ValueError("DAE derivative must be a finite layout-sized vector")
    n_raw, p_raw, phi = model.physical_fields(coordinate)
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
            "thermionic interface caps are outside the no-interface DAE slice"
        )

    # assemble_rhs pins ohmic boundary reservoirs before transport. Keep their
    # algebraic log coordinates in F, but do not differentiate carrier rates
    # with respect to values the production RHS discards.
    n = np.array(n_raw, copy=True)
    p = np.array(p_raw, copy=True)
    n[[0, -1]] = (material.n_L, material.n_R)
    p[[0, -1]] = (material.p_L, material.p_R)
    electron_face, hole_face = build_carrier_face_jacobians(model, n, p, phi)

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

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []

    def add(row: int, column: int, value: float) -> None:
        number = float(value)
        if not np.isfinite(number):
            raise DAEStructuredJacobianCapabilityError(
                "structured DAE assembly produced a non-finite entry"
            )
        if number != 0.0:
            rows.append(row)
            columns.append(column)
            values.append(number)

    # Four ohmic carrier constraints.
    for index in (0, count - 1, count, 2 * count - 1):
        add(index, index, 1.0)

    # Differential storage and local recombination blocks.
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

    def distribute_face(
        face: int,
        local: ScharfetterGummelFaceJacobian,
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
                    local.density_left_derivative[face] * density[face],
                )
            )
        right_node = face + 1
        if 0 < right_node < count - 1:
            face_columns.append(
                (
                    density_offset + right_node,
                    local.density_right_derivative[face] * density[right_node],
                )
            )
        face_columns.extend(
            (
                (
                    2 * count + face,
                    local.potential_left_derivative[face],
                ),
                (
                    2 * count + right_node,
                    local.potential_right_derivative[face],
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
        distribute_face(face, electron_face, n, 0, electron=True)
        distribute_face(face, hole_face, p, count, electron=False)

    # Exact finite-volume Poisson rows.
    potential_offset = 2 * count
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
            "structured DAE matrix contains non-finite entries"
        )
    return DAEStructuredStateJacobian(
        matrix=matrix,
        nonzero_count=int(matrix.nnz),
        field_mobility_active=bool(material.has_field_mobility),
        minimum_bulk_srh_denominator_s_m3=(
            float(np.min(denominator)) if denominator.size else float("inf")
        ),
        electron_current_faces_A_m2=np.asarray(electron_face.flux, dtype=float),
        hole_current_faces_A_m2=np.asarray(hole_face.flux, dtype=float),
    )


__all__ = [
    "DAEStructuredJacobianCapabilityError",
    "DAEStructuredStateJacobian",
    "build_carrier_face_jacobians",
    "build_structured_state_jacobian",
]
