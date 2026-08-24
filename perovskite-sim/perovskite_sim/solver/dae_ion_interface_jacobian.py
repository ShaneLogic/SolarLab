"""Structured analytic tangent for the combined ion/interface DAE slice."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from perovskite_sim.constants import Q
from perovskite_sim.physics.ion_migration import (
    IonFaceFluxJacobian,
    ion_face_flux_jacobian,
)
from perovskite_sim.solver.dae_interface_jacobian import (
    AlgebraicInterfaceJacobianCapabilityError,
    AlgebraicInterfaceLocalLinearization,
    build_algebraic_interface_structured_backward_euler_jacobian,
    build_algebraic_interface_structured_state_jacobian,
)
from perovskite_sim.solver.dae_interface_states import (
    AlgebraicInterfaceDAELayout,
    AlgebraicInterfaceStateDAE,
)
from perovskite_sim.solver.dae_ion_interface_states import (
    SingleIonAlgebraicInterfaceDAE,
)


class IonInterfaceJacobianCapabilityError(ValueError):
    """The operating point is outside the smooth combined tangent slice."""


def _readonly(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise IonInterfaceJacobianCapabilityError(
            f"{name} must be finite with shape {shape}"
        )
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class IonInterfaceStructuredJacobian:
    """Sparse combined tangent with carrier, ion, and interface evidence."""

    matrix: csr_matrix
    nonzero_count: int
    ion_steric_diffusion_only: bool
    minimum_bulk_srh_denominator_s_m3: float
    minimum_positive_ion_occupation_margin: float
    local_interface: AlgebraicInterfaceLocalLinearization
    electron_current_faces_A_m2: np.ndarray
    hole_current_faces_A_m2: np.ndarray
    positive_ion_particle_flux_faces_m2_s: np.ndarray


def _interface_view(
    model: SingleIonAlgebraicInterfaceDAE,
    coordinate: np.ndarray,
) -> tuple[AlgebraicInterfaceStateDAE, np.ndarray, np.ndarray]:
    """Drop only the ion coordinate while preserving every shared scale."""
    combined = model.layout
    count = combined.node_count
    interface_count = combined.interface_count
    interface_state_count = combined.interface_state_count
    size = 3 * count + interface_state_count
    differential = np.zeros(size, dtype=bool)
    differential[1 : count - 1] = True
    differential[count + 1 : 2 * count - 1] = True
    algebraic = ~differential
    for mask in (differential, algebraic):
        mask.setflags(write=False)
    layout = AlgebraicInterfaceDAELayout(
        node_count=count,
        interface_count=interface_count,
        electron_reference_m3=combined.electron_reference_m3,
        hole_reference_m3=combined.hole_reference_m3,
        interface_reference_m3=combined.interface_reference_m3,
        interface_capacity_m3=combined.interface_capacity_m3,
        interface_logit_reference=combined.interface_logit_reference,
        electron_rate_scale_m3_s=combined.electron_rate_scale_m3_s,
        hole_rate_scale_m3_s=combined.hole_rate_scale_m3_s,
        interface_flux_scale_m2_s=combined.interface_flux_scale_m2_s,
        poisson_scale_C_m2=combined.poisson_scale_C_m2,
        potential_scale_V=combined.potential_scale_V,
        differential_mask=differential,
        algebraic_mask=algebraic,
    )
    view = AlgebraicInterfaceStateDAE(
        grid_m=model.grid_m,
        stack=model.stack,
        material=model.material,
        layout=layout,
        V_app_V=model.V_app_V,
        illuminated=model.illuminated,
        interface_residual_tolerance=model.interface_residual_tolerance,
    )
    index_map = np.concatenate(
        (
            np.arange(2 * count, dtype=int),
            np.arange(
                combined.interface_slice.start,
                combined.interface_slice.stop,
                dtype=int,
            ),
            np.arange(
                combined.potential_slice.start,
                combined.potential_slice.stop,
                dtype=int,
            ),
        )
    )
    combined_coordinate = np.asarray(coordinate, dtype=float)
    if combined_coordinate.shape != (combined.size,) or not np.all(
        np.isfinite(combined_coordinate)
    ):
        raise ValueError("ion-interface coordinate must be finite and layout-sized")
    interface_coordinate = np.array(combined_coordinate[index_map], copy=True)
    return view, interface_coordinate, index_map


def _ion_face_columns(
    face: int,
    tangent: IonFaceFluxJacobian,
    ion_coordinate_derivative_m3: np.ndarray,
    *,
    ion_offset: int,
    potential_offset: int,
) -> tuple[tuple[int, float], ...]:
    right = face + 1
    return (
        (
            ion_offset + face,
            tangent.density_left_derivative[face] * ion_coordinate_derivative_m3[face],
        ),
        (
            ion_offset + right,
            tangent.density_right_derivative[face]
            * ion_coordinate_derivative_m3[right],
        ),
        (
            potential_offset + face,
            tangent.potential_left_derivative[face],
        ),
        (
            potential_offset + right,
            tangent.potential_right_derivative[face],
        ),
    )


def _assemble_ion_interface_structured_jacobian(
    model: SingleIonAlgebraicInterfaceDAE,
    coordinate: np.ndarray,
    *,
    derivative: np.ndarray | None,
    backward_euler_dt_s: float | None,
) -> IonInterfaceStructuredJacobian:
    if (derivative is None) == (backward_euler_dt_s is None):
        raise ValueError("provide exactly one of derivative or backward_euler_dt_s")
    layout = model.layout
    count = layout.node_count
    value = np.asarray(coordinate, dtype=float)
    if value.shape != (layout.size,) or not np.all(np.isfinite(value)):
        raise ValueError("ion-interface coordinate must be finite and layout-sized")
    if derivative is not None:
        rate = np.asarray(derivative, dtype=float)
        if rate.shape != (layout.size,) or not np.all(np.isfinite(rate)):
            raise ValueError("ion-interface derivative must be finite and layout-sized")
        dt_s = None
    else:
        dt_s = float(backward_euler_dt_s)
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("backward_euler_dt_s must be finite and positive")
        rate = None

    n, p, positive_ion, interface_state, phi = model.physical_fields(value)
    interface_model, interface_coordinate, index_map = _interface_view(model, value)
    interface_fields = interface_model.physical_fields(interface_coordinate)
    for combined_field, view_field in zip(
        (n, p, interface_state, phi),
        interface_fields,
    ):
        if not np.array_equal(combined_field, view_field):
            raise IonInterfaceJacobianCapabilityError(
                "ion-free interface coordinate view changed a shared physical field"
            )
    try:
        if rate is None:
            base = build_algebraic_interface_structured_backward_euler_jacobian(
                interface_model,
                interface_coordinate,
                dt_s,
            )
        else:
            interface_rate = np.asarray(rate[index_map], dtype=float)
            base = build_algebraic_interface_structured_state_jacobian(
                interface_model,
                interface_coordinate,
                interface_rate,
            )
    except AlgebraicInterfaceJacobianCapabilityError as exc:
        raise IonInterfaceJacobianCapabilityError(str(exc)) from exc

    material = model.material
    ion_tangent = ion_face_flux_jacobian(
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
    if not np.all(ion_tangent.differentiable_faces):
        faces = tuple(np.flatnonzero(~ion_tangent.differentiable_faces).tolist())
        raise IonInterfaceJacobianCapabilityError(
            f"positive-ion steric law is non-differentiable on faces {faces}"
        )

    ion_slope = model.positive_ion_coordinate_derivative_m3(value)
    occupation = positive_ion / layout.positive_ion_site_limit_m3
    occupation_margin = float(np.min(np.minimum(occupation, 1.0 - occupation)))
    if not math.isfinite(occupation_margin) or occupation_margin <= 0.0:
        raise IonInterfaceJacobianCapabilityError(
            "positive-ion logit map must stay strictly inside the site limit"
        )

    base_coo = base.matrix.tocoo()
    rows = index_map[base_coo.row].tolist()
    columns = index_map[base_coo.col].tolist()
    values = base_coo.data.tolist()

    def add(row: int, column: int, entry: float) -> None:
        number = float(entry)
        if not math.isfinite(number):
            raise IonInterfaceJacobianCapabilityError(
                "combined structured DAE assembly produced a non-finite entry"
            )
        if number != 0.0:
            rows.append(row)
            columns.append(column)
            values.append(number)

    ion_offset = layout.positive_ion_slice.start
    potential_offset = layout.potential_slice.start
    assert ion_offset is not None
    assert potential_offset is not None
    if rate is None:
        ion_storage_tangent = ion_slope / dt_s
    else:
        ion_second_derivative = ion_slope * (1.0 - 2.0 * occupation)
        ion_storage_tangent = ion_second_derivative * rate[layout.positive_ion_slice]
    for node in range(count):
        add(
            ion_offset + node,
            ion_offset + node,
            ion_storage_tangent[node] / layout.positive_ion_rate_scale_m3_s[node],
        )

    for face in range(count - 1):
        right = face + 1
        face_columns = _ion_face_columns(
            face,
            ion_tangent,
            ion_slope,
            ion_offset=ion_offset,
            potential_offset=potential_offset,
        )
        left_coefficient = 1.0 / (
            material.dx_cell[face] * layout.positive_ion_rate_scale_m3_s[face]
        )
        right_coefficient = -1.0 / (
            material.dx_cell[right] * layout.positive_ion_rate_scale_m3_s[right]
        )
        for column, tangent in face_columns:
            add(ion_offset + face, column, left_coefficient * tangent)
            add(ion_offset + right, column, right_coefficient * tangent)

    widths = material.poisson_factor.h_cell
    for local_index, node in enumerate(range(1, count - 1)):
        add(
            potential_offset + node,
            ion_offset + node,
            Q
            * ion_slope[node]
            * widths[local_index]
            / layout.poisson_scale_C_m2[local_index],
        )

    matrix = coo_matrix(
        (values, (rows, columns)),
        shape=(layout.size, layout.size),
        dtype=float,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if not np.all(np.isfinite(matrix.data)):
        raise IonInterfaceJacobianCapabilityError(
            "combined structured DAE matrix contains non-finite entries"
        )
    return IonInterfaceStructuredJacobian(
        matrix=matrix,
        nonzero_count=int(matrix.nnz),
        ion_steric_diffusion_only=bool(material.ion_steric_diffusion_only),
        minimum_bulk_srh_denominator_s_m3=(base.minimum_bulk_srh_denominator_s_m3),
        minimum_positive_ion_occupation_margin=occupation_margin,
        local_interface=base.local_interface,
        electron_current_faces_A_m2=base.electron_current_faces_A_m2,
        hole_current_faces_A_m2=base.hole_current_faces_A_m2,
        positive_ion_particle_flux_faces_m2_s=_readonly(
            ion_tangent.flux,
            (count - 1,),
            "positive-ion particle flux faces",
        ),
    )


def build_ion_interface_structured_state_jacobian(
    model: SingleIonAlgebraicInterfaceDAE,
    coordinate: np.ndarray,
    derivative: np.ndarray,
) -> IonInterfaceStructuredJacobian:
    """Assemble exact smooth ``dF/dq`` for the combined DAE topology."""
    return _assemble_ion_interface_structured_jacobian(
        model,
        coordinate,
        derivative=derivative,
        backward_euler_dt_s=None,
    )


def build_ion_interface_structured_backward_euler_jacobian(
    model: SingleIonAlgebraicInterfaceDAE,
    coordinate: np.ndarray,
    dt_s: float,
) -> IonInterfaceStructuredJacobian:
    """Assemble the complete physical-density backward-Euler tangent."""
    return _assemble_ion_interface_structured_jacobian(
        model,
        coordinate,
        derivative=None,
        backward_euler_dt_s=dt_s,
    )


__all__ = [
    "IonInterfaceJacobianCapabilityError",
    "IonInterfaceStructuredJacobian",
    "build_ion_interface_structured_backward_euler_jacobian",
    "build_ion_interface_structured_state_jacobian",
]
