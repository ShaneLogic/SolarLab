from __future__ import annotations

import numpy as np
import pytest

from perovskite_sim.physics.ion_migration import (
    ion_face_flux,
    ion_face_flux_jacobian,
)


def _node_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    matrix = np.zeros((left.size, left.size + 1), dtype=float)
    for face in range(left.size):
        matrix[face, face] += left[face]
        matrix[face, face + 1] += right[face]
    return matrix


@pytest.mark.parametrize(
    ("diffusion_only", "drift_sign", "shared_partner", "near_limit"),
    (
        (False, 1.0, False, False),
        (False, 1.0, False, True),
        (True, 1.0, False, False),
        (True, 1.0, False, True),
        (True, -1.0, True, False),
        (True, -1.0, True, True),
    ),
)
def test_ion_face_flux_jacobian_matches_independent_finite_difference(
    diffusion_only,
    drift_sign,
    shared_partner,
    near_limit,
):
    phi = np.array([0.01, -0.015, 0.035])
    dx = np.array([7.0e-9, 2.0e-8])
    diffusion = np.array([1.1e-17, 3.2e-18])
    node_limit = np.array([1.6e27, 1.7e27, 1.8e27])
    density = (
        node_limit * np.array([0.72, 0.78, 0.70])
        if near_limit
        else np.array([1.2e25, 2.1e25, 1.7e25])
    )
    partner = None
    if shared_partner:
        partner = (
            node_limit * np.array([0.12, 0.10, 0.15])
            if near_limit
            else np.array([3.0e24, 4.0e24, 2.0e24])
        )
    face_limit = 0.5 * (node_limit[:-1] + node_limit[1:])
    thermal_voltage = 0.0257
    keywords = {
        "steric_diffusion_only": diffusion_only,
        "P_lim_node": node_limit,
        "P_other_node": partner,
        "drift_sign": drift_sign,
    }
    local = ion_face_flux_jacobian(
        phi,
        density,
        dx,
        diffusion,
        thermal_voltage,
        face_limit,
        **keywords,
    )
    expected_density = _node_matrix(
        local.density_left_derivative,
        local.density_right_derivative,
    )
    expected_potential = _node_matrix(
        local.potential_left_derivative,
        local.potential_right_derivative,
    )
    expected_partner = _node_matrix(
        local.partner_left_derivative,
        local.partner_right_derivative,
    )

    density_difference = np.empty_like(expected_density)
    potential_difference = np.empty_like(expected_potential)
    partner_difference = np.empty_like(expected_partner)
    for node in range(density.size):
        density_step = density[node] * 1.0e-6
        density_plus = density.copy()
        density_minus = density.copy()
        density_plus[node] += density_step
        density_minus[node] -= density_step
        density_difference[:, node] = (
            ion_face_flux(
                phi,
                density_plus,
                dx,
                diffusion,
                thermal_voltage,
                face_limit,
                **keywords,
            )
            - ion_face_flux(
                phi,
                density_minus,
                dx,
                diffusion,
                thermal_voltage,
                face_limit,
                **keywords,
            )
        ) / (2.0 * density_step)

        potential_step = 1.0e-7
        potential_plus = phi.copy()
        potential_minus = phi.copy()
        potential_plus[node] += potential_step
        potential_minus[node] -= potential_step
        potential_difference[:, node] = (
            ion_face_flux(
                potential_plus,
                density,
                dx,
                diffusion,
                thermal_voltage,
                face_limit,
                **keywords,
            )
            - ion_face_flux(
                potential_minus,
                density,
                dx,
                diffusion,
                thermal_voltage,
                face_limit,
                **keywords,
            )
        ) / (2.0 * potential_step)

        if partner is None:
            partner_difference[:, node] = 0.0
        else:
            partner_step = partner[node] * 1.0e-6
            partner_plus = partner.copy()
            partner_minus = partner.copy()
            partner_plus[node] += partner_step
            partner_minus[node] -= partner_step
            partner_difference[:, node] = (
                ion_face_flux(
                    phi,
                    density,
                    dx,
                    diffusion,
                    thermal_voltage,
                    face_limit,
                    steric_diffusion_only=diffusion_only,
                    P_lim_node=node_limit,
                    P_other_node=partner_plus,
                    drift_sign=drift_sign,
                )
                - ion_face_flux(
                    phi,
                    density,
                    dx,
                    diffusion,
                    thermal_voltage,
                    face_limit,
                    steric_diffusion_only=diffusion_only,
                    P_lim_node=node_limit,
                    P_other_node=partner_minus,
                    drift_sign=drift_sign,
                )
            ) / (2.0 * partner_step)

    assert np.all(local.differentiable_faces)
    np.testing.assert_array_equal(
        local.flux,
        ion_face_flux(
            phi,
            density,
            dx,
            diffusion,
            thermal_voltage,
            face_limit,
            **keywords,
        ),
    )
    for finite_difference, expected in (
        (density_difference, expected_density),
        (potential_difference, expected_potential),
        (partner_difference, expected_partner),
    ):
        np.testing.assert_allclose(
            finite_difference,
            expected,
            rtol=8.0e-7,
            atol=float(np.max(np.abs(expected))) * 2.0e-9 + 1.0e-30,
        )


@pytest.mark.parametrize("diffusion_only", [False, True])
def test_ion_face_flux_jacobian_marks_clipping_kinks(diffusion_only):
    kwargs = {
        "steric_diffusion_only": diffusion_only,
        "P_lim_node": np.ones(3),
    }
    active = ion_face_flux_jacobian(
        np.zeros(3),
        np.zeros(3),
        np.ones(2),
        np.ones(2),
        0.025,
        np.ones(2),
        **kwargs,
    )
    inactive = ion_face_flux_jacobian(
        np.zeros(3),
        np.zeros(3),
        np.ones(2),
        np.zeros(2),
        0.025,
        np.ones(2),
        **kwargs,
    )
    upper_kink = ion_face_flux_jacobian(
        np.zeros(3),
        np.full(3, 0.999999),
        np.ones(2),
        np.ones(2),
        0.025,
        np.ones(2),
        **kwargs,
    )

    assert not np.any(active.differentiable_faces)
    assert np.all(inactive.differentiable_faces)
    assert not np.any(upper_kink.differentiable_faces)


def test_blocking_flux_jacobian_preserves_dual_cell_inventory():
    x = np.array([0.0, 0.2, 0.7, 1.0])
    phi = np.array([0.0, 0.02, -0.01, 0.03])
    positive = np.array([0.12, 0.18, 0.16, 0.14])
    negative = np.array([0.08, 0.07, 0.09, 0.06])
    node_limit = np.ones(4)
    local = ion_face_flux_jacobian(
        phi,
        positive,
        np.diff(x),
        np.array([0.8, 1.1, 0.6]),
        0.025,
        np.ones(3),
        steric_diffusion_only=True,
        P_lim_node=node_limit,
        P_other_node=negative,
    )
    dx_cell = np.array(
        [
            x[1] - x[0],
            0.5 * (x[2] - x[0]),
            0.5 * (x[3] - x[1]),
            x[3] - x[2],
        ]
    )
    for face_matrix in (
        _node_matrix(
            local.density_left_derivative,
            local.density_right_derivative,
        ),
        _node_matrix(
            local.partner_left_derivative,
            local.partner_right_derivative,
        ),
        _node_matrix(
            local.potential_left_derivative,
            local.potential_right_derivative,
        ),
    ):
        padded = np.vstack(
            [
                np.zeros((1, face_matrix.shape[1])),
                face_matrix,
                np.zeros((1, face_matrix.shape[1])),
            ]
        )
        rate_jacobian = -(padded[1:] - padded[:-1]) / dx_cell[:, None]
        inventory_derivative = dx_cell @ rate_jacobian
        scale = max(float(np.max(np.abs(face_matrix))), 1.0)
        np.testing.assert_allclose(
            inventory_derivative,
            0.0,
            rtol=0.0,
            atol=64.0 * np.finfo(float).eps * scale,
        )
