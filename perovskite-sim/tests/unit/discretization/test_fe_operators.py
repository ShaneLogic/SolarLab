import numpy as np
import pytest
from perovskite_sim.discretization.fe_operators import (
    bernoulli,
    bernoulli_derivative,
    sg_flux_n,
    sg_flux_p,
    sg_fluxes_n,
    sg_fluxes_n_jacobian,
    sg_fluxes_p,
    sg_fluxes_p_jacobian,
)


def test_bernoulli_at_zero():
    assert bernoulli(np.array([0.0]))[0] == pytest.approx(1.0)


def test_bernoulli_large_positive():
    # B(x) → 0 for large positive x
    assert bernoulli(np.array([50.0]))[0] == pytest.approx(0.0, abs=1e-10)


def test_bernoulli_large_negative():
    # B(-x) → |x| for large |x| (drift dominates)
    x = np.array([-20.0])
    assert bernoulli(x)[0] == pytest.approx(20.0, rel=1e-6)


def test_bernoulli_symmetry():
    x = np.array([1.5])
    # B(x)*exp(x) == B(-x)
    assert (bernoulli(x) * np.exp(x))[0] == pytest.approx(bernoulli(-x)[0], rel=1e-10)


@pytest.mark.parametrize("value", [-20.0, -2.0, -1.0e-7, 0.0, 1.0e-7, 2.0, 20.0])
def test_bernoulli_derivative_matches_centered_difference(value):
    step = 1.0e-6
    finite_difference = float(
        (
            bernoulli(np.array([value + step]))
            - bernoulli(np.array([value - step]))
        )[0]
        / (2.0 * step)
    )
    assert bernoulli_derivative(np.array([value]))[0] == pytest.approx(
        finite_difference,
        rel=2.0e-7,
        abs=2.0e-10,
    )


def test_bernoulli_derivative_has_finite_asymptotic_limits():
    derivative = bernoulli_derivative(np.array([-800.0, 800.0]))

    np.testing.assert_array_equal(derivative, np.array([-1.0, 0.0]))


@pytest.mark.parametrize(
    ("flux_function", "jacobian_function"),
    (
        (sg_fluxes_n, sg_fluxes_n_jacobian),
        (sg_fluxes_p, sg_fluxes_p_jacobian),
    ),
)
def test_vector_sg_face_jacobian_matches_independent_finite_difference(
    flux_function,
    jacobian_function,
):
    phi = np.array([0.01, -0.02, 0.04])
    density = np.array([1.2e20, 3.4e21, 7.8e19])
    dx = np.array([8.0e-9, 2.3e-8])
    diffusion = np.array([2.0e-6, 7.0e-5])
    thermal_voltage = 0.0257
    local = jacobian_function(
        phi,
        density,
        dx,
        diffusion,
        thermal_voltage,
    )

    expected_density = np.zeros((dx.size, density.size))
    expected_potential = np.zeros_like(expected_density)
    for node in range(density.size):
        if node < density.size - 1:
            expected_density[node, node] += local.density_left_derivative[node]
            expected_potential[node, node] += local.potential_left_derivative[node]
        if node > 0:
            expected_density[node - 1, node] += (
                local.density_right_derivative[node - 1]
            )
            expected_potential[node - 1, node] += (
                local.potential_right_derivative[node - 1]
            )

    density_difference = np.empty_like(expected_density)
    potential_difference = np.empty_like(expected_potential)
    for node in range(density.size):
        density_step = density[node] * 1.0e-6
        density_plus = density.copy()
        density_minus = density.copy()
        density_plus[node] += density_step
        density_minus[node] -= density_step
        density_difference[:, node] = (
            flux_function(
                phi,
                density_plus,
                dx,
                diffusion,
                thermal_voltage,
            )
            - flux_function(
                phi,
                density_minus,
                dx,
                diffusion,
                thermal_voltage,
            )
        ) / (2.0 * density_step)

        potential_step = 1.0e-7
        potential_plus = phi.copy()
        potential_minus = phi.copy()
        potential_plus[node] += potential_step
        potential_minus[node] -= potential_step
        potential_difference[:, node] = (
            flux_function(
                potential_plus,
                density,
                dx,
                diffusion,
                thermal_voltage,
            )
            - flux_function(
                potential_minus,
                density,
                dx,
                diffusion,
                thermal_voltage,
            )
        ) / (2.0 * potential_step)

    np.testing.assert_allclose(
        local.flux,
        flux_function(phi, density, dx, diffusion, thermal_voltage),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        density_difference,
        expected_density,
        rtol=3.0e-7,
        atol=float(np.max(np.abs(expected_density))) * 1.0e-9,
    )
    np.testing.assert_allclose(
        potential_difference,
        expected_potential,
        rtol=3.0e-7,
        atol=float(np.max(np.abs(expected_potential))) * 1.0e-9,
    )


def test_sg_flux_n_equilibrium():
    """Electron current is zero at thermal equilibrium."""
    V_T = 0.025852
    phi = np.array([0.0, 0.1])   # 100 mV potential difference
    xi = (phi[1] - phi[0]) / V_T
    n_eq = np.array([1e18, 1e18 * np.exp(xi)])  # Boltzmann distribution
    h = 100e-9
    D_n = 5.17e-6  # m²/s
    J = sg_flux_n(phi, n_eq, h, D_n, V_T)
    assert abs(J) < 1e-10 * abs(n_eq[0])


def test_sg_flux_p_equilibrium():
    """Hole current is zero at thermal equilibrium."""
    V_T = 0.025852
    phi = np.array([0.0, 0.1])
    p_eq = np.array([1e18, 1e18 * np.exp(-(phi[1]-phi[0])/V_T)])
    h = 100e-9
    D_p = 5.17e-6
    J = sg_flux_p(phi, p_eq, h, D_p, V_T)
    assert abs(J) < 1e-10 * abs(p_eq[0])
