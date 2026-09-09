"""Physical limiting cases of the reference mesh and SG profile reconstruction."""

import numpy as np
import pytest

from perovskite_sim.constants import V_T
from perovskite_sim.discretization.fe_operators import sg_fluxes_n, sg_fluxes_p
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.physics.generation import dual_cell_widths
from perovskite_sim.solver.mol import build_material_arrays
from perovskite_sim.validation.foundation_reference_refinement import (
    _balanced_reference_grid,
    _sample_sg_density,
)


@pytest.mark.parametrize("total", [88, 176, 352])
def test_balanced_mesh_preserves_physical_absorber_volume(total):
    stack = load_device_from_yaml("tests/fixtures/configs/tpv_physical_reference.yaml")
    counts = np.array([3, 16, 3]) * (total // 22)
    grid, alphas = _balanced_reference_grid(stack.layers, counts)
    material = build_material_arrays(grid, stack)
    assert len(grid) == total + 1
    assert min(alphas) > 0.0
    for boundary in np.cumsum(counts)[:-1]:
        assert grid[boundary] - grid[boundary - 1] == pytest.approx(
            grid[boundary + 1] - grid[boundary], rel=1e-9
        )
    measure = np.sum(dual_cell_widths(grid)[material.alpha > 0.0])
    assert measure == pytest.approx(stack.layers[1].thickness, rel=1e-12)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_zero_field_has_linear_density_not_linear_log_density(sign):
    grid = np.array([0.0, 1e-8])
    positions = np.linspace(0.0, grid[-1], 31)
    density = np.array([0.1, 1e18])
    actual = _sample_sg_density(
        grid, density, np.zeros(2), positions, V_T, drift_sign=sign
    )
    expected = density[0] + (density[1] - density[0]) * positions / grid[-1]
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=0.0)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_zero_current_reproduces_the_boltzmann_exponential(sign):
    grid = np.array([0.0, 1e-8])
    potential = np.array([0.0, 0.05])
    density = 1e15 * np.exp(sign * potential / V_T)
    positions = np.linspace(0.0, grid[-1], 31)
    actual = _sample_sg_density(
        grid, density, potential, positions, V_T, drift_sign=sign
    )
    expected = 1e15 * np.exp(sign * potential[-1] * positions / grid[-1] / V_T)
    np.testing.assert_allclose(actual, expected, rtol=3e-15)


@pytest.mark.parametrize("sign,current", [(1.0, sg_fluxes_n), (-1.0, sg_fluxes_p)])
def test_subdivision_preserves_the_same_nonzero_face_current(sign, current):
    grid = np.array([0.0, 1e-8])
    potential = np.array([0.0, 0.02])
    density = np.array([1e15, 3e15])
    positions = np.linspace(0.0, grid[-1], 31)
    sampled = _sample_sg_density(
        grid, density, potential, positions, V_T, drift_sign=sign
    )
    expected = current(potential, density, np.diff(grid), 1e-6, V_T)[0]
    actual = current(
        np.interp(positions, grid, potential), sampled, np.diff(positions), 1e-6, V_T
    )
    assert expected != 0.0
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)
