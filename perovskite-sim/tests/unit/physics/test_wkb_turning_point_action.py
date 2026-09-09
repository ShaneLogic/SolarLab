"""Analytic actions detect lost turning-point flanks and unresolved triangles."""

import math

import numpy as np
import pytest

from perovskite_sim.constants import Q
from perovskite_sim.physics.wkb_tunneling import (
    ELECTRON_MASS_KG,
    HBAR_J_S,
    windowed_wkb_action,
    wkb_action,
)


def test_single_forbidden_node_is_a_finite_triangle_not_a_transparent_path():
    x = np.array([0.0, 5e-9, 10e-9])
    barrier = np.array([0.0, 0.3, 0.0])
    energy = 0.1
    width = 10e-9 * (0.3 - energy) / 0.3
    peak_kappa = math.sqrt(2 * 0.2 * ELECTRON_MASS_KG * (0.3 - energy) * Q) / HBAR_J_S
    expected = 2.0 / 3.0 * width * peak_kappa
    actual = windowed_wkb_action(x, barrier, energy, 0.2, 0)
    assert actual > 0.0
    assert actual == pytest.approx(expected, rel=2e-14)


@pytest.mark.parametrize("nodes", [3, 5, 11, 41])
def test_linear_triangular_barrier_integrates_to_its_closed_form(nodes):
    x = np.linspace(0.0, 10e-9, nodes)
    barrier = 0.3 * (1 - np.abs(x - 5e-9) / 5e-9)
    energy = 0.07
    expected = (2.0 / 3.0 * 10e-9 * (0.3 - energy) / 0.3
                * math.sqrt(2 * 0.2 * ELECTRON_MASS_KG * (0.3 - energy) * Q) / HBAR_J_S)
    assert wkb_action(x, barrier, energy, 0.2) == pytest.approx(expected, rel=2e-13)


def test_window_does_not_include_a_second_disconnected_barrier():
    x = np.arange(7, dtype=float) * 5e-9
    barrier = np.array([0.0, 0.3, 0.0, 0.0, 0.0, 0.6, 0.0])
    first = windowed_wkb_action(x, barrier, 0.1, 0.2, 0)
    isolated = wkb_action(x[:3], barrier[:3], 0.1, 0.2)
    assert first == pytest.approx(isolated, rel=1e-14)
    assert first < wkb_action(x, barrier, 0.1, 0.2)


def test_a_nonlinear_barrier_still_requires_spatial_convergence():
    height, width, mass = 0.3, 10e-9, 0.2
    exact = (math.pi * width / 4
             * math.sqrt(2 * mass * ELECTRON_MASS_KG * height * Q) / HBAR_J_S)
    errors = []
    for nodes in (31, 61, 121, 241):
        x = np.linspace(-width / 2, width / 2, nodes)
        barrier = height * (1 - (2 * x / width)**2)
        errors.append(abs(wkb_action(x, barrier, 0.0, mass) / exact - 1))
    orders = np.log2(np.asarray(errors[:-1]) / errors[1:])
    assert errors[-1] < 1e-4
    assert np.all(orders > 1.8)
