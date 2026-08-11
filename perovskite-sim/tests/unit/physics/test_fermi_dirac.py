"""Numerical limits for the Fermi-Dirac interface supply functions."""
from __future__ import annotations

import math

import pytest

from perovskite_sim.physics.fermi_dirac import (
    fermi_dirac_half,
    fermi_dirac_half_log_derivative,
    fermi_dirac_one,
    fermi_dirac_zero,
    inverse_fermi_dirac_half,
)


@pytest.mark.parametrize(
    ("eta", "expected"),
    [
        (-10.0, 4.5399201053e-5),
        (0.0, 0.7651470246),
        (2.0, 2.8237212774),
        (10.0, 24.0846569646),
    ],
)
def test_fermi_dirac_half_matches_reference_quadrature(eta, expected):
    assert fermi_dirac_half(eta) == pytest.approx(expected, rel=2.0e-7)


@pytest.mark.parametrize("ratio", [1.0e-12, 0.01, 0.1, 1.0, 3.0, 10.0, 100.0])
def test_inverse_half_round_trips_density_ratio(ratio):
    eta = inverse_fermi_dirac_half(ratio)
    assert fermi_dirac_half(eta) == pytest.approx(ratio, rel=2.0e-6)


def test_fermi_dirac_one_has_boltzmann_and_degenerate_limits():
    dilute_eta = -12.0
    assert fermi_dirac_one(dilute_eta) == pytest.approx(
        math.exp(dilute_eta),
        rel=2.0e-6,
    )
    assert fermi_dirac_one(0.0) == pytest.approx(
        math.pi**2 / 12.0,
        rel=1.0e-12,
    )
    eta = 50.0
    assert fermi_dirac_one(eta) == pytest.approx(
        0.5 * eta * eta + math.pi**2 / 6.0,
        rel=1.0e-12,
    )


def test_fermi_integrals_are_monotone():
    values = [-8.0, -2.0, 0.0, 2.0, 8.0]
    half = [fermi_dirac_half(value) for value in values]
    one = [fermi_dirac_one(value) for value in values]
    assert all(right > left for left, right in zip(half, half[1:]))
    assert all(right > left for left, right in zip(one, one[1:]))


@pytest.mark.parametrize("eta", [-12.0, -2.0, 0.3, 4.0, 30.0])
def test_fermi_dirac_one_derivative_is_fermi_dirac_zero(eta):
    step = 1.0e-6
    finite_difference = (
        fermi_dirac_one(eta + step) - fermi_dirac_one(eta - step)
    ) / (2.0 * step)
    assert fermi_dirac_zero(eta) == pytest.approx(
        finite_difference,
        rel=2.0e-7,
        abs=1.0e-12,
    )


@pytest.mark.parametrize("eta", [-12.0, -2.0, 0.3, 4.0, 30.0])
def test_half_log_derivative_matches_implemented_constitutive_law(eta):
    step = 1.0e-6
    finite_difference = (
        math.log(fermi_dirac_half(eta + step))
        - math.log(fermi_dirac_half(eta - step))
    ) / (2.0 * step)
    assert fermi_dirac_half_log_derivative(eta) == pytest.approx(
        finite_difference,
        rel=2.0e-6,
        abs=1.0e-10,
    )
