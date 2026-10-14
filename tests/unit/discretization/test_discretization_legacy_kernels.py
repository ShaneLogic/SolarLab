"""Historical Bernoulli, Scharfetter-Gummel, face-mean and divergence kernels.

M class: every kernel here is compared bit-for-bit with the legacy engine
(read-only import). The N-class stable evaluations have their own test file.
"""

from __future__ import annotations

import numpy as np
import pytest

from perovskite_sim.discretization import fe_operators as legacy
from perovskite_sim.physics import physical_control_volume as legacy_physical
from perovskite_sim.solver import mol as legacy_mol
from perovskite_sim.twod import poisson_2d as legacy_poisson_2d

from solarlab.discretization import (
    arithmetic_face_mean, bernoulli, bernoulli_derivative, flux_divergence,
    guarded_harmonic_face_mean, harmonic_face_mean, harmonic_mean, legacy_sg_current_n,
    legacy_sg_current_n_jacobian, legacy_sg_current_p, legacy_sg_current_p_jacobian,
    series_face_capacitance,
)
from solarlab.discretization.face_average import VACUUM_PERMITTIVITY_F_M

SPECIAL = np.array([0.0, 5e-9, 1e-8, 2e-8, 9.99e-5, 1e-4, 1.01e-4, 1e-3, 0.1, 0.5, 1.0, 2.0,
                    10.0, 36.0, 49.9, 50.0, 50.1, 100.0, 300.0, 699.0, 700.0, 701.0, 745.0])
ARGUMENTS = np.concatenate([SPECIAL, -SPECIAL, np.geomspace(1e-10, 740.0, 300),
                            -np.geomspace(1e-10, 740.0, 300)])


def same_bits(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def test_historical_bernoulli_value_and_derivative_are_bit_identical():
    assert same_bits(bernoulli(ARGUMENTS), legacy.bernoulli(ARGUMENTS))
    assert same_bits(bernoulli_derivative(ARGUMENTS), legacy.bernoulli_derivative(ARGUMENTS))


def test_historical_sg_currents_and_jacobians_are_bit_identical():
    rng = np.random.default_rng(7)
    phi = np.cumsum(rng.normal(scale=0.08, size=12))
    n = 10.0 ** rng.uniform(14, 24, size=12)
    dx = rng.uniform(1e-9, 5e-8, size=11)
    D = rng.uniform(1e-6, 1e-3, size=11)
    vt = 0.025851999786435535
    assert same_bits(legacy_sg_current_n(phi, n, dx, D, vt), legacy.sg_fluxes_n(phi, n, dx, D, vt))
    assert same_bits(legacy_sg_current_p(phi, n, dx, D, vt), legacy.sg_fluxes_p(phi, n, dx, D, vt))
    for new, old in ((legacy_sg_current_n_jacobian(phi, n, dx, D, vt),
                      legacy.sg_fluxes_n_jacobian(phi, n, dx, D, vt)),
                     (legacy_sg_current_p_jacobian(phi, n, dx, 2e-4, vt),
                      legacy.sg_fluxes_p_jacobian(phi, n, dx, 2e-4, vt))):
        for name in ("flux", "density_left_derivative", "density_right_derivative",
                     "potential_left_derivative", "potential_right_derivative"):
            assert same_bits(getattr(new, name), getattr(old, name)), name
    with pytest.raises(ValueError, match="positive spacing"):
        legacy_sg_current_n_jacobian(phi, n, -dx, D, vt)


def test_face_means_and_divergence_match_their_historical_expressions():
    rng = np.random.default_rng(3)
    values = rng.uniform(1.0, 30.0, 9)
    assert same_bits(harmonic_face_mean(values),
                     2.0 * values[:-1] * values[1:] / (values[:-1] + values[1:]))
    assert same_bits(harmonic_mean(values[:-1], values[1:]),
                     legacy_poisson_2d._harmonic_mean(values[:-1], values[1:]))
    with_zero = np.array([0.0, 0.0, 2.0, 0.0, 3.0])
    assert same_bits(guarded_harmonic_face_mean(with_zero),
                     legacy_mol._harmonic_face_average(with_zero))
    assert np.array_equal(guarded_harmonic_face_mean(with_zero)[:2], [0.0, 0.0])
    assert same_bits(arithmetic_face_mean(values), 0.5 * (values[:-1] + values[1:]))
    flux, widths = rng.normal(size=8), rng.uniform(0.1, 1.0, 9)
    assert same_bits(flux_divergence(flux, widths, left_flux=0.3, right_flux=-0.2),
                     legacy_physical.flux_divergence(flux, widths, left_flux=0.3, right_flux=-0.2))
    with pytest.raises(ValueError, match="must align"):
        flux_divergence(flux, widths[:-1])


def test_series_capacitance_reduces_to_the_harmonic_mean_at_the_face_midpoint():
    h, eps_l, eps_r = 4.0e-9, 3.0, 24.0
    series = series_face_capacitance(h / 2, eps_l, h / 2, eps_r)
    harmonic = VACUUM_PERMITTIVITY_F_M * harmonic_mean(eps_l, eps_r) / h
    assert series == pytest.approx(float(harmonic), rel=4e-16)
    off_midpoint = series_face_capacitance(h / 4, eps_l, 3 * h / 4, eps_r)
    assert off_midpoint == pytest.approx(
        VACUUM_PERMITTIVITY_F_M / (h / 4 / eps_l + 3 * h / 4 / eps_r), rel=4e-16)
    with pytest.raises(ValueError, match="positive"):
        series_face_capacitance(0.0, eps_l, h, eps_r)
