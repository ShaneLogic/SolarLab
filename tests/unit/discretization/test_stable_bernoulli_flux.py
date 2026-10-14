"""Stable paired Bernoulli evaluation and unified SG particle flux (N class).

Checked against an independent 80-digit Decimal oracle; the difference from the
historical element-wise values and electrical currents is bounded explicitly.
"""

from __future__ import annotations

from decimal import Decimal, localcontext

import numpy as np
import pytest

from perovskite_sim.discretization import fe_operators as legacy

from solarlab.discretization.particle_flux import sg_particle_flux
from solarlab.discretization.stable_bernoulli import bernoulli_pair

Q = 1.602176634e-19
TINY = np.finfo(float).tiny
SPECIAL = np.array([0.0, 5e-9, 1e-8, 2e-8, 9.99e-5, 1e-4, 1.01e-4, 1e-3, 0.1, 0.5, 1.0, 2.0,
                    10.0, 36.0, 49.9, 50.0, 50.1, 100.0, 300.0, 699.0, 700.0, 701.0, 745.0])
ARGUMENTS = np.concatenate([SPECIAL, -SPECIAL, np.geomspace(1e-10, 740.0, 300),
                            -np.geomspace(1e-10, 740.0, 300)])


def same_bits(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def decimal_bernoulli(x):
    """Independent oracle: B(x) and B'(x) at the exact binary64 argument."""
    with localcontext() as context:
        context.prec = 80
        value = Decimal(float(x))
        if value == 0:
            return Decimal(1), Decimal(-1) / 2
        e = value.exp()
        t = e - 1
        return value / t, (t - value * e) / (t * t)


def relative_error(computed, reference):
    with localcontext() as context:
        context.prec = 80
        return float(abs((Decimal(float(computed)) - reference) / reference))


@pytest.mark.parametrize("x", ARGUMENTS[::7])
def test_bernoulli_pair_matches_the_decimal_oracle_in_both_directions(x):
    pair = bernoulli_pair(np.array([x]))
    for computed, argument, index in ((pair.forward[0], x, 0), (pair.backward[0], -x, 0),
                                      (pair.forward_derivative[0], x, 1),
                                      (pair.backward_derivative[0], -x, 1)):
        reference = decimal_bernoulli(argument)[index]
        if abs(reference) >= Decimal(TINY):
            assert relative_error(computed, reference) <= 8e-16, (x, index)
        else:  # subnormal small branch: absolute accuracy at the subnormal spacing
            assert abs(Decimal(float(computed)) - reference) <= Decimal(4 * 5e-324)


def test_small_branch_first_keeps_b_of_50_where_large_branch_subtraction_loses_it():
    x = np.array([50.0, 100.0, 300.0, 700.0])
    pair = bernoulli_pair(x)
    # Differencing the large branch cannot recover the small one in binary64...
    assert np.all(legacy.bernoulli(-x) - x == 0.0)
    # ...while the pair evaluates it first and keeps its normal-range value.
    for value, argument in zip(pair.forward, x):
        assert value > 0.0
        assert relative_error(value, decimal_bernoulli(argument)[0]) <= 6e-16
    # The large branch is recovered by one stable addition of positives.
    assert np.array_equal(pair.backward, x + pair.forward)


def test_bernoulli_pair_parity_identities_and_direction_swap():
    pair = bernoulli_pair(ARGUMENTS)
    swapped = bernoulli_pair(-ARGUMENTS)
    assert same_bits(swapped.forward, pair.backward)
    assert same_bits(swapped.backward, pair.forward)
    assert same_bits(swapped.forward_derivative, pair.backward_derivative)
    # B(-x) - B(x) = x up to the rounding of the large branch.
    gap = np.abs((pair.backward - pair.forward) - ARGUMENTS)
    assert np.all(gap <= np.spacing(np.maximum(pair.backward, pair.forward)))
    # B'(-x) = -1 - B'(x), exactly as constructed on the large branch.
    large = ARGUMENTS < 0
    assert np.array_equal(pair.forward_derivative[large], -(1.0 + pair.backward_derivative[large]))
    assert pair.forward[ARGUMENTS == 0][0] == 1.0
    assert pair.forward_derivative[ARGUMENTS == 0][0] == -0.5
    with pytest.raises(ValueError, match="finite"):
        bernoulli_pair(np.array([np.inf]))


def test_bernoulli_pair_differs_from_the_historical_values_only_within_n_class_bounds():
    """N-class comparison: record the actual deviation from the legacy evaluation."""
    pair = bernoulli_pair(ARGUMENTS)
    old_forward, old_backward = legacy.bernoulli(ARGUMENTS), legacy.bernoulli(-ARGUMENTS)
    cut = ARGUMENTS > 700.0  # historical value is exactly 0 there
    assert np.all(old_forward[cut] == 0.0) and np.all(pair.forward[cut] > 0.0)
    keep = ~cut & (np.abs(old_forward) >= TINY)
    ulps = np.abs(pair.forward[keep] - old_forward[keep]) / np.spacing(np.abs(old_forward[keep]))
    assert ulps.max() <= 4.0
    keep_b = ~(-ARGUMENTS > 700.0) & (np.abs(old_backward) >= TINY)
    ulps_b = (np.abs(pair.backward[keep_b] - old_backward[keep_b])
              / np.spacing(np.abs(old_backward[keep_b])))
    assert ulps_b.max() <= 4.0
    old_derivative = legacy.bernoulli_derivative(ARGUMENTS)
    relative = (np.abs(pair.forward_derivative - old_derivative)
                / np.maximum(np.abs(old_derivative), TINY))
    normal = np.abs(old_derivative) >= TINY
    # The historical quotient loses ~1e-12 near |x| = 1e-4; the pair does not.
    assert relative[normal].max() <= 4e-12


def test_bernoulli_derivative_matches_a_centered_difference_of_the_pair():
    for x in (-30.0, -3.0, -0.2, -1e-3, 1e-3, 0.2, 3.0, 30.0):
        h = 1e-6 * max(1.0, abs(x))
        pair = bernoulli_pair(np.array([x - h, x, x + h]))
        central = (pair.forward[2] - pair.forward[0]) / (2 * h)
        assert central == pytest.approx(pair.forward_derivative[1], rel=1e-7, abs=1e-12)


def test_unified_particle_flux_limits_signs_and_legacy_currents():
    vt = 0.025851999786435535
    rng = np.random.default_rng(11)
    left, right = 10.0 ** rng.uniform(15, 23, 40), 10.0 ** rng.uniform(15, 23, 40)
    step = rng.uniform(-60, 60, 40)
    k = rng.uniform(1e-2, 1e3, 40)
    flux = sg_particle_flux(left, right, step, k)
    # Zero field: pure diffusion, exactly k*(c_L - c_R).
    zero = sg_particle_flux(left, right, np.zeros(40), k)
    assert np.array_equal(zero.flux, k * (left - right))
    # Reversing the face orientation reverses the flux exactly.
    reverse = sg_particle_flux(right, left, -step, k)
    assert np.array_equal(reverse.flux, -flux.flux)
    # Thermal equilibrium c_R = c_L*exp(-D) carries no net flux.
    equilibrium = sg_particle_flux(left, left * np.exp(-step / 4), step / 4, k)
    scale = k * (bernoulli_pair(step / 4).forward * left)
    assert np.all(np.abs(equilibrium.flux) <= 1e-13 * scale)
    # Unified fluxes against the historical electrical currents (N class).
    phi = np.cumsum(rng.normal(scale=0.1, size=41))
    n, dx, D = 10.0 ** rng.uniform(15, 23, 41), rng.uniform(1e-9, 1e-8, 40), 1e-4
    xi = (phi[1:] - phi[:-1]) / vt
    electrons = sg_particle_flux(n[:-1], n[1:], -xi, D / dx).flux
    holes = sg_particle_flux(n[:-1], n[1:], xi, D / dx).flux
    j_n, j_p = legacy.sg_fluxes_n(phi, n, dx, D, vt), legacy.sg_fluxes_p(phi, n, dx, D, vt)
    magnitude_n = Q * D / dx * (legacy.bernoulli(xi) * n[1:] + legacy.bernoulli(-xi) * n[:-1])
    magnitude_p = Q * D / dx * (legacy.bernoulli(xi) * n[:-1] + legacy.bernoulli(-xi) * n[1:])
    assert np.all(np.abs(-Q * electrons - j_n) <= 1e-14 * magnitude_n)
    assert np.all(np.abs(Q * holes - j_p) <= 1e-14 * magnitude_p)


def test_unified_particle_flux_derivatives_against_a_decimal_oracle():
    left, right, step, k = 3.0e20, 7.0e21, 2.5, 40.0
    flux = sg_particle_flux(np.array([left]), np.array([right]), np.array([step]), k)
    b_plus, d_plus = decimal_bernoulli(step)
    b_minus, d_minus = decimal_bernoulli(-step)
    with localcontext() as context:
        context.prec = 80
        L, R, K = Decimal(left), Decimal(right), Decimal(k)
        expected = {
            "flux": K * (b_plus * L - b_minus * R),
            "left_density_derivative": K * b_plus,
            "right_density_derivative": -K * b_minus,
            "energy_step_derivative": K * (d_plus * L + d_minus * R),
            "coefficient_derivative": b_plus * L - b_minus * R,
        }
    for name, reference in expected.items():
        assert relative_error(getattr(flux, name)[0], reference) <= 1e-14, name
    with pytest.raises(ValueError, match="non-negative"):
        sg_particle_flux(np.ones(1), np.ones(1), np.zeros(1), -1.0)
    with pytest.raises(ValueError, match="one face shape"):
        sg_particle_flux(np.ones(2), np.ones(1), np.zeros(2), 1.0)
